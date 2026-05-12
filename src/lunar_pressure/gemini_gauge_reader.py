from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from .config import LunarPressureConfig
from .observation_contract import get_image
from .schemas import GaugeReading


class GaugeReaderError(RuntimeError):
    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


@dataclass(frozen=True)
class GaugeReadResult:
    reading: GaugeReading
    raw_response: str
    image_key: str


class GaugeReader(Protocol):
    def read(self, observation: Mapping[str, Any]) -> GaugeReadResult:
        ...


class GeminiGaugeReader:
    """Gemini-backed gauge reader.

    Gemini is intentionally confined to visual pressure reading. The returned
    JSON must validate as ``GaugeReading`` before the planner sees it.
    """

    def __init__(self, config: LunarPressureConfig):
        self.config = config
        template = Path(config.gemini_prompt_file).read_text(encoding="utf-8")
        self.prompt = render_prompt_template(
            template,
            {
                "line_id": config.line_id,
                "gauge_id": config.gauge_id,
                "visual_hard_min_mpa": config.visual_hard_min_mpa,
                "visual_hard_max_mpa": config.visual_hard_max_mpa,
                "confidence_threshold": config.gemini_confidence_threshold,
            },
        )
        self._client = None
        self._types = None

    def read(self, observation: Mapping[str, Any]) -> GaugeReadResult:
        image_key, image = get_image(
            observation,
            self.config.gemini_primary_image,
            self.config.gemini_fallback_image,
        )
        image_bytes, mime_type = image_to_bytes(image)
        raw = self._call_gemini(image_bytes, mime_type)
        reading = parse_gauge_reading(raw)
        return GaugeReadResult(reading=reading, raw_response=raw, image_key=image_key)

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GaugeReaderError(
                "google-genai is not installed. Install project dependencies before using GeminiGaugeReader."
            ) from exc
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._types = types

    def _call_gemini(self, image_bytes: bytes, mime_type: str) -> str:
        self._ensure_client()
        assert self._client is not None
        assert self._types is not None
        part = self._types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = self._client.models.generate_content(
            model=self.config.gemini_model,
            contents=[self.prompt, part],
            config=self._types.GenerateContentConfig(
                temperature=self.config.gemini_temperature,
                response_mime_type="application/json",
                system_instruction="Return only a single strict JSON object that matches the requested schema. Do not include markdown fences or any prose.",
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            raise GaugeReaderError("Gemini returned an empty response")
        return str(text)


def image_to_bytes(image: Any) -> tuple[bytes, str]:
    if isinstance(image, (bytes, bytearray)):
        raw = bytes(image)
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return raw, "image/png"
        return raw, "image/jpeg"
    if isinstance(image, str):
        path = Path(image)
        data = path.read_bytes()
        suffix = path.suffix.lower()
        if suffix == ".png":
            return data, "image/png"
        return data, "image/jpeg"

    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise GaugeReaderError(f"Expected image array rank 2 or 3, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        raise GaugeReaderError("Pillow is required to encode numpy images for Gemini") from exc

    if array.ndim == 3 and array.shape[-1] == 3:
        mode = "RGB"
    elif array.ndim == 3 and array.shape[-1] == 4:
        mode = "RGBA"
    else:
        mode = "L"

    import io

    buffer = io.BytesIO()
    pil_image = Image.fromarray(array, mode=mode)
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffer, format="JPEG")
    return buffer.getvalue(), "image/jpeg"


def parse_gauge_reading(raw_text: str) -> GaugeReading:
    payload = extract_json_object(raw_text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise GaugeReaderError(f"Gemini response was not valid JSON: {exc}", raw_response=raw_text) from exc
    try:
        return GaugeReading.model_validate(data)
    except Exception as exc:
        raise GaugeReaderError(f"Gemini JSON did not match GaugeReading schema: {exc}", raw_response=raw_text) from exc


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    raise GaugeReaderError("No JSON object found in Gemini response", raw_response=text)


def render_prompt_template(template: str, values: Mapping[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered
