import pytest

from lunar_pressure.gemini_gauge_reader import (
    GaugeReaderError,
    parse_gauge_reading,
    render_prompt_template,
)


def test_parse_strict_gemini_json():
    reading = parse_gauge_reading(
        """
        {
          "line_id": "Line-A",
          "gauge_id": "G1",
          "value_mpa": 0.11,
          "confidence": 0.9,
          "raw_text": "needle near 0.11",
          "need_retry": false,
          "risk_flags": []
        }
        """
    )

    assert reading.value_mpa == 0.11
    assert reading.is_usable(0.8)


def test_parse_json_fence_and_reject_bad_schema():
    reading = parse_gauge_reading(
        """```json
        {"line_id":"Line-A","gauge_id":"G1","value_mpa":0.08,"confidence":0.81,"raw_text":"","need_retry":false,"risk_flags":[]}
        ```"""
    )

    assert reading.value_mpa == 0.08

    with pytest.raises(GaugeReaderError):
        parse_gauge_reading('{"value_mpa": 0.1, "confidence": 2.0}')


def test_parse_failure_carries_raw_response():
    """GaugeReaderError.raw_response must equal the input text that failed to parse."""
    bad_text = '{"value_mpa": 0.1, "confidence": 2.0}'
    with pytest.raises(GaugeReaderError) as exc_info:
        parse_gauge_reading(bad_text)
    assert exc_info.value.raw_response == bad_text


def test_prompt_template_preserves_json_braces():
    template = '{\n  "line_id": "{line_id}",\n  "value_mpa": <number>\n}\nrange {min} to {max}'

    rendered = render_prompt_template(
        template,
        {"line_id": "Line-A", "min": 0.0, "max": 0.2},
    )

    assert '"line_id": "Line-A"' in rendered
    assert "range 0.0 to 0.2" in rendered
    assert rendered.strip().startswith("{")
    assert rendered.splitlines()[2].strip() == '"value_mpa": <number>'
