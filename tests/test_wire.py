import numpy as np

from lunar_pressure.wire import MsgpackNumpyCodec


def test_wire_codec_round_trips_numpy_arrays():
    codec = MsgpackNumpyCodec()
    payload = {"array": np.arange(8, dtype=np.float32), "prompt": "hello"}

    decoded = codec.unpack(codec.pack(payload))

    np.testing.assert_allclose(decoded["array"], payload["array"])
    assert decoded["prompt"] == "hello"

