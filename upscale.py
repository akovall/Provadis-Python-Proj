"""Optional AI upscaling of small drawings before vectorizing (Real-ESRGAN animevideo v3 via ONNX Runtime).

The model makes edges sharp and colors flat, so the tracer gets clean outlines instead of blurry ones.
"""

from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime
except ImportError:  # the feature is optional, everything else works without it
    onnxruntime = None

MODEL = Path(__file__).parent / "models" / "realesr-animevideov3.onnx"
SCALE = 4
MAX_INPUT_SIDE = 1024  # bigger images already have enough pixels, and the model gets slow on them

session = None


def skip_reason(image: np.ndarray) -> str | None:
    """Why the image cannot be upscaled, or None if it can."""
    if onnxruntime is None:
        return "onnxruntime is not installed"
    if not MODEL.is_file():
        return f"model file {MODEL.name} is missing"
    if max(image.shape[:2]) > MAX_INPUT_SIDE:
        return f"the image is larger than {MAX_INPUT_SIDE} px"
    return None


def upscale(image: np.ndarray) -> np.ndarray:
    """Upscale a BGRA image 4x. The model only sees the colors; the alpha channel is resized normally."""
    global session
    if session is None:
        session = onnxruntime.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    rgb = cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]), cv2.COLOR_BGR2RGB)
    output = session.run(None, {"input": rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255})[0][0]
    bgr = cv2.cvtColor(np.clip(output.transpose(1, 2, 0) * 255 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    alpha = cv2.resize(image[:, :, 3], (width * SCALE, height * SCALE), interpolation=cv2.INTER_CUBIC)
    return np.dstack([bgr, alpha])
