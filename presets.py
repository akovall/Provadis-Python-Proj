"""Ready-made settings for common kinds of images, and a guess which one fits an image."""

import cv2
import numpy as np

PRESETS = {
    "logo": {"colors": 8, "merge": 12.0, "detail": 6, "speckle": 4, "remove_background": True},
    "cartoon": {"colors": 16, "merge": 8.0, "detail": 4, "speckle": 4, "remove_background": True},
    "photo": {"colors": 32, "merge": 4.0, "detail": 5, "speckle": 2, "remove_background": False},
}
SAMPLE_SIDE = 256
FLAT_GRADIENT = 8  # Sobel magnitude below this counts as a flat area
PHOTO_FLAT_SHARE = 0.3  # graphics are mostly flat areas, photos almost never
LOGO_COLORS = 12  # a logo is almost entirely covered by this many colors


def detect(image: np.ndarray) -> str:
    """Guess the preset for a BGRA image: photos have few flat areas, logos have few colors."""
    alpha = image[:, :, 3:4] / 255
    bgr = (image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    height, width = bgr.shape[:2]
    scale = SAMPLE_SIDE / max(height, width)
    small = cv2.resize(bgr, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    if (gradient < FLAT_GRADIENT).mean() < PHOTO_FLAT_SHARE:
        return "photo"
    # 4 bits per channel, so anti-aliased edges and noise do not count as extra colors
    codes = (small >> 4).reshape(-1, 3).astype(np.int32) @ np.array([256, 16, 1])
    counts = np.sort(np.bincount(codes))[::-1]
    colors = np.searchsorted(np.cumsum(counts), 0.95 * counts.sum()) + 1
    return "logo" if colors <= LOGO_COLORS else "cartoon"
