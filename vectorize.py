import re

import cv2
import numpy as np
import vtracer

from curves import refit
from preprocess import MAX_SIDE, UPSCALE


def lerp(low: float, high: float, detail: int) -> float:
    """Map detail 1..10 onto low..high."""
    return low + (high - low) * (min(max(detail, 1), 10) - 1) / 9


def trace(image: np.ndarray, speckle: int = 4) -> str:
    """Trace a flat-color image to a raw SVG with many short Bezier segments."""
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("cannot encode image to PNG")
    return vtracer.convert_raw_image_to_svg(
        png.tobytes(),
        img_format="png",
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=speckle * UPSCALE,
        color_precision=8,
        layer_difference=1,
        corner_threshold=60,
        length_threshold=4.0,
        splice_threshold=45,
        max_iterations=10,
        path_precision=2,
    )


def finish(raw: str, image: np.ndarray, detail: int = 4) -> str:
    """Refit the traced outlines. Lower `detail` gives fewer Bezier points and rounder curves."""
    height, width = image.shape[:2]
    # The pixel values below suit the largest images. Smaller images get proportionally less
    # smoothing, otherwise details only a few pixels wide (e.g. claws) are rounded away.
    scale = min(1.0, max(height, width) / (MAX_SIDE * UPSCALE))
    svg = refit(
        raw,
        error=6.0 * 0.1 ** ((min(max(detail, 1), 10) - 1) / 9) * scale,  # 6 px down to 0.6 px, geometric steps
        corner_angle=lerp(115, 50, detail),
        sigma=lerp(4.0, 0.5, detail) * scale,
    )
    background = ""
    if image[:, :, 3].min() == 255:
        # Smooth curves also round off the image frame, so paint the border color behind everything.
        background = '<rect width="100%" height="100%" fill="#{2:02x}{1:02x}{0:02x}"/>'.format(*border_color(image))
    return re.sub(
        r'width="\d+" height="\d+">',
        f'width="{width // UPSCALE}" height="{height // UPSCALE}" viewBox="0 0 {width} {height}">\n{background}',
        svg,
        count=1,
    )


def vectorize(image: np.ndarray, detail: int = 4, speckle: int = 4) -> str:
    """Trace a flat-color image to SVG. Higher `detail` keeps more corners and Bezier points."""
    return finish(trace(image, speckle), image, detail)


def border_color(image: np.ndarray) -> np.ndarray:
    """Most common BGR color along the image edges."""
    bgr = image[:, :, :3]
    border = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]])
    colors, counts = np.unique(border, axis=0, return_counts=True)
    return colors[np.argmax(counts)]


def stats(svg: str) -> dict:
    return {
        "paths": svg.count("<path"),
        "curves": len(re.findall(r"C", svg)),
        "bytes": len(svg.encode("utf-8")),
    }
