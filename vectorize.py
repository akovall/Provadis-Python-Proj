import re

import cv2
import numpy as np
import vtracer

from preprocess import UPSCALE


def lerp(low: float, high: float, detail: int) -> float:
    """Map detail 1..10 onto low..high."""
    return low + (high - low) * (min(max(detail, 1), 10) - 1) / 9


def vectorize(image: np.ndarray, detail: int = 4, speckle: int = 4) -> str:
    """Trace a flat-color image to SVG. Higher `detail` keeps more corners and Bezier points."""
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("cannot encode image to PNG")
    svg = vtracer.convert_raw_image_to_svg(
        png.tobytes(),
        img_format="png",
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=speckle * UPSCALE,
        color_precision=8,
        layer_difference=1,
        corner_threshold=round(lerp(150, 45, detail)),
        length_threshold=lerp(10, 3.5, detail),
        splice_threshold=round(lerp(90, 30, detail)),
        max_iterations=10,
        path_precision=2,
    )
    height, width = image.shape[:2]
    # Smooth corner settings also round off the image frame, so paint the border color behind everything.
    background = '<rect width="100%" height="100%" fill="#{2:02x}{1:02x}{0:02x}"/>'.format(*border_color(image))
    return re.sub(
        r'width="\d+" height="\d+">',
        f'width="{width // UPSCALE}" height="{height // UPSCALE}" viewBox="0 0 {width} {height}">\n{background}',
        svg,
        count=1,
    )


def border_color(image: np.ndarray) -> np.ndarray:
    """Most common BGR color along the image edges."""
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    colors, counts = np.unique(border, axis=0, return_counts=True)
    return colors[np.argmax(counts)]


def stats(svg: str) -> dict:
    return {
        "paths": svg.count("<path"),
        "curves": len(re.findall(r"C", svg)),
        "bytes": len(svg.encode("utf-8")),
    }
