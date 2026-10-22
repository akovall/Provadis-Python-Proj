"""Recognize layers that are simple forms (ellipses, circles, rectangles) and draw them as exact SVG shapes.

A traced ellipse is a path whose outline wobbles a little. When a layer matches an ellipse or a
rectangle almost exactly, it is drawn as a real <ellipse>, <circle> or <rect> instead: perfectly
smooth and without Bezier points. Only the visible part of a layer has to match; where later
layers lie on top (a cheek partly behind an eye), the form may continue underneath.
"""

import cv2
import numpy as np

MATCH = 0.95  # overlap (intersection over union) the visible part needs with the form
MIN_AREA = 400  # smaller layers are traced (in traced pixels)
MAX_HOLES = 0.02  # holes showing what is underneath cannot be part of a simple form
ROUND = 0.97  # an ellipse with axes this close to equal is drawn as a circle
MIN_EDGE = 20  # a partly hidden layer needs at least this many visible edge pixels to fit a form


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)


def moment_ellipse(mask: np.ndarray) -> dict:
    """The ellipse with the same area and spread as the region."""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    spread = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / m["m00"]
    values, vectors = np.linalg.eigh(spread)
    return {
        "kind": "ellipse", "cx": m["m10"] / m["m00"], "cy": m["m01"] / m["m00"],
        "rx": 2 * np.sqrt(max(values[1], 0)), "ry": 2 * np.sqrt(max(values[0], 0)),
        "angle": float(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))),
    }


def edge_ellipse(points: np.ndarray) -> dict:
    """The ellipse through the given edge points (for a layer partly hidden behind others)."""
    (cx, cy), (width, height), angle = cv2.fitEllipse(points.astype(np.float32))
    return {"kind": "ellipse", "cx": cx, "cy": cy, "rx": width / 2, "ry": height / 2, "angle": angle}


def rectangle(points: np.ndarray) -> dict:
    (cx, cy), (w, h), angle = cv2.minAreaRect(points.astype(np.float32))
    return {"kind": "rect", "cx": cx, "cy": cy, "w": w, "h": h, "angle": angle}


def draw(shape: dict, size: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(size, np.uint8)
    if shape["kind"] == "ellipse":
        cv2.ellipse(canvas, ((shape["cx"], shape["cy"]), (2 * shape["rx"], 2 * shape["ry"]), shape["angle"]), 1, -1)
    else:
        box = cv2.boxPoints(((shape["cx"], shape["cy"]), (shape["w"], shape["h"]), shape["angle"]))
        cv2.fillPoly(canvas, [np.round(box).astype(np.int32)], 1)
    return canvas > 0


def inside(shape: dict, size: tuple[int, int]) -> bool:
    """Whether the form lies within the area it was checked in."""
    if shape["kind"] == "ellipse":
        corners = cv2.ellipse2Poly(
            (round(shape["cx"]), round(shape["cy"])), (round(shape["rx"]), round(shape["ry"])),
            round(shape["angle"]), 0, 360, 5,
        ).astype(float)
    else:
        corners = cv2.boxPoints(((shape["cx"], shape["cy"]), (shape["w"], shape["h"]), shape["angle"]))
    height, width = size
    return bool((corners[:, 0] >= -1).all() and (corners[:, 0] <= width + 1).all()
                and (corners[:, 1] >= -1).all() and (corners[:, 1] <= height + 1).all())


def fit(layer: dict) -> dict | None:
    """The simple form matching this layer, in image pixels, or None."""
    region, free = layer["region"], layer["free"]
    if region.sum() < MIN_AREA:
        return None
    holes = layer["shape"] & ~region & ~free
    if holes.sum() > MAX_HOLES * region.sum():
        return None
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = max(contours, key=len)[:, 0]
    near_free = cv2.dilate(free.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hidden = near_free[edge[:, 1], edge[:, 0]]
    visible_edge = edge[~hidden]
    if len(visible_edge) < MIN_EDGE:
        return None
    candidates = [edge_ellipse(visible_edge) if hidden.any() else moment_ellipse(region), rectangle(visible_edge)]
    seen = ~free  # under later layers anything goes
    best, score = None, 0.0
    for shape in candidates:
        if not inside(shape, region.shape):
            continue  # the part outside could not be checked against the picture
        value = overlap(draw(shape, region.shape) & seen, region & seen)
        if value > score:
            best, score = shape, value
    if score < MATCH:
        return None
    best["cx"] += layer["x"]
    best["cy"] += layer["y"]
    return best


def to_svg(shape: dict, color: list[int]) -> str:
    b, g, r = color
    fill = f'fill="#{r:02x}{g:02x}{b:02x}"'
    cx, cy = shape["cx"], shape["cy"]
    rotate = f' transform="rotate({shape["angle"]:.1f} {cx:.1f} {cy:.1f})"' if abs(shape["angle"]) > 0.5 else ""
    if shape["kind"] == "ellipse":
        rx, ry = shape["rx"], shape["ry"]
        if min(rx, ry) / max(rx, ry) >= ROUND:
            return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{(rx + ry) / 2:.1f}" {fill}/>'
        return f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"{rotate} {fill}/>'
    w, h = shape["w"], shape["h"]
    return f'<rect x="{cx - w / 2:.1f}" y="{cy - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}"{rotate} {fill}/>'
