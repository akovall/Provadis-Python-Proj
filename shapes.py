"""Find regions that are simple shapes (ellipses, circles, rectangles) and draw them as exact SVG shapes.

A traced ellipse is a path whose outline wobbles a little. When a single-color region matches an
ellipse or rectangle almost exactly, it is taken out of the image before tracing and drawn as a
real <ellipse>, <circle> or <rect> instead, so it is perfectly smooth and needs no Bezier points.
"""

import cv2
import numpy as np

from preprocess import give_to_neighbors

MATCH = 0.95  # overlap (intersection over union) a region needs with the shape
MIN_AREA = 400  # smaller regions are left to the tracer (in traced pixels)
MAX_HOLES = 0.02  # regions with holes would hide what is inside them, so allow almost none
ROUND = 0.97  # an ellipse with axes this close to equal is drawn as a circle
MARGIN = 2


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)


def fit_ellipse(mask: np.ndarray) -> tuple[float, dict]:
    """The ellipse with the same area and spread as the region (from its moments)."""
    m = cv2.moments(mask, binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    spread = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / m["m00"]
    values, vectors = np.linalg.eigh(spread)
    rx, ry = 2 * np.sqrt(max(values[1], 0)), 2 * np.sqrt(max(values[0], 0))
    angle = float(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])))
    drawn = np.zeros_like(mask)
    cv2.ellipse(drawn, ((cx, cy), (2 * rx, 2 * ry), angle), 1, -1)
    return overlap(drawn, mask), {"kind": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry, "angle": angle}


def fit_rectangle(mask: np.ndarray, contour: np.ndarray) -> tuple[float, dict]:
    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    drawn = np.zeros_like(mask)
    cv2.fillPoly(drawn, [np.round(cv2.boxPoints(((cx, cy), (w, h), angle))).astype(np.int32)], 1)
    return overlap(drawn, mask), {"kind": "rect", "cx": cx, "cy": cy, "w": w, "h": h, "angle": angle}


def fit_partial_ellipse(mask: np.ndarray, hidden: np.ndarray) -> tuple[float, dict]:
    """Ellipse fitted only to the visible edge of a region that is partly behind other shapes."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=len)[:, 0]
    near_hidden = cv2.dilate(hidden.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    visible_edge = points[~near_hidden[points[:, 1], points[:, 0]]]
    if len(visible_edge) < 20:
        return 0.0, {}
    (cx, cy), (width, height), angle = cv2.fitEllipse(visible_edge.astype(np.float32))
    drawn = np.zeros_like(mask)
    cv2.ellipse(drawn, ((cx, cy), (width, height), angle), 1, -1)
    seen = ~hidden  # behind the other shape anything goes
    score = overlap(drawn.astype(bool) & seen, mask.astype(bool) & seen)
    return score, {"kind": "ellipse", "cx": cx, "cy": cy, "rx": width / 2, "ry": height / 2, "angle": angle}


def extract(image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Take simple shapes out of a flat-color BGRA image.

    Returns the image with the shapes' pixels given to the neighboring colors, and the shapes
    in image pixels in drawing order. A second pass finds shapes partly hidden behind the
    shapes of the first pass (a cheek behind an eye); those are drawn underneath them.
    """
    opaque = image[:, :, 3] == 255
    bgr = np.ascontiguousarray(image[:, :, :3])
    packed = (bgr[:, :, 0].astype(np.int32) << 16) | (bgr[:, :, 1].astype(np.int32) << 8) | bgr[:, :, 2]
    packed[~opaque] = -1
    taken = np.zeros(packed.shape, bool)
    regions_by_color = {}
    for color in np.unique(packed):
        if color >= 0:
            regions_by_color[color] = cv2.connectedComponentsWithStats((packed == color).astype(np.uint8), connectivity=8)

    def candidates():
        for color, (count, regions, boxes, _) in regions_by_color.items():
            for region in range(1, count):
                x, y, w, h, area = boxes[region]
                if area < MIN_AREA or x == 0 or y == 0 or x + w == packed.shape[1] or y + h == packed.shape[0]:
                    continue  # too small, or cut off by the image edge
                inside = regions[y:y + h, x:x + w] == region
                if taken[y:y + h, x:x + w][inside].any():
                    continue
                yield color, x, y, w, h, area, inside

    def accept(shape, color, x, y, area, inside, w, h):
        shape["cx"] += x - MARGIN
        shape["cy"] += y - MARGIN
        shape["color"] = [(color >> 16) & 255, (color >> 8) & 255, color & 255]
        shape["area"] = int(area)
        taken[y:y + h, x:x + w] |= inside
        return shape

    on_top = []
    for color, x, y, w, h, area, inside in candidates():
        mask = np.pad(inside.astype(np.uint8), MARGIN)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 1, -1)
        if 1 - area / filled.sum() > MAX_HOLES:
            continue
        score, shape = max([fit_ellipse(mask), fit_rectangle(mask, max(contours, key=len))], key=lambda c: c[0])
        if score >= MATCH:
            on_top.append(accept(shape, color, x, y, area, inside, w, h))
    if not on_top:
        return image, []

    first_pass = taken.copy()
    underneath = []
    for color, x, y, w, h, area, inside in candidates():
        y0, y1 = max(y - MARGIN, 0), min(y + h + MARGIN, packed.shape[0])
        x0, x1 = max(x - MARGIN, 0), min(x + w + MARGIN, packed.shape[1])
        hidden = np.zeros((h + 2 * MARGIN, w + 2 * MARGIN), bool)
        hidden[y0 - y + MARGIN:y1 - y + MARGIN, x0 - x + MARGIN:x1 - x + MARGIN] = first_pass[y0:y1, x0:x1]
        if not hidden.any():
            continue  # not next to a shape from the first pass
        score, shape = fit_partial_ellipse(np.pad(inside.astype(np.uint8), MARGIN), hidden)
        if score >= MATCH:
            underneath.append(accept(shape, color, x, y, area, inside, w, h))
    by_size = lambda s: -s["area"]  # noqa: E731 - larger shapes first, so smaller ones are drawn on top
    return give_to_neighbors(image, taken), sorted(underneath, key=by_size) + sorted(on_top, key=by_size)


def to_svg(shapes: list[dict]) -> str:
    elements = []
    for s in shapes:
        b, g, r = s["color"]
        fill = f'fill="#{r:02x}{g:02x}{b:02x}"'
        rotate = f' transform="rotate({s["angle"]:.1f} {s["cx"]:.1f} {s["cy"]:.1f})"' if abs(s["angle"]) > 0.5 else ""
        if s["kind"] == "ellipse" and min(s["rx"], s["ry"]) / max(s["rx"], s["ry"]) >= ROUND:
            radius = (s["rx"] + s["ry"]) / 2
            elements.append(f'<circle cx="{s["cx"]:.1f}" cy="{s["cy"]:.1f}" r="{radius:.1f}" {fill}/>')
        elif s["kind"] == "ellipse":
            elements.append(
                f'<ellipse cx="{s["cx"]:.1f}" cy="{s["cy"]:.1f}" rx="{s["rx"]:.1f}" ry="{s["ry"]:.1f}"{rotate} {fill}/>'
            )
        else:
            elements.append(
                f'<rect x="{s["cx"] - s["w"] / 2:.1f}" y="{s["cy"] - s["h"] / 2:.1f}" '
                f'width="{s["w"]:.1f}" height="{s["h"]:.1f}"{rotate} {fill}/>'
            )
    return "\n".join(elements)
