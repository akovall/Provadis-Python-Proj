"""Split a flat-color image into stacked layers, from the biggest shape at the bottom to the smallest on top.

Every connected area of one color is a layer. Where smaller layers (drawn later, on top) cover
part of a bigger one, the bigger one is not cut out: its holes and the notches made by the
shapes on top are filled in, because nobody sees them. So a face stays one whole shape with the
eyes lying on it, instead of a face with eye-shaped holes.
"""

import cv2
import numpy as np

from preprocess import UPSCALE, give_to_neighbors, to_lab

FILL_IN = 0.15  # notches up to this share of a layer's size (sqrt of its area) are filled in under later layers
MAX_FILL_IN = 40  # but never wider than this (in traced pixels)
CLOSING_STEPS = 6  # filling in is done on a reduced copy with a kernel of about this radius, for speed
OVERLAP = 4  # every layer reaches this far under the layers on top of it, so no gaps open between them
FRINGE_WIDTH = 12  # anti-aliasing fringes are thinner than this (in traced pixels) ...
FRINGE_AREA = 20000  # ... and smaller than this
FRINGE_NEIGHBOR = 5  # ... and lie between two areas at least this many times bigger
FRINGE_BLEND = 10  # ... with a color within this ΔE of a blend of those two


def regions(image: np.ndarray, speckle: int) -> tuple[np.ndarray, list[dict]]:
    """Connected areas of one color, with specks given to their neighbors first.

    Returns an id map (-1 for transparent) and the areas, biggest first.
    """
    opaque = image[:, :, 3] == 255
    if speckle > 0:
        min_area = (speckle * UPSCALE) ** 2
        specks = np.zeros(opaque.shape, bool)
        for color, ids, boxes in components(image, opaque):
            specks |= (boxes[:, 4] < min_area)[ids] & (ids > 0)
        if specks.any():
            image = merge_specks(image, specks)
    image = merge_fringes(image, opaque)
    ids_map = np.full(opaque.shape, -1, np.int32)
    found = []
    for color, ids, boxes in components(image, opaque):
        for index in range(1, len(boxes)):
            x, y, w, h, area = boxes[index]
            ids_map[y:y + h, x:x + w][ids[y:y + h, x:x + w] == index] = len(found)
            found.append({"color": color, "box": (x, y, w, h), "area": int(area)})
    order = np.argsort([-r["area"] for r in found], kind="stable")
    rank = np.empty(len(found), np.int32)
    rank[order] = np.arange(len(found))
    ranked = np.where(ids_map >= 0, rank[np.maximum(ids_map, 0)], -1)
    return ranked, [found[i] for i in order]


def merge_fringes(image: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """Split thin anti-aliasing fringes (a mid color between two big areas) between those two areas.

    Drawn as their own small layers they would lie on top of the edge and make it look bumpy.
    A piece of a thin line is not a fringe: its neighbors are not both much bigger than itself.
    """
    pieces = list(components(image, opaque))
    area_map = np.zeros(opaque.shape, np.int64)
    for _, ids, boxes in pieces:
        area_map[ids > 0] = boxes[ids[ids > 0], 4]
    lab = to_lab(image)
    result = image.copy()
    for color, ids, boxes in pieces:
        for index in range(1, len(boxes)):
            x, y, w, h, area = boxes[index]
            if area >= FRINGE_AREA or min(w, h) < 1:
                continue
            y0, y1, x0, x1 = max(y - 2, 0), y + h + 2, max(x - 2, 0), x + w + 2
            mask = ids[y0:y1, x0:x1] == index
            if cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 3).max() * 2 > FRINGE_WIDTH:
                continue  # too thick
            ring = (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~mask
            ring &= opaque[y0:y1, x0:x1] & (area_map[y0:y1, x0:x1] >= FRINGE_NEIGHBOR * area)
            if ring.sum() < 4:
                continue
            ring_colors = image[y0:y1, x0:x1][ring]
            values, counts = np.unique(ring_colors, axis=0, return_counts=True)
            if len(values) < 2:
                continue
            first, second = np.argsort(-counts)[:2]
            if counts[second] < 0.15 * counts.sum():
                continue  # big area on one side only
            a = to_lab(values[first].reshape(1, 1, 4))[0, 0]
            b = to_lab(values[second].reshape(1, 1, 4))[0, 0]
            own = lab[y0:y1, x0:x1][mask].mean(axis=0)
            segment = b - a
            t = float(np.dot(own - a, segment) / max(np.dot(segment, segment), 1e-9))
            if not 0.1 < t < 0.9 or np.linalg.norm(a + t * segment - own) > FRINGE_BLEND:
                continue  # not a blend of the two
            # Each pixel goes to the side whose color it is closer to, so the edge stays in the middle.
            pixels = lab[y0:y1, x0:x1][mask]
            to_a = np.linalg.norm(pixels - a, axis=1) <= np.linalg.norm(pixels - b, axis=1)
            crop = result[y0:y1, x0:x1]
            crop[mask] = np.where(to_a[:, None], values[first], values[second])
    return result


def merge_specks(image: np.ndarray, specks: np.ndarray) -> np.ndarray:
    """Give each speck the most similar color touching it (not just the nearest pixel's color),
    so a piece of a faint line joins the rest of the line instead of the background."""
    count, ids = cv2.connectedComponents(specks.astype(np.uint8), connectivity=8)
    around = cv2.dilate(ids.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
    ring = (around > 0) & ~specks & (image[:, :, 3] == 255)
    owner = around[ring]
    lab = to_lab(image).reshape(-1, 3)
    speck_lab = np.zeros((count, 3), np.float32)
    np.add.at(speck_lab, ids[specks], lab[specks.ravel()])
    speck_lab /= np.maximum(np.bincount(ids[specks], minlength=count), 1)[:, None]
    distance = np.linalg.norm(lab[ring.ravel()] - speck_lab[owner], axis=1)
    best = np.full(count, np.inf, np.float32)
    np.minimum.at(best, owner, distance)
    choice = np.full(count, -1, np.int64)
    winners = np.nonzero(distance <= best[owner])[0]
    choice[owner[winners]] = np.flatnonzero(ring.ravel())[winners]
    result = give_to_neighbors(image, specks)  # fallback for specks that only touch other specks
    chosen = specks & (choice[ids] >= 0)
    result.reshape(-1, image.shape[2])[chosen.ravel()] = image.reshape(-1, image.shape[2])[choice[ids[chosen]]]
    return result


def components(image: np.ndarray, opaque: np.ndarray):
    bgr = np.ascontiguousarray(image[:, :, :3])
    packed = (bgr[:, :, 0].astype(np.int32) << 16) | (bgr[:, :, 1].astype(np.int32) << 8) | bgr[:, :, 2]
    packed[~opaque] = -1
    for color in np.unique(packed):
        if color < 0:
            continue
        _, ids, boxes, _ = cv2.connectedComponentsWithStats((packed == color).astype(np.uint8), connectivity=8)
        yield [(color >> 16) & 255, (color >> 8) & 255, color & 255], ids, boxes


def fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return filled > 0


def close(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological closing (fills notches up to `radius`), done on a reduced copy for speed."""
    factor = max(1, radius // CLOSING_STEPS)
    small = cv2.resize(mask.astype(np.uint8), (mask.shape[1] // factor + 1, mask.shape[0] // factor + 1),
                       interpolation=cv2.INTER_AREA)
    steps = max(1, radius // factor)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * steps + 1, 2 * steps + 1))
    closed = cv2.morphologyEx(np.pad(small, steps), cv2.MORPH_CLOSE, kernel)[steps:-steps, steps:-steps]
    return cv2.resize(closed, mask.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0


def build(image: np.ndarray, over: np.ndarray | None = None, speckle: int = 4) -> list[dict]:
    """Layers from bottom (biggest) to top (smallest).

    `over` marks pixels covered by something drawn above all layers (the outlines), so layers
    may extend under them too. Each layer has its color, its position (x, y), its own pixels
    (`region`), the pixels it may cover because later layers hide them (`free`), and the filled-in
    shape to draw (`shape`), all cropped to the layer's box.
    """
    ranked, found = regions(image, speckle)
    if over is not None:
        ranked = np.where(over & (ranked >= 0), len(found), ranked)  # outlines count as the very last layer
    layers = []
    height, width = ranked.shape
    for index, layer in enumerate(found):
        x, y, w, h = layer["box"]
        radius = int(min(MAX_FILL_IN, max(2, FILL_IN * np.sqrt(layer["area"]))))
        # Room around the layer, so a form continuing under the layers on top can still be checked.
        margin = max(radius, max(w, h) // 3)
        x0, y0 = max(x - margin, 0), max(y - margin, 0)
        x1, y1 = min(x + w + margin, width), min(y + h + margin, height)
        crop = ranked[y0:y1, x0:x1]
        region = crop == index
        free = crop > index  # drawn later, on top: anything may be underneath
        reach = cv2.dilate(region.astype(np.uint8), np.ones((2 * OVERLAP + 1,) * 2, np.uint8)) > 0
        shape = region | ((fill_holes(close(region, radius)) | reach) & free)
        layers.append({"color": layer["color"], "x": x0, "y": y0, "region": region, "free": free, "shape": shape})
    return layers


def contours(layer: dict) -> list[np.ndarray]:
    """Outer edges and holes of a layer's shape, as (x, y) loops in image pixels."""
    found, _ = cv2.findContours(layer["shape"].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    offset = np.array([layer["x"], layer["y"]], float) + 0.5
    return [c[:, 0].astype(float) + offset for c in found if len(c) >= 8]
