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
MAX_REGIONS = 1000  # images with more areas than this (busy photos) take too long to split into layers


def count_regions(image: np.ndarray, speckle: int) -> int:
    """About how many layers `build` would make: the areas of one color bigger than a speck."""
    min_area = (speckle * UPSCALE) ** 2
    return sum(int((boxes[1:, 4] >= min_area).sum()) for _, _, boxes in components(image, image[:, :, 3] == 255))


def check_complexity(image: np.ndarray, speckle: int) -> None:
    """Refuse images that would give too many layers, before any of the slow steps run."""
    count = count_regions(image, speckle)
    if count > MAX_REGIONS:
        raise ValueError(
            f"the image is too detailed to split into layers ({count} areas, at most {MAX_REGIONS}); "
            "try fewer colors or remove bigger specks"
        )


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


def overlaps(a: dict, b: dict) -> bool:
    """Whether two layers' shapes (cropped masks at x, y) share any pixel."""
    ah, aw = a["shape"].shape
    bh, bw = b["shape"].shape
    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x1, y1 = min(a["x"] + aw, b["x"] + bw), min(a["y"] + ah, b["y"] + bh)
    if x0 >= x1 or y0 >= y1:
        return False
    a_part = a["shape"][y0 - a["y"]:y1 - a["y"], x0 - a["x"]:x1 - a["x"]]
    b_part = b["shape"][y0 - b["y"]:y1 - b["y"], x0 - b["x"]:x1 - b["x"]]
    return bool((a_part & b_part).any())


def group(stack: list[dict]) -> list[list[int]]:
    """Combine layers of the same color into one, where that does not change the picture.

    `stack` is in drawing order. A layer joins an earlier group of its color when no layer
    drawn in between overlaps it, so moving it down to that group hides nothing. Separate
    lines of one color (e.g. on the mushroom's cap) become one layer this way.
    Returns the groups in drawing order, as lists of indices into `stack`.
    """
    groups: list[dict] = []
    group_of: list[int] = []
    for index, item in enumerate(stack):
        joined = None
        for g in range(len(groups) - 1, -1, -1):
            candidate = groups[g]
            if candidate["color"] != item["color"]:
                continue
            # Layers whose group is drawn above the candidate group would now end up above this one.
            between = (j for j in range(candidate["members"][0] + 1, index) if group_of[j] > g)
            if not any(overlaps(stack[j], item) for j in between):
                joined = g
                break
        if joined is None:
            groups.append({"color": item["color"], "members": [index]})
            group_of.append(len(groups) - 1)
        else:
            groups[joined]["members"].append(index)
            group_of.append(joined)
    return [g["members"] for g in groups]


def lab_of(color) -> np.ndarray:
    return to_lab(np.array(color, np.uint8).reshape(1, 1, 3))[0, 0]


def ring_colors(image: np.ndarray, item: dict, blocked: np.ndarray) -> list[np.ndarray]:
    """Colors touching a layer's shape, leaving out pixels in `blocked`."""
    region = item["region"]
    h, w = region.shape
    window = (slice(item["y"], item["y"] + h), slice(item["x"], item["x"] + w))
    crop = image[window]
    ring = (cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~region
    ring &= (crop[:, :, 3] == 255) & ~blocked[window]
    return [c for c in np.unique(crop[ring][:, :3], axis=0) if list(c) != list(item["color"])]


def simplify(image: np.ndarray, stack: list[dict], groups: list[list[int]], remove: int) -> np.ndarray:
    """Remove about `remove` layers, choosing the ones whose loss changes the picture least.

    Two kinds of steps: recolor one layer with the most similar color touching it, or merge a
    whole color into the most similar other color (that removes all of its layers at once, and
    is the only way to get fewer layers than colors). Each step costs area x color difference;
    the cheapest steps per removed layer are taken. The color covering most of the picture stays.
    """
    by_color: dict[tuple, list[int]] = {}
    for g, members in enumerate(groups):
        by_color.setdefault(tuple(stack[members[0]]["color"]), []).append(g)
    palette = list(by_color)
    nothing = np.zeros(image.shape[:2], bool)
    area = lambda g: sum(int(stack[i]["region"].sum()) for i in groups[g])  # noqa: E731
    keep = max(palette, key=lambda c: sum(area(g) for g in by_color[c]))

    steps = []  # (cost per removed layer, layers removed, kind, payload)
    for g in range(len(groups)):
        item = stack[groups[g][0]]
        if tuple(item["color"]) == keep and len(by_color[keep]) == 1:
            continue
        around = [c for i in groups[g] for c in ring_colors(image, stack[i], nothing)]
        if around:
            own = lab_of(item["color"])
            cost = area(g) * min(np.linalg.norm(lab_of(c) - own) for c in around)
            steps.append((cost, 1, "layer", g))
    for color, color_groups in by_color.items():
        if color == keep or len(palette) < 2:
            continue
        own = lab_of(color)
        nearest = min((c for c in palette if c != color), key=lambda c: np.linalg.norm(lab_of(c) - own))
        cost = sum(area(g) for g in color_groups) * np.linalg.norm(lab_of(nearest) - own)
        steps.append((cost / len(color_groups), len(color_groups), "color", color))

    chosen_layers, chosen_colors, removed = set(), set(), 0
    for cost, count, kind, payload in sorted(steps, key=lambda s: s[0]):
        if removed >= remove:
            break
        if kind == "layer":
            if payload in chosen_layers or tuple(stack[groups[payload][0]]["color"]) in chosen_colors:
                continue
            chosen_layers.add(payload)
        else:
            if any(tuple(stack[groups[g][0]]["color"]) == payload for g in chosen_layers):
                continue
            chosen_colors.add(payload)
        removed += count

    # Pixels that change color in this round; their old colors are not offered as new ones.
    gone = np.zeros(image.shape[:2], bool)
    targets = []
    for g in list(chosen_layers) + [g for c in chosen_colors for g in by_color[c]]:
        for index in groups[g]:
            item = stack[index]
            h, w = item["region"].shape
            gone[item["y"]:item["y"] + h, item["x"]:item["x"] + w] |= item["region"]
            targets.append(index)
    result = image.copy()
    for index in targets:
        item = stack[index]
        color = tuple(item["color"])
        if color in chosen_colors:
            own = lab_of(color)
            options = [c for c in palette if c != color and c not in chosen_colors]
            new = min(options, key=lambda c: np.linalg.norm(lab_of(c) - own)) if options else None
        else:
            around = ring_colors(image, item, gone)
            own = lab_of(color)
            new = min(around, key=lambda c: np.linalg.norm(lab_of(c) - own)) if around else None
        if new is not None:
            h, w = item["region"].shape
            result[item["y"]:item["y"] + h, item["x"]:item["x"] + w][item["region"], :3] = new
    return result
