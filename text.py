"""Find lettering and draw every copy of a letter from one shared shape.

Traced letters come out a little different each time: two R's of the same word get their own
bumps. Here the letters are taken out of the image before tracing. Letters that are the same
(same color, size and shape, even turned along curved text) are put on top of each other and
averaged into one clean shape, and every copy is drawn from that shape, turned and moved to its
place. So the font keeps its shapes and all copies of a letter look exactly alike.

A letter is a small area of one color with thin strokes, lying on one background color, with
other such areas of about the same size next to it (a line of text).
"""

import cv2
import numpy as np

from curves import refit_loop, to_path
from preprocess import give_to_neighbors

MIN_AREA = 60  # smaller areas are specks, not letters (in traced pixels)
MIN_SIZE = 0.006  # a letter is at least this share of the image size ...
MAX_SIZE = 0.15  # ... and at most this share
MAX_STROKE = 0.35  # stroke width / letter size; dots and blobs are thicker
MIN_FILL = 0.25  # share of its (turned) box a letter covers; thin curved slivers cover less
BACKGROUND = 0.8  # this share of the pixels around a letter has one color (the background)
NEIGHBOR = 2.5  # letters of a line are at most this many letter sizes apart (center to center)
SIZE_RATIO = 2.0  # neighbors in a line differ in size at most this much
MIN_LETTERS = 3  # a line of text has at least this many letters
MAX_LETTERS = 600  # more candidates than this: not lettering (or too slow to compare)
AMBIGUOUS = 0.5  # lines steeper than this (|up.y|) can be read from either side
SEARCH = 12  # turning searched around the guessed one when matching two letters (degrees)
SAME = 0.82  # overlap (intersection over union) for two letters to count as the same
FINE = 2  # the shared shape is built at this many times the traced resolution
BLUR = 1.0  # and smoothed by this much (traced pixels) before its edge is taken
MAX_SCALE = 3.0  # the same letter may appear up to this many times bigger (another font size)


def candidates(image: np.ndarray) -> list[dict]:
    """Areas of one color that could be letters, each with its mask, center, size and background."""
    opaque = image[:, :, 3] == 255
    bgr = np.ascontiguousarray(image[:, :, :3])
    packed = (bgr[:, :, 0].astype(np.int32) << 16) | (bgr[:, :, 1].astype(np.int32) << 8) | bgr[:, :, 2]
    packed[~opaque] = -1
    side = max(image.shape[:2])
    height, width = packed.shape
    found = []
    for color in np.unique(packed):
        if color < 0:
            continue
        count, ids, boxes, centers = cv2.connectedComponentsWithStats((packed == color).astype(np.uint8), connectivity=8)
        for index in range(1, count):
            x, y, w, h, area = boxes[index]
            if area < MIN_AREA or not MIN_SIZE * side <= max(w, h) <= MAX_SIZE * side:
                continue
            x0, y0, x1, y1 = max(x - 2, 0), max(y - 2, 0), min(x + w + 2, width), min(y + h + 2, height)
            mask = ids[y0:y1, x0:x1] == index
            loops, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            perimeter = sum(cv2.arcLength(loop, True) for loop in loops)
            (_, _), (rw, rh), _ = cv2.minAreaRect(np.argwhere(mask)[:, ::-1].astype(np.float32))
            size = max(rw, rh, 1.0)
            if 2 * area / max(perimeter, 1) > MAX_STROKE * size:
                continue  # a solid shape: strokes are thin
            if area < MIN_FILL * max(rw, 1.0) * max(rh, 1.0):
                continue  # a sliver (grass, hair), not a letter
            ring = (cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~mask
            around, counts = np.unique(packed[y0:y1, x0:x1][ring], return_counts=True)
            if counts.max() < BACKGROUND * counts.sum():
                continue  # not lying on one color
            found.append({
                "color": [(color >> 16) & 255, (color >> 8) & 255, color & 255], "key": int(color),
                "background": int(around[counts.argmax()]), "x": x0, "y": y0, "mask": mask,
                "center": centers[index], "size": size, "area": int(area),
            })
    return found


def lines(found: list[dict]) -> tuple[list[dict], list[list[int]], list[int]]:
    """The candidates that form lines of text, each one's neighbors in its line, and its line number."""
    count = len(found)
    links: list[list[int]] = [[] for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            a, b = found[i], found[j]
            if a["key"] != b["key"] or a["background"] != b["background"]:
                continue
            if max(a["size"], b["size"]) > SIZE_RATIO * min(a["size"], b["size"]):
                continue
            if np.linalg.norm(a["center"] - b["center"]) <= NEIGHBOR * max(a["size"], b["size"]):
                links[i].append(j)
                links[j].append(i)
    # Lines are the connected groups of linked candidates, kept when long enough.
    keep, line_of, seen = [], {}, set()
    for start in range(count):
        if start in seen:
            continue
        group, todo = [], [start]
        seen.add(start)
        while todo:
            i = todo.pop()
            group.append(i)
            for j in links[i]:
                if j not in seen:
                    seen.add(j)
                    todo.append(j)
        if len(group) >= MIN_LETTERS:
            keep += group
            line_of.update((i, start) for i in group)
    keep.sort()
    new = {old: new for new, old in enumerate(keep)}
    return [found[i] for i in keep], [[new[j] for j in links[i]] for i in keep], [line_of[i] for i in keep]


def tilt(glyph: dict, neighbors: list[dict]) -> tuple[float, bool]:
    """How far the letter is turned (degrees, clockwise) from upright, guessed from its line.

    The line runs through the letter and its neighbors; up is across it, on the side of the top of
    the image. Where the line is steep that side is a guess, so the letter is then ambiguous.
    """
    points = np.array([glyph["center"]] + [n["center"] for n in neighbors])
    spread = np.cov((points - points.mean(axis=0)).T) if len(points) > 2 else None
    if spread is None:
        direction = points[1] - points[0]
    else:
        values, vectors = np.linalg.eigh(spread)
        direction = vectors[:, 1]
    up = np.array([direction[1], -direction[0]]) / (np.linalg.norm(direction) + 1e-9)
    if up[1] > 0:
        up = -up
    return float(np.degrees(np.arctan2(up[0], -up[1]))), abs(up[1]) < AMBIGUOUS


def pose(glyph: dict, angle: float, radius: int, scale: float = 1.0) -> np.ndarray:
    """The letter turned by `angle` (counterclockwise) and scaled, centered on a square canvas, as a 0..1 mask."""
    cx, cy = glyph["center"][0] - glyph["x"], glyph["center"][1] - glyph["y"]
    turn = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, scale)
    turn[:, 2] += (radius - cx, radius - cy)
    return cv2.warpAffine(glyph["mask"].astype(np.float32), turn, (2 * radius + 1,) * 2, flags=cv2.INTER_LINEAR)


def radius(glyph: dict) -> int:
    """Half the canvas size that holds the letter at any turn."""
    points = np.argwhere(glyph["mask"])[:, ::-1] + (glyph["x"], glyph["y"])
    return int(np.ceil(np.linalg.norm(points - glyph["center"], axis=1).max())) + 2


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a >= 0.5, b >= 0.5
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)


def match(rep: dict, glyph: dict) -> tuple[float, float] | None:
    """The turn and scale that put `glyph` onto `rep` (upright), or None if they are not the same letter."""
    scale = float(np.sqrt(rep["area"] / glyph["area"]))
    if rep["key"] != glyph["key"] or not 1 / MAX_SCALE <= scale <= MAX_SCALE:
        return None
    size = max(rep["radius"], int(np.ceil(glyph["radius"] * scale)))
    target = pose(rep, rep["turn"], size)
    starts = [glyph["turn"]] + ([glyph["turn"] + 180] if rep["ambiguous"] or glyph["ambiguous"] else [])
    best, score = None, 0.0
    for start in starts:
        # Coarse search, then a finer one around the best coarse turn.
        coarse = max(start + np.arange(-SEARCH, SEARCH + 1, 4.0), key=lambda a: overlap(target, pose(glyph, a, size, scale)))
        for angle in coarse + np.arange(-3, 4, 1.0):
            value = overlap(target, pose(glyph, angle, size, scale))
            if value > score:
                best, score = float(angle), value
    return (best, scale) if score >= SAME else None


def extract(image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Find the lettering in a flat-color BGRA image and take it out.

    Returns the image with the letters painted over in their background color, and the letters
    as {"color", "loops", "places"}: one shared shape (pixel-edge loops around (0, 0), upright)
    and where each copy goes as (x, y, turn in degrees clockwise, scale).
    """
    found, links, line_of = lines(candidates(image))
    if not found or len(found) > MAX_LETTERS:
        return image, []
    for i, glyph in enumerate(found):
        glyph["turn"], glyph["ambiguous"] = tilt(glyph, [found[j] for j in links[i]])
        glyph["radius"] = radius(glyph)
    # Each letter joins the first group whose first letter it matches.
    # Bigger letters go first, so the shared shapes come from the most detailed copies.
    groups: list[list[tuple[int, float, float]]] = []
    for i in sorted(range(len(found)), key=lambda i: -found[i]["area"]):
        for group in groups:
            fit = match(found[group[0][0]], found[i])
            if fit is not None:
                group.append((i, *fit))
                break
        else:
            groups.append([(i, found[i]["turn"], 1.0)])
    # Text repeats letters; a "line" where no letter has a twin anywhere is something else (leaves,
    # specks of a drawing that happen to lie in a row), so it is left to the normal tracing.
    texts = {line_of[i] for group in groups if len(group) > 1 for i, _, _ in group}
    groups = [[m for m in group if line_of[m[0]] in texts] for group in groups]
    groups = [group for group in groups if group]
    found = [glyph if line_of[i] in texts else None for i, glyph in enumerate(found)]
    letters = []
    for group in groups:
        size = max(int(np.ceil(found[i]["radius"] * scale)) for i, _, scale in group)
        # Averaged at FINE times the size and blurred a little, so the pixel steps and bumps of
        # the single copies cancel out and the shared edge runs between them.
        mean = np.mean([pose(found[i], turn, FINE * size, FINE * scale) for i, turn, scale in group], axis=0)
        shape = cv2.GaussianBlur(mean, (0, 0), FINE * BLUR) >= 0.5
        loops, _ = cv2.findContours(shape.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        loops = [(loop[:, 0].astype(float) + 0.5) / FINE - size for loop in loops if len(loop) >= 8 * FINE]
        if loops:
            places = [(*found[i]["center"], turn, 1 / scale) for i, turn, scale in group]
            letters.append({"color": found[group[0][0]]["color"], "loops": loops, "places": places})
    covered = np.zeros(image.shape[:2], bool)
    for glyph in filter(None, found):
        h, w = glyph["mask"].shape
        covered[glyph["y"]:glyph["y"] + h, glyph["x"]:glyph["x"] + w] |= glyph["mask"]
    return give_to_neighbors(image, covered), letters


def place(curves: list[np.ndarray], x: float, y: float, turn: float, scale: float) -> list[np.ndarray]:
    """Bezier curves of the shared shape, turned back (clockwise by `turn`), scaled and moved to (x, y)."""
    matrix = cv2.getRotationMatrix2D((0, 0), -turn, scale)[:, :2]
    return [c @ matrix.T + (x, y) for c in curves]


def to_svg(letter: dict, error: float, corner_angle: float, sigma: float) -> list[str]:
    """One path per copy, all from the same curves."""
    b, g, r = letter["color"]
    loops = [refit_loop(loop, error, corner_angle, sigma) for loop in letter["loops"]]
    return [
        f'<path d="{to_path([place(curves, x, y, turn, scale) for curves in loops])}" fill="#{r:02x}{g:02x}{b:02x}"/>'
        for x, y, turn, scale in letter["places"]
    ]
