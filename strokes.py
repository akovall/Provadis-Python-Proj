"""Find outlines (dark, long, thin shapes) and redraw them smoothly with their original thickness.

A traced outline is a filled shape whose two edges are smoothed separately, so its thickness
wobbles. Here such shapes are taken out of the image before tracing. Each one is reduced to a
centerline, its thickness is measured at every point and smoothed along the line, and it is
drawn as a ribbon around the smooth centerline, so thick and thin parts stay as in the original.
"""

import cv2
import numpy as np

from curves import refit_loop, smooth, to_path
from preprocess import give_to_neighbors

DARK = 50  # Lab lightness; only colors darker than this can be outlines
MIN_LENGTH = 6  # an outline is at least this many times longer than it is wide
FILL_RATIO = 1.8  # area / (width * length); a solid blob has much more area than a line
MIN_AREA = 200  # ignore tiny pieces (in traced pixels)
MIN_SPAN = 0.03  # an outline is at least this share of the image size long
MIN_WIDTH = 4  # thinner pieces (in traced pixels) are usually anti-aliased line tips; they stay filled
MAX_WIDTH = 0.015  # outlines are at most this share of the image size wide
BLOB = 2.5  # parts thicker than this many times the line width stay filled shapes (e.g. a nose on a smile)
SPACING = 2.0  # distance between centerline points, in traced pixels
CAP_CORNER = 150  # ribbon edges only get a sharp corner where they turn more than this (degrees)


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: reduce a shape to a 1-pixel-wide centerline."""
    image = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
            p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
            p8, p9 = image[1:-1, :-2], image[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                side = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                side = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (image[1:-1, 1:-1] == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & side
            if remove.any():
                image[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return image[1:-1, 1:-1].astype(bool)


OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def centerlines(skeleton: np.ndarray) -> list[tuple[np.ndarray, bool, bool]]:
    """Split a skeleton into polylines (y, x) between ends and junctions.

    Returns (points, closed, dead_end) for each line; dead_end lines have a free end.
    """
    pixels = set(zip(*np.nonzero(skeleton)))

    def linked(p):
        # A diagonal pixel only counts when no straight neighbor already connects the two;
        # otherwise every stair step of the skeleton would look like a junction.
        result = []
        for dy, dx in OFFSETS:
            q = (p[0] + dy, p[1] + dx)
            if q not in pixels:
                continue
            if dy and dx and ((p[0] + dy, p[1]) in pixels or (p[0], p[1] + dx) in pixels):
                continue
            result.append(q)
        return result

    neighbors = {p: linked(p) for p in pixels}
    nodes = {p for p, n in neighbors.items() if len(n) != 2}
    lines, used = [], set()

    def walk(start, step):
        path = [start, step]
        previous, current = start, step
        while current not in nodes:
            following = [n for n in neighbors[current] if n != previous and n not in path[-3:]]
            if not following:
                break
            previous, current = current, following[0]
            path.append(current)
        return path

    for node in nodes:
        for step in neighbors[node]:
            if (node, step) in used:
                continue
            path = walk(node, step)
            used.add((node, step))
            if len(path) > 1:
                used.add((path[-1], path[-2]))
            spur = len(neighbors[path[0]]) == 1 or len(neighbors[path[-1]]) == 1
            lines.append((np.array(path, float), False, spur))
    walked = {p for line, _, _ in lines for p in map(tuple, line.astype(int))}
    for start in pixels - walked - nodes:  # loops without any end or junction
        if start in walked or len(neighbors[start]) != 2:
            continue
        path = walk(start, neighbors[start][0])
        walked.update(path)
        lines.append((np.array(path, float), True, False))
    return lines


def resample(points: np.ndarray, widths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced points along a polyline, with the widths carried along."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate([[0], np.cumsum(lengths)])
    targets = np.linspace(0, distance[-1], max(2, int(distance[-1] / SPACING) + 1))
    resampled = np.column_stack([np.interp(targets, distance, points[:, k]) for k in range(2)])
    return resampled, np.interp(targets, distance, widths)


def normals(points: np.ndarray, closed: bool) -> np.ndarray:
    if closed:
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    else:
        tangent = np.gradient(points, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


def cap(center: np.ndarray, normal: np.ndarray, radius: float) -> np.ndarray:
    """Round end: half circle from the `normal` side round to the opposite side."""
    steps = max(4, int(np.pi * radius / SPACING))
    angles = np.arctan2(normal[1], normal[0]) - np.linspace(0, np.pi, steps)[1:-1]
    return center + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def ribbon(points: np.ndarray, widths: np.ndarray, closed: bool, sigma: float) -> list[np.ndarray]:
    """Outline loops of a line drawn with the given (smoothed) width at each point."""
    width_sigma = 2 + 2 * sigma  # thickness changes more slowly than direction
    if closed:
        points, widths = resample(np.vstack([points, points[:1]]), np.append(widths, widths[0]))
        points, widths = points[:-1], widths[:-1]
        if len(points) < 4:
            return []
        points = smooth(points, sigma, closed=True)
        widths = smooth(widths[:, None], width_sigma, closed=True)[:, 0]
        side = normals(points, closed=True) * widths[:, None] / 2
        return [points + side, (points - side)[::-1]]  # opposite directions, so the middle stays open
    points, widths = resample(points, widths)
    if len(points) < 3:
        return []
    points = smooth(points, sigma, closed=False)
    widths = smooth(widths[:, None], width_sigma, closed=False)[:, 0]
    normal = normals(points, closed=False)
    side = normal * widths[:, None] / 2
    end_cap = cap(points[-1], normal[-1], widths[-1] / 2)
    start_cap = cap(points[0], -normal[0], widths[0] / 2)
    return [np.vstack([points + side, end_cap, (points - side)[::-1], start_cap])]


def extract(image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Take outlines out of a flat-color BGRA image.

    Returns the image with outline pixels given to the neighboring colors, and the outlines
    as {"color", "width", "lines"} where lines are (points as x, y, closed, widths) in image pixels.
    """
    opaque = image[:, :, 3] == 255
    bgr = np.ascontiguousarray(image[:, :, :3])
    packed = (bgr[:, :, 0].astype(np.int32) << 16) | (bgr[:, :, 1].astype(np.int32) << 8) | bgr[:, :, 2]
    packed[~opaque] = -1
    colors = [c for c in np.unique(packed) if c >= 0]

    def unpack(color: int) -> np.ndarray:
        return np.array([(color >> 16) & 255, (color >> 8) & 255, color & 255], np.uint8)

    def lightness(color: int) -> float:
        return cv2.cvtColor(unpack(color).reshape(1, 1, 3).astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[0, 0, 0]

    outline_pixels = np.zeros(packed.shape, bool)
    band = np.zeros(packed.shape, bool)  # area around the strokes found so far
    found = []
    # Darkest first, so the lighter fringe of a line is not mistaken for a second line next to it.
    for color in sorted((c for c in colors if lightness(c) < DARK), key=lightness):
        bgr_color = unpack(color)
        count, regions, boxes, _ = cv2.connectedComponentsWithStats((packed == color).astype(np.uint8), connectivity=8)
        for region in range(1, count):
            x, y, w, h, area = boxes[region]
            if area < MIN_AREA:
                continue
            mask = (regions[y:y + h, x:x + w] == region).astype(np.uint8)
            if band[y:y + h, x:x + w][mask > 0].mean() > 0.5:
                continue  # the fringe of a stroke found before
            distance = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            # Thin at half size: the centerline gets smoothed anyway, and it is 4x faster.
            small = cv2.resize(mask, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA) > 0
            skeleton = thin(small)
            length = 2 * skeleton.sum()
            if length == 0:
                continue
            big = cv2.resize(skeleton.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            # The centerline comes from a half-size image, so look for the true center next to each point.
            half_width = cv2.dilate(distance, np.ones((5, 5), np.uint8))
            values = half_width[big & (mask > 0)]
            if len(values) == 0:
                continue
            width = max(1.0, 2 * float(np.median(values)) - 1)
            if not MIN_WIDTH <= width <= MAX_WIDTH * max(image.shape[:2]):
                continue  # too thin (anti-aliasing) or too thick to be an outline
            # Thick parts (blobs attached to the line) stay as filled shapes.
            core = (distance > BLOB * width / 2).astype(np.uint8)
            radius = int(np.ceil(BLOB * width / 2))
            blob = (cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0) & (mask > 0)
            line = (mask > 0) & ~blob
            small_blob = cv2.resize(blob.astype(np.uint8), skeleton.shape[::-1], interpolation=cv2.INTER_NEAREST)
            skeleton &= small_blob == 0
            at_blob = cv2.dilate(small_blob, np.ones((5, 5), np.uint8)) > 0
            length = 2 * skeleton.sum()
            too_short = length < max(MIN_LENGTH * width, MIN_SPAN * max(image.shape[:2]))
            if too_short or line.sum() > FILL_RATIO * width * length:
                continue  # a solid shape, not a line
            lines = []
            for points, closed, dead_end in centerlines(skeleton):
                ends = points[[0, -1]].astype(int)
                joins_blob = at_blob[ends[:, 0], ends[:, 1]].any()
                if dead_end and len(points) * 2 < 2 * width and not joins_blob:
                    continue  # short spur at a corner
                xy = points[:, ::-1] * 2 + 0.5 + np.array([x, y])
                rows = np.clip((points[:, 0] * 2).astype(int), 0, h - 1)
                cols = np.clip((points[:, 1] * 2).astype(int), 0, w - 1)
                # Thickness at each point as in the original; capped so a junction does not swell.
                widths = np.clip(2 * half_width[rows, cols] - 1, 1, 2 * width)
                lines.append((xy, closed, widths))
            if lines:
                found.append({"color": bgr_color, "width": width, "lines": lines})
                outline_pixels[y:y + h, x:x + w] |= line
                reach = int(width) + 1
                band[y:y + h, x:x + w] |= cv2.dilate(line.astype(np.uint8), np.ones((reach, reach), np.uint8)) > 0
    if not found:
        return image, []
    # Small pieces of any color along the edge of a stroke (leftover anti-aliasing) would show as
    # specks next to it: hide them too. Pieces further out stay, so nothing opens a gap.
    near = cv2.dilate(outline_pixels.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for color in colors:
        rest = ((packed == color) & ~outline_pixels).astype(np.uint8)
        count, regions, boxes, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
        if count < 2:
            continue
        inside = np.bincount(regions[near & (rest > 0)], minlength=count)
        hide = ((boxes[:, 4] < MIN_AREA) & (inside > 0.5 * boxes[:, 4])) | (inside > 0.7 * boxes[:, 4])
        hide[0] = False
        outline_pixels |= hide[regions]
    # The fill shapes meet under the outline.
    return give_to_neighbors(image, outline_pixels), found


def to_svg(outlines: list[dict], error: float, sigma: float) -> str:
    """SVG paths for the outlines: each line is one filled ribbon with smooth edges."""
    paths = []
    for outline in outlines:
        b, g, r = outline["color"]
        for points, closed, widths in outline["lines"]:
            loops = [refit_loop(loop, error, CAP_CORNER, 0) for loop in ribbon(points, widths, closed, sigma)]
            if loops:
                paths.append(f'<path d="{to_path(loops)}" fill="#{r:02x}{g:02x}{b:02x}"/>')
    return "\n".join(paths)
