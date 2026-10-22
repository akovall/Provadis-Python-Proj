import cv2
import numpy as np

WORK_SIDE = 1024  # images are processed at about this size, larger ones shrunk and small ones enlarged
MAX_ENLARGE = 4
UPSCALE = 2
SAMPLE_SIZE = 20000
MODE_KERNEL = 5
SHARPEN_STEPS = 2  # toggle-filter passes; each pass sharpens soft edges by about 1 pixel
BACKGROUND_TOLERANCE = 12  # ΔE from the border colors that still counts as background


def to_bgra(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (image / 257).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def load(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return to_bgra(image)


def decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("cannot read image")
    return to_bgra(image)


def work_scale(image: np.ndarray) -> float:
    return min(WORK_SIDE / max(image.shape[:2]), MAX_ENLARGE)


def resize(image: np.ndarray) -> np.ndarray:
    """Bring the image to the working size; enlarging small images keeps thin lines from being smoothed away."""
    height, width = image.shape[:2]
    scale = work_scale(image)
    if abs(scale - 1) < 0.05:
        return image
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=interpolation)


def smooth(image: np.ndarray) -> np.ndarray:
    """Flatten textures and JPEG noise while keeping edges sharp. The alpha channel is kept as is."""
    bgr = cv2.pyrMeanShiftFiltering(np.ascontiguousarray(image[:, :, :3]), 8, 18)
    return np.dstack([bgr, image[:, :, 3]])


def prepare(image: np.ndarray) -> np.ndarray:
    """Color-independent steps; the result can be reused for any color settings."""
    scale = work_scale(image)
    image = resize(image)
    if scale > 1:
        # Blur away the enlarged pixel staircase; sharpening then turns it into a smooth edge.
        sigma = 0.6 * scale
        image = np.dstack([cv2.GaussianBlur(image[:, :, :3], (0, 0), sigma), image[:, :, 3]])
    # Enlarging widens the soft edges too, so they need more sharpening passes.
    steps = SHARPEN_STEPS + max(0, round(scale) - 1)
    return sharpen_edges(smooth(image), steps)


def to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]).astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


def border_colors(lab: np.ndarray) -> np.ndarray:
    """The main colors along the image edges (several, e.g. for a painted checkerboard)."""
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    count = min(3, len(np.unique(border, axis=0)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(border, count, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.flatten(), minlength=count) / len(border)
    return centers[shares > 0.1]


def foreground(prepared: np.ndarray, remove_background: bool) -> np.ndarray:
    """Mask of the object pixels.

    A real alpha channel is used when the image has one. Otherwise, with `remove_background`,
    pixels close to the border colors that are connected to the image edge become background.
    Areas of the same color inside the object (e.g. white details) are not connected to the
    edge, so they stay part of the object.
    """
    alpha = prepared[:, :, 3]
    if alpha.min() < 128:
        return alpha >= 128
    if not remove_background:
        return np.ones(alpha.shape, bool)
    lab = to_lab(prepared)
    distance = np.min([np.linalg.norm(lab - center, axis=2) for center in border_colors(lab)], axis=0)
    candidate = (distance < BACKGROUND_TOLERANCE).astype(np.uint8)
    _, regions = cv2.connectedComponents(candidate, connectivity=4)
    edge = np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
    background = np.isin(regions, edge[edge > 0])
    if background.all():
        return np.ones(alpha.shape, bool)
    return ~background


def palette(lab: np.ndarray, colors: int) -> tuple[np.ndarray, np.ndarray]:
    """K-means on a random sample of the given Lab pixels. Returns cluster centers and their pixel counts."""
    rng = np.random.default_rng(0)
    sample = lab[rng.choice(len(lab), min(SAMPLE_SIZE, len(lab)), replace=False)]
    colors = min(colors, len(np.unique(sample, axis=0)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(sample, colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=colors).astype(np.float64)
    return centers.astype(np.float64), counts


def merge_colors(centers: np.ndarray, counts: np.ndarray, threshold: float) -> np.ndarray:
    """Merge the closest pair of colors until all are at least `threshold` ΔE apart."""
    centers, counts = list(centers), list(counts)
    while len(centers) > 1:
        array = np.array(centers)
        distances = np.linalg.norm(array[:, None] - array[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        if distances[i, j] >= threshold:
            break
        total = counts[i] + counts[j]
        centers[i] = (centers[i] * counts[i] + centers[j] * counts[j]) / total
        counts[i] = total
        del centers[j], counts[j]
    return np.array(centers, dtype=np.float32)


def assign(lab: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Label every pixel with its nearest center."""
    best = np.full(lab.shape[:2], np.inf, np.float32)
    labels = np.zeros(lab.shape[:2], np.uint8)
    for index, center in enumerate(centers):
        # |x - c|² minus the constant |x|²; cv2.transform computes x·c much faster than numpy
        distance = float(center @ center) - 2 * cv2.transform(lab, center.reshape(1, 3))
        closer = distance < best
        best[closer] = distance[closer]
        labels[closer] = index
    return labels


def sharpen_edges(image: np.ndarray, iterations: int) -> np.ndarray:
    """Toggle filter: each pixel takes the darkest or lightest color around it, whichever is closer.

    Soft anti-aliased edges collapse into a sharp step in the middle, so they do not turn into
    thin bands of in-between colors. A thin line keeps its core, because the core is already the
    darkest (or lightest) color around it.
    """
    bgr = np.ascontiguousarray(image[:, :, :3])
    height, width = bgr.shape[:2]
    for _ in range(iterations):
        light = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        padded = cv2.copyMakeBorder(bgr, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        padded_light = cv2.copyMakeBorder(light, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        dark, dark_light = bgr.copy(), light.copy()
        bright, bright_light = bgr.copy(), light.copy()
        for dy in range(3):
            for dx in range(3):
                color = padded[dy:dy + height, dx:dx + width]
                value = padded_light[dy:dy + height, dx:dx + width]
                darker = value < dark_light
                dark[darker], dark_light[darker] = color[darker], value[darker]
                brighter = value > bright_light
                bright[brighter], bright_light[brighter] = color[brighter], value[brighter]
        to_dark = (light - dark_light) < (bright_light - light)
        flat = (bright_light - dark_light) < 8  # leave flat areas and gentle gradients alone
        bgr = np.where(flat[..., None], bgr, np.where(to_dark[..., None], dark, bright))
    return np.dstack([bgr, image[:, :, 3]])


def mode_filter(labels: np.ndarray, count: int) -> np.ndarray:
    """Replace each pixel with the most common label around it, removing stair-steps and specks."""
    best = np.full(labels.shape, -1.0, np.float32)
    result = labels.copy()
    for index in range(count):
        votes = cv2.blur((labels == index).astype(np.float32), (MODE_KERNEL, MODE_KERNEL))
        more = votes > best
        best[more] = votes[more]
        result[more] = index
    return result


def quantize(
    prepared: np.ndarray, colors: int = 16, merge: float = 8.0, remove_background: bool = True
) -> tuple[np.ndarray, int]:
    """Reduce to at most `colors` flat colors, merging ones closer than `merge` ΔE, at UPSCALE resolution.

    Returns a BGRA image (background fully transparent) and the number of colors left after merging.
    """
    mask = foreground(prepared, remove_background)
    if not mask.any():
        raise ValueError("the image has no visible pixels")
    centers, counts = palette(to_lab(prepared)[mask], colors)
    centers = merge_colors(centers, counts, merge)
    height, width = prepared.shape[:2]
    size = (width * UPSCALE, height * UPSCALE)
    # Linear (not cubic) so the enlarged edges never overshoot into darker or lighter colors.
    large = cv2.resize(prepared, size, interpolation=cv2.INTER_LINEAR)
    large_mask = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) >= 0.5
    labels = assign(to_lab(large), centers)
    transparent = len(centers)
    labels[~large_mask] = transparent
    labels = mode_filter(labels, transparent + 1)
    bgr = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_Lab2BGR).reshape(-1, 3)
    bgra = np.zeros((transparent + 1, 4), np.uint8)
    bgra[:transparent, :3] = np.clip(bgr * 255 + 0.5, 0, 255)
    bgra[:transparent, 3] = 255
    return bgra[labels], transparent


def give_to_neighbors(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Recolor `pixels` with the nearest other pixel's color, so the shapes around meet underneath."""
    _, nearest = cv2.distanceTransformWithLabels(pixels.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    kept = ~pixels
    lookup = np.zeros((nearest.max() + 1, image.shape[2]), np.uint8)
    lookup[nearest[kept]] = image[kept]
    result = image.copy()
    result[pixels] = lookup[nearest[pixels]]
    return result
