import cv2
import numpy as np

MAX_SIDE = 1024
UPSCALE = 2
SAMPLE_SIZE = 20000
MODE_KERNEL = 5


def load(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot read image")
    return image


def resize(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def smooth(image: np.ndarray) -> np.ndarray:
    """Flatten textures and JPEG noise while keeping edges sharp."""
    return cv2.pyrMeanShiftFiltering(image, 8, 18)


def prepare(image: np.ndarray) -> np.ndarray:
    """Color-independent steps; the result can be reused for any color settings."""
    return smooth(resize(image))


def to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


def palette(lab: np.ndarray, colors: int) -> tuple[np.ndarray, np.ndarray]:
    """K-means on a random pixel sample. Returns cluster centers and their pixel counts."""
    pixels = lab.reshape(-1, 3)
    rng = np.random.default_rng(0)
    sample = pixels[rng.choice(len(pixels), min(SAMPLE_SIZE, len(pixels)), replace=False)]
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


def quantize(prepared: np.ndarray, colors: int = 16, merge: float = 8.0) -> tuple[np.ndarray, int]:
    """Reduce to at most `colors` flat colors, merging ones closer than `merge` ΔE, at UPSCALE resolution.

    Returns the flat-color image and the number of colors left after merging.
    """
    centers, counts = palette(to_lab(prepared), colors)
    centers = merge_colors(centers, counts, merge)
    height, width = prepared.shape[:2]
    large = cv2.resize(prepared, (width * UPSCALE, height * UPSCALE), interpolation=cv2.INTER_CUBIC)
    labels = mode_filter(assign(to_lab(large), centers), len(centers))
    bgr = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_Lab2BGR).reshape(-1, 3)
    return np.clip(bgr * 255 + 0.5, 0, 255).astype(np.uint8)[labels], len(centers)
