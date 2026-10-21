import cv2
import numpy as np

MAX_SIDE = 1024
COLORS = 24


def load(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def resize(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, 5, 50, 50)


def reduce_colors(image: np.ndarray) -> np.ndarray:
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, COLORS, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.uint8)[labels.flatten()].reshape(image.shape)


def preprocess(path: str) -> np.ndarray:
    return reduce_colors(denoise(resize(load(path))))
