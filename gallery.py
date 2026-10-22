"""Gallery of conversions, kept in memory.

At startup one background thread converts every image in img/ with its auto-detected preset.
Conversions from the main page are added on top as history, one tile per uploaded image.
The converter has priority: the thread only goes on while no converter request is running.
"""

import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import resvg_py

from preprocess import foreground, load, prepare, quantize, resize
from presets import PRESETS, detect
from vectorize import stats, vectorize

IMAGES = Path(__file__).parent / "img"
EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_HISTORY = 20
THUMBNAIL_SIDE = 480
# Thin outlines shifted by a pixel already add a few points of error, so the limits are not tiny
GOOD_ERROR = 5  # hard to tell apart from the original
OKAY_ERROR = 12  # clearly the same picture, but simplified (e.g. visible color bands in photos)
PRIVATE = {"svg", "original", "reference", "prepared"}  # large fields, served separately

tiles: OrderedDict[str, dict] = OrderedDict()  # history (newest first), then the img/ folder
lock = threading.Lock()
jobs: Queue = Queue()
busy = 0  # converter requests running right now
idle = threading.Condition()


@contextmanager
def priority():
    """Mark a converter request as running, so the gallery thread waits until it is done."""
    global busy
    with idle:
        busy += 1
    try:
        yield
    finally:
        with idle:
            busy -= 1
            idle.notify_all()


def wait_for_idle() -> None:
    with idle:
        idle.wait_for(lambda: busy == 0)


def on_white(image: np.ndarray) -> np.ndarray:
    alpha = image[:, :, 3:4] / 255
    return image[:, :, :3] * alpha + 255 * (1 - alpha)


def measure(svg: str, reference: np.ndarray, prepared: np.ndarray, remove_background: bool) -> float:
    """Mean color difference (0-255) between the rendered SVG and the BGRA reference, both on white.

    With a removed background only the object counts: the pixels that preprocessing kept as the
    object, plus everything the SVG paints. Otherwise the transparent background would be compared
    with the original background color and count as error.
    """
    height, width = reference.shape[:2]
    png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height)
    rendered = cv2.imdecode(np.frombuffer(bytes(png), np.uint8), cv2.IMREAD_UNCHANGED)
    difference = np.abs(on_white(rendered) - on_white(reference))
    region = foreground(prepared, remove_background) | (rendered[:, :, 3] >= 128)
    return float(difference[region].mean())


def rate(error: float) -> str:
    return "good" if error < GOOD_ERROR else "okay" if error < OKAY_ERROR else "poor"


def thumbnail(image: np.ndarray) -> bytes:
    height, width = image.shape[:2]
    scale = min(1, THUMBNAIL_SIDE / max(height, width))
    small = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", small)[1].tobytes()


def preset_name(settings: dict) -> str:
    return next((name for name, values in PRESETS.items() if values == settings), "custom")


def new_tile(tile_id: str, name: str, source: str) -> dict:
    return {
        "id": tile_id, "name": name, "source": source, "status": "queued", "message": "",
        "preset": None, "auto": False, "settings": None, "version": 0,
        "colors": None, "paths": None, "curves": None, "bytes": None, "seconds": None, "error": None, "rating": None,
        "svg": None, "original": None, "reference": None, "prepared": None,
    }


def update(tile_id: str, **fields) -> None:
    with lock:
        if tile_id in tiles:
            tiles[tile_id].update(fields)


def convert_file(tile_id: str, path: Path) -> None:
    update(tile_id, status="working")
    start = time.perf_counter()
    image = load(str(path))
    preset = detect(image)
    settings = PRESETS[preset]
    prepared = prepare(image)
    wait_for_idle()  # also give way between the slow steps, not only between images
    flat, colors = quantize(prepared, settings["colors"], settings["merge"], settings["remove_background"])
    wait_for_idle()
    svg = vectorize(flat, settings["detail"], settings["speckle"])
    seconds = time.perf_counter() - start
    reference = resize(image)
    error = round(measure(svg, reference, prepared, settings["remove_background"]), 2)
    update(
        tile_id, status="done", preset=preset, auto=True, settings=settings, colors=colors, seconds=round(seconds, 1),
        error=error, rating=rate(error), svg=svg, original=thumbnail(reference), version=1, **stats(svg),
    )


def measure_tile(tile_id: str, version: int) -> None:
    with lock:
        tile = tiles.get(tile_id)
        if tile is None or tile["version"] != version:
            return  # removed, or replaced by a newer result that queued its own job
        svg, reference, prepared, settings = tile["svg"], tile["reference"], tile["prepared"], tile["settings"]
    error = round(measure(svg, reference, prepared, settings["remove_background"]), 2)
    with lock:
        if tile["version"] == version:
            tile.update(error=error, rating=rate(error), status="done")


def record(
    upload_id: str, name: str, reference: np.ndarray, prepared: np.ndarray, svg: str, settings: dict, colors: int
) -> None:
    """Add or update the history tile of an uploaded image; its error is measured in the background."""
    with lock:
        tile = tiles.get(upload_id)
        if tile is None:
            tile = new_tile(upload_id, name, "history")
            tile.update(original=thumbnail(reference), reference=reference, prepared=prepared)
            tiles[upload_id] = tile
            tiles.move_to_end(upload_id, last=False)
            history = [key for key, item in tiles.items() if item["source"] == "history"]
            for key in history[MAX_HISTORY:]:
                del tiles[key]
        tile.update(
            status="measuring", preset=preset_name(settings), settings=settings, colors=colors,
            error=None, rating=None, svg=svg, version=tile["version"] + 1, **stats(svg),
        )
        version = tile["version"]
    jobs.put((measure_tile, upload_id, version))


def listing() -> dict:
    with lock:
        public = [{key: value for key, value in tile.items() if key not in PRIVATE} for tile in tiles.values()]
    return {"tiles": public, "pending": sum(tile["status"] in ("queued", "working", "measuring") for tile in public)}


def file(tile_id: str, field: str) -> str | bytes | None:
    with lock:
        tile = tiles.get(tile_id)
        return tile[field] if tile else None


def work() -> None:
    while True:
        function, tile_id, *args = jobs.get()
        wait_for_idle()
        try:
            function(tile_id, *args)
        except Exception as error:  # one broken image must not stop the whole gallery
            update(tile_id, status="failed", message=str(error))


def start() -> None:
    """Queue every image in img/ and start the background thread."""
    if IMAGES.is_dir():
        for path in sorted(IMAGES.iterdir(), key=lambda path: path.name.lower()):
            if path.suffix.lower() in EXTENSIONS:
                tile = new_tile(uuid.uuid4().hex, path.name, "img")
                with lock:
                    tiles[tile["id"]] = tile
                jobs.put((convert_file, tile["id"], path))
    threading.Thread(target=work, daemon=True).start()
