import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import gallery
from preprocess import decode, foreground, prepare, quantize, resize
from presets import PRESETS, detect
from quality import choose_detail, choose_merge, difference
from vectorize import Traced, finish, stats, trace

STATIC = Path(__file__).parent / "static"
MAX_IMAGES = 8
RECOMMENDED_COLORS = 32  # upper limit; the recommended merge decides how many are left


@asynccontextmanager
async def lifespan(app: FastAPI):
    gallery.start()
    yield


app = FastAPI(title="Image to SVG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Uploaded images, prepared once. Each entry also caches the last quantized image and
# the last raw trace, so moving the detail slider only re-runs the curve refit.
images: OrderedDict[str, dict] = OrderedDict()
lock = threading.Lock()


class Settings(BaseModel):
    id: str
    colors: int = Field(16, ge=2, le=64)
    merge: float = Field(8.0, ge=0, le=60)
    detail: int = Field(4, ge=1, le=10)
    speckle: int = Field(4, ge=0, le=32)
    remove_background: bool = True
    outlines: bool = True
    shapes: bool = True
    max_layers: int | None = Field(None, ge=2, le=500)
    seed: int = Field(0, ge=0)  # another seed gives another variant of the colors and layers


class RecommendRequest(BaseModel):
    id: str
    speckle: int = Field(4, ge=0, le=32)
    remove_background: bool = True
    outlines: bool = True
    shapes: bool = True


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/presets")
def presets() -> dict:
    return PRESETS


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict:
    try:
        image = decode(file.file.read())
    except ValueError as error:
        raise HTTPException(400, str(error))
    with gallery.priority():
        preset = detect(image)
        prepared = prepare(image)
    image_id = uuid.uuid4().hex
    with lock:
        images[image_id] = {
            "prepared": prepared,
            "reference": resize(image),  # same size as `prepared`, for the gallery and the match score
            "size": (image.shape[1], image.shape[0]),
            "name": file.filename or "image",
            "masks": {},
            "key": None, "quantized": None, "colors": 0,
            "trace_key": None, "raw": None,
        }
        while len(images) > MAX_IMAGES:
            images.popitem(last=False)
    height, width = image.shape[:2]
    return {"id": image_id, "width": width, "height": height, "preset": preset}


def get_entry(image_id: str) -> dict:
    with lock:
        entry = images.get(image_id)
    if entry is None:
        raise HTTPException(404, "image not found, please upload it again")
    return entry


def get_mask(entry: dict, remove_background: bool) -> np.ndarray:
    if remove_background not in entry["masks"]:
        entry["masks"][remove_background] = foreground(entry["prepared"], remove_background)
    return entry["masks"][remove_background]


def reference(entry: dict, remove_background: bool) -> np.ndarray:
    """The original image as the SVG should look: the removed background is transparent."""
    image = entry["reference"].copy()
    image[:, :, 3] = np.where(get_mask(entry, remove_background), image[:, :, 3], 0)
    return image


def get_trace(
    entry: dict,
    colors: int,
    merge: float,
    remove_background: bool,
    speckle: int,
    outlines: bool,
    shapes: bool,
    max_layers: int | None = None,
    seed: int = 0,
) -> Traced:
    """Quantize and trace, reusing the cached results when the settings did not change."""
    key = (colors, merge, remove_background, seed)
    if entry["key"] != key:
        try:
            entry["quantized"], entry["colors"] = quantize(entry["prepared"], colors, merge, remove_background, seed)
        except ValueError as error:
            raise HTTPException(400, str(error))
        entry["key"] = key
    trace_key = (key, speckle, outlines, shapes, max_layers)
    if entry["trace_key"] != trace_key:
        entry["raw"] = trace(entry["quantized"], speckle, outlines, shapes, max_layers)
        entry["trace_key"] = trace_key
    return entry["raw"]


def result(entry: dict, svg: str, remove_background: bool) -> dict:
    match = difference(svg, reference(entry, remove_background))
    return {
        "svg": svg, "colors": entry["colors"], "match": round(match["delta"], 2),
        "auto_layers": entry["raw"]["auto_layers"], **stats(svg),
    }


@app.post("/api/vectorize")
def convert(settings: Settings) -> dict:
    entry = get_entry(settings.id)
    with gallery.priority():
        raw = get_trace(
            entry, settings.colors, settings.merge, settings.remove_background, settings.speckle,
            settings.outlines, settings.shapes, settings.max_layers, settings.seed,
        )
        svg = finish(raw, entry["quantized"], settings.detail, entry["size"])
        response = result(entry, svg, settings.remove_background)
    gallery.record(
        settings.id, entry["name"], entry["reference"], entry["prepared"], svg, settings.model_dump(exclude={"id"}), entry["colors"]
    )
    return response


@app.post("/api/recommend")
def recommend(request: RecommendRequest) -> dict:
    """Try several settings, compare each with the original and pick the simplest close one."""
    entry = get_entry(request.id)
    with gallery.priority():
        mask = get_mask(entry, request.remove_background)
        if not mask.any():
            raise HTTPException(400, "the image has no visible pixels")
        merge = choose_merge(entry["prepared"], mask, RECOMMENDED_COLORS)
        raw = get_trace(
            entry, RECOMMENDED_COLORS, merge, request.remove_background, request.speckle, request.outlines, request.shapes
        )
        svgs = {}

        def render(detail: int) -> str:
            svgs[detail] = finish(raw, entry["quantized"], detail, entry["size"])
            return svgs[detail]

        detail, _ = choose_detail(render, reference(entry, request.remove_background))
        settings = {"colors": RECOMMENDED_COLORS, "merge": merge, "detail": detail, "speckle": request.speckle}
        return {"settings": settings, **result(entry, svgs[detail], request.remove_background)}


@app.get("/gallery")
def gallery_page() -> FileResponse:
    return FileResponse(STATIC / "gallery.html")


@app.get("/api/gallery")
def gallery_tiles() -> dict:
    return gallery.listing()


@app.get("/api/gallery/{tile_id}/original.png")
def gallery_original(tile_id: str) -> Response:
    data = gallery.file(tile_id, "original")
    if data is None:
        raise HTTPException(404, "not in the gallery")
    return Response(data, media_type="image/png")


@app.get("/api/gallery/{tile_id}/full.png")
def gallery_full(tile_id: str) -> Response:
    data = gallery.full_image(tile_id)
    if data is None:
        raise HTTPException(404, "not in the gallery")
    return Response(data, media_type="image/png")


@app.get("/api/gallery/{tile_id}/image.svg")
def gallery_svg(tile_id: str) -> Response:
    data = gallery.file(tile_id, "svg")
    if data is None:
        raise HTTPException(404, "not in the gallery")
    return Response(data, media_type="image/svg+xml")
