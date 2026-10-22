import threading
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from preprocess import decode, prepare, quantize
from presets import PRESETS, detect
from vectorize import finish, stats, trace

STATIC = Path(__file__).parent / "static"
MAX_IMAGES = 8

app = FastAPI(title="Image to SVG")

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
    preset = detect(image)
    prepared = prepare(image)
    image_id = uuid.uuid4().hex
    with lock:
        images[image_id] = {"prepared": prepared, "key": None, "quantized": None, "colors": 0, "trace_key": None, "raw": None}
        while len(images) > MAX_IMAGES:
            images.popitem(last=False)
    height, width = prepared.shape[:2]
    return {"id": image_id, "width": width, "height": height, "preset": preset}


@app.post("/api/vectorize")
def convert(settings: Settings) -> dict:
    with lock:
        entry = images.get(settings.id)
    if entry is None:
        raise HTTPException(404, "image not found, please upload it again")
    key = (settings.colors, settings.merge, settings.remove_background)
    if entry["key"] != key:
        try:
            entry["quantized"], entry["colors"] = quantize(
                entry["prepared"], settings.colors, settings.merge, settings.remove_background
            )
        except ValueError as error:
            raise HTTPException(400, str(error))
        entry["key"] = key
    trace_key = (key, settings.speckle)
    if entry["trace_key"] != trace_key:
        entry["raw"] = trace(entry["quantized"], settings.speckle)
        entry["trace_key"] = trace_key
    svg = finish(entry["raw"], entry["quantized"], settings.detail)
    return {"svg": svg, "colors": entry["colors"], **stats(svg)}
