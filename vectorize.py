import re

import numpy as np

import layers
import shapes
import strokes
from curves import refit_all
from preprocess import UPSCALE, WORK_SIDE

# Layers from bottom to top ({"color", "form" or "loops"}) and the outlines drawn over them.
Traced = tuple[list[dict], list[dict]]
SMOOTH_FLOOR = 2.0  # minimum smoothing (in edge points), pixel edges have stair-steps
SVG_HEAD = '<?xml version="1.0" encoding="UTF-8"?>\n<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '


def lerp(low: float, high: float, detail: int) -> float:
    """Map detail 1..10 onto low..high."""
    return low + (high - low) * (min(max(detail, 1), 10) - 1) / 9


def trace(image: np.ndarray, speckle: int = 4, outlines: bool = True, simple_shapes: bool = True) -> Traced:
    """Split a flat-color image into stacked layers and find their forms.

    Each layer is either a simple form (ellipse, circle, rectangle) or a set of pixel-edge loops
    that `finish` fits with curves. With `outlines`, dark thin lines are found first; the layers
    extend under them and they are drawn on top of everything.
    """
    over, found_outlines = strokes.extract(image) if outlines else (None, [])
    built = layers.build(image, over, speckle)
    forms = [shapes.fit(layer) if simple_shapes else None for layer in built]
    # Position in the stack; a layer can move just below a form that covers it (see below).
    position = [float(i) for i in range(len(built))]
    if simple_shapes:
        # Bigger layers go first, but a smaller one can lie *under* a bigger form in the picture
        # (a cheek behind an eye). If a layer becomes a simple form once the neighboring forms
        # may lie on top of it, it moves just below them.
        owner = np.full(image.shape[:2], -1, np.int32)
        for index, (layer, form) in enumerate(zip(built, forms)):
            if form:
                h, w = layer["region"].shape
                owner[layer["y"]:layer["y"] + h, layer["x"]:layer["x"] + w][layer["region"]] = index
        for index, layer in enumerate(built):
            if forms[index]:
                continue
            h, w = layer["region"].shape
            crop = owner[layer["y"]:layer["y"] + h, layer["x"]:layer["x"] + w]
            covering = (crop >= 0) & (crop < index)
            if covering.any():
                form = shapes.fit({**layer, "free": layer["free"] | covering})
                if form:
                    forms[index] = form
                    position[index] = crop[covering].min() - 0.5
    stacked = []
    for index in sorted(range(len(built)), key=lambda i: position[i]):
        layer, form = built[index], forms[index]
        if form:
            stacked.append({"color": layer["color"], "form": form})
        else:
            loops = layers.contours(layer)
            if loops:
                stacked.append({"color": layer["color"], "loops": loops})
    return stacked, found_outlines


def finish(traced: Traced, image: np.ndarray, detail: int = 4, size: tuple[int, int] | None = None) -> str:
    """Draw the layers bottom to top with smooth curves, then the outlines over them.

    Lower `detail` gives fewer Bezier points and rounder curves. `size` is the (width, height)
    the SVG should display at, normally the original image size.
    """
    stacked, found_outlines = traced
    height, width = image.shape[:2]
    # The pixel values below suit the largest images. Smaller images get proportionally less
    # smoothing, otherwise details only a few pixels wide (e.g. claws) are rounded away.
    scale = min(1.0, max(height, width) / (WORK_SIDE * UPSCALE))
    error = 6.0 * 0.1 ** ((min(max(detail, 1), 10) - 1) / 9) * scale  # 6 px down to 0.6 px, geometric steps
    # Edges come straight from pixels, so always smooth away at least the pixel stair-steps.
    sigma = max(SMOOTH_FLOOR, lerp(4.0, 0.5, detail) * scale)
    traced_layers = [layer for layer in stacked if "loops" in layer]
    paths = iter(refit_all([layer["loops"] for layer in traced_layers], error, lerp(115, 50, detail), sigma))
    elements = []
    for layer in stacked:
        if "form" in layer:
            elements.append(shapes.to_svg(layer["form"], layer["color"]))
        else:
            b, g, r = layer["color"]
            elements.append(f'<path d="{next(paths)}" fill="#{r:02x}{g:02x}{b:02x}" fill-rule="evenodd"/>')
    outline_svg = strokes.to_svg(found_outlines, error, sigma)
    if outline_svg:
        elements.append(outline_svg)
    display_width, display_height = size or (width // UPSCALE, height // UPSCALE)
    return (
        SVG_HEAD + f'width="{display_width}" height="{display_height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements) + "\n</svg>\n"
    )


def vectorize(
    image: np.ndarray,
    detail: int = 4,
    speckle: int = 4,
    size: tuple[int, int] | None = None,
    outlines: bool = True,
    simple_shapes: bool = True,
) -> str:
    """Convert a flat-color image to SVG. Higher `detail` keeps more corners and Bezier points."""
    return finish(trace(image, speckle, outlines, simple_shapes), image, detail, size)


def stats(svg: str) -> dict:
    return {
        "paths": len(re.findall(r"<(?:path|ellipse|circle|rect) ", svg)),
        "curves": len(re.findall(r"C", svg)),
        "bytes": len(svg.encode("utf-8")),
    }
