import re

import numpy as np

import layers
import shapes
import strokes
from curves import refit_all
from preprocess import UPSCALE, WORK_SIDE

# {"layers": bottom to top, each {"color", "items": [{"form"} or {"loops"}]},
#  "outlines": drawn over them, "auto_layers": how many layers there were before simplifying}
Traced = dict
SMOOTH_FLOOR = 2.0  # minimum smoothing (in edge points), pixel edges have stair-steps
SVG_HEAD = '<?xml version="1.0" encoding="UTF-8"?>\n<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
LAYER_SLACK = 3  # the layer count may end up this far from the requested one
SIMPLIFY_ROUNDS = 5
AREAS_PER_LAYER = 10  # with a layer target, the picture is first cut down to about this many areas per layer
COLORS_PER_LAYER = 0.7  # ... and to this many colors per layer (a layer has one color)
LOOPS_PER_JOB = 150  # curve fitting is split into jobs of about this many edges, so all CPU cores help


def lerp(low: float, high: float, detail: int) -> float:
    """Map detail 1..10 onto low..high."""
    return low + (high - low) * (min(max(detail, 1), 10) - 1) / 9


def stack_layers(image: np.ndarray, over: np.ndarray | None, simple_shapes: bool) -> list[dict]:
    """Layers in drawing order, each with its simple form if it has one."""
    built = layers.build(image, over)
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
    return [{**built[i], "form": forms[i]} for i in sorted(range(len(built)), key=lambda i: position[i])]


def trace(
    image: np.ndarray,
    speckle: int = 4,
    outlines: bool = True,
    simple_shapes: bool = True,
    max_layers: int | None = None,
    separate: bool = False,
) -> Traced:
    """Split a flat-color image into stacked layers and find their forms.

    Each shape is either a simple form (ellipse, circle, rectangle) or a set of pixel-edge loops
    that `finish` fits with curves; shapes of one color at the same depth share a layer. With
    `outlines`, dark thin lines are found first; the layers extend under them and they are
    drawn on top of everything. With `max_layers`, the least important layers are blended into
    their surroundings until about that many are left.

    Without `separate` (the default) there is no layering: all colors end up in one layer,
    which is fast even for big photos; outlines, simple shapes and `max_layers` then do not apply.
    Pictures too detailed to split (busy photos, see layers.can_layer) also stay one layer;
    the result then has "too_detailed" set.
    """
    too_detailed = separate and not layers.can_layer(image, speckle)
    if not separate or too_detailed:
        flat = layers.flat(image, speckle)
        return {"layers": [], "flat": flat, "outlines": [], "auto_layers": 1, "too_detailed": too_detailed}
    if max_layers:
        # Building every layer first and removing most of them afterwards would be slow; cutting
        # the picture down to a few areas per wanted layer first is quick.
        image = layers.reduce_colors(image, max(2, round(COLORS_PER_LAYER * max_layers)))
        image = layers.reduce_regions(image, AREAS_PER_LAYER * max_layers, (speckle * UPSCALE) ** 2)
    image = layers.clean(image, speckle)  # once; the rounds below rebuild from this
    over, found_outlines = strokes.extract(image) if outlines else (None, [])
    outline_layers = len({tuple(o["color"]) for o in found_outlines})
    stack = stack_layers(image, over, simple_shapes)
    groups = layers.group(stack)
    result = collect(stack, groups)
    auto_layers = len(result) + outline_layers
    effect = 1.0  # how many layers disappeared per removed layer in the last round
    for _ in range(SIMPLIFY_ROUNDS):
        count = len(result) + outline_layers
        extra = count - (max_layers or 0)
        if not max_layers or extra <= LAYER_SLACK:
            break
        # A recolored shape does not always join an existing layer, so removing one layer can
        # lower the count by less than one; remove more when the last round fell short.
        remove = min(int(np.ceil(extra / max(effect, 0.5))), extra)
        image = layers.simplify(image, stack, groups, remove)
        stack = stack_layers(image, over, simple_shapes)
        groups = layers.group(stack)
        result = collect(stack, groups)
        effect = max(count - (len(result) + outline_layers), 0) / remove
    return {"layers": result, "flat": [], "outlines": found_outlines, "auto_layers": auto_layers, "too_detailed": False}


def collect(stack: list[dict], groups: list[list[int]]) -> list[dict]:
    """The layers to draw: each group's forms and pixel-edge loops (empty groups are dropped)."""
    result = []
    for members in groups:
        items = []
        for index in members:
            item = stack[index]
            if item["form"]:
                items.append({"form": item["form"]})
            else:
                loops = layers.contours(item)
                if loops:
                    items.append({"loops": loops})
        if items:
            result.append({"color": stack[members[0]]["color"], "items": items})
    return result


def bounds(points: np.ndarray, pad: float = 0) -> tuple[float, float, float, float]:
    return points[:, 0].min() - pad, points[:, 1].min() - pad, points[:, 0].max() + pad, points[:, 1].max() + pad


def item_bounds(item: dict) -> tuple[float, float, float, float]:
    if "loops" in item:
        return bounds(np.vstack(item["loops"]))
    form = item["form"]
    reach = max(form.get("rx", 0), form.get("ry", 0), np.hypot(form.get("w", 0), form.get("h", 0)) / 2)
    return form["cx"] - reach, form["cy"] - reach, form["cx"] + reach, form["cy"] + reach


def box_attribute(boxes: list[tuple[float, float, float, float]]) -> str:
    """The area a layer covers, so the layers panel can zoom its thumbnail onto it."""
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return f"{x0:.0f} {y0:.0f} {x1 - x0:.0f} {y1 - y0:.0f}"


def layer_name(items: list[dict]) -> str:
    kinds = {item["form"]["kind"] if "form" in item else "shape" for item in items}
    kind = kinds.pop() if len(kinds) == 1 else "shape"
    kind = {"ellipse": "Ellipse", "rect": "Rectangle", "shape": "Shape"}[kind]
    return kind if len(items) == 1 else f"{kind}s ×{len(items)}"


def finish(traced: Traced, image: np.ndarray, detail: int = 4, size: tuple[int, int] | None = None) -> str:
    """Draw the layers bottom to top with smooth curves, then the outlines over them.

    Every layer is an SVG group with data-layer (its number) and data-name, so the web page
    can list and hide them. Lower `detail` gives fewer Bezier points and rounder curves.
    `size` is the (width, height) the SVG should display at, normally the original image size.
    """
    height, width = image.shape[:2]
    # The pixel values below suit the largest images. Smaller images get proportionally less
    # smoothing, otherwise details only a few pixels wide (e.g. claws) are rounded away.
    scale = min(1.0, max(height, width) / (WORK_SIDE * UPSCALE))
    error = 6.0 * 0.1 ** ((min(max(detail, 1), 10) - 1) / 9) * scale  # 6 px down to 0.6 px, geometric steps
    # Edges come straight from pixels, so always smooth away at least the pixel stair-steps.
    sigma = max(SMOOTH_FLOOR, lerp(4.0, 0.5, detail) * scale)
    traced_items = [item for layer in traced["layers"] for item in layer["items"] if "loops" in item]
    # One color of a photo can have thousands of edges: split it into several jobs.
    flat_jobs = [(i, item["loops"][start:start + LOOPS_PER_JOB])
                 for i, item in enumerate(traced["flat"]) for start in range(0, len(item["loops"]), LOOPS_PER_JOB)]
    jobs = [item["loops"] for item in traced_items] + [loops for _, loops in flat_jobs]
    fitted = refit_all(jobs, error, lerp(115, 50, detail), sigma)
    paths = iter(fitted[:len(traced_items)])
    flat_paths = [""] * len(traced["flat"])
    for (i, _), d in zip(flat_jobs, fitted[len(traced_items):]):
        flat_paths[i] += d
    groups = []
    for layer in traced["layers"]:
        b, g, r = layer["color"]
        fill = f"#{r:02x}{g:02x}{b:02x}"
        parts = []
        for item in layer["items"]:
            if "form" in item:
                parts.append(shapes.to_svg(item["form"], layer["color"]))
            else:
                parts.append(f'<path d="{next(paths)}" fill="{fill}"/>')
        boxes = [item_bounds(item) for item in layer["items"]]
        groups.append((layer_name(layer["items"]), box_attribute(boxes), parts))
    if traced["flat"]:
        parts = []
        for item, d in zip(traced["flat"], flat_paths):
            b, g, r = item["color"]
            parts.append(f'<path d="{d}" fill="#{r:02x}{g:02x}{b:02x}"/>')
        boxes = [item_bounds(item) for item in traced["flat"]]
        groups.append((f"All colors ×{len(parts)}", box_attribute(boxes), parts))
    by_color: dict[tuple, list[dict]] = {}
    for outline in traced["outlines"]:
        by_color.setdefault(tuple(outline["color"]), []).append(outline)
    for outlines in by_color.values():
        parts = [part for part in (strokes.to_svg([o], error, sigma) for o in outlines) if part]
        boxes = [bounds(points, o["width"]) for o in outlines for points, _, _ in o["lines"]]
        groups.append(("Outline", box_attribute(boxes), parts))
    body = "\n".join(
        f'<g data-layer="{number}" data-name="{name}" data-box="{box}">\n' + "\n".join(parts) + "\n</g>"
        for number, (name, box, parts) in enumerate(groups, 1)
    )
    display_width, display_height = size or (width // UPSCALE, height // UPSCALE)
    return (
        SVG_HEAD + f'width="{display_width}" height="{display_height}" viewBox="0 0 {width} {height}">\n'
        + body + "\n</svg>\n"
    )


def vectorize(
    image: np.ndarray,
    detail: int = 4,
    speckle: int = 4,
    size: tuple[int, int] | None = None,
    outlines: bool = True,
    simple_shapes: bool = True,
    max_layers: int | None = None,
    separate: bool = False,
) -> str:
    """Convert a flat-color image to SVG. Higher `detail` keeps more corners and Bezier points."""
    return finish(trace(image, speckle, outlines, simple_shapes, max_layers, separate), image, detail, size)


def stats(svg: str) -> dict:
    return {
        "paths": len(re.findall(r"<(?:path|ellipse|circle|rect) ", svg)),
        "layers": svg.count("<g data-layer="),
        "curves": len(re.findall(r"C", svg)),
        "bytes": len(svg.encode("utf-8")),
    }
