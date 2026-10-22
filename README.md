# Provadis-Python-Proj

A small tool that converts raster images (`.jpg`, `.jpeg`, `.png`) to vector graphics (`.svg`), with a command line and a web page with live sliders.

## How it works

1. `preprocess.py` prepares the image with OpenCV:
   - separates the object from the background: a real alpha channel is used if the image has one, otherwise the background is the area of border-like colors connected to the image edge (same-colored areas inside the object stay),
   - brings the image to a working size of about 1024 px (large images are shrunk, small ones enlarged so thin lines survive),
   - flattens textures and JPEG noise with mean-shift filtering, then lightly sharpens soft anti-aliased edges with a toggle filter,
   - picks a color palette with k-means in Lab color space and merges colors that look alike (ΔE threshold),
   - upscales 2× and assigns every pixel to a palette color, then runs a majority filter to remove stair-steps and specks.
2. `layers.py` splits the flat-color image into stacked layers, from the biggest area at the bottom to the smallest on top. Layers are not cut out where smaller ones lie on them: a face stays one whole shape with the eyes lying on top. Thin anti-aliasing fringes and specks are merged into their neighbors first. Shapes of the same color at the same depth (for example the separate lines on the mushroom's cap) share one layer. With a layer limit, the least important layers are blended into their surroundings until about that many are left.
3. `shapes.py` checks every layer against simple forms: layers that are ellipses, circles or rectangles (for example eyes) become exact SVG `<ellipse>`, `<circle>` or `<rect>` elements. Only the visible part has to match, so a shape partly behind another one is recognized too.
4. `curves.py` draws all other layers with as few Bezier curves as possible: it finds real corners (they stay sharp), smooths the parts between them and fits curves within an error tolerance. The curve detail setting controls the tolerance, so it decides how many Bezier points are kept.
5. `strokes.py` finds outlines (dark, long, thin shapes like contour lines or a mouth) and draws them on top of everything, around a smooth centerline. The thickness is measured from the original at every point, so lines stay thick or thin where they were, just without the wobble.
6. `quality.py` draws the SVG back into pixels to measure how closely it matches the original, and picks the recommended settings.
7. `presets.py` holds ready-made settings for logos, cartoons and photos and guesses which one fits an image (see [Presets](#presets)).
8. `vectorize.py` puts it all together, `main.py` is the command line, `app.py` is the web app (FastAPI) and `static/index.html` is its page.
9. `gallery.py` and `static/gallery.html` are the gallery (see [Gallery](#gallery)).
10. `static/compare.js` (with `compare.css`) is the original-vs-SVG view with the split handle and click-to-zoom, used by the main page and the gallery viewer; `static/export.js` saves PNGs.

## Installation

Requires Python 3.10 or newer.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `cmd` instead.

## Usage

Activate the virtual environment first (see above).

### macOS / Linux

```bash
python main.py photo.png                      # saves photo.svg next to the input
python main.py photo.jpg --output result.svg
python main.py logo.png --preset logo          # pick the preset yourself
python main.py photo.jpg --detail 2 --merge 15 # smoother curves, fewer colors
python main.py --help
```

### Windows

```powershell
python main.py photo.png                      # saves photo.svg next to the input
python main.py photo.jpg --output result.svg
python main.py logo.png --preset logo          # pick the preset yourself
python main.py photo.jpg --detail 2 --merge 15 # smoother curves, fewer colors
python main.py --help
```

| Option | Default | Meaning |
|---|---|---|
| `--preset` | auto | `logo`, `cartoon` or `photo`. `auto` guesses it from the image. |
| `--detail` | from preset | Curve detail 1–10. Lower values use fewer Bezier points and give smoother curves. |
| `--merge` | from preset | Colors closer than this ΔE are combined into one. |
| `--colors` | from preset | Maximum number of colors. |
| `--speckle` | from preset | Shapes smaller than this many pixels are dropped. |
| `--keep-background` | off | Keep the background instead of making it transparent. |
| `--seed` | 0 | Another number gives another variant of the colors and layers. |
| `--no-shapes` | off | Trace ellipses, circles and rectangles like any other shape. |
| `--no-outlines` | off | Trace dark outlines like any other shape instead of redrawing them smoothly. |

### Presets

| Preset | Colors | Merge | Detail | Speckle | Background | For |
|---|---|---|---|---|---|---|
| `logo` | 8 | 12 | 6 | 4 | removed | Logos and icons with a few flat colors |
| `cartoon` | 16 | 8 | 4 | 4 | removed | Drawings and cartoon characters |
| `photo` | 32 | 4 | 5 | 2 | kept | Photos and detailed scenes |

`auto` picks `photo` when less than 30% of the image is flat (photos have texture almost everywhere), otherwise `logo` when about 12 colors cover 95% of the image, otherwise `cartoon`. Photo SVGs are large (often around 1 MB) and take a few seconds.

### Web page

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000, drop an image on the page and move the sliders. The **Layers** panel on the right lists every layer, top layer first, with a thumbnail; the eye button hides a layer (hidden layers are left out of the download). The **Layers** slider sets about how many layers the SVG should have (within ±3). **Regenerate layers** keeps the settings and makes another variant of the colors and layers. **Recommended for this image** tries several settings, compares each with the original and picks the simplest one that still matches closely. The **Match** value is the average color difference (ΔE) between the SVG and the original; under about 2 is hard to see. Drag across the preview to compare the original with the SVG, then click **Download SVG**.
Open http://127.0.0.1:8000 and drop an image on the page. A preset is picked automatically; switch it with the Logo / Cartoon / Photo buttons or move the sliders. **Recommended for this image** tries several settings, compares each with the original and picks the simplest one that still matches closely. The **Match** value is the average color difference (ΔE) between the SVG and the original; under about 2 is hard to see. Drag across the preview to compare the original with the SVG, then click **Download SVG**.

### PNG export

Both the main page and every gallery tile can also save the result as a PNG: pick a size (Ã—1, Ã—2, Ã—4 or Ã—8 of the image size; the list shows the pixel size) and click **Download PNG** / **PNG**. The browser draws the SVG straight at that size, so a PNG made from the vector stays sharp at Ã—8, unlike an enlarged raster image. Transparent backgrounds stay transparent. Sizes above 8192 px per side are disabled because browsers cannot draw such large images.

### Gallery

Open http://127.0.0.1:8000/gallery (or the **Gallery** link on the main page). When the server starts, every image in `img/` is converted in the background with its auto-detected preset; the gallery shows a loader until they are done. Each image you convert on the main page appears on top as history and is updated when you change its settings.

Click a tile's pictures to open it full screen with the same split view and zoom as the main page (click to zoom in, drag to move, drag the ↔ handle to compare). Use ← / → or the side buttons to go to the previous or next tile, and Esc, × or a click on the dark background to close it. The viewer has the same SVG and PNG download buttons as the tiles.

The **Tiles / Charts / Table** switch at the top shows the same results in three ways. **Charts** has one bar chart per number (error, file size, curves, paths, colors, time) with every image in the same order, best error first; hover or focus a bar to see its exact value and to highlight that image in all charts. The error chart marks the good and okay limits. **Table** lists every image with all numbers at once.

Every tile shows the original next to the SVG, the preset and these numbers:

| Number | Meaning |
|---|---|
| Error | Average color difference between the SVG and the original, 0â€“255, lower is better. With a removed background only the object counts (the pixels kept as the object plus everything the SVG paints), so the transparent background is not compared with the original background color. The rating in brackets is **good** below 5, **okay** below 12 and **poor** above. |
| Colors, Paths, Curves | Size of the vector drawing. |
| Size, Time | SVG file size and how long the conversion took (only for `img/`). |

The main converter always has priority: the gallery only works while no conversion from the main page is running. The gallery lives in memory, so it is rebuilt on every server start.


To leave the virtual environment, run `deactivate`.
