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
6. `text.py` (with **Has text**) finds lettering and draws every copy of a letter from one shared shape (see [Text](#text)).
7. `quality.py` draws the SVG back into pixels to measure how closely it matches the original.
8. `presets.py` holds ready-made settings for logos, cartoons and photos and guesses which one fits an image (see [Presets](#presets)).
9. `vectorize.py` puts it all together, `main.py` is the command line, `app.py` is the web app (FastAPI) and `static/index.html` is its page.
10. `gallery.py` and `static/gallery.html` are the gallery (see [Gallery](#gallery)).
11. `static/compare.js` (with `compare.css`) is the original-vs-SVG view with the split handle and click-to-zoom, used by the main page and the gallery viewer; `static/export.js` saves PNGs.
12. `upscale.py` is the optional AI upscaling (see [AI upscale](#ai-upscale)); the model is in `models/`.

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
| `--layers` | off | Separate into stacked layers (slower; needed for outlines and simple shapes). Too detailed pictures stay one layer. |
| `--no-shapes` | off | Trace ellipses, circles and rectangles like any other shape. |
| `--no-outlines` | off | Trace dark outlines like any other shape instead of redrawing them smoothly. |
| `--text` | off | The image has lettering: every copy of a letter is drawn from one shared shape (see [Text](#text)). |
| `--ai-upscale` | off | Upscale small drawings 4× with Real-ESRGAN before tracing (see [AI upscale](#ai-upscale)). |

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

Open http://127.0.0.1:8000 and drop an image on the page.

The page has two modes, switched at the top of the left panel and remembered by the browser:

- **Newbie** (the default on the first visit): everything is automatic. Only the Logo / Cartoon / Photo buttons, **Remove background**, the numbers and the downloads are shown. The SVG is one layer, and **AI upscale** is switched on whenever the image allows it (up to 1024 px), except with the Photo preset; a line under the presets says whether it is on. Switching back to Newbie drops your own settings and uses the automatic ones again.
- **Pro**: all settings described below. **✎ Edit** from the gallery always opens Pro, because the image comes with its own settings.

In Pro mode a preset is picked automatically; switch it with the Logo / Cartoon / Photo buttons or move the sliders. The SVG starts as **one layer with all colors**, which is fast even for big photos. **Separate into layers** stacks the shapes from big to small (starting at about 15 layers); then the **Layers** slider sets about how many layers there are (within ±3, all the way right keeps every layer), **Regenerate layers** makes another variant, and Smooth outlines / Detect simple shapes become available. Pictures that are too detailed for layers (more than 1000 areas, like busy photos) stay one layer, and the page says so. The **Layers** panel on the right lists every layer, top layer first; the eye button hides a layer (hidden layers are left out of the download). The **Match** value is the average color difference (ΔE) between the SVG and the original; under about 2 is hard to see. Drag across the preview to compare the original with the SVG, then click **Download SVG**.

### AI upscale

Small or blurry drawings and logos trace better when they are first upscaled by a neural network: Real-ESRGAN **animevideo v3** (`models/realesr-animevideov3.onnx`, 2.5 MB) makes edges sharp and colors flat, so thin outlines, letters and claws survive. It runs on the CPU with `onnxruntime` and needs no graphics card.

- Main page: tick **AI upscale**. The note under it shows how long the upscale took. Untick it to compare.
- Command line: `python main.py drawing.png --ai-upscale`.
- Gallery: results with and without AI get separate history tiles (the AI one is marked **History · AI**), so their error and the other numbers can be compared directly.

It is only offered for images up to 1024 px; larger ones already have enough pixels. It helps drawings and logos most, and small lettering: the edges of the upscaled image are already sharp, so the usual edge sharpening is skipped and every edge pixel goes to the nearest real color, which keeps letters straight with sharp corners instead of wavy (for example the text on `img/umbrella.png`). On photos and soft watercolor pictures the gain is small and the model changes the style (darker outlines, flatter shading). Typical time on a laptop CPU: well under a second for 256 px, about a second for 512 px, about 3 s for 880 × 660 px and 5 s for 1024 × 1024 px, which needs about 2.5 GB of memory (measured on Apple Silicon; older Intel laptops can be a few times slower).

In a test with drawings shrunk to 256 px and saved as JPEG, the error of the SVG against the full-size original dropped from 14.5 to 9.2 on the logo, from 15.9 to 10.4 on the dino and from 8.6 to 6.0 on the pixel-art figure.

### Text

Tick **Has text** (command line: `--text`) when the picture has lettering. Traced letters normally come out a little different each time, so two R's of the same word get their own bumps. With this option the letters are found first: small areas of one color with thin strokes, lying on one background color, in a row with other letters of about the same size. Every letter is turned upright along its line, so curved text (like on `img/umbrella.png`) works too. Copies of the same letter, also in another font size, are laid on top of each other and averaged into one clean shape, and every copy is drawn from that shape, turned and scaled back into place. The letters are drawn on top as a **Text** layer.

On `img/umbrella.png` all 19 letters are found and the R's, O's, A's and L's share their shapes. It works best together with **AI upscale**, which gives the letters sharp edges first. A row of shapes only counts as text if at least one of its letters repeats, so drawings are not mistaken for text; words without any repeated letter, and letters that touch each other, are traced normally. It adds about 1–1.5 s.

### PNG export

Both the main page and every gallery tile can also save the result as a PNG: pick a size (Ã—1, Ã—2, Ã—4 or Ã—8 of the image size; the list shows the pixel size) and click **Download PNG** / **PNG**. The browser draws the SVG straight at that size, so a PNG made from the vector stays sharp at Ã—8, unlike an enlarged raster image. Transparent backgrounds stay transparent. Sizes above 8192 px per side are disabled because browsers cannot draw such large images.

### Gallery

Open http://127.0.0.1:8000/gallery (or the **Gallery** link on the main page). When the server starts, every image in `img/` is converted in the background with its auto-detected preset; the gallery shows a loader until they are done. Each image you convert on the main page appears on top as history and is updated when you change its settings.

Click a tile's pictures to open it full screen with the same split view and zoom as the main page (click to zoom in, drag to move, drag the ↔ handle to compare). Use ← / → or the side buttons to go to the previous or next tile, and Esc, × or a click on the dark background to close it. The viewer has the same SVG and PNG download buttons as the tiles.

**✎ Edit** (on a tile or in the viewer) opens the image on the main page with the same settings, to keep working on it: the original at full size, the sliders, the checkboxes, the layer count and the variant. Your changes then appear as a new history tile; the one you opened stays as it was. It also works for an image that failed, for example one too detailed to split into layers, so you can retry it with fewer colors.

The **Tiles / Charts / Table** switch at the top shows the same results in three ways. **Charts** has one bar chart per number (error, file size, curves, paths, colors, time) with every image in the same order, best error first; hover or focus a bar to see its exact value and to highlight that image in all charts. The error chart marks the good and okay limits. **Table** lists every image with all numbers at once.

Every tile shows the original next to the SVG, the preset and these numbers:

| Number | Meaning |
|---|---|
| Error | Average color difference between the SVG and the original, 0â€“255, lower is better. With a removed background only the object counts (the pixels kept as the object plus everything the SVG paints), so the transparent background is not compared with the original background color. The rating in brackets is **good** below 5, **okay** below 12 and **poor** above. |
| Colors, Paths, Curves | Size of the vector drawing. |
| Size, Time | SVG file size and how long the conversion took (only for `img/`). |

The main converter always has priority: the gallery only works while no conversion from the main page is running. The gallery lives in memory, so it is rebuilt on every server start.


To leave the virtual environment, run `deactivate`.
