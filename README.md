# Provadis-Python-Proj

A small tool that converts raster images (`.jpg`, `.jpeg`, `.png`) to vector graphics (`.svg`), with a command line and a web page with live sliders.

## How it works

1. `preprocess.py` prepares the image with OpenCV:
   - separates the object from the background: a real alpha channel is used if the image has one, otherwise the background is the area of border-like colors connected to the image edge (same-colored areas inside the object stay),
   - shrinks large images and flattens textures and JPEG noise with mean-shift filtering,
   - picks a color palette with k-means in Lab color space and merges colors that look alike (ΔE threshold),
   - upscales 2× and assigns every pixel to a palette color, then runs a majority filter to remove stair-steps and specks.
2. `vectorize.py` traces the flat-color image with [vtracer](https://github.com/visioncortex/vtracer).
3. `curves.py` refits every outline with as few Bezier curves as possible: it finds real corners, smooths the parts between them and fits curves within an error tolerance. The curve detail setting controls the tolerance, so it decides how many Bezier points are kept.
4. `presets.py` holds ready-made settings for logos, cartoons and photos and guesses which one fits an image (see [Presets](#presets)).
5. `main.py` is the command line, `app.py` is the web app (FastAPI) and `static/index.html` is its page.
6. `gallery.py` and `static/gallery.html` are the gallery (see [Gallery](#gallery)).

## Installation

Requires Python 3.10–3.13. vtracer crashes on Python 3.14.

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

Open http://127.0.0.1:8000 and drop an image on the page. A preset is picked automatically; switch it with the Logo / Cartoon / Photo buttons or move the sliders. Drag across the preview to compare the original with the SVG, then click **Download SVG**.

### Gallery

Open http://127.0.0.1:8000/gallery (or the **Gallery** link on the main page). When the server starts, every image in `img/` is converted in the background with its auto-detected preset; the gallery shows a loader until they are done. Each image you convert on the main page appears on top as history and is updated when you change its settings.

Every tile shows the original next to the SVG, the preset and these numbers:

| Number | Meaning |
|---|---|
| Error | Average color difference between the SVG and the original, 0–255, lower is better. With a removed background only the object counts (the pixels kept as the object plus everything the SVG paints), so the transparent background is not compared with the original background color. The rating in brackets is **good** below 5, **okay** below 12 and **poor** above. |
| Colors, Paths, Curves | Size of the vector drawing. |
| Size, Time | SVG file size and how long the conversion took (only for `img/`). |

The main converter always has priority: the gallery only works while no conversion from the main page is running. The gallery lives in memory, so it is rebuilt on every server start.

To leave the virtual environment, run `deactivate`.
