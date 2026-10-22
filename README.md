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
4. `main.py` is the command line, `app.py` is the web app (FastAPI) and `static/index.html` is its page.

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
python main.py photo.jpg --detail 2 --merge 15 # smoother curves, fewer colors
python main.py --help
```

### Windows

```powershell
python main.py photo.png                      # saves photo.svg next to the input
python main.py photo.jpg --output result.svg
python main.py photo.jpg --detail 2 --merge 15 # smoother curves, fewer colors
python main.py --help
```

| Option | Default | Meaning |
|---|---|---|
| `--detail` | 4 | Curve detail 1–10. Lower values use fewer Bezier points and give smoother curves. |
| `--merge` | 8 | Colors closer than this ΔE are combined into one. |
| `--colors` | 16 | Maximum number of colors. |
| `--speckle` | 4 | Shapes smaller than this many pixels are dropped. |
| `--keep-background` | off | Keep the background instead of making it transparent. |

### Web page

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000, drop an image on the page and move the sliders. Drag across the preview to compare the original with the SVG, then click **Download SVG**.

To leave the virtual environment, run `deactivate`.
