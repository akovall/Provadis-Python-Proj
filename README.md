# Provadis-Python-Proj

A small command-line tool that converts raster images (`.jpg`, `.jpeg`, `.png`) to vector graphics (`.svg`).

## How it works

1. `preprocess.py` prepares the image with OpenCV: it shrinks large images, removes noise and reduces the number of colors.
2. `vectorize.py` traces the image with [vtracer](https://github.com/visioncortex/vtracer) and saves the SVG.
3. `main.py` handles the command-line arguments.

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
python main.py --help
```

### Windows

```powershell
python main.py photo.png                      # saves photo.svg next to the input
python main.py photo.jpg --output result.svg
python main.py --help
```

To leave the virtual environment, run `deactivate`.
