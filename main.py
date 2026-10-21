import argparse
import sys
from pathlib import Path

from preprocess import preprocess
from vectorize import vectorize

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a JPG or PNG image to SVG.")
    parser.add_argument("image", help="path to a .jpg, .jpeg or .png file")
    parser.add_argument("--output", help="path to the output .svg file (default: next to the input image)")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.is_file():
        sys.exit(f"Error: file not found: {path}")
    if path.suffix.lower() not in EXTENSIONS:
        sys.exit("Error: only .jpg, .jpeg and .png files are supported")

    output = args.output or str(path.with_suffix(".svg"))
    try:
        image = preprocess(str(path))
        print(f"SVG saved to {vectorize(image, output)}")
    except (ValueError, OSError) as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
