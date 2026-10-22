import argparse
import sys
from pathlib import Path

from preprocess import load, prepare, quantize
from vectorize import vectorize

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a JPG or PNG image to SVG.")
    parser.add_argument("image", help="path to a .jpg, .jpeg or .png file")
    parser.add_argument("--output", help="path to the output .svg file (default: next to the input image)")
    parser.add_argument("--colors", type=int, default=16, help="maximum number of colors (default: 16)")
    parser.add_argument("--merge", type=float, default=8.0, help="merge colors closer than this ΔE (default: 8)")
    parser.add_argument("--detail", type=int, default=4, help="curve detail 1-10, higher keeps more points (default: 4)")
    parser.add_argument("--speckle", type=int, default=4, help="drop shapes smaller than this many pixels (default: 4)")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.is_file():
        sys.exit(f"Error: file not found: {path}")
    if path.suffix.lower() not in EXTENSIONS:
        sys.exit("Error: only .jpg, .jpeg and .png files are supported")

    output = args.output or str(path.with_suffix(".svg"))
    try:
        image, _ = quantize(prepare(load(str(path))), args.colors, args.merge)
        with open(output, "w", encoding="utf-8") as file:
            file.write(vectorize(image, args.detail, args.speckle))
        print(f"SVG saved to {output}")
    except (ValueError, OSError) as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
