import argparse
import sys
from pathlib import Path

from preprocess import load, prepare, quantize
from presets import PRESETS, detect
from vectorize import vectorize

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a JPG or PNG image to SVG.")
    parser.add_argument("image", help="path to a .jpg, .jpeg or .png file")
    parser.add_argument("--output", help="path to the output .svg file (default: next to the input image)")
    parser.add_argument(
        "--preset", choices=["auto", *PRESETS], default="auto",
        help="settings for the kind of image (default: auto, guessed from the image)",
    )
    # The options below override the preset's values
    parser.add_argument("--colors", type=int, help="maximum number of colors (default: from the preset)")
    parser.add_argument("--merge", type=float, help="merge colors closer than this ΔE (default: from the preset)")
    parser.add_argument("--detail", type=int, help="curve detail 1-10, higher keeps more points (default: from the preset)")
    parser.add_argument("--speckle", type=int, help="drop shapes smaller than this many pixels (default: from the preset)")
    parser.add_argument("--keep-background", action="store_true", help="keep the background instead of making it transparent")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.is_file():
        sys.exit(f"Error: file not found: {path}")
    if path.suffix.lower() not in EXTENSIONS:
        sys.exit("Error: only .jpg, .jpeg and .png files are supported")

    output = args.output or str(path.with_suffix(".svg"))
    try:
        image = load(str(path))
        preset = detect(image) if args.preset == "auto" else args.preset
        settings = PRESETS[preset] | {name: value for name, value in vars(args).items() if name in PRESETS[preset] and value is not None}
        if args.keep_background:
            settings["remove_background"] = False
        print(f"Preset: {preset}" + (" (auto)" if args.preset == "auto" else ""))
        flat, _ = quantize(prepare(image), settings["colors"], settings["merge"], settings["remove_background"])
        with open(output, "w", encoding="utf-8") as file:
            file.write(vectorize(flat, settings["detail"], settings["speckle"]))
        print(f"SVG saved to {output}")
    except (ValueError, OSError) as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
