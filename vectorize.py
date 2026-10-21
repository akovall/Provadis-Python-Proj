import cv2
import numpy as np
import vtracer


def vectorize(image: np.ndarray, output_path: str) -> str:
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("cannot encode image to PNG")
    svg = vtracer.convert_raw_image_to_svg(
        png.tobytes(),
        img_format="png",
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=4,
        color_precision=6,
        layer_difference=16,
    )
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(svg)
    return output_path
