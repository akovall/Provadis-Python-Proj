"""Measure how closely an SVG matches the original image."""

import re

import cv2
import numpy as np

from curves import parse
from preprocess import to_lab

WRONG = 10  # a pixel this many ΔE away from the original counts as clearly wrong


def number(tag: str, name: str) -> float:
    match = re.search(rf' {name}="([-\d.e]+)"', tag)
    return float(match.group(1)) if match else 0.0


def rasterize(svg: str, width: int, height: int) -> np.ndarray:
    """Draw the SVG's background, paths and simple shapes, in order, into a BGRA image of the given size."""
    canvas = np.zeros((height, width, 4), np.uint8)
    view = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    scale = width / float(view.group(1)) if view else 1.0
    for tag in re.findall(r"<(?:path|ellipse|circle|rect) [^>]*>", svg):
        fill = re.search(r'fill="#([0-9a-fA-F]{6})"', tag)
        if not fill:
            continue
        color = (*bytes.fromhex(fill.group(1))[::-1], 255)
        rotate = re.search(r"rotate\(([-\d.e]+)", tag)
        angle = float(rotate.group(1)) if rotate else 0.0
        if 'width="100%"' in tag:
            canvas[:] = color
        elif tag.startswith("<path"):
            d = re.search(r'd="([^"]*)"', tag)
            move = re.search(r"translate\(([-\d.e]+),([-\d.e]+)\)", tag)
            offset = np.array([float(move.group(1)), float(move.group(2))]) if move else np.zeros(2)
            loops = [np.round((loop + offset) * scale * 16).astype(np.int32) for loop in parse(d.group(1))]
            if loops:
                cv2.fillPoly(canvas, loops, color, cv2.LINE_8, shift=4)
        elif tag.startswith("<rect"):
            w, h = number(tag, "width"), number(tag, "height")
            center = ((number(tag, "x") + w / 2) * scale, (number(tag, "y") + h / 2) * scale)
            box = cv2.boxPoints((center, (w * scale, h * scale), angle))
            cv2.fillPoly(canvas, [np.round(box * 16).astype(np.int32)], color, cv2.LINE_8, shift=4)
        else:
            center = (number(tag, "cx") * scale, number(tag, "cy") * scale)
            rx = number(tag, "rx") or number(tag, "r")
            ry = number(tag, "ry") or number(tag, "r")
            cv2.ellipse(canvas, (center, (2 * rx * scale, 2 * ry * scale), angle), color, -1)
    return canvas


def on_white(bgra: np.ndarray) -> np.ndarray:
    alpha = bgra[:, :, 3:4].astype(np.float32) / 255
    return (bgra[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)


def difference(svg: str, reference: np.ndarray) -> dict[str, float]:
    """Compare the SVG with the reference BGRA image, both on white.

    `delta` is the mean ΔE (color closeness), `wrong` the share of clearly wrong pixels in %
    (shape closeness: it grows when outlines drift). Lower is closer for both.
    """
    height, width = reference.shape[:2]
    drawn = to_lab(on_white(rasterize(svg, width, height)))
    distance = np.linalg.norm(drawn - to_lab(on_white(reference)), axis=2)
    return {"delta": float(distance.mean()), "wrong": float(100 * np.mean(distance > WRONG))}
