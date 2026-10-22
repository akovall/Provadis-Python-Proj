"""Refit traced outlines with as few cubic Bezier curves as possible.

vtracer splits outlines into many short segments. Here every outline is sampled into
a dense polyline, real corners are detected, the parts between corners are smoothed
and then refitted with Schneider's curve-fitting algorithm (Graphics Gems, 1990).
"""

import re

import numpy as np

SPACING = 1.5  # distance between resampled points, in traced pixels
WINDOW = 4  # points on each side used to measure the turning angle


def parse(d: str) -> list[np.ndarray]:
    """Split an absolute M/C/L/Z path into closed polylines."""
    loops, points, current = [], [], None
    for command, args in re.findall(r"([MCLZ])([^MCLZ]*)", d):
        numbers = [float(n) for n in args.replace(",", " ").split()]
        if command == "M":
            if len(points) > 2:
                loops.append(np.array(points))
            current = np.array(numbers[:2])
            points = [current]
        elif command == "L":
            for i in range(0, len(numbers), 2):
                current = np.array(numbers[i:i + 2])
                points.append(current)
        elif command == "C":
            for i in range(0, len(numbers), 6):
                p1, p2, p3 = (np.array(numbers[i + j:i + j + 2]) for j in (0, 2, 4))
                steps = max(2, int(np.linalg.norm(p1 - current) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)))
                t = np.linspace(0, 1, steps + 1)[1:, None]
                points.extend(bezier(np.array([current, p1, p2, p3]), t))
                current = p3
        elif command == "Z" and len(points) > 2:
            loops.append(np.array(points))
            points = []
    if len(points) > 2:
        loops.append(np.array(points))
    return loops


def resample(loop: np.ndarray) -> np.ndarray:
    """Evenly spaced points along a closed polyline."""
    closed = np.vstack([loop, loop[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.concatenate([[0], np.cumsum(lengths)])
    count = max(8, int(distance[-1] / SPACING))
    targets = np.linspace(0, distance[-1], count, endpoint=False)
    return np.column_stack([np.interp(targets, distance, closed[:, k]) for k in range(2)])


def corners(points: np.ndarray, angle: float) -> list[int]:
    """Indices where the outline turns sharper than `angle` degrees."""
    before = points - np.roll(points, WINDOW, axis=0)
    after = np.roll(points, -WINDOW, axis=0) - points
    cos = np.sum(before * after, axis=1) / (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1) + 1e-9)
    turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    result = []
    for i in np.argsort(-turn):
        if turn[i] < angle:
            break
        if all(min(abs(i - j), len(points) - abs(i - j)) > 2 * WINDOW for j in result):
            result.append(int(i))
    return sorted(result)


def smooth(points: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    """Gaussian smoothing along the outline; open pieces keep their end points."""
    if sigma <= 0 or len(points) < 5:
        return points
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    mode = "wrap" if closed else "edge"
    padded = np.pad(points, ((radius, radius), (0, 0)), mode=mode)
    result = np.column_stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(2)])
    if not closed:
        result[0], result[-1] = points[0], points[-1]
    return result


def bezier(control: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t.ravel()
    s = 1 - t
    return np.column_stack([s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t]) @ control


def unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-9 else vector


def end_tangent(points: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Direction leaving one end of a piece, averaged over a few points for stability."""
    if reverse:
        points = points[::-1]
    return unit(points[min(WINDOW, len(points) - 1)] - points[0])


def chord_params(points: np.ndarray) -> np.ndarray:
    distance = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    return distance / distance[-1] if distance[-1] > 0 else np.linspace(0, 1, len(points))


def generate(points: np.ndarray, u: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Least-squares control points for fixed end points and tangent directions."""
    first, last = points[0], points[-1]
    s = 1 - u
    a1 = (3 * s**2 * u)[:, None] * left
    a2 = (3 * s * u**2)[:, None] * right
    rest = points - bezier(np.array([first, first, last, last]), u[:, None])
    c00, c01, c11 = np.sum(a1 * a1), np.sum(a1 * a2), np.sum(a2 * a2)
    x0, x1 = np.sum(a1 * rest), np.sum(a2 * rest)
    det = c00 * c11 - c01 * c01
    chord = np.linalg.norm(last - first)
    alpha1 = (x0 * c11 - x1 * c01) / det if abs(det) > 1e-12 else 0
    alpha2 = (c00 * x1 - c01 * x0) / det if abs(det) > 1e-12 else 0
    # Handles longer than the outline itself overshoot on thin slivers; fall back to a safe guess.
    arc = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    if alpha1 < 1e-6 * chord or alpha2 < 1e-6 * chord or max(alpha1, alpha2) > arc / 2:
        alpha1 = alpha2 = chord / 3
    return np.array([first, first + left * alpha1, last + right * alpha2, last])


def reparameterize(control: np.ndarray, points: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One Newton-Raphson step moving each parameter closer to its point."""
    t = u[:, None]
    s = 1 - t
    d1 = 3 * (s**2 * (control[1] - control[0]) + 2 * s * t * (control[2] - control[1]) + t**2 * (control[3] - control[2]))
    d2 = 6 * (s * (control[2] - 2 * control[1] + control[0]) + t * (control[3] - 2 * control[2] + control[1]))
    diff = bezier(control, t) - points
    numerator = np.sum(diff * d1, axis=1)
    denominator = np.sum(d1 * d1 + diff * d2, axis=1)
    step = np.where(np.abs(denominator) > 1e-12, numerator / np.where(denominator == 0, 1, denominator), 0)
    return np.clip(u - step, 0, 1)


def fit(points: np.ndarray, left: np.ndarray, right: np.ndarray, error: float) -> list[np.ndarray]:
    """Fit one piece with cubic curves, splitting where the error is largest."""
    if len(points) < 3:
        chord = np.linalg.norm(points[-1] - points[0]) / 3
        return [np.array([points[0], points[0] + left * chord, points[-1] + right * chord, points[-1]])]
    u = chord_params(points)
    for _ in range(5):
        control = generate(points, u, left, right)
        distances = np.sum((bezier(control, u[:, None]) - points) ** 2, axis=1)
        split = int(np.argmax(distances))
        if distances[split] <= error**2:
            return [control]
        if distances[split] > (4 * error) ** 2:
            break
        u = reparameterize(control, points, u)
    split = min(max(split, 1), len(points) - 2)
    middle = unit(points[split - 1] - points[split + 1])
    return fit(points[:split + 1], left, middle, error) + fit(points[split:], -middle, right, error)


def refit_loop(loop: np.ndarray, error: float, corner_angle: float, sigma: float) -> list[np.ndarray]:
    points = resample(loop)
    found = corners(points, corner_angle)
    if not found:
        points = smooth(points, sigma, closed=True)
        # Start at an arbitrary point with the same tangent on both sides so the joint is smooth.
        tangent = unit(points[1] - points[-1])
        closed = np.vstack([points, points[:1]])
        return fit(closed, tangent, -tangent, error)
    points = np.roll(points, -found[0], axis=0)
    cuts = [c - found[0] for c in found] + [len(points)]
    closed = np.vstack([points, points[:1]])
    curves = []
    for start, end in zip(cuts, cuts[1:]):
        piece = smooth(closed[start:end + 1], sigma, closed=False)
        curves += fit(piece, end_tangent(piece), end_tangent(piece, reverse=True), error)
    return curves


def to_path(loops: list[list[np.ndarray]]) -> str:
    parts = []
    for curves in loops:
        start = curves[0][0]
        parts.append(f"M{start[0]:.1f} {start[1]:.1f}")
        for c in curves:
            parts.append("C" + " ".join(f"{x:.1f} {y:.1f}" for x, y in c[1:]))
        parts.append("Z")
    return "".join(parts)


def refit(svg: str, error: float, corner_angle: float, sigma: float) -> str:
    """Replace every path's outline with a smoother one that has fewer Bezier points."""
    def replace(match: re.Match) -> str:
        loops = [refit_loop(loop, error, corner_angle, sigma) for loop in parse(match.group(1))]
        return f'd="{to_path(loops)}"' if loops else match.group(0)

    return re.sub(r'd="([^"]*)"', replace, svg)
