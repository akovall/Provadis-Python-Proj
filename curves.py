"""Fit shape edges with as few cubic Bezier curves as possible.

Every edge is sampled into a dense polyline, real corners are detected (they stay sharp),
the parts between corners are smoothed and then fitted with Schneider's curve-fitting
algorithm (Graphics Gems, 1990).
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

SPACING = 2.0  # distance between resampled points, in traced pixels
LONG_SPACING = 3.0  # on long edges, where a pixel more or less is not visible
LONG_EDGE = 800  # edges longer than this many pixels count as long
WINDOW = 3  # points on each side used to measure the turning angle
SHARPNESS = 15  # degrees of turning at which the error tolerance and smoothing are halved
SMALL_LOOP = 120  # edges with fewer pixel points get a quick smooth spline instead of the full fit
RELATIVE_ERROR = 0.04  # tolerance never exceeds this share of a shape's size, so small shapes keep detail


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


def resample(loop: np.ndarray, spacing: float = SPACING) -> np.ndarray:
    """Evenly spaced points along a closed polyline."""
    closed = np.vstack([loop, loop[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.concatenate([[0], np.cumsum(lengths)])
    count = max(8, int(distance[-1] / spacing))
    targets = np.linspace(0, distance[-1], count, endpoint=False)
    return np.column_stack([np.interp(targets, distance, closed[:, k]) for k in range(2)])


def turning(points: np.ndarray) -> np.ndarray:
    """How many degrees the closed outline turns around each point."""
    before = points - np.roll(points, WINDOW, axis=0)
    after = np.roll(points, -WINDOW, axis=0) - points
    cos = np.sum(before * after, axis=1) / (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def corners(turn: np.ndarray, angle: float) -> list[int]:
    """Indices where the outline turns sharper than `angle` degrees."""
    result = []
    for i in np.argsort(-turn):
        if turn[i] < angle:
            break
        if all(min(abs(i - j), len(turn) - abs(i - j)) > 2 * WINDOW for j in result):
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
    result = np.column_stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(points.shape[1])])
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


def fit(points: np.ndarray, left: np.ndarray, right: np.ndarray, error: np.ndarray) -> list[np.ndarray]:
    """Fit one piece with cubic curves, splitting where the error is largest.

    `error` is the allowed distance for each point, so sharp parts can get a tighter fit.
    """
    if len(points) < 3:
        chord = np.linalg.norm(points[-1] - points[0]) / 3
        return [np.array([points[0], points[0] + left * chord, points[-1] + right * chord, points[-1]])]
    u = chord_params(points)
    for _ in range(2):  # more refinement passes barely save curves but cost a lot of time
        control = generate(points, u, left, right)
        ratio = np.sum((bezier(control, u[:, None]) - points) ** 2, axis=1) / error**2
        split = int(np.argmax(ratio))
        if ratio[split] <= 1:
            return [control]
        if ratio[split] > 16:
            break
        u = reparameterize(control, points, u)
    split = min(max(split, 1), len(points) - 2)
    middle = unit(points[split - 1] - points[split + 1])
    return fit(points[:split + 1], left, middle, error[:split + 1]) + fit(points[split:], -middle, right, error[split:])


def spline(loop: np.ndarray) -> list[np.ndarray]:
    """Closed Catmull-Rom spline through a few evenly spaced points, as cubic Bezier curves.

    Small shapes have no corners worth keeping, and this is many times faster than fitting.
    """
    points = resample(loop)
    points = smooth(points[::max(1, len(points) // max(4, len(points) // 3))], 1.0, closed=True)
    before, after, after2 = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0), np.roll(points, -2, axis=0)
    first = points + (after - before) / 6
    second = after - (after2 - points) / 6
    return [np.array(c) for c in zip(points, first, second, after)]


def refit_loop(loop: np.ndarray, error: float, corner_angle: float, sigma: float) -> list[np.ndarray]:
    if len(loop) < SMALL_LOOP:
        return spline(loop)
    # Long edges are sampled less densely (much less work); the smoothing stays the same in pixels.
    spacing = LONG_SPACING if len(loop) > LONG_EDGE else SPACING
    sigma *= SPACING / spacing
    points = resample(loop, spacing)
    turn = turning(points)
    # Sharp parts get less smoothing and a tighter fit (more points); gentle curves get fewer points.
    keep = (turn / SHARPNESS / (1 + turn / SHARPNESS))[:, None]
    size = np.min(points.max(axis=0) - points.min(axis=0))
    tolerance = np.minimum(error / (1 + turn / SHARPNESS), max(0.5, RELATIVE_ERROR * size))
    found = corners(turn, corner_angle)
    if not found:
        points = keep * points + (1 - keep) * smooth(points, sigma, closed=True)
        # Start at an arbitrary point with the same tangent on both sides so the joint is smooth.
        tangent = unit(points[1] - points[-1])
        closed = np.vstack([points, points[:1]])
        return fit(closed, tangent, -tangent, np.append(tolerance, tolerance[0]))
    points, keep, tolerance = (np.roll(a, -found[0], axis=0) for a in (points, keep, tolerance))
    cuts = [c - found[0] for c in found] + [len(points)]
    closed = np.vstack([points, points[:1]])
    keep, tolerance = np.vstack([keep, keep[:1]]), np.append(tolerance, tolerance[0])
    curves = []
    for start, end in zip(cuts, cuts[1:]):
        piece = closed[start:end + 1]
        piece = keep[start:end + 1] * piece + (1 - keep[start:end + 1]) * smooth(piece, sigma, closed=False)
        curves += fit(piece, end_tangent(piece), end_tangent(piece, reverse=True), tolerance[start:end + 1])
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


_pool: ProcessPoolExecutor | None = None


def pool() -> ProcessPoolExecutor:
    """Worker processes shared by all calls; started on first use."""
    global _pool
    if _pool is None:
        # One math thread per worker: the workers already use all cores, more threads only compete.
        # The workers are started fresh, so they read these settings when they import numpy.
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        _pool = ProcessPoolExecutor(max_workers=os.cpu_count())
    return _pool


def refit_loops(loops: list[np.ndarray], error: float, corner_angle: float, sigma: float) -> str:
    """Smooth path data for one shape given as pixel-edge loops (outer edges and holes)."""
    return to_path([refit_loop(loop, error, corner_angle, sigma) for loop in loops])


def refit_all(shapes: list[list[np.ndarray]], error: float, corner_angle: float, sigma: float) -> list[str]:
    """Path data for many shapes; they are independent, so they are fitted in parallel on all CPU cores."""
    work = partial(refit_loops, error=error, corner_angle=corner_angle, sigma=sigma)
    if len(shapes) <= 8:
        return [work(loops) for loops in shapes]
    # Start the biggest shapes first so no worker is left with a big one at the end.
    order = sorted(range(len(shapes)), key=lambda i: -sum(len(loop) for loop in shapes[i]))
    futures = {i: pool().submit(work, shapes[i]) for i in order}
    return [futures[i].result() for i in range(len(shapes))]
