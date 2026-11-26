import time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from core.star_generator import build_star_polygon


@dataclass
class OptimizationResult:
    best_polygon: np.ndarray
    best_area: float
    best_z: np.ndarray
    variants: List[Tuple[np.ndarray, float, np.ndarray]]
    checked_count: int
    elapsed_ms: float


def _canonical_order(order: np.ndarray) -> Tuple[int, ...]:
    zero_pos = int(np.where(order == 0)[0][0])
    rotated = np.roll(order, -zero_pos)
    return tuple(rotated.tolist())


def _line_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray):
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2 * p3[0] + b2 * p3[1]

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    return np.array([x, y], dtype=float)


def _generate_candidates(points: np.ndarray, exact_mode: bool) -> List[np.ndarray]:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)

    candidates = [points.mean(axis=0), np.median(points, axis=0)]
    grid_side = 6 if exact_mode else 4
    xs = np.linspace(min_xy[0] + 0.08 * span[0], max_xy[0] - 0.08 * span[0], grid_side)
    ys = np.linspace(min_xy[1] + 0.08 * span[1], max_xy[1] - 0.08 * span[1], grid_side)
    for x in xs:
        for y in ys:
            candidates.append(np.array([x, y], dtype=float))

    if exact_mode:
        rng = np.random.default_rng(7)
        n = len(points)
        for _ in range(min(120, n * n)):
            idx = rng.choice(n, size=4, replace=False) if n >= 4 else None
            if idx is None:
                break
            pt = _line_intersection(points[idx[0]], points[idx[1]], points[idx[2]], points[idx[3]])
            if pt is not None:
                if np.all(pt >= min_xy) and np.all(pt <= max_xy):
                    candidates.append(pt + rng.uniform(-1e-3, 1e-3, size=2) * span)

    return candidates


def find_min_area_star_polygon(
    points: np.ndarray,
    exact_threshold: int = 20,
) -> OptimizationResult:
    t0 = time.perf_counter()
    n = len(points)
    exact_mode = n <= exact_threshold

    candidates = _generate_candidates(points, exact_mode)
    seen_orders = set()
    variants: List[Tuple[np.ndarray, float, np.ndarray]] = []
    best_poly = None
    best_area = float("inf")
    best_z = points.mean(axis=0)
    checked = 0

    for z in candidates:
        poly, area, z_used, order = build_star_polygon(points, z)
        if poly is None:
            continue
        checked += 1
        if exact_mode:
            key = _canonical_order(order)
            if key not in seen_orders:
                seen_orders.add(key)
                variants.append((poly, area, z_used))
        if area < best_area:
            best_area = area
            best_poly = poly
            best_z = z_used

    span = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-6)
    step = 0.15 * span
    directions = np.array(
        [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]],
        dtype=float,
    )

    for _ in range(4):
        improved = False
        for d in directions:
            z_try = best_z + d * step
            poly, area, z_used, order = build_star_polygon(points, z_try)
            if poly is None:
                continue
            checked += 1
            if exact_mode:
                key = _canonical_order(order)
                if key not in seen_orders:
                    seen_orders.add(key)
                    variants.append((poly, area, z_used))
            if area < best_area:
                best_area = area
                best_poly = poly
                best_z = z_used
                improved = True
        if not improved:
            step *= 0.5

    if exact_mode:
        variants.sort(key=lambda item: item[1])
    elif best_poly is not None:
        variants = [(best_poly, best_area, best_z)]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return OptimizationResult(
        best_polygon=best_poly,
        best_area=best_area,
        best_z=best_z,
        variants=variants,
        checked_count=checked,
        elapsed_ms=elapsed_ms,
    )