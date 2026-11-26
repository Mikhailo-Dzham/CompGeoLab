from typing import Optional, Tuple
import numpy as np
from core.primitives import polygon_area


def build_star_polygon(
    points: np.ndarray,
    z: np.ndarray,
    eps: float = 1e-12,
    max_attempts: int = 6,
) -> Tuple[Optional[np.ndarray], float, np.ndarray, Optional[np.ndarray]]:
    if len(points) < 3:
        return None, float("inf"), np.array(z, dtype=float), None

    z_curr = np.array(z, dtype=float)
    rng = np.random.default_rng(42)
    order = None

    for _ in range(max_attempts):
        dx = points[:, 0] - z_curr[0]
        dy = points[:, 1] - z_curr[1]
        angles = np.arctan2(dy, dx)
        dists = dx * dx + dy * dy
        order = np.lexsort((dists, angles))
        sorted_angles = angles[order]

        diffs = np.diff(sorted_angles)
        wrap_gap = 2.0 * np.pi - (sorted_angles[-1] - sorted_angles[0])
        if max(float(np.max(diffs)), float(wrap_gap)) >= np.pi - 1e-9:
            return None, float("inf"), z_curr, None

        if np.all(diffs > eps):
            ordered = points[order]
            return ordered, polygon_area(ordered), z_curr, order

        z_curr = z_curr + rng.uniform(-1e-5, 1e-5, size=2)

    ordered = points[order]
    return ordered, polygon_area(ordered), z_curr, order