import time
from typing import List, Tuple
import numpy as np
from core.star_generator import build_star_polygon
from core.optimizer import find_min_area_star_polygon
from utils.point_generator import generate_uniform


def run_benchmarks(sizes: List[int] = None) -> List[Tuple[int, float, float]]:
    if sizes is None:
        sizes = [100, 500, 1000, 2500, 5000, 10000]

    results = []
    for n in sizes:
        pts = generate_uniform(n, seed=123)
        t0 = time.perf_counter()
        build_star_polygon(pts, pts.mean(axis=0))
        t_single = (time.perf_counter() - t0) * 1000.0

        res = find_min_area_star_polygon(pts)
        results.append((n, t_single, res.elapsed_ms))
    return results


if __name__ == "__main__":
    print(f"{'N':>7} | {'1 побудова (мс)':>16} | {'Пошук S_min (мс)':>17}")
    print("-" * 47)
    for n, t_s, t_opt in run_benchmarks():
        print(f"{n:7d} | {t_s:16.2f} | {t_opt:17.2f}")