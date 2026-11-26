from typing import Optional
import numpy as np


def generate_uniform(n: int, low: float = 5.0, high: float = 95.0, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, 2))


def generate_normal(n: int, center: float = 50.0, scale: float = 15.0, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=center, scale=scale, size=(n, 2))
    return np.clip(pts, 2.0, 98.0)


def generate_ring(n: int, radius: float = 35.0, noise: float = 8.0, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radii = radius + rng.uniform(-noise, noise, size=n)
    x = 50.0 + radii * np.cos(angles)
    y = 50.0 + radii * np.sin(angles)
    return np.column_stack((x, y))