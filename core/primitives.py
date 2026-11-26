import numpy as np


def orientation(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return float((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def find_interior_point(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True
    return False


def has_self_intersections(polygon: np.ndarray) -> bool:
    n = len(polygon)
    if n < 4:
        return False
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = polygon[j], polygon[(j + 1) % n]
            if segments_intersect(a, b, c, d):
                return True
    return False