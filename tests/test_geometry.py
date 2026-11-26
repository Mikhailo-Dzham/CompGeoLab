import unittest
import numpy as np
from core.primitives import polygon_area, has_self_intersections
from core.star_generator import build_star_polygon
from core.optimizer import find_min_area_star_polygon
from utils.point_generator import generate_uniform


class TestStarPolygon(unittest.TestCase):
    def test_square_area(self):
        square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        self.assertAlmostEqual(polygon_area(square), 100.0)

    def test_simple_star_no_intersections(self):
        pts = generate_uniform(20, seed=1)
        res = find_min_area_star_polygon(pts)
        self.assertIsNotNone(res.best_polygon)
        self.assertEqual(len(res.best_polygon), 20)
        self.assertFalse(has_self_intersections(res.best_polygon))

    def test_exterior_z_rejected(self):
        pts = generate_uniform(15, seed=2)
        poly, _, _, _ = build_star_polygon(pts, np.array([-50.0, -50.0]))
        self.assertIsNone(poly)

    def test_collinear_ray_handled(self):
        pts = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [10.0, 30.0], [30.0, 10.0]])
        poly, area, _, _ = build_star_polygon(pts, np.array([15.0, 15.0]))
        self.assertIsNotNone(poly)
        self.assertGreater(area, 0.0)

    def test_large_scale_performance(self):
        pts = generate_uniform(10000, seed=42)
        res = find_min_area_star_polygon(pts)
        self.assertIsNotNone(res.best_polygon)
        self.assertEqual(len(res.best_polygon), 10000)
        self.assertLess(res.elapsed_ms, 3000.0)


if __name__ == "__main__":
    unittest.main()