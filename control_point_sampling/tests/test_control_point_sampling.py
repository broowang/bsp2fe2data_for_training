from __future__ import annotations

import unittest

import bspmap
import numpy as np

from src.control_point_sampling.control_point_sampling import (
    ControlPointSamplingSpec,
    build_sampled_bsp_surface,
    extract_sampled_control_point_case,
    sample_control_points_from_surface,
)


class TestControlPointSampling(unittest.TestCase):
    def test_sample_control_points_from_numpy_surface_preserves_corners(self) -> None:
        grid = np.zeros((3, 4, 3), dtype=np.float64)
        for i in range(3):
            for j in range(4):
                grid[i, j] = np.array([float(i), float(j), float(i + j)], dtype=np.float64)

        sampled = sample_control_points_from_surface(grid, (5, 6))

        self.assertEqual(sampled.shape, (5, 6, 3))
        np.testing.assert_allclose(sampled[0, 0], grid[0, 0])
        np.testing.assert_allclose(sampled[-1, -1], grid[-1, -1])

    def test_extract_sampled_control_point_case_exports_expected_fields(self) -> None:
        outer = np.zeros((4, 4, 3), dtype=np.float64)
        inner = np.ones((3, 5, 3), dtype=np.float64)

        sample = extract_sampled_control_point_case(
            [outer, inner],
            sampling_spec=ControlPointSamplingSpec(target_cavity_index=1, control_grid_shape=(6, 7)),
            case_key="demo_case",
        )

        self.assertEqual(tuple(sample["inner_ctrl_sampled"].shape), (6, 7, 3))
        self.assertEqual(tuple(sample["outer_ctrl_sampled"].shape), (6, 7, 3))
        self.assertEqual(tuple(sample["control_grid_shape"].tolist()), (6, 7))
        self.assertEqual(sample["case_key"].item(), "demo_case")

    def test_build_sampled_bsp_surface_roundtrip(self) -> None:
        basis = [
            bspmap.BasisClamped(num_cps=4, degree=3),
            bspmap.BasisClamped(num_cps=5, degree=3),
        ]
        control_points = np.arange(4 * 5 * 3, dtype=np.float64).reshape(4, 5, 3)
        surface = bspmap.BSP(
            basis=basis,
            degree=3,
            size=(4, 5),
            control_points=control_points.reshape(-1, 3),
        )

        sampled_surface = build_sampled_bsp_surface(surface, (6, 7))

        self.assertEqual(tuple(sampled_surface.size), (6, 7))
        sampled_points = np.asarray(sampled_surface.control_points, dtype=np.float64).reshape(6, 7, 3)
        self.assertEqual(sampled_points.shape, (6, 7, 3))


if __name__ == "__main__":
    unittest.main()
