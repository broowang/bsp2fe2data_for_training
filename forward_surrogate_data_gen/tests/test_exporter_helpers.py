from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for bootstrap_path in (PROJECT_ROOT, REPO_ROOT):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))

import bspmap

from src.forward_surrogate_data_gen.exporter import (
    collect_surface_cases,
    sample_surface_points_from_control_surface,
    validate_v_direction_boundary_plane,
)


class TestExporterHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = REPO_ROOT / "bsp2fe" / "tests" / "data"

    def test_collect_surface_cases(self) -> None:
        cases = collect_surface_cases(self.data_dir)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_key, "iter-100")
        self.assertEqual(len(cases[0].surface_files), 5)

    def test_surface_sampling_shapes(self) -> None:
        surface = bspmap.BSP.load(str(self.data_dir / "Surface-1_iter-100.npz"))
        sampled_surface, uv_grid = sample_surface_points_from_control_surface(surface, (64, 64))

        self.assertEqual(sampled_surface.shape, (64, 64, 3))
        self.assertEqual(uv_grid.shape, (64, 64, 2))

    def test_v_plane_validation(self) -> None:
        surface = bspmap.BSP.load(str(self.data_dir / "Surface-1_iter-100.npz"))
        result = validate_v_direction_boundary_plane(surface, tol=1.0e-4, surface_label="surface_1", enforce=False)
        self.assertIn("first_layer_max_abs_distance", result)
        self.assertIn("last_layer_max_abs_distance", result)
        self.assertIn("is_coplanar", result)


if __name__ == "__main__":
    unittest.main()
