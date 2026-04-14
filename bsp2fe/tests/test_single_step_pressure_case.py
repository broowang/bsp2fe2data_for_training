from __future__ import annotations

import os
import unittest
from pathlib import Path

import bspmap
import torch

from bsp2fe import build_parametric_pneumatic_model


class TestSingleStepPressureCase(unittest.TestCase):
    def test_surface_1_pressure_006_mpa(self) -> None:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        torch.set_default_dtype(torch.float64)

        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "tests" / "data"
        tmp_root = project_root / ".tmp_tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_workdir = tmp_root / "single_step_pressure_case"
        case_workdir.mkdir(parents=True, exist_ok=True)

        # Input convention: index 0 is outer shell, index >=1 are cavities.
        surface_files = sorted(data_dir.glob("Surface-*_iter-100.npz"), key=lambda p: int(p.stem.split("-")[1].split("_")[0]))
        self.assertGreaterEqual(len(surface_files), 2, "Need at least one outer and one inner surface.")

        surfaces = [bspmap.BSP.load(str(p)) for p in surface_files]

        cavity_count = len(surfaces) - 1
        pressure_values = [0.06] + [0.0] * (cavity_count - 1)

        model = build_parametric_pneumatic_model(
            surfaces=surfaces,
            working_dir=case_workdir,
            mesh_size=1.2,
            mu=0.48,
            kappa=4.82,
            pressure_values=pressure_values,
        )

        self.assertEqual(len(model.cavity_surface_names), cavity_count)

        first_load = model.fe.assembly.get_load(model.load_names[0])
        self.assertAlmostEqual(float(first_load.pressure), 0.06, places=12)

        for i in range(1, cavity_count):
            load_i = model.fe.assembly.get_load(model.load_names[i])
            self.assertAlmostEqual(float(load_i.pressure), 0.0, places=12)

        # This case test focuses on data loading + FE model build + pressure assignment.
        # Full nonlinear solve is intentionally not executed here to keep test runtime stable.

        result = model.fe.solve()


if __name__ == "__main__":
    unittest.main()
