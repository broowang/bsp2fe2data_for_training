from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[3]
for bootstrap_path in (PROJECT_ROOT_BOOTSTRAP, REPO_ROOT_BOOTSTRAP):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))

from src.forward_surrogate_data_gen.exporter import run_forward_dataset_export


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate forward-surrogate training samples from grouped Surface-*.npz files."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory that contains Surface-*.npz files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory used to store exported samples.")
    parser.add_argument(
        "--pressures",
        type=float,
        nargs="+",
        required=True,
        help="Pressure values to apply to the target cavity, one exported sample per pressure value.",
    )
    parser.add_argument("--target-cavity-index", type=int, default=1, help="1-based cavity index. surfaces[0] is the outer shell.")
    parser.add_argument("--mesh-size", type=float, default=1.2, help="Maximum Gmsh mesh size.")
    parser.add_argument("--mu", type=float, default=0.48, help="NeoHookean mu parameter.")
    parser.add_argument("--kappa", type=float, default=4.82, help="NeoHookean kappa parameter.")
    parser.add_argument("--solver-tol-error", type=float, default=1.0e-5, help="torchfea nonlinear solve tolerance.")
    parser.add_argument("--sample-grid-height", type=int, default=64, help="Sample height for input surface points and output displacement field.")
    parser.add_argument("--sample-grid-width", type=int, default=64, help="Sample width for input surface points and output displacement field.")
    parser.add_argument(
        "--exclude-raw-fe-mesh",
        action="store_true",
        help="Do not store raw FE surface nodes and triangles inside each exported NPZ.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute samples even when the output NPZ already exists.",
    )
    parser.add_argument(
        "--v-plane-tol",
        type=float,
        default=1.0e-6,
        help="Tolerance used to validate that the first and last v-direction control-point layers are coplanar.",
    )
    parser.add_argument(
        "--enforce-v-plane",
        action="store_true",
        help="Fail the export when the v-direction boundary plane check is not satisfied.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_forward_dataset_export(
        input_root=args.input_root,
        output_root=args.output_root,
        pressure_values=args.pressures,
        mesh_size=args.mesh_size,
        mu=args.mu,
        kappa=args.kappa,
        target_cavity_index=args.target_cavity_index,
        sample_grid_shape=(args.sample_grid_height, args.sample_grid_width),
        solver_tol_error=args.solver_tol_error,
        include_raw_fe_mesh=not args.exclude_raw_fe_mesh,
        skip_existing=not args.overwrite,
        v_plane_tolerance=args.v_plane_tol,
        enforce_v_plane=args.enforce_v_plane,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
