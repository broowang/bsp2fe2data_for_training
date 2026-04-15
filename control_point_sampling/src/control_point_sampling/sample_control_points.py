from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.control_point_sampling.control_point_sampling import run_control_point_sampling_export


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample raw B-spline control lattices into fixed-size Surface-*.npz files."
    )
    parser.add_argument("--input-root", required=True, help="Root directory that contains Surface-*.npz files.")
    parser.add_argument("--output-root", required=True, help="Directory used to store sampled control-point outputs.")
    parser.add_argument("--target-cavity-index", type=int, default=1, help="1-based cavity index. surfaces[0] is the outer shell.")
    parser.add_argument("--control-grid-height", type=int, default=64, help="Sample height for the resampled control grid.")
    parser.add_argument("--control-grid-width", type=int, default=64, help="Sample width for the resampled control grid.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute samples even when the output files already exist.",
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
    summary = run_control_point_sampling_export(
        input_root=args.input_root,
        output_root=args.output_root,
        target_cavity_index=args.target_cavity_index,
        control_grid_shape=(args.control_grid_height, args.control_grid_width),
        skip_existing=not args.overwrite,
        v_plane_tolerance=args.v_plane_tol,
        enforce_v_plane=args.enforce_v_plane,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
