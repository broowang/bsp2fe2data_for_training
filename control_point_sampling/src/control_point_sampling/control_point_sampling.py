from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence

import bspmap
from bspmap.basis import BasisFactory
import numpy as np


DEFAULT_CONTROL_GRID_SHAPE = (64, 64)


@dataclass(slots=True)
class SurfaceData:
    control_points: np.ndarray
    degree_u: int
    degree_v: int


@dataclass(slots=True)
class ControlPointSamplingSpec:
    target_cavity_index: int = 1
    control_grid_shape: tuple[int, int] = DEFAULT_CONTROL_GRID_SHAPE
    v_plane_tolerance: float = 1.0e-6
    enforce_v_plane: bool = False


@dataclass(slots=True)
class ControlPointSamplingCase:
    case_key: str
    surface_files: list[Path]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _surface_to_data(surface: bspmap.BSP | np.ndarray) -> SurfaceData:
    if isinstance(surface, bspmap.BSP):
        cp = np.asarray(surface.control_points, dtype=np.float64)
        cp = cp.reshape(surface.size[0], surface.size[1], 3)
        return SurfaceData(control_points=cp, degree_u=int(surface.degree), degree_v=int(surface.degree))

    cp = np.asarray(surface, dtype=np.float64)
    if cp.ndim != 3 or cp.shape[-1] != 3:
        raise ValueError("Numpy surface input must have shape (num_v, num_u, 3).")
    return SurfaceData(control_points=cp, degree_u=3, degree_v=3)


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        raise ValueError("At least 3 points are required to fit a plane.")

    center = pts.mean(axis=0)
    centered = pts - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular_values > 1.0e-10))
    if rank < 2:
        raise ValueError("Cannot fit a stable plane because the points are nearly collinear.")

    normal = vh[-1]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1.0e-12:
        raise ValueError("Cannot fit a stable plane because the normal is ill-defined.")
    normal = normal / normal_norm
    signed_distances = (pts - center) @ normal
    return center, normal, signed_distances


def validate_v_direction_boundary_plane(
    surface: bspmap.BSP | np.ndarray,
    *,
    tol: float = 1.0e-6,
    surface_label: str | None = None,
    enforce: bool = False,
) -> dict[str, Any]:
    data = _surface_to_data(surface)
    control_points = data.control_points
    if control_points.shape[0] < 2:
        raise ValueError("The surface must contain at least two v-direction layers.")

    first_layer = control_points[:, 0, :]
    last_layer = control_points[:, -1, :]

    plane_point, plane_normal, first_distances = _fit_plane(first_layer)
    last_distances = (last_layer.reshape(-1, 3) - plane_point) @ plane_normal

    first_max = float(np.max(np.abs(first_distances)))
    last_max = float(np.max(np.abs(last_distances)))
    max_distance = max(first_max, last_max)
    is_coplanar = max_distance <= float(tol)
    if enforce and not is_coplanar:
        label = surface_label or "surface"
        raise ValueError(
            f"{label}: v-direction first/last layers are not coplanar within tolerance {tol:g}. "
            f"first_layer_max={first_max:.3e}, last_layer_max={last_max:.3e}"
        )

    return {
        "surface_label": surface_label or "surface",
        "plane_point": plane_point.astype(np.float64),
        "plane_normal": plane_normal.astype(np.float64),
        "first_layer_max_abs_distance": first_max,
        "last_layer_max_abs_distance": last_max,
        "v_plane_tolerance": float(tol),
        "is_coplanar": bool(is_coplanar),
    }


def validate_surface_collection_for_export(
    surfaces: Sequence[bspmap.BSP | np.ndarray],
    *,
    tol: float = 1.0e-6,
    enforce: bool = False,
) -> list[dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    for surface_index, surface in enumerate(surfaces):
        validations.append(
            validate_v_direction_boundary_plane(
                surface,
                tol=tol,
                surface_label=f"surface_{surface_index}",
                enforce=enforce,
            )
        )
    return validations


def _resample_grid(grid: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError("Expected grid shape (H, W, 3).")

    src_h, src_w, _ = grid.shape
    dst_h, dst_w = output_shape
    if src_h == dst_h and src_w == dst_w:
        return grid.copy()

    y = np.linspace(0.0, src_h - 1, dst_h, dtype=np.float64)
    x = np.linspace(0.0, src_w - 1, dst_w, dtype=np.float64)

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = y - y0
    wx = x - x0

    out = np.empty((dst_h, dst_w, 3), dtype=np.float64)
    for iy in range(dst_h):
        gy0 = grid[y0[iy], x0, :]
        gy1 = grid[y1[iy], x0, :]
        gy2 = grid[y0[iy], x1, :]
        gy3 = grid[y1[iy], x1, :]

        row0 = (1.0 - wy[iy]) * gy0 + wy[iy] * gy1
        row1 = (1.0 - wy[iy]) * gy2 + wy[iy] * gy3
        out[iy, :, :] = (1.0 - wx)[:, None] * row0 + wx[:, None] * row1
    return out


def sample_control_points_from_surface(
    surface: bspmap.BSP | np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    data = _surface_to_data(surface)
    sampled = _resample_grid(data.control_points, grid_shape)
    return sampled.astype(np.float64)


def _get_basis_type_names(surface: bspmap.BSP | np.ndarray, input_dimension: int) -> list[str]:
    if isinstance(surface, bspmap.BSP):
        basis_tuple = getattr(surface, "_basis", None)
        if basis_tuple is not None:
            return [type(basis).__name__ for basis in basis_tuple]
    return ["Basis"] * input_dimension


def build_sampled_bsp_surface(
    surface: bspmap.BSP | np.ndarray,
    grid_shape: tuple[int, int],
) -> bspmap.BSP:
    sampled_control_points = sample_control_points_from_surface(surface, grid_shape)
    surface_data = _surface_to_data(surface)
    basis_type_names = _get_basis_type_names(surface, input_dimension=len(grid_shape))
    basis = [
        BasisFactory.create(basis_type=basis_type_names[dim], num_cps=grid_shape[dim], degree=surface_data.degree_u)
        for dim in range(len(grid_shape))
    ]
    return bspmap.BSP(
        basis=basis,
        degree=surface_data.degree_u,
        size=grid_shape,
        control_points=sampled_control_points.reshape(-1, 3),
    )


def extract_sampled_control_point_case(
    surfaces: Sequence[bspmap.BSP | np.ndarray],
    *,
    sampling_spec: ControlPointSamplingSpec | None = None,
    case_key: str | None = None,
    surface_validations: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if sampling_spec is None:
        sampling_spec = ControlPointSamplingSpec()

    if sampling_spec.target_cavity_index < 1:
        raise ValueError("target_cavity_index must be >= 1 because surfaces[0] is the outer shell.")
    if sampling_spec.target_cavity_index >= len(surfaces):
        raise ValueError(
            f"target_cavity_index={sampling_spec.target_cavity_index} out of range for {len(surfaces)} surfaces."
        )

    outer_surface = surfaces[0]
    inner_surface = surfaces[sampling_spec.target_cavity_index]

    outer_raw = _surface_to_data(outer_surface).control_points.astype(np.float64)
    inner_raw = _surface_to_data(inner_surface).control_points.astype(np.float64)

    outer_ctrl_sampled = sample_control_points_from_surface(outer_surface, sampling_spec.control_grid_shape)
    inner_ctrl_sampled = sample_control_points_from_surface(inner_surface, sampling_spec.control_grid_shape)

    sample: dict[str, Any] = {
        "case_key": np.array("" if case_key is None else case_key),
        "target_cavity_index": np.array(sampling_spec.target_cavity_index, dtype=np.int64),
        "control_grid_shape": np.array(sampling_spec.control_grid_shape, dtype=np.int64),
        "inner_ctrl_raw": inner_raw,
        "outer_ctrl_raw": outer_raw,
        "inner_ctrl_sampled": inner_ctrl_sampled,
        "outer_ctrl_sampled": outer_ctrl_sampled,
        "inner_ctrl_raw_shape": np.array(inner_raw.shape, dtype=np.int64),
        "outer_ctrl_raw_shape": np.array(outer_raw.shape, dtype=np.int64),
    }

    if surface_validations is not None:
        sample["surface_validation_json"] = np.array(
            json.dumps(surface_validations, ensure_ascii=True, default=_json_default)
        )

    return sample


def save_sampled_control_point_case(
    sample: dict[str, Any],
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays_to_save = dict(sample)
    if metadata is not None:
        arrays_to_save["metadata_json"] = np.array(
            json.dumps(metadata, ensure_ascii=True, sort_keys=True, default=_json_default)
        )
    np.savez_compressed(output_path, **arrays_to_save)
    return output_path


def collect_surface_cases(input_root: str | Path) -> list[ControlPointSamplingCase]:
    input_root = Path(input_root)
    grouped: dict[str, dict[int, Path]] = {}

    for surface_file in sorted(input_root.rglob("Surface-*.npz")):
        match = re.fullmatch(r"Surface-(\d+)(?:_(.+))?", surface_file.stem)
        if match is None:
            continue
        surface_index = int(match.group(1))
        case_token = match.group(2) or "default"
        relative_parent = surface_file.parent.relative_to(input_root).as_posix()
        case_key = case_token if relative_parent == "." else f"{relative_parent}::{case_token}"
        grouped.setdefault(case_key, {})
        if surface_index in grouped[case_key]:
            raise ValueError(f"Duplicate surface index {surface_index} detected in case '{case_key}'.")
        grouped[case_key][surface_index] = surface_file

    cases: list[ControlPointSamplingCase] = []
    for case_key in sorted(grouped):
        indexed_files = grouped[case_key]
        if 0 not in indexed_files:
            continue
        surface_files = [indexed_files[idx] for idx in sorted(indexed_files)]
        cases.append(ControlPointSamplingCase(case_key=case_key, surface_files=surface_files))
    return cases


def _sanitize_case_key(case_key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", case_key.strip())
    slug = slug.strip("-")
    return slug or "case"


def run_control_point_sampling_export(
    *,
    input_root: str | Path,
    output_root: str | Path,
    target_cavity_index: int = 1,
    control_grid_shape: tuple[int, int] = DEFAULT_CONTROL_GRID_SHAPE,
    skip_existing: bool = True,
    v_plane_tolerance: float = 1.0e-6,
    enforce_v_plane: bool = False,
) -> dict[str, Any]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    samples_dir = output_root / "samples"
    surfaces_dir = output_root / "surface_files"
    samples_dir.mkdir(parents=True, exist_ok=True)
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    sampling_spec = ControlPointSamplingSpec(
        target_cavity_index=target_cavity_index,
        control_grid_shape=control_grid_shape,
        v_plane_tolerance=v_plane_tolerance,
        enforce_v_plane=enforce_v_plane,
    )

    manifest_path = output_root / "manifest.jsonl"
    failure_path = output_root / "failures.jsonl"
    manifest_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    cases = collect_surface_cases(input_root)
    for case in cases:
        try:
            surfaces = [bspmap.BSP.load(str(path)) for path in case.surface_files]
            surface_validations = validate_surface_collection_for_export(
                surfaces,
                tol=sampling_spec.v_plane_tolerance,
                enforce=sampling_spec.enforce_v_plane,
            )
            if target_cavity_index >= len(surfaces):
                raise ValueError(
                    f"target_cavity_index={target_cavity_index} missing in case with {len(surfaces)} surfaces"
                )

            sample_name = f"{_sanitize_case_key(case.case_key)}__cavity{target_cavity_index}__ctrlsampled.npz"
            sample_path = samples_dir / sample_name
            if skip_existing and sample_path.exists():
                manifest_records.append(
                    {
                        "case_key": case.case_key,
                        "sample_path": str(sample_path),
                        "status": "skipped_existing",
                    }
                )
                continue

            sample = extract_sampled_control_point_case(
                surfaces,
                sampling_spec=sampling_spec,
                case_key=case.case_key,
                surface_validations=surface_validations,
            )
            sampled_surfaces = [
                build_sampled_bsp_surface(surface, sampling_spec.control_grid_shape)
                for surface in surfaces
            ]
            metadata = {
                "case_key": case.case_key,
                "target_cavity_index": int(target_cavity_index),
                "control_grid_shape": list(control_grid_shape),
                "v_plane_tolerance": float(v_plane_tolerance),
                "enforce_v_plane": bool(enforce_v_plane),
                "surface_validations": surface_validations,
            }
            save_sampled_control_point_case(sample, sample_path, metadata=metadata)

            exported_surface_paths: list[str] = []
            for surface_index, sampled_surface in enumerate(sampled_surfaces):
                relative_surface_path = case.surface_files[surface_index].relative_to(input_root)
                surface_output_path = surfaces_dir / relative_surface_path
                surface_output_path.parent.mkdir(parents=True, exist_ok=True)
                sampled_surface.save(str(surface_output_path))
                exported_surface_paths.append(str(surface_output_path))

            manifest_records.append(
                {
                    "case_key": case.case_key,
                    "sample_path": str(sample_path),
                    "surface_paths": exported_surface_paths,
                    "status": "ok",
                    **metadata,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure_records.append(
                {
                    "case_key": case.case_key,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    manifest_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, sort_keys=True, default=_json_default) + "\n"
            for record in manifest_records
        ),
        encoding="utf-8",
    )
    failure_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, sort_keys=True, default=_json_default) + "\n"
            for record in failure_records
        ),
        encoding="utf-8",
    )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "case_count": len(cases),
        "success_count": sum(1 for record in manifest_records if record["status"] == "ok"),
        "skipped_existing_count": sum(1 for record in manifest_records if record["status"] == "skipped_existing"),
        "failure_count": len(failure_records),
        "manifest_path": str(manifest_path),
        "failure_path": str(failure_path),
        "surface_output_root": str(surfaces_dir),
        "control_grid_shape": list(control_grid_shape),
        "target_cavity_index": int(target_cavity_index),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary
