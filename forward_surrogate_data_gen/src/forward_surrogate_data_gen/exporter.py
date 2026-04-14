from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import bspmap
import numpy as np
import pyvista as pv
import torch

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[3]
BSP2FE_ROOT_BOOTSTRAP = REPO_ROOT_BOOTSTRAP / "bsp2fe"      #确保路径正确
for bootstrap_path in (REPO_ROOT_BOOTSTRAP, BSP2FE_ROOT_BOOTSTRAP):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))

from bsp2fe import ParametricPneumaticModel, build_parametric_pneumatic_model   #调用bsp2fe来进行有限元求解


DEFAULT_SAMPLE_GRID_SHAPE = (64, 64)  #采样分辨率64x64个采样点


@dataclass(slots=True)
class SurfaceData:    #整理B样条曲面
    control_points: np.ndarray  #控制点
    degree_u: int     #u方向
    degree_v: int     #v方向


@dataclass(slots=True)
class ForwardDatasetExportSpec:  #单次导出规则
    target_cavity_index: int = 1
    sample_grid_shape: tuple[int, int] = DEFAULT_SAMPLE_GRID_SHAPE #网格形状
    include_raw_fe_mesh: bool = True  #保存原始 FE 网格
    v_plane_tolerance: float = 1.0e-6  #v向平面检查 第一层和最后一层要保证在同一个平面内
    enforce_v_plane: bool = False


@dataclass(slots=True)
class ForwardDatasetCase:
    case_key: str
    surface_files: list[Path]


def _surface_to_data(surface: bspmap.BSP | np.ndarray) -> SurfaceData:  #把输入曲面统一成同一种内部表示
    if isinstance(surface, bspmap.BSP):
        cp = np.asarray(surface.control_points, dtype=np.float64)
        cp = cp.reshape(surface.size[0], surface.size[1], 3)
        return SurfaceData(control_points=cp, degree_u=int(surface.degree), degree_v=int(surface.degree))

    cp = np.asarray(surface, dtype=np.float64)
    if cp.ndim != 3 or cp.shape[-1] != 3:
        raise ValueError("Numpy surface input must have shape (num_v, num_u, 3).")
    return SurfaceData(control_points=cp, degree_u=3, degree_v=3)


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  #这个是用来拟合平面的
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)   #任意形状输入拉平成 N x 3
    if pts.shape[0] < 3:
        raise ValueError("At least 3 points are required to fit a plane.")

    center = pts.mean(axis=0)
    centered = pts - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)  #这里用 SVD 判断这些点是否足够形成一个稳定平面
    rank = int(np.sum(singular_values > 1.0e-10))
    if rank < 2:
        raise ValueError("Cannot fit a stable plane because the points are nearly collinear.")

    normal = vh[-1]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1.0e-12:
        raise ValueError("Cannot fit a stable plane because the normal is ill-defined.")
    normal = normal / normal_norm       #做一个简单的归一化
    signed_distances = (pts - center) @ normal
    return center, normal, signed_distances


def validate_v_direction_boundary_plane(       #这个用来v向首尾层平面检查
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


    first_layer = control_points[:, 0, :]    #取首层和末层：[:,0,:] 与 [:,-1,:]
    last_layer = control_points[:, -1, :]

    plane_point, plane_normal, first_distances = _fit_plane(first_layer)
    last_distances = (last_layer.reshape(-1, 3) - plane_point) @ plane_normal

    first_max = float(np.max(np.abs(first_distances)))
    last_max = float(np.max(np.abs(last_distances)))
    max_distance = max(first_max, last_max)
    is_coplanar = max_distance <= float(tol) #取最大偏差并判断是否共面
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


def validate_surface_collection_for_export(     #对每一个曲面都做一遍检查
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


def _is_closed_on_axis(control_points: np.ndarray, axis: int, tol: float = 1.0e-6) -> bool:  #判断某个方向是不是闭合的
    if axis == 0:
        seam_error = np.linalg.norm(control_points[0, :, :] - control_points[-1, :, :], axis=-1)
    elif axis == 1:
        seam_error = np.linalg.norm(control_points[:, 0, :] - control_points[:, -1, :], axis=-1)
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return bool(np.max(seam_error) < tol)


def _build_parametric_grid(      #生成规则网格
    grid_shape: tuple[int, int],
    *,
    closed_axis0: bool,
    closed_axis1: bool,
) -> tuple[np.ndarray, np.ndarray]:      #闭合方向不取终点 非闭合方向取终点
    axis0 = np.linspace(0.0, 1.0, grid_shape[0], endpoint=not closed_axis0, dtype=np.float64)
    axis1 = np.linspace(0.0, 1.0, grid_shape[1], endpoint=not closed_axis1, dtype=np.float64)
    mesh0, mesh1 = np.meshgrid(axis0, axis1, indexing="ij")
    uv_grid = np.stack([mesh0, mesh1], axis=-1)
    return uv_grid, uv_grid.reshape(-1, 2)


def _resample_grid(grid: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:     #这个是双线性重采样函数，当输入不是 bspmap.BSP，而只是普通 numpy 网格时才会走这里，基本用不到
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


def sample_surface_points_from_control_surface(    #从控制点中进行采样
    surface: bspmap.BSP | np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    data = _surface_to_data(surface)
    uv_grid, uv_flat = _build_parametric_grid(
        grid_shape,
        closed_axis0=_is_closed_on_axis(data.control_points, axis=0),
        closed_axis1=_is_closed_on_axis(data.control_points, axis=1),   #建立参数网格
    )

    if isinstance(surface, bspmap.BSP):    #如果是 bspmap，直接用 surface.map(uv_flat) 在原始 B-spline 曲面上采样
        sampled_points = surface.map(uv_flat).reshape(grid_shape[0], grid_shape[1], 3)
    else:   #是普通numpy网格的话就要回退了
        sampled_points = _resample_grid(data.control_points, grid_shape)

    return sampled_points.astype(np.float64), uv_grid.astype(np.float64)


def _polydata_to_triangles(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:    #从FE结果里把目标内腔表面网格取出来
    if mesh.n_cells == 0:
        raise ValueError("Surface mesh is empty.")

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size % 4 != 0:
        raise ValueError("Expected triangular PolyData faces encoded as 4-tuples.")
    faces = faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected all faces to be triangles.")
    return np.asarray(mesh.points, dtype=np.float64), faces[:, 1:4]


def extract_surface_mesh_arrays(
    model: ParametricPneumaticModel,   #源自于FE脚本
    surface_name: str,
    *,
    use_deformed_configuration: bool,
) -> tuple[np.ndarray, np.ndarray]:
    instance = model.fe.assembly.get_instance("final_model")
    rgc = model.fe.assembly.RGC if use_deformed_configuration else None
    mesh = instance.get_mesh(RGC=rgc, surf_name=surface_name)
    return _polydata_to_triangles(mesh)


def _barycentric_coordinates(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]          #计算重心坐标
    c = triangles[:, 2, :]

    v0 = b - a
    v1 = c - a
    v2 = points - a

    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)

    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1.0e-12, 1.0e-12, denom)

    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2

    weights = np.stack([w0, w1, w2], axis=1)
    weights = np.nan_to_num(weights, nan=0.0)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_sum = np.where(np.abs(weights_sum) < 1.0e-12, 1.0, weights_sum)
    return weights / weights_sum


def _transfer_displacement_to_sample_points(
    *,
    rest_points: np.ndarray,          #把 FE 结果转移到我们定义好的固定采样点上
    rest_triangles: np.ndarray,
    deformed_points: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    faces = np.hstack(
        [
            np.full((rest_triangles.shape[0], 1), 3, dtype=np.int64),
            rest_triangles.astype(np.int64),
        ]
    )
    rest_mesh = pv.PolyData(rest_points, faces)
    cell_ids, closest_points = rest_mesh.find_closest_cell(query_points, return_closest_point=True)
    cell_ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    closest_points = np.asarray(closest_points, dtype=np.float64).reshape(-1, 3)

    tri_rest = rest_points[rest_triangles[cell_ids]]
    tri_def = deformed_points[rest_triangles[cell_ids]]
    bary = _barycentric_coordinates(closest_points, tri_rest)

    aligned_deformed = np.einsum("ni,nij->nj", bary, tri_def)
    transferred_disp = aligned_deformed - closest_points
    return closest_points, aligned_deformed, transferred_disp


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def extract_forward_surrogate_sample(           #构造用于正向模型训练的样本
    model: ParametricPneumaticModel,
    surfaces: Sequence[bspmap.BSP | np.ndarray],
    pressure_values: Sequence[float],
    *,
    export_spec: ForwardDatasetExportSpec | None = None,
    case_key: str | None = None,
    surface_validations: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if export_spec is None:
        export_spec = ForwardDatasetExportSpec()

    if export_spec.target_cavity_index < 1:
        raise ValueError("target_cavity_index must be >= 1 because surfaces[0] is the outer shell.")
    if export_spec.target_cavity_index >= len(surfaces):
        raise ValueError(
            f"target_cavity_index={export_spec.target_cavity_index} out of range for {len(surfaces)} surfaces."
        )

    target_surface_name = f"surface_{export_spec.target_cavity_index}_All"
    if target_surface_name not in model.cavity_surface_names:
        raise ValueError(f"Target cavity surface '{target_surface_name}' was not generated.")

    outer_surface = surfaces[0]
    inner_surface = surfaces[export_spec.target_cavity_index]

    outer_raw = _surface_to_data(outer_surface).control_points.astype(np.float64)
    inner_raw = _surface_to_data(inner_surface).control_points.astype(np.float64)    #把原始控制点保存成 outer_raw / inner_raw

    inner_surface_samples, uv_grid = sample_surface_points_from_control_surface(
        inner_surface,
        export_spec.sample_grid_shape,
    )
    outer_surface_samples, _ = sample_surface_points_from_control_surface(
        outer_surface,
        export_spec.sample_grid_shape,
    )

    rest_points, rest_triangles = extract_surface_mesh_arrays(
        model,
        target_surface_name,
        use_deformed_configuration=False,
    )
    deformed_points, deformed_triangles = extract_surface_mesh_arrays(
        model,
        target_surface_name,
        use_deformed_configuration=True,
    )

    if not np.array_equal(rest_triangles, deformed_triangles):
        raise RuntimeError("Rest/deformed surface triangulations differ; cannot transfer displacement consistently.")

    aligned_rest_flat, aligned_def_flat, sampled_disp_flat = _transfer_displacement_to_sample_points(  #执行位移传递
        rest_points=rest_points,
        rest_triangles=rest_triangles,
        deformed_points=deformed_points,
        query_points=inner_surface_samples.reshape(-1, 3),
    )

    inner_rest_surface = inner_surface_samples.astype(np.float64)
    inner_disp_field = sampled_disp_flat.reshape(*export_spec.sample_grid_shape, 3).astype(np.float64)
    inner_def_surface = (inner_rest_surface + inner_disp_field).astype(np.float64)
    inner_fe_aligned_rest_surface = aligned_rest_flat.reshape(*export_spec.sample_grid_shape, 3).astype(np.float64)
    inner_fe_aligned_def_surface = aligned_def_flat.reshape(*export_spec.sample_grid_shape, 3).astype(np.float64)

    target_pressure = float(pressure_values[export_spec.target_cavity_index - 1])
    sample: dict[str, Any] = {
        "case_key": np.array("" if case_key is None else case_key),
        "target_surface_name": np.array(target_surface_name),
        "target_cavity_index": np.array(export_spec.target_cavity_index, dtype=np.int64),
        "pressure": np.array([target_pressure], dtype=np.float64),
        "inner_ctrl_raw": inner_raw,
        "outer_ctrl_raw": outer_raw,
        "inner_surface_samples": inner_surface_samples.astype(np.float64),
        "outer_surface_samples": outer_surface_samples.astype(np.float64),
        "inner_rest_surface": inner_rest_surface,
        "inner_def_surface": inner_def_surface,
        "inner_disp_field": inner_disp_field,
        "inner_fe_aligned_rest_surface": inner_fe_aligned_rest_surface,
        "inner_fe_aligned_def_surface": inner_fe_aligned_def_surface,
        "surface_uv_grid": uv_grid.astype(np.float64),
        "sample_grid_shape": np.array(export_spec.sample_grid_shape, dtype=np.int64),
        "pressure_values_all_cavities": np.asarray(pressure_values, dtype=np.float64),
        "inner_ctrl_raw_shape": np.array(inner_raw.shape, dtype=np.int64),
        "outer_ctrl_raw_shape": np.array(outer_raw.shape, dtype=np.int64),
    }

    if surface_validations is not None:
        sample["surface_validation_json"] = np.array(json.dumps(surface_validations, ensure_ascii=True, default=_json_default))

    if export_spec.include_raw_fe_mesh:
        sample["inner_fe_rest_points"] = rest_points.astype(np.float64)
        sample["inner_fe_rest_triangles"] = rest_triangles.astype(np.int64)
        sample["inner_fe_def_points"] = deformed_points.astype(np.float64)
        sample["inner_fe_def_triangles"] = deformed_triangles.astype(np.int64)

    return sample


def save_forward_surrogate_sample(
    sample: dict[str, Any],
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays_to_save = dict(sample)
    if metadata is not None:
        arrays_to_save["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=True, sort_keys=True, default=_json_default))
    np.savez_compressed(output_path, **arrays_to_save)
    return output_path


def generate_forward_surrogate_sample(           #完整流程函数
    surfaces: Sequence[bspmap.BSP | np.ndarray],
    pressure_values: Sequence[float],
    *,
    working_dir: str | Path,
    mesh_size: float,
    mu: float,
    kappa: float,
    output_path: str | Path | None = None,
    export_spec: ForwardDatasetExportSpec | None = None,
    solver_tol_error: float = 1.0e-5,
    case_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if export_spec is None:
        export_spec = ForwardDatasetExportSpec()

    surface_validations = validate_surface_collection_for_export(
        surfaces,
        tol=export_spec.v_plane_tolerance,
        enforce=export_spec.enforce_v_plane,
    )

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = build_parametric_pneumatic_model(           #这里调用 FE脚本
            surfaces=surfaces,
            working_dir=working_dir,
            mesh_size=mesh_size,
            mu=mu,
            kappa=kappa,
            pressure_values=pressure_values,
            solver_tol_error=solver_tol_error,
            validate_v_boundary_plane=export_spec.enforce_v_plane,    # v 向平面检查
            v_boundary_plane_tol=export_spec.v_plane_tolerance,
        )
        result = model.fe.solve()
        if isinstance(result, bool):
            raise RuntimeError("FE solve did not return a StaticResult object.")
    finally:
        torch.set_default_dtype(default_dtype)

    sample = extract_forward_surrogate_sample(
        model=model,
        surfaces=surfaces,
        pressure_values=pressure_values,
        export_spec=export_spec,
        case_key=case_key,
        surface_validations=surface_validations,
    )

    metadata = {
        "case_key": case_key,
        "mesh_size": float(mesh_size),
        "mu": float(mu),
        "kappa": float(kappa),
        "pressure_values": [float(p) for p in pressure_values],
        "solver_tol_error": float(solver_tol_error),
        "solver_total_time": float(result.total_time),
        "target_surface_name": str(sample["target_surface_name"].item()),
        "target_cavity_index": int(sample["target_cavity_index"].item()),
        "sample_grid_shape": list(export_spec.sample_grid_shape),
        "include_raw_fe_mesh": bool(export_spec.include_raw_fe_mesh),
        "v_plane_tolerance": float(export_spec.v_plane_tolerance),
        "enforce_v_plane": bool(export_spec.enforce_v_plane),
        "surface_validations": surface_validations,
    }
    if output_path is not None:
        save_forward_surrogate_sample(sample, output_path, metadata=metadata)
    return sample, metadata


def collect_surface_cases(input_root: str | Path) -> list[ForwardDatasetCase]:  #扫描输入目录下所有 Surface-*.npz
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

    cases: list[ForwardDatasetCase] = []
    for case_key in sorted(grouped):
        indexed_files = grouped[case_key]
        if 0 not in indexed_files:
            continue
        surface_files = [indexed_files[idx] for idx in sorted(indexed_files)]
        cases.append(ForwardDatasetCase(case_key=case_key, surface_files=surface_files))
    return cases


def _load_surface_case(case: ForwardDatasetCase) -> list[bspmap.BSP]:
    return [bspmap.BSP.load(str(path)) for path in case.surface_files]


def _sanitize_case_key(case_key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", case_key.strip())
    slug = slug.strip("-")
    return slug or "case"


def _pressure_tag(pressure_value: float) -> str:
    return f"{pressure_value:.6f}".replace("-", "m").replace(".", "p")


def run_forward_dataset_export(
    *,
    input_root: str | Path,
    output_root: str | Path,
    pressure_values: Iterable[float],
    mesh_size: float,
    mu: float,
    kappa: float,
    target_cavity_index: int = 1,
    sample_grid_shape: tuple[int, int] = DEFAULT_SAMPLE_GRID_SHAPE,
    solver_tol_error: float = 1.0e-5,
    include_raw_fe_mesh: bool = True,
    skip_existing: bool = True,
    v_plane_tolerance: float = 1.0e-6,
    enforce_v_plane: bool = False,
) -> dict[str, Any]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    samples_dir = output_root / "samples"
    work_dir = output_root / "work"
    samples_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    export_spec = ForwardDatasetExportSpec(
        target_cavity_index=target_cavity_index,
        sample_grid_shape=sample_grid_shape,
        include_raw_fe_mesh=include_raw_fe_mesh,
        v_plane_tolerance=v_plane_tolerance,
        enforce_v_plane=enforce_v_plane,
    )

    manifest_path = output_root / "manifest.jsonl"
    failure_path = output_root / "failures.jsonl"
    manifest_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    cases = collect_surface_cases(input_root)
    for case in cases:
        surfaces = _load_surface_case(case)
        if target_cavity_index >= len(surfaces):
            failure_records.append(
                {
                    "case_key": case.case_key,
                    "reason": f"target_cavity_index={target_cavity_index} missing in case with {len(surfaces)} surfaces",
                }
            )
            continue

        case_slug = _sanitize_case_key(case.case_key)
        for pressure_value in pressure_values:
            pressure_vector = [0.0] * (len(surfaces) - 1)
            pressure_vector[target_cavity_index - 1] = float(pressure_value)

            sample_name = f"{case_slug}__cavity{target_cavity_index}__p_{_pressure_tag(float(pressure_value))}.npz"
            sample_path = samples_dir / sample_name
            if skip_existing and sample_path.exists():
                manifest_records.append(
                    {
                        "case_key": case.case_key,
                        "pressure": float(pressure_value),
                        "sample_path": str(sample_path),
                        "status": "skipped_existing",
                    }
                )
                continue

            sample_work_dir = work_dir / sample_path.stem
            sample_work_dir.mkdir(parents=True, exist_ok=True)
            try:
                _, metadata = generate_forward_surrogate_sample(
                    surfaces=surfaces,
                    pressure_values=pressure_vector,
                    working_dir=sample_work_dir,
                    mesh_size=mesh_size,
                    mu=mu,
                    kappa=kappa,
                    output_path=sample_path,
                    export_spec=export_spec,
                    solver_tol_error=solver_tol_error,
                    case_key=case.case_key,
                )
                manifest_records.append(
                    {
                        "case_key": case.case_key,
                        "pressure": float(pressure_value),
                        "sample_path": str(sample_path),
                        "work_dir": str(sample_work_dir),
                        "status": "ok",
                        **metadata,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failure_records.append(
                    {
                        "case_key": case.case_key,
                        "pressure": float(pressure_value),
                        "sample_path": str(sample_path),
                        "work_dir": str(sample_work_dir),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True, sort_keys=True, default=_json_default) + "\n" for record in manifest_records),
        encoding="utf-8",
    )
    failure_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True, sort_keys=True, default=_json_default) + "\n" for record in failure_records),
        encoding="utf-8",
    )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "case_count": len(cases),
        "requested_pressures": [float(v) for v in pressure_values],
        "success_count": sum(1 for record in manifest_records if record["status"] == "ok"),
        "skipped_existing_count": sum(1 for record in manifest_records if record["status"] == "skipped_existing"),
        "failure_count": len(failure_records),
        "manifest_path": str(manifest_path),
        "failure_path": str(failure_path),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary
