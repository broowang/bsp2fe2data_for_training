from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile
from typing import Sequence

import numpy as np
import bspmap
import gmsh
import torch
import torchfea


@dataclass(slots=True)
class SurfaceData:
    control_points: np.ndarray
    degree_u: int
    degree_v: int


@dataclass(slots=True)
class ParametricPneumaticModel:
    fe: torchfea.FEAController
    inp: torchfea.FEA_INP
    inp_path: Path
    cavity_surface_names: list[str]
    load_names: list[str]

    def set_material(self, mu: float, kappa: float) -> None:
        """Apply a single NeoHookean material to all elements in final_model."""
        part = self.fe.assembly.get_part("final_model")
        mat = torchfea.materials.NeoHookean(mu=float(mu), kappa=float(kappa))
        for element in part.elems.values():
            if hasattr(element, "set_materials"):
                element.set_materials(mat)

    def setup_cavity_pressures(self, pressure_values: Sequence[float]) -> None:
        """Create pressure loads for all cavity surfaces in one step."""
        if len(self.cavity_surface_names) == 0:
            raise ValueError("No cavity surfaces found. Need at least one inner surface (surface_1_All+).")
        num_cavity = len(self.cavity_surface_names)
        if len(pressure_values) != num_cavity:
            raise ValueError(f"pressure_values size mismatch: expected {num_cavity}, got {len(pressure_values)}.")

        self.load_names = []
        for cavity_idx, pressure in enumerate(pressure_values):
            load_name = f"pressure_cavity{cavity_idx+1}"
            pressure_load = torchfea.model.loads.Pressure(
                instance_name="final_model",
                surface_set=self.cavity_surface_names[cavity_idx],
                pressure=float(pressure),
            )
            self.fe.assembly.add_load(pressure_load, load_name)
            self.load_names.append(load_name)

    def set_pressure_values(self, pressure_values: Sequence[float]) -> None:
        """Update pressure values in-place for one step."""
        if len(pressure_values) != len(self.cavity_surface_names):
            raise ValueError("pressures size must equal cavity count.")

        for i, p in enumerate(pressure_values):
            load = self.fe.assembly.get_load(self.load_names[i])
            load.pressure = float(p)

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


def _validate_v_boundary_plane(surface_data: SurfaceData, *, tol: float = 1.0e-6, label: str = "surface") -> None:
    control_points = surface_data.control_points
    if control_points.shape[0] < 2:
        raise ValueError(f"{label}: surface must contain at least two v-direction layers.")

    # In the current project data layout, the second axis corresponds to the
    # v-direction boundary layers that should remain coplanar.
    first_layer = control_points[:, 0, :]
    last_layer = control_points[:, -1, :]

    plane_point, plane_normal, first_distances = _fit_plane(first_layer)
    last_distances = (last_layer.reshape(-1, 3) - plane_point) @ plane_normal

    first_max = float(np.max(np.abs(first_distances)))
    last_max = float(np.max(np.abs(last_distances)))
    if max(first_max, last_max) > tol:
        raise ValueError(
            f"{label}: v-direction first/last layers are not coplanar within tolerance {tol:g}. "
            f"first_layer_max={first_max:.3e}, last_layer_max={last_max:.3e}"
        )


class _BSplineSolidGenerator:
    def __init__(self, control_points: np.ndarray, degree_u: int = 3, degree_v: int = 3) -> None:
        self.control_points = control_points
        self.degree_u = degree_u
        self.degree_v = degree_v

    def export_step(self, output_path: Path) -> None:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.model.add("bspline_solid")

            self._build_solid()
            gmsh.write(str(output_path))
        finally:
            gmsh.finalize()

    def _build_solid(self) -> None:
        p0 = self.control_points
        du = self.degree_u
        dv = self.degree_v

        if np.linalg.norm(p0[0, 0] - p0[0, -1]) < 1e-6:
            poles_to_use = p0[:, :-1]
        else:
            poles_to_use = p0

        num_poles_v = poles_to_use.shape[0]
        num_poles_u = poles_to_use.shape[1]

        point_tags = np.zeros((num_poles_v, num_poles_u), dtype=np.int32)
        for j in range(num_poles_v):
            for i in range(num_poles_u):
                pt = poles_to_use[j, i]
                point_tags[j, i] = gmsh.model.occ.addPoint(float(pt[0]), float(pt[1]), float(pt[2]))

        side_point_tags: list[int] = []
        for j in range(num_poles_v):
            for i in range(num_poles_u):
                side_point_tags.append(int(point_tags[j, i]))
            for k in range(du):
                side_point_tags.append(int(point_tags[j, k]))

        num_extended_poles_u = num_poles_u + du
        num_knots_u = num_extended_poles_u + du + 1
        knots_u = [float(i) for i in range(num_knots_u)]
        mults_u = [1] * len(knots_u)

        num_knots_v = num_poles_v - dv + 1
        knots_v = []
        mults_v = []
        for i in range(num_knots_v):
            knots_v.append(float(i))
            if i == 0 or i == num_knots_v - 1:
                mults_v.append(dv + 1)
            else:
                mults_v.append(1)

        raw_surface = gmsh.model.occ.addBSplineSurface(
            side_point_tags,
            num_extended_poles_u,
            -1,
            du,
            dv,
            [],
            knots_u,
            knots_v,
            mults_u,
            mults_v,
        )

        u_min = float(du)
        u_max = float(du + num_poles_u)
        v_min = knots_v[0]
        v_max = knots_v[-1]

        p1 = gmsh.model.occ.addPoint(u_min, v_min, 0)
        p2 = gmsh.model.occ.addPoint(u_max, v_min, 0)
        p3 = gmsh.model.occ.addPoint(u_max, v_max, 0)
        p4 = gmsh.model.occ.addPoint(u_min, v_max, 0)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        side_surface = gmsh.model.occ.addTrimmedSurface(raw_surface, [loop], wire3D=False)

        gmsh.model.occ.synchronize()

        boundaries = gmsh.model.getBoundary([(2, side_surface)], recursive=False)
        cap_faces: list[int] = []
        for _, tag in boundaries:
            _, nodes = gmsh.model.getAdjacencies(1, tag)
            if len(nodes) < 2:
                continue
            c1 = np.array(gmsh.model.getValue(0, nodes[0], []))
            c2 = np.array(gmsh.model.getValue(0, nodes[-1], []))
            if np.linalg.norm(c1 - c2) > 1e-5:
                continue
            wire = gmsh.model.occ.addWire([tag])
            try:
                cap_faces.append(gmsh.model.occ.addPlaneSurface([wire]))
            except Exception:
                cap_faces.append(gmsh.model.occ.addSurfaceFilling(wire))

        surface_tags = [side_surface] + cap_faces
        volume_tag: int | None = None

        try:
            shell = gmsh.model.occ.addSurfaceLoop(surface_tags)
            volume_tag = gmsh.model.occ.addVolume([shell])
        except Exception:
            healed = gmsh.model.occ.healShapes(
                [(2, t) for t in surface_tags],
                tolerance=1e-1,
                fixDegenerated=True,
                fixSmallEdges=False,
                fixSmallFaces=False,
                sewFaces=True,
                makeSolids=True,
            )
            for dim, tag in healed:
                if dim == 3:
                    volume_tag = tag
                    break

        gmsh.model.occ.synchronize()
        if volume_tag is None:
            raise RuntimeError("Failed to create closed BSpline solid.")

        # Isolate the solid via temporary BRep to purge historical/orphan entities.
        keep_entities = {(3, volume_tag)}
        all_entities = gmsh.model.getEntities()
        to_remove = [e for e in all_entities if e not in keep_entities and e[0] <= 2]
        if to_remove:
            gmsh.model.occ.remove(to_remove, recursive=False)
            gmsh.model.occ.synchronize()

        fd, temp_brep = tempfile.mkstemp(suffix=".brep")
        os.close(fd)
        try:
            gmsh.write(temp_brep)
            gmsh.clear()
            gmsh.model.occ.importShapes(temp_brep)
            gmsh.model.occ.synchronize()
        finally:
            if os.path.exists(temp_brep):
                os.remove(temp_brep)

        vols = gmsh.model.getEntities(3)
        if len(vols) == 0:
            raise RuntimeError("BRep isolation failed: no volume found after re-import.")
        volume_tag = vols[0][1]

        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        try:
            healed = gmsh.model.occ.healShapes(
                [(3, volume_tag)],
                tolerance=1e-5,
                fixDegenerated=True,
                fixSmallEdges=True,
                fixSmallFaces=True,
                sewFaces=True,
                makeSolids=True,
            )
            if healed and healed[0][0] == 3:
                volume_tag = healed[0][1]
        except Exception:
            pass

        gmsh.model.occ.synchronize()

        keep_set = {(3, volume_tag)}
        for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], recursive=True):
            keep_set.add((dim, tag))

        candidates = gmsh.model.getEntities()
        to_delete = [e for e in candidates if e not in keep_set]
        if to_delete:
            gmsh.model.occ.remove(to_delete, recursive=False)
            gmsh.model.occ.synchronize()


class _MeshGenerator:
    def __init__(self, mesh_size_min: float, mesh_size_max: float) -> None:
        self.mesh_size_min = mesh_size_min
        self.mesh_size_max = mesh_size_max
        self.files_map: dict[int, Path] = {}
        self.sorted_indices: list[int] = []
        self.surface_tags_by_index: dict[int, list[int]] = {}

    def run(self, working_dir: Path, output_inp: Path) -> None:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.NumThreads", 0)
            gmsh.option.setNumber("General.Verbosity", 2)
            gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size_max)

            self._scan_directory(working_dir)
            self._load_and_process_files()
            self._construct_volume()
            gmsh.model.mesh.generate(3)
            self._export_inp(output_inp)
        finally:
            gmsh.finalize()

    def _scan_directory(self, directory: Path) -> None:
        for file in directory.glob("__surface-*.stp"):
            idx = int(file.stem.split("-")[-1])
            self.files_map[idx] = file
        if 0 not in self.files_map:
            raise FileNotFoundError("Missing outer surface __surface-0.stp.")
        self.sorted_indices = sorted(self.files_map)

    @staticmethod
    def _surface_tags() -> set[int]:
        return {tag for dim, tag in gmsh.model.getEntities(2) if dim == 2}

    def _load_and_process_files(self) -> None:
        for idx in self.sorted_indices:
            before = self._surface_tags()
            gmsh.model.occ.importShapes(str(self.files_map[idx]))
            gmsh.model.occ.synchronize()
            vols = gmsh.model.getEntities(3)
            if vols:
                gmsh.model.occ.remove(vols, recursive=False)
                gmsh.model.occ.synchronize()
            after = self._surface_tags()
            self.surface_tags_by_index[idx] = list(after - before)

    def _construct_volume(self) -> None:
        if not self.surface_tags_by_index.get(0):
            raise RuntimeError("Surface index 0 has no geometric faces.")

        loops: list[int] = [gmsh.model.geo.addSurfaceLoop(self.surface_tags_by_index[0])]
        for idx in self.sorted_indices:
            if idx == 0:
                continue
            tags = self.surface_tags_by_index.get(idx, [])
            if tags:
                loops.append(gmsh.model.geo.addSurfaceLoop(tags))

        vol = gmsh.model.geo.addVolume(loops)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(3, [vol], name="Volume_All")

    def _generate_abaqus_surface_payload(self) -> str:
        tet_tags, tet_node_tags = gmsh.model.mesh.getElementsByType(4)
        if len(tet_tags) == 0:
            return ""

        tet_nodes = np.asarray(tet_node_tags, dtype=np.int64).reshape(-1, 4)
        tet_tags_arr = np.asarray(tet_tags, dtype=np.int64)
        face_map: dict[frozenset[int], tuple[int, str]] = {}
        face_defs = [([0, 1, 2], "S1"), ([0, 3, 1], "S2"), ([1, 3, 2], "S3"), ([2, 3, 0], "S4")]
        for cols, face_name in face_defs:
            faces = tet_nodes[:, cols]
            for face_nodes, etag in zip(faces, tet_tags_arr):
                face_map[frozenset(int(n) for n in face_nodes)] = (int(etag), face_name)

        payload_lines: list[str] = []
        for idx, surf_tags in self.surface_tags_by_index.items():
            by_face: dict[str, list[int]] = {"S1": [], "S2": [], "S3": [], "S4": []}
            surface_nodes: set[int] = set()

            for surface_tag in surf_tags:
                tri_tags, tri_node_tags = gmsh.model.mesh.getElementsByType(2, tag=surface_tag)
                if len(tri_tags) == 0:
                    continue
                tri_nodes = np.asarray(tri_node_tags, dtype=np.int64).reshape(-1, 3)
                for tri in tri_nodes:
                    tri_tuple = tuple(int(n) for n in tri)
                    surface_nodes.update(tri_tuple)
                    mapped = face_map.get(frozenset(tri_tuple))
                    if mapped is None:
                        continue
                    etag, face_id = mapped
                    by_face[face_id].append(etag)

            if not any(by_face.values()):
                continue

            surf_name = f"surface_{idx}_All"
            active_faces: list[str] = []
            for face_id, elem_ids in by_face.items():
                if not elem_ids:
                    continue
                set_name = f"_{surf_name}_{face_id}"
                active_faces.append(f"{set_name}, {face_id}")
                payload_lines.append(f"*ELSET, ELSET={set_name}, INTERNAL")
                for i in range(0, len(elem_ids), 16):
                    payload_lines.append(", ".join(str(v) for v in elem_ids[i : i + 16]))

            payload_lines.append(f"*SURFACE, TYPE=ELEMENT, NAME={surf_name}")
            payload_lines.extend(active_faces)

            if surface_nodes:
                payload_lines.append(f"*Nset, nset={surf_name}")
                nodes_sorted = sorted(surface_nodes)
                for i in range(0, len(nodes_sorted), 16):
                    payload_lines.append(", ".join(str(v) for v in nodes_sorted[i : i + 16]))

        return "\n".join(payload_lines)

    def _export_inp(self, output_path: Path) -> None:
        gmsh.write(str(output_path))
        payload = self._generate_abaqus_surface_payload()

        lines = output_path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines.insert(2, "*Part, name=final_model\n")
        if payload:
            lines.append("\n")
            lines.append(payload + "\n")
        lines.append("*End Part\n")
        output_path.write_text("".join(lines), encoding="utf-8")


def build_torchfea_model_from_bspline_surfaces(
    surfaces: Sequence[bspmap.BSP | np.ndarray],
    working_dir: str | Path,
    mesh_size: float,
    output_inp_name: str = "TopOptRun.inp",
    validate_v_boundary_plane: bool = False,
    v_boundary_plane_tol: float = 1.0e-6,
) -> tuple[torchfea.FEAController, torchfea.FEA_INP, Path]:
    """Build a torchfea model from multiple B-spline surfaces.

    The first surface is treated as the outer shell and remaining surfaces
    are treated as cavities.
    """
    if not surfaces:
        raise ValueError("At least one surface is required.")
    if mesh_size <= 0:
        raise ValueError("mesh_size must be > 0.")

    workdir = Path(working_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    for i, surface in enumerate(surfaces):
        data = _surface_to_data(surface)
        if validate_v_boundary_plane:
            _validate_v_boundary_plane(data, tol=v_boundary_plane_tol, label=f"surface_{i}")
        stp_file = workdir / f"__surface-{i}.stp"
        _BSplineSolidGenerator(
            control_points=data.control_points,
            degree_u=data.degree_u,
            degree_v=data.degree_v,
        ).export_step(stp_file)

    inp_path = workdir / output_inp_name
    _MeshGenerator(mesh_size_min=mesh_size * 0.5, mesh_size_max=mesh_size).run(workdir, inp_path)

    inp = torchfea.FEA_INP()
    inp.read_inp(str(inp_path))
    fe = _create_fe_from_inp(inp)
    return fe, inp, inp_path


def _create_fe_from_inp(inp: torchfea.FEA_INP) -> torchfea.FEAController:
    default_device = torch.device("cpu")
    default_dtype = torch.float64

    model = inp.part["final_model"]
    nodes = model.nodes[:, 1:]
    part = torchfea.Part(torch.from_numpy(nodes).to(default_device).to(default_dtype))

    for surface_name, surface in model.surfaces.items():
        sf_now = []
        for sf in surface:
            sf_now.append((sf[0], sf[1]))
        part.add_surface_set(surface_name, sf_now)

    if hasattr(model, "sets_nodes"):
        for set_name, node_indices in model.sets_nodes.items():
            part.set_nodes[set_name] = np.unique(np.array(list(node_indices)))

    elem_type = "C3D4"
    if elem_type not in model.elems:
        keys = list(model.elems.keys())
        if len(keys) == 0:
            raise RuntimeError("No element blocks found in INP final_model.")
        elem_type = keys[0]

    elem_table = model.elems[elem_type]
    elems = elem_table[:, 1:]
    elems_index = elem_table[:, 0]
    element = torchfea.elements.initialize_element(
        element_type=elem_type,
        elems_index=torch.from_numpy(elems_index).to(default_device),
        elems=torch.from_numpy(elems).to(default_device),
        part=part,
    )
    part.add_element(element)

    assembly = torchfea.Assembly()
    assembly.add_part(part=part, name="final_model")
    assembly.add_instance(instance=torchfea.Instance(part), name="final_model")

    fe = torchfea.FEAController()
    fe.assembly = assembly
    fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=1e-3)
    return fe


def extract_surface_names(inp: torchfea.FEA_INP, include_outer: bool = True) -> list[str]:
    """Extract generated surface set names like surface_0_All, surface_1_All, ..."""
    surface_dict = inp.part["final_model"].surfaces
    indexed: list[tuple[int, str]] = []
    for name in surface_dict.keys():
        match = re.fullmatch(r"surface_(\d+)_All", name)
        if match is None:
            continue
        idx = int(match.group(1))
        if include_outer or idx > 0:
            indexed.append((idx, name))
    indexed.sort(key=lambda x: x[0])
    return [name for _, name in indexed]


def build_parametric_pneumatic_model(
    surfaces: Sequence[bspmap.BSP | np.ndarray],
    working_dir: str | Path,
    mesh_size: float,
    mu: float,
    kappa: float,
    pressure_values: Sequence[float],
    output_inp_name: str = "TopOptRun.inp",
    solver_tol_error: float = 1e-5,
    validate_v_boundary_plane: bool = False,
    v_boundary_plane_tol: float = 1.0e-6,
) -> ParametricPneumaticModel:
    """Build lightweight parametric model with material and one-step cavity pressures.

    Convention:
    - surfaces[0]: outer shell
    - surfaces[1:]: cavities
    """
    fe, inp, inp_path = build_torchfea_model_from_bspline_surfaces(
        surfaces=surfaces,
        working_dir=working_dir,
        mesh_size=mesh_size,
        output_inp_name=output_inp_name,
        validate_v_boundary_plane=validate_v_boundary_plane,
        v_boundary_plane_tol=v_boundary_plane_tol,
    )

    fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=float(solver_tol_error))

    cavity_surface_names = extract_surface_names(inp=inp, include_outer=False)
    model = ParametricPneumaticModel(
        fe=fe,
        inp=inp,
        inp_path=inp_path,
        cavity_surface_names=cavity_surface_names,
        load_names=[],
    )
    model.set_material(mu=mu, kappa=kappa)
    model.setup_cavity_pressures(pressure_values)
    return model
