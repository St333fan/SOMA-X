# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


@dataclass
class JointInfo:
    child_body: str
    parent_body: str
    location_in_parent: np.ndarray
    orientation_in_parent: np.ndarray
    location_in_child: np.ndarray
    orientation_in_child: np.ndarray
    coordinates: dict[str, float]
    axes: list[dict]


@dataclass
class MarkerInfo:
    name: str
    body: str
    location_local: np.ndarray
    location_ground: np.ndarray


def clean_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def numbers(text: str | None) -> np.ndarray:
    if text is None:
        return np.zeros(0, dtype=np.float64)
    vals = [float(v) for v in text.split()]
    return np.asarray(vals, dtype=np.float64)


def child_text(node: ET.Element, name: str, default: str = "") -> str:
    child = node.find(name)
    return child.text if child is not None and child.text is not None else default


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def euler_xyz(angles: np.ndarray) -> np.ndarray:
    rx, ry, rz = angles if angles.size == 3 else (0.0, 0.0, 0.0)
    return rot_axis([1, 0, 0], rx) @ rot_axis([0, 1, 0], ry) @ rot_axis([0, 0, 1], rz)


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if rotation is not None:
        out[:3, :3] = rotation
    if translation is not None:
        out[:3, 3] = translation
    return out


def eval_function(function_node: ET.Element | None, q: float) -> float:
    if function_node is None or len(function_node) == 0:
        return q
    fn = list(function_node)[0]
    tag = fn.tag
    if tag == "LinearFunction":
        coeff = numbers(child_text(fn, "coefficients"))
        if coeff.size >= 2:
            return float(coeff[0] * q + coeff[1])
        if coeff.size == 1:
            return float(coeff[0] * q)
        return q
    if tag == "Constant":
        vals = numbers(child_text(fn, "value"))
        return float(vals[0]) if vals.size else 0.0
    if tag in {"SimmSpline", "NaturalCubicSpline", "PiecewiseLinearFunction"}:
        xs = numbers(child_text(fn, "x"))
        ys = numbers(child_text(fn, "y"))
        if xs.size and ys.size and xs.size == ys.size:
            return float(np.interp(q, xs, ys))
    return q


def parse_axis(axis_node: ET.Element) -> dict:
    coords = child_text(axis_node, "coordinates").split()
    axis = numbers(child_text(axis_node, "axis"))
    return {
        "name": axis_node.get("name", ""),
        "coordinates": coords,
        "axis": axis if axis.size == 3 else np.zeros(3),
        "function": axis_node.find("function"),
    }


def parse_model(path: Path) -> tuple[dict[str, JointInfo], list[MarkerInfo]]:
    root = ET.parse(path).getroot()

    joints: dict[str, JointInfo] = {}
    body_set = root.find(".//BodySet/objects")
    if body_set is not None:
        for body in body_set.findall("Body"):
            body_name = body.get("name", "")
            joint_parent = body.find("Joint")
            if joint_parent is None or len(joint_parent) == 0:
                continue
            joint = list(joint_parent)[0]
            parent_body = child_text(joint, "parent_body").strip()
            coords = {}
            coord_set = joint.find("CoordinateSet/objects")
            if coord_set is not None:
                for coord in coord_set.findall("Coordinate"):
                    coords[coord.get("name", "")] = float(numbers(child_text(coord, "default_value", "0"))[0])
            axes = []
            spatial = joint.find("SpatialTransform")
            if spatial is not None:
                axes = [parse_axis(axis) for axis in spatial.findall("TransformAxis")]

            joints[body_name] = JointInfo(
                child_body=body_name,
                parent_body=parent_body,
                location_in_parent=numbers(child_text(joint, "location_in_parent")),
                orientation_in_parent=numbers(child_text(joint, "orientation_in_parent")),
                location_in_child=numbers(child_text(joint, "location")),
                orientation_in_child=numbers(child_text(joint, "orientation")),
                coordinates=coords,
                axes=axes,
            )

    body_transforms = compute_body_transforms(joints)

    markers = []
    for marker in root.findall(".//Marker"):
        name = marker.get("name", "")
        body = child_text(marker, "socket_parent_frame", child_text(marker, "body")).strip()
        body = body.removeprefix("/bodyset/").strip()
        local = numbers(child_text(marker, "location"))
        if local.size != 3:
            continue
        body_tf = body_transforms.get(body, np.eye(4))
        ground = (body_tf @ np.array([local[0], local[1], local[2], 1.0]))[:3]
        markers.append(MarkerInfo(name=name, body=body, location_local=local, location_ground=ground))

    return joints, markers


def coordinate_transform(joint: JointInfo) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    for axis_info in joint.axes:
        coords = axis_info["coordinates"]
        q = joint.coordinates.get(coords[0], 0.0) if coords else 0.0
        value = eval_function(axis_info["function"], q)
        if axis_info["name"].startswith("rotation"):
            out = out @ transform(rotation=rot_axis(axis_info["axis"], value))
        elif axis_info["name"].startswith("translation"):
            out = out @ transform(translation=axis_info["axis"] * value)
    return out


def joint_child_transform(joint: JointInfo) -> np.ndarray:
    parent_frame = transform(euler_xyz(joint.orientation_in_parent), joint.location_in_parent)
    child_frame = transform(euler_xyz(joint.orientation_in_child), joint.location_in_child)
    return parent_frame @ coordinate_transform(joint) @ np.linalg.inv(child_frame)


def compute_body_transforms(joints: dict[str, JointInfo]) -> dict[str, np.ndarray]:
    transforms = {"ground": np.eye(4, dtype=np.float64)}

    pending = dict(joints)
    while pending:
        progressed = False
        for body, joint in list(pending.items()):
            if joint.parent_body not in transforms:
                continue
            transforms[body] = transforms[joint.parent_body] @ joint_child_transform(joint)
            del pending[body]
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(pending))
            raise RuntimeError(f"Could not resolve body transforms for: {unresolved}")

    return transforms


def create_marker_scene(markers: list[MarkerInfo], radius: float) -> trimesh.Scene:
    scene = trimesh.Scene()
    palette = {
        "pelvis": [235, 160, 55, 255],
        "thorax": [50, 130, 230, 255],
        "humerus": [60, 180, 95, 255],
        "humerus_l": [60, 180, 95, 255],
        "ulna": [220, 80, 80, 255],
        "ulna_l": [220, 80, 80, 255],
        "radius": [160, 90, 220, 255],
        "radius_l": [160, 90, 220, 255],
    }
    for i, marker in enumerate(markers):
        name = f"marker_{i:03d}_{clean_name(marker.name)}__body_{clean_name(marker.body)}"
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        sphere.apply_translation(marker.location_ground)
        sphere.visual.face_colors = palette.get(marker.body, [245, 220, 70, 255])
        scene.add_geometry(sphere, geom_name=name, node_name=name)
    return scene


def write_manifest(path_json: Path, path_csv: Path, markers: list[MarkerInfo]) -> None:
    rows = []
    for i, marker in enumerate(markers):
        rows.append(
            {
                "index": i,
                "name": marker.name,
                "body": marker.body,
                "location_local_body": marker.location_local.tolist(),
                "location_ground_default_pose": marker.location_ground.tolist(),
                "unit": "meters",
            }
        )

    path_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a named marker-only mesh from an OpenSim .osim model.")
    parser.add_argument("osim", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--marker-radius", type=float, default=0.018)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or args.osim.stem

    _, markers = parse_model(args.osim)
    scene = create_marker_scene(markers, args.marker_radius)

    glb_path = args.out_dir / f"{stem}_markers_named.glb"
    obj_path = args.out_dir / f"{stem}_markers_combined.obj"
    json_path = args.out_dir / f"{stem}_markers_manifest.json"
    csv_path = args.out_dir / f"{stem}_markers_manifest.csv"

    scene.export(glb_path)
    scene.dump(concatenate=True).export(obj_path)
    write_manifest(json_path, csv_path, markers)

    positions = np.asarray([m.location_ground for m in markers])
    print(f"wrote {glb_path.resolve()}")
    print(f"wrote {obj_path.resolve()}")
    print(f"wrote {json_path.resolve()}")
    print(f"wrote {csv_path.resolve()}")
    print(f"markers: {len(markers)}")
    print(f"bounds min: {positions.min(axis=0)}")
    print(f"bounds max: {positions.max(axis=0)}")


if __name__ == "__main__":
    main()
