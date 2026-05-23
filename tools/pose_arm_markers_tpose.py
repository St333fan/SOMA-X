# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh


RIGHT_ARM_MARKERS = {"RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN"}
LEFT_ARM_MARKERS = {"LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN"}


def rotation_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotate_point(point: np.ndarray, pivot: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return pivot + rot @ (point - pivot)


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(rows: list[dict], out_json: Path, out_csv: Path) -> None:
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def create_scene(rows: list[dict], radius: float) -> trimesh.Scene:
    scene = trimesh.Scene()
    for row in rows:
        name = f"marker_{int(row['index']):03d}_{clean_name(row['name'])}__body_{clean_name(row['body'])}"
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        sphere.apply_translation(np.asarray(row["location_ground_default_pose"], dtype=np.float64))
        if row["name"] in RIGHT_ARM_MARKERS:
            sphere.visual.face_colors = [220, 80, 80, 255]
        elif row["name"] in LEFT_ARM_MARKERS:
            sphere.visual.face_colors = [60, 130, 230, 255]
        else:
            sphere.visual.face_colors = [245, 220, 70, 255]
        scene.add_geometry(sphere, geom_name=name, node_name=name)
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate arm/hand marker spheres upward into a T-pose.")
    parser.add_argument("input_glb", type=Path)
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("--out-glb", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--right-degrees", type=float, default=-90.0)
    parser.add_argument("--left-degrees", type=float, default=90.0)
    parser.add_argument("--arm-translate", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--right-arm-translate", type=float, nargs=3, default=None)
    parser.add_argument("--left-arm-translate", type=float, nargs=3, default=None)
    parser.add_argument("--marker-radius", type=float, default=0.018)
    args = parser.parse_args()

    rows = load_manifest(args.input_manifest)
    by_name = {row["name"]: row for row in rows}
    right_pivot = np.asarray(by_name["RSHO"]["location_ground_default_pose"], dtype=np.float64)
    left_pivot = np.asarray(by_name["LSHO"]["location_ground_default_pose"], dtype=np.float64)
    right_rot = rotation_z(args.right_degrees)
    left_rot = rotation_z(args.left_degrees)
    right_translate = np.asarray(args.right_arm_translate or args.arm_translate, dtype=np.float64)
    left_translate = np.asarray(args.left_arm_translate or args.arm_translate, dtype=np.float64)

    for row in rows:
        marker = row["name"]
        point = np.asarray(row["location_ground_default_pose"], dtype=np.float64)
        row["location_ground_default_pose_before_arm_tpose"] = point.tolist()
        if marker in RIGHT_ARM_MARKERS:
            row["location_ground_default_pose"] = (
                rotate_point(point, right_pivot, right_rot) + right_translate
            ).tolist()
            row["arm_tpose_transform"] = (
                f"right_arm_rotate_global_z_{args.right_degrees:g}deg_about_RSHO"
                f"_then_translate_{right_translate[0]:g}_{right_translate[1]:g}_{right_translate[2]:g}"
            )
        elif marker in LEFT_ARM_MARKERS:
            row["location_ground_default_pose"] = (
                rotate_point(point, left_pivot, left_rot) + left_translate
            ).tolist()
            row["arm_tpose_transform"] = (
                f"left_arm_rotate_global_z_{args.left_degrees:g}deg_about_LSHO"
                f"_then_translate_{left_translate[0]:g}_{left_translate[1]:g}_{left_translate[2]:g}"
            )
        else:
            row["arm_tpose_transform"] = "unchanged"

    scene = create_scene(rows, args.marker_radius)
    args.out_glb.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.out_glb)
    write_manifest(rows, args.out_json, args.out_csv)

    print(f"wrote {args.out_glb.resolve()}")
    print(f"wrote {args.out_json.resolve()}")
    print(f"wrote {args.out_csv.resolve()}")
    print(f"rotated right arm markers: {sorted(RIGHT_ARM_MARKERS)}")
    print(f"rotated left arm markers: {sorted(LEFT_ARM_MARKERS)}")


if __name__ == "__main__":
    main()
