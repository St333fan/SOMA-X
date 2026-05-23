# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh


def rotation_y_right_handed(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c = np.cos(radians)
    s = np.sin(radians)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    return out


def scale_matrix(scale: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] *= scale
    return out


def transform_point(point: list[float], matrix: np.ndarray) -> list[float]:
    p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    return (matrix @ p)[:3].tolist()


def translation_matrix(translation: list[float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.asarray(translation, dtype=np.float64)
    return out


def write_manifest(
    in_path: Path,
    out_json: Path,
    out_csv: Path,
    matrix: np.ndarray,
    scale: float,
    y_degrees: float,
    translation: list[float],
) -> None:
    rows = json.loads(in_path.read_text(encoding="utf-8"))
    for row in rows:
        row["location_ground_default_pose_before_transform"] = row["location_ground_default_pose"]
        row["location_ground_default_pose"] = transform_point(
            row["location_ground_default_pose"], matrix
        )
        row["transform_applied"] = (
            f"uniform_scale_{scale:g}_then_right_handed_y_rotation_{y_degrees:g}deg"
            f"_then_translation_{translation[0]:g}_{translation[1]:g}_{translation[2]:g}"
        )

    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform a marker GLB and manifest.")
    parser.add_argument("input_glb", type=Path)
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("--out-glb", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0629)
    parser.add_argument("--y-degrees", type=float, default=90.0)
    parser.add_argument("--translate", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    args = parser.parse_args()

    args.out_glb.parent.mkdir(parents=True, exist_ok=True)
    matrix = (
        translation_matrix(args.translate)
        @ rotation_y_right_handed(args.y_degrees)
        @ scale_matrix(args.scale)
    )

    scene = trimesh.load(args.input_glb)
    scene.apply_transform(matrix)
    scene.export(args.out_glb)
    write_manifest(
        args.input_manifest,
        args.out_json,
        args.out_csv,
        matrix,
        args.scale,
        args.y_degrees,
        args.translate,
    )

    bounds = scene.bounds
    print(f"wrote {args.out_glb.resolve()}")
    print(f"wrote {args.out_json.resolve()}")
    print(f"wrote {args.out_csv.resolve()}")
    print(f"bounds min: {bounds[0]}")
    print(f"bounds max: {bounds[1]}")


if __name__ == "__main__":
    main()
