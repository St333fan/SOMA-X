# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class Joint:
    name: str
    offset: np.ndarray
    parent: int | None
    channels: list[str] = field(default_factory=list)
    children: list[int] = field(default_factory=list)


@dataclass
class BVHSkeleton:
    joints: list[Joint]
    motion: np.ndarray


def _clean_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def q_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def q_inv(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out / np.maximum(np.sum(q * q, axis=-1, keepdims=True), 1e-12)


def q_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], axis=-1)
    return q_mul(q_mul(q, qv), q_inv(q))[..., :3]


def axis_quat(axis: str, degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    half = radians * 0.5
    out = np.zeros(4, dtype=np.float64)
    out[3] = np.cos(half)
    out["XYZ".index(axis.upper())] = np.sin(half)
    return out


def euler_to_quat(order: str, values_deg: list[float]) -> np.ndarray:
    q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for axis, degrees in zip(order.upper(), values_deg):
        q = q_mul(q, axis_quat(axis, degrees))
    return q_normalize(q)


def parse_bvh(path: Path) -> BVHSkeleton:
    joints: list[Joint] = []
    stack: list[int] = []
    pending_joint: int | None = None
    end_site_count: dict[int, int] = {}

    lines = path.read_text(encoding="utf-8").splitlines()
    in_hierarchy = False
    pending_end_site = False
    motion_start = None

    for line_no, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line == "HIERARCHY":
            in_hierarchy = True
            continue
        if line == "MOTION":
            motion_start = line_no
            break
        if not in_hierarchy:
            continue

        parts = line.split()
        keyword = parts[0]

        if keyword in {"ROOT", "JOINT"}:
            parent = stack[-1] if stack else None
            name = parts[1]
            joint = Joint(name=name, offset=np.zeros(3, dtype=np.float64), parent=parent)
            joints.append(joint)
            idx = len(joints) - 1
            if parent is not None:
                joints[parent].children.append(idx)
            pending_joint = idx
            pending_end_site = False
            continue

        if line == "End Site":
            if not stack:
                raise ValueError("End Site without parent joint")
            parent = stack[-1]
            count = end_site_count.get(parent, 0) + 1
            end_site_count[parent] = count
            suffix = "EndSite" if count == 1 else f"EndSite{count}"
            name = f"{joints[parent].name}_{suffix}"
            joint = Joint(name=name, offset=np.zeros(3, dtype=np.float64), parent=parent)
            joints.append(joint)
            idx = len(joints) - 1
            joints[parent].children.append(idx)
            pending_joint = idx
            pending_end_site = True
            continue

        if keyword == "OFFSET":
            target = pending_joint if pending_joint is not None else (stack[-1] if stack else None)
            if target is None:
                raise ValueError(f"OFFSET without active joint: {line}")
            joints[target].offset = np.array([float(v) for v in parts[1:4]])
            continue

        if keyword == "CHANNELS":
            target = pending_joint if pending_joint is not None else (stack[-1] if stack else None)
            if target is None:
                raise ValueError(f"CHANNELS without active joint: {line}")
            joints[target].channels = parts[2:]
            continue

        if line == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
            continue

        if line == "}":
            if stack:
                stack.pop()
            pending_end_site = False
            continue

    if motion_start is None:
        return BVHSkeleton(joints=joints, motion=np.zeros((0, 0), dtype=np.float64))

    motion_rows = []
    for raw in lines[motion_start + 1 :]:
        line = raw.strip()
        if not line or line.startswith("Frames:") or line.startswith("Frame Time:"):
            continue
        if re.match(r"^-?\d", line):
            motion_rows.append([float(v) for v in re.split(r"\s+", line)])

    return BVHSkeleton(joints=joints, motion=np.asarray(motion_rows, dtype=np.float64))


def offset_world_positions(joints: list[Joint], scale: float) -> np.ndarray:
    positions = np.zeros((len(joints), 3), dtype=np.float64)
    for i, joint in enumerate(joints):
        parent_pos = positions[joint.parent] if joint.parent is not None else 0.0
        positions[i] = parent_pos + joint.offset * scale
    return positions


def world_positions(skel: BVHSkeleton, scale: float, frame: int | None) -> np.ndarray:
    joints = skel.joints
    if frame is None or skel.motion.size == 0:
        return offset_world_positions(joints, scale)

    positions = np.zeros((len(joints), 3), dtype=np.float64)
    rotations = np.zeros((len(joints), 4), dtype=np.float64)
    rotations[:, 3] = 1.0
    row = skel.motion[frame]
    col = 0

    for i, joint in enumerate(joints):
        local_pos = joint.offset.copy()
        rot_order = []
        rot_values = []
        values = row[col : col + len(joint.channels)]
        col += len(joint.channels)

        for channel, value in zip(joint.channels, values):
            axis = channel[0].upper()
            if "position" in channel:
                local_pos["XYZ".index(axis)] = value
            elif "rotation" in channel:
                rot_order.append(axis)
                rot_values.append(value)

        local_q = euler_to_quat("".join(rot_order), rot_values) if rot_values else rotations[i]
        if joint.parent is None:
            rotations[i] = local_q
            positions[i] = local_pos * scale
        else:
            parent_q = rotations[joint.parent]
            rotations[i] = q_mul(parent_q, local_q)
            positions[i] = positions[joint.parent] + q_apply(parent_q, local_pos * scale)

    return positions


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float) -> trimesh.Trimesh | None:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return None
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction / length)
    transform[:3, 3] = (start + end) * 0.5
    return trimesh.creation.cylinder(radius=radius, height=length, sections=12, transform=transform)


def create_scene(
    joints: list[Joint],
    positions: np.ndarray,
    joint_radius: float,
    bone_radius: float,
) -> trimesh.Scene:
    scene = trimesh.Scene()

    for i, joint in enumerate(joints):
        name = f"joint_{i:03d}_{_clean_name(joint.name)}"
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=joint_radius)
        sphere.apply_translation(positions[i])
        sphere.visual.face_colors = [40, 120, 255, 255]
        scene.add_geometry(sphere, geom_name=name, node_name=name)

    for i, joint in enumerate(joints):
        if joint.parent is None:
            continue
        parent = joints[joint.parent]
        bone = cylinder_between(positions[joint.parent], positions[i], bone_radius)
        name = f"bone_{joint.parent:03d}_{i:03d}_{_clean_name(parent.name)}__to__{_clean_name(joint.name)}"
        if bone is None:
            bone = trimesh.creation.box(extents=[bone_radius * 2.0] * 3)
            bone.apply_translation(positions[i])
            name = f"{name}__zero_length"
        bone.visual.face_colors = [235, 160, 55, 255]
        scene.add_geometry(bone, geom_name=name, node_name=name)

    return scene


def write_manifest(
    path_json: Path,
    path_csv: Path,
    joints: list[Joint],
    positions: np.ndarray,
    source_unit: str,
    output_unit: str,
    scale: float,
) -> None:
    rows = []
    for i, joint in enumerate(joints):
        parent_name = joints[joint.parent].name if joint.parent is not None else ""
        rows.append(
            {
                "index": i,
                "name": joint.name,
                "parent_index": joint.parent,
                "parent_name": parent_name,
                "offset_source": joint.offset.tolist(),
                "position_output": positions[i].tolist(),
                "source_unit": source_unit,
                "output_unit": output_unit,
                "scale": scale,
            }
        )

    path_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a named stick-figure mesh from a BVH skeleton.")
    parser.add_argument("bvh", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--scale", type=float, default=0.01, help="BVH source units to output units. Default cm to m.")
    parser.add_argument("--frame", type=int, default=0, help="Motion frame to use for FK positions. Use -1 to sum raw offsets only.")
    parser.add_argument("--joint-radius", type=float, default=0.012)
    parser.add_argument("--bone-radius", type=float, default=0.006)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or args.bvh.stem

    skel = parse_bvh(args.bvh)
    joints = skel.joints
    frame = None if args.frame < 0 else args.frame
    positions = world_positions(skel, args.scale, frame)
    scene = create_scene(joints, positions, args.joint_radius, args.bone_radius)

    glb_path = args.out_dir / f"{stem}_stick_named.glb"
    obj_path = args.out_dir / f"{stem}_stick_combined.obj"
    json_path = args.out_dir / f"{stem}_stick_manifest.json"
    csv_path = args.out_dir / f"{stem}_stick_manifest.csv"

    scene.export(glb_path)
    scene.dump(concatenate=True).export(obj_path)
    write_manifest(json_path, csv_path, joints, positions, "BVH units", "meters", args.scale)

    bone_count = sum(1 for joint in joints if joint.parent is not None)
    print(f"wrote {glb_path.resolve()}")
    print(f"wrote {obj_path.resolve()}")
    print(f"wrote {json_path.resolve()}")
    print(f"wrote {csv_path.resolve()}")
    print(f"joints/points: {len(joints)}, parent-child links: {bone_count}")


if __name__ == "__main__":
    main()
