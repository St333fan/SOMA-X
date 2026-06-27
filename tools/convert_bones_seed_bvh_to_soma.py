# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Convert a BONES-SEED SOMA BVH motion into a complete SOMA animation NPZ.

The BONES-SEED BVHs contain the public 78-joint SOMA hierarchy, including a
virtual ``Root``.  ``SOMALayer.pose`` instead consumes 77 rotations beginning
at ``Hips`` plus a separate Hips translation.  This tool folds Root into Hips,
converts BVH centimeters to meters, and combines the motion with the matching
MHR identity/scale parameters.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class Joint:
    name: str
    parent: int | None
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channels: list[str] = field(default_factory=list)
    channel_start: int = 0


@dataclass
class BVH:
    joints: list[Joint]
    motion: np.ndarray
    frame_time: float


def parse_bvh(path: Path) -> BVH:
    lines = path.read_text(encoding="utf-8").splitlines()
    joints: list[Joint] = []
    stack: list[int] = []
    pending_joint: int | None = None
    channel_count = 0
    motion_line: int | None = None

    for line_number, raw in enumerate(lines):
        line = raw.strip()
        if not line or line == "HIERARCHY":
            continue
        if line == "MOTION":
            motion_line = line_number
            break

        parts = line.split()
        keyword = parts[0]
        if keyword in {"ROOT", "JOINT"}:
            parent = stack[-1] if stack else None
            joints.append(Joint(name=parts[1], parent=parent))
            pending_joint = len(joints) - 1
        elif line == "End Site":
            raise ValueError(
                f"{path}: unnamed BVH End Sites are not supported; "
                "BONES-SEED SOMA files should use named end joints."
            )
        elif keyword == "OFFSET":
            target = pending_joint if pending_joint is not None else stack[-1]
            joints[target].offset = np.asarray(parts[1:4], dtype=np.float64)
        elif keyword == "CHANNELS":
            target = pending_joint if pending_joint is not None else stack[-1]
            count = int(parts[1])
            channels = parts[2:]
            if len(channels) != count:
                raise ValueError(f"{path}: malformed CHANNELS line: {line}")
            joints[target].channels = channels
            joints[target].channel_start = channel_count
            channel_count += count
        elif line == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
        elif line == "}":
            if stack:
                stack.pop()

    if motion_line is None:
        raise ValueError(f"{path}: missing MOTION section")

    declared_frames: int | None = None
    frame_time: float | None = None
    rows: list[list[float]] = []
    for raw in lines[motion_line + 1 :]:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Frames:"):
            declared_frames = int(line.split(":", 1)[1])
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split(":", 1)[1])
        elif re.match(r"^[+-]?(?:\d|\.\d)", line):
            row = [float(value) for value in line.split()]
            if len(row) != channel_count:
                raise ValueError(
                    f"{path}: motion row has {len(row)} values, expected {channel_count}"
                )
            rows.append(row)

    if frame_time is None or frame_time <= 0:
        raise ValueError(f"{path}: missing or invalid Frame Time")
    if declared_frames is not None and declared_frames != len(rows):
        raise ValueError(
            f"{path}: declares {declared_frames} frames but contains {len(rows)} rows"
        )
    if not rows:
        raise ValueError(f"{path}: contains no motion frames")

    return BVH(
        joints=joints,
        motion=np.asarray(rows, dtype=np.float64),
        frame_time=frame_time,
    )


def _axis_rotation(axis: str, angles: np.ndarray) -> np.ndarray:
    result = np.zeros((angles.shape[0], 3, 3), dtype=np.float64)
    c = np.cos(angles)
    s = np.sin(angles)
    if axis == "X":
        result[:, 0, 0] = 1
        result[:, 1, 1] = c
        result[:, 1, 2] = -s
        result[:, 2, 1] = s
        result[:, 2, 2] = c
    elif axis == "Y":
        result[:, 0, 0] = c
        result[:, 0, 2] = s
        result[:, 1, 1] = 1
        result[:, 2, 0] = -s
        result[:, 2, 2] = c
    elif axis == "Z":
        result[:, 0, 0] = c
        result[:, 0, 1] = -s
        result[:, 1, 0] = s
        result[:, 1, 1] = c
        result[:, 2, 2] = 1
    else:
        raise ValueError(f"Unknown rotation axis {axis!r}")
    return result


def joint_local_motion(bvh: BVH) -> tuple[np.ndarray, np.ndarray]:
    """Return local rotations and translations in the BVH's native units."""
    num_frames = bvh.motion.shape[0]
    num_joints = len(bvh.joints)
    rotations = np.broadcast_to(np.eye(3), (num_frames, num_joints, 3, 3)).copy()
    translations = np.empty((num_frames, num_joints, 3), dtype=np.float64)

    for joint_index, joint in enumerate(bvh.joints):
        translations[:, joint_index] = joint.offset
        start = joint.channel_start
        values = bvh.motion[:, start : start + len(joint.channels)]
        for channel_index, channel in enumerate(joint.channels):
            axis = channel[0].upper()
            if channel.endswith("position"):
                translations[:, joint_index, "XYZ".index(axis)] = values[:, channel_index]
            elif channel.endswith("rotation"):
                axis_matrix = _axis_rotation(axis, np.deg2rad(values[:, channel_index]))
                rotations[:, joint_index] = rotations[:, joint_index] @ axis_matrix
            else:
                raise ValueError(f"Unsupported BVH channel {channel!r}")
    return rotations, translations


def load_public_joint_names(definition_path: Path) -> list[str]:
    with definition_path.open("r", encoding="utf-8") as handle:
        definition = json.load(handle)
    names = definition["public_rig_derivation"]["main_joint_names"]
    if len(names) != 78 or names[:2] != ["Root", "Hips"]:
        raise ValueError(
            f"{definition_path}: expected 78 public SOMA joints beginning with Root, Hips"
        )
    return names


def validate_hierarchy(joints: Sequence[Joint], expected_names: Sequence[str], path: Path) -> None:
    names = [joint.name for joint in joints]
    if names != list(expected_names):
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(names, expected_names, strict=False)
                )
                if actual != expected
            ),
            min(len(names), len(expected_names)),
        )
        actual = names[mismatch] if mismatch < len(names) else "<missing>"
        expected = expected_names[mismatch] if mismatch < len(expected_names) else "<none>"
        raise ValueError(
            f"{path}: BVH is not the expected public SOMA hierarchy. "
            f"Joint {mismatch}: got {actual!r}, expected {expected!r}; "
            f"counts are {len(names)} and {len(expected_names)}."
        )

    if joints[0].parent is not None or joints[1].parent != 0:
        raise ValueError(f"{path}: expected Root -> Hips at the top of the hierarchy")


def load_mhr_shape(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = {"identity_params", "scale_params"} - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing MHR shape arrays: {sorted(missing)}")
        identity = np.asarray(data["identity_params"], dtype=np.float32)
        scale = np.asarray(data["scale_params"], dtype=np.float32)
    if identity.shape != (1, 45):
        raise ValueError(f"{path}: expected identity_params shape (1, 45), got {identity.shape}")
    if scale.shape != (1, 68):
        raise ValueError(f"{path}: expected scale_params shape (1, 68), got {scale.shape}")
    return identity, scale


def infer_shape_path(bvh_path: Path) -> Path:
    resolved = bvh_path.resolve()
    dataset_root = next(
        (parent for parent in resolved.parents if (parent / "soma_shapes").is_dir()),
        None,
    )
    if dataset_root is None:
        raise ValueError(
            f"Could not find a BONES-SEED root containing soma_shapes above {bvh_path}. "
            "Pass --shape explicitly."
        )

    lower_parts = {part.lower() for part in resolved.parts}
    if "soma_uniform" in lower_parts:
        shape_path = dataset_root / "soma_shapes" / "soma_base_fit_mhr_params.npz"
    elif "soma_proportional" in lower_parts:
        match = re.search(r"__(A\d+)(?:_M)?$", bvh_path.stem, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(
                f"Could not infer actor ID from proportional BVH name {bvh_path.name!r}. "
                "Pass --shape explicitly."
            )
        actor_id = match.group(1).upper()
        shape_path = (
            dataset_root
            / "soma_shapes"
            / "soma_proportion_fit_mhr_params"
            / f"{actor_id}.npz"
        )
    else:
        raise ValueError(
            f"Could not tell whether {bvh_path} is SOMA Uniform or Proportional. "
            "Pass --shape explicitly."
        )

    if not shape_path.is_file():
        raise FileNotFoundError(f"Inferred shape file does not exist: {shape_path}")
    return shape_path


def convert(
    bvh_path: Path,
    shape_path: Path,
    output_path: Path,
    definition_path: Path,
) -> None:
    bvh = parse_bvh(bvh_path)
    public_names = load_public_joint_names(definition_path)
    validate_hierarchy(bvh.joints, public_names, bvh_path)
    identity, scale = load_mhr_shape(shape_path)
    rotations, translations = joint_local_motion(bvh)

    # SOMALayer omits the virtual Root. Fold its transform into Hips so Hips
    # remains the global body rotation/translation expected by SOMALayer.pose.
    #
    # BVH channel rotations are absolute local joint-frame rotations: the
    # bind/joint orientation is already baked into them (the BONES-SEED base
    # BVH therefore contains non-zero rotations in its T-pose).  Marking these
    # as relative would make SOMALayer apply joint orient a second time.
    root_rotation = rotations[:, 0]
    hips_rotation = rotations[:, 1]
    poses = rotations[:, 1:].copy()
    poses[:, 0] = root_rotation @ hips_rotation
    hips_local_translation = translations[:, 1]
    root_translation = translations[:, 0]
    transl_cm = root_translation + np.einsum(
        "nij,nj->ni", root_rotation, hips_local_translation
    )
    transl_m = transl_cm * 0.01

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        poses=poses.astype(np.float32),
        transl=transl_m.astype(np.float32),
        joint_names=np.asarray(public_names[1:]),
        identity_model_type=np.asarray("mhr"),
        identity_coeffs=identity,
        scale_params=scale,
        rotation_repr=np.asarray("matrix"),
        absolute_pose=np.bool_(True),
        unit=np.asarray("meters"),
        keep_root=np.bool_(False),
        fps=np.float32(1.0 / bvh.frame_time),
        frame_time=np.float32(bvh.frame_time),
        source_bvh=np.asarray(str(bvh_path.resolve())),
        source_shape=np.asarray(str(shape_path.resolve())),
    )

    print(f"Saved: {output_path}")
    print(f"  frames: {poses.shape[0]} ({1.0 / bvh.frame_time:.3f} fps)")
    print(f"  poses: {poses.shape} (matrix, absolute local, no virtual Root)")
    print(f"  transl: {transl_m.shape} (meters)")
    print(f"  identity_coeffs: {identity.shape}")
    print(f"  scale_params: {scale.shape}")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert a BONES-SEED SOMA BVH and MHR shape NPZ to SOMA motion NPZ."
    )
    parser.add_argument("bvh", type=Path, help="BONES-SEED SOMA Uniform/Proportional BVH.")
    parser.add_argument("output", type=Path, help="Output SOMA motion .npz.")
    parser.add_argument(
        "--shape",
        type=Path,
        default=None,
        help=(
            "soma_base_fit_mhr_params.npz or actor-specific MHR shape .npz. "
            "By default it is inferred from the BONES-SEED directory and filename."
        ),
    )
    parser.add_argument(
        "--definition",
        type=Path,
        default=repo_root / "assets" / "SOMA_procedural_transforms.json",
        help="SOMA procedural definition used to validate public joint order.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    shape_path = args.shape if args.shape is not None else infer_shape_path(args.bvh)
    convert(args.bvh, shape_path, args.output, args.definition)


if __name__ == "__main__":
    main()
