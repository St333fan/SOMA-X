# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from soma import SOMALayer


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a SOMA mesh from MHR parameter .npz data.")
    parser.add_argument(
        "--params",
        type=Path,
        default=Path(r"C:/Users/stlec/Projects/bones-seed/soma_shapes/soma_base_fit_mhr_params.npz"),
        help="Input .npz with identity_params and scale_params.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("out/soma_from_mhr_params.obj"),
        help="Output mesh path, for example .obj, .ply, .stl, or .glb.",
    )
    parser.add_argument("--data-root", default="./assets", help="SOMA assets directory.")
    parser.add_argument(
        "--device",
        default="auto",
        help='Torch device. Use "auto", "cpu", or e.g. "cuda:0".',
    )
    parser.add_argument(
        "--apply-correctives",
        action="store_true",
        help="Apply SOMA pose correctives. Off by default for a neutral shape mesh.",
    )
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    data = np.load(args.params, allow_pickle=False)
    identity = torch.from_numpy(data["identity_params"]).float().to(device)
    scale = torch.from_numpy(data["scale_params"]).float().to(device)

    soma = SOMALayer(
        data_root=args.data_root,
        identity_model_type="mhr",
        device=device,
        mode="torch",
    ).to(device)

    poses = torch.zeros(identity.shape[0], 77, 3, device=device)
    transl = torch.zeros(identity.shape[0], 3, device=device)

    with torch.no_grad():
        out = soma(
            poses,
            identity,
            scale_params=scale,
            transl=transl,
            apply_correctives=args.apply_correctives,
        )

    vertices = out["vertices"][0].detach().cpu().numpy()
    faces = soma.faces.detach().cpu().numpy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(args.out)

    print(f"wrote {args.out.resolve()}")
    print(f"vertices: {len(vertices)}")
    print(f"faces: {len(faces)}")


if __name__ == "__main__":
    main()
