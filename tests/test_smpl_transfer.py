# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from soma.smpl.transfer import _adapt_identity_coeffs


def _layer(*, model_spec, topology_family, num_identity_coeffs):
    return SimpleNamespace(
        model_spec=model_spec,
        topology_family=topology_family,
        num_identity_coeffs=num_identity_coeffs,
        bind_shape=torch.empty(1, dtype=torch.float32),
        device=torch.device("cpu"),
    )


def test_cross_family_identity_defaults_to_neutral_target():
    source = _layer(model_spec="soma", topology_family=None, num_identity_coeffs=45)
    target = _layer(model_spec="smplx", topology_family="body", num_identity_coeffs=10)
    source_identity = torch.arange(45, dtype=torch.float32).unsqueeze(0)

    target_identity = _adapt_identity_coeffs(source_identity, source, target)

    torch.testing.assert_close(target_identity, torch.zeros(1, 10))


def test_smpl_family_identity_is_reused_between_smpl_variants():
    source = _layer(model_spec="smpl", topology_family="body", num_identity_coeffs=16)
    target = _layer(model_spec="smplx", topology_family="body", num_identity_coeffs=10)
    source_identity = torch.arange(16, dtype=torch.float32).unsqueeze(0)

    target_identity = _adapt_identity_coeffs(source_identity, source, target)

    torch.testing.assert_close(target_identity, source_identity[:, :10])
