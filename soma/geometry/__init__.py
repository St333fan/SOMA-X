# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advanced geometry and rigging utilities used throughout SOMA-X.

The `soma.geometry` package contains the lower-level building blocks behind
skinning, skeleton fitting, interpolation, Laplacian mesh editing, and several
Warp-accelerated kernels. Most users will interact with these features through
:obj:`~soma.soma.SOMALayer` or :obj:`~soma.pose_inversion.PoseInversion`, but
the modules here are available for custom fitting and retargeting pipelines.
"""
