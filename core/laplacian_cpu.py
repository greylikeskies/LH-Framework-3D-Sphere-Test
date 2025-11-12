# -----------------------------------------------------------
# © 2025 Jeff Boylan
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# DOI: 10.5281/zenodo.17589521
# -----------------------------------------------------------


import numpy as np

def laplacian(A):
    """Discrete 2D Laplacian with mirror boundaries"""
    return (
        -4 * A
        + np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
        + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
    )

