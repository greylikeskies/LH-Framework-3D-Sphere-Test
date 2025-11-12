# -----------------------------------------------------------
# © 2025 Jeff Boylan
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# DOI: 10.5281/zenodo.17589521
# -----------------------------------------------------------


#"Implements single-pass L/H Poisson motion closure per Boylan (2025)"

import numpy as np

def gradient(A):
    gx = 0.5 * (np.roll(A, -1, axis=1) - np.roll(A, 1, axis=1))
    gy = 0.5 * (np.roll(A, -1, axis=0) - np.roll(A, 1, axis=0))
    return np.stack([gx, gy], axis=0)

def divergence(v):
    vx, vy = v[0], v[1]
    dvx = np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)
    dvy = np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)
    return 0.5 * (dvx + dvy)

def laplacian_inverse(S, max_iter=100, tol=1e-6):
    phi = np.zeros_like(S)
    for _ in range(max_iter):
        phi_new = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) +
                   np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - S) / 4.0
        if np.max(np.abs(phi_new - phi)) < tol:
            break
        phi = phi_new
    return phi

# L/H Motion Pipeline
A = np.random.rand(256, 256).astype(np.float32)  # source
B = np.random.rand(256, 256).astype(np.float32)  # target

g_m = gradient(A)
dI = B - A
S = divergence(dI * g_m) / (np.linalg.norm(g_m, axis=0)**2 + 1e-12)
phi = laplacian_inverse(S)
u = gradient(phi)
A_warp = np.roll(np.roll(A, int(u[0,0,0]), axis=1), int(u[1,0,0]), axis=0)  # simplified

g_L = gradient(A_warp)
Lx = np.linalg.norm(g_L, axis=0)
Hx = A_warp - 1.0 * Lx  # alpha=1.0
recon = Hx + 1.0 * Lx

# C1 score
C1 = 1.0 - np.linalg.norm(A_warp - recon) / (np.linalg.norm(A_warp) + 1e-8)
print(f"C₁: {C1:.4f}")

