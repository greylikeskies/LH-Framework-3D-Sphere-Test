# -----------------------------------------------------------
# © 2025 Jeff Boylan
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# DOI: 10.5281/zenodo.17589521
# -----------------------------------------------------------

#L/H Poisson Cube Test - notice the rounding corners.

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import time, os, math
import matplotlib.pyplot as plt

# ----------------------------- Core 3D Ops -----------------------------
def gradient_3d(A):
    gx = 0.5 * (np.roll(A, -1, 2) - np.roll(A, 1, 2))
    gy = 0.5 * (np.roll(A, -1, 1) - np.roll(A, 1, 1))
    gz = 0.5 * (np.roll(A, -1, 0) - np.roll(A, 1, 0))
    return gx, gy, gz

def divergence_3d(px, py, pz):
    dx = 0.5 * (np.roll(px, -1, 2) - np.roll(px, 1, 2))
    dy = 0.5 * (np.roll(py, -1, 1) - np.roll(py, 1, 1))
    dz = 0.5 * (np.roll(pz, -1, 0) - np.roll(pz, 1, 0))
    return dx + dy + dz

def poisson_phi_3d(A, B, lam=1e-2):
    dI = B - A
    gx, gy, gz = gradient_3d(A)
    denom = gx*gx + gy*gy + gz*gz + lam
    return divergence_3d(dI*gx, dI*gy, dI*gz) / np.maximum(denom, 1e-12)

def warp_3d(A, phi):
    gx, gy, gz = gradient_3d(phi)
    D, H, W = A.shape
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    z2 = np.clip(zz + gz, 0, D-1)
    y2 = np.clip(yy + gy, 0, H-1)
    x2 = np.clip(xx + gx, 0, W-1)
    coords = np.vstack([z2.ravel(), y2.ravel(), x2.ravel()])
    return map_coordinates(A, coords, order=1, mode='nearest').reshape(D, H, W)

# ----------------------------- Metrics -----------------------------
def conservation_score_3d(A, B):
    return 1.0 - abs(A.mean() - B.mean())

def l2(A, B):
    return np.sqrt(np.mean((A - B)**2))

def mae(A, B):
    return np.mean(np.abs(A - B))

def psnr(A, B):
    mse = np.mean((A - B)**2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))

# ----------------------------- Cube generator -----------------------------
def cube_volume(D=64, H=64, W=64, center=None, half_size=(10,10,10), intensity=1.0):
    if center is None:
        center = (D/2, H/2, W/2)
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    mask = (
        (np.abs(x - center[2]) <= half_size[2]) &
        (np.abs(y - center[1]) <= half_size[1]) &
        (np.abs(z - center[0]) <= half_size[0])
    )
    vol = np.zeros((D,H,W), np.float32)
    vol[mask] = intensity
    return gaussian_filter(vol, sigma=0.5)  # small softening

def make_scene(D=64, H=64, W=64, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    cA = (D/2 + rng.integers(-4,4), H/2 + rng.integers(-4,4), W/2 + rng.integers(-4,4))
    hA = (rng.integers(8,12), rng.integers(8,12), rng.integers(8,12))
    cB = (D/2 + rng.integers(-8,8), H/2 + rng.integers(-8,8), W/2 + rng.integers(-8,8))
    hB = (rng.integers(7,13), rng.integers(7,13), rng.integers(7,13))
    A = cube_volume(D,H,W,cA,hA)
    B = cube_volume(D,H,W,cB,hB)
    return A, B

# ----------------------------- Visualization -----------------------------
def plot_slices(A, B, phi, warped):
    mid = A.shape[0]//2
    error = np.abs(B - warped)
    fig, axes = plt.subplots(1,5, figsize=(16,4))
    titles = ["A[zmid]", "B[zmid]", "φ[zmid]", "Warped[zmid]", "Error[zmid]"]
    for ax, data, title in zip(axes, [A,B,phi,warped,error], titles):
        im = ax.imshow(data[mid], cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# ----------------------------- Battery -----------------------------
def run_trial(D=64, H=64, W=64, lam=1e-2, visualize=True):
    A, B = make_scene(D,H,W)
    t0 = time.perf_counter()
    phi = poisson_phi_3d(A,B,lam)
    warped = warp_3d(A,phi)
    t = time.perf_counter() - t0
    if visualize:
        plot_slices(A,B,phi,warped)
    print(f"Conservation={conservation_score_3d(A,warped):.6f}, "
          f"PSNR={psnr(B,warped):.2f} dB, Time={t:.3f}s")

if __name__ == "__main__":
    run_trial(D=64, H=64, W=64, lam=1e-2)
