# -----------------------------------------------------------
# © 2025 Jeff Boylan
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# DOI: 10.5281/zenodo.17589521
# -----------------------------------------------------------

#L/H Torus Propagation (Non-Temporal Visualization)
#Single-pass Green's function convolution on a periodic domain.
#The frames generated below are not time stepped motion, but spatial equalization toward equilibrium.

import os, time, math, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# If this lives in a separate file, import your functions like:
# from your_module import (
#     gradient_3d, divergence_3d, poisson_phi_3d, warp_3d,
#     conservation_score_3d, l2, mae, psnr,
#     torus_volume, make_scene, plot_slices, circulation_midloop
# )

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

from scipy.ndimage import map_coordinates, gaussian_filter

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

# ----------------------------- Torus generator -----------------------------
def torus_volume(D=64, H=64, W=64, center=None, R=16, r=6, intensity=1.0):
    if center is None:
        center = (D/2, H/2, W/2)
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    x, y, z = x-center[2], y-center[1], z-center[0]
    val = (np.sqrt(x**2 + y**2) - R)**2 + z**2
    mask = val <= r**2
    vol = np.zeros((D,H,W), np.float32)
    vol[mask] = intensity
    return gaussian_filter(vol, sigma=0.8)

def make_scene(D=64, H=64, W=64, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    R1, r1 = rng.uniform(14,18), rng.uniform(5,8)
    R2, r2 = rng.uniform(13,19), rng.uniform(4,8)
    cA = (D/2 + rng.integers(-3,3), H/2 + rng.integers(-3,3), W/2 + rng.integers(-3,3))
    cB = (D/2 + rng.integers(-6,6), H/2 + rng.integers(-6,6), W/2 + rng.integers(-6,6))
    A = torus_volume(D,H,W,cA,R1,r1)
    B = torus_volume(D,H,W,cB,R2,r2)
    return A,B

# ----------------------------- Circulation diagnostic -----------------------------
def circulation_midloop(phi, r_frac=0.35, n=2048, slice_index=None):
    """∮ ∇φ · dl on a circular contour in the mid z-slice."""
    if slice_index is None:
        slice_index = phi.shape[0] // 2
    sl = phi[slice_index]
    gy, gx = np.gradient(sl)
    h, w = sl.shape
    cx, cy = w / 2.0, h / 2.0
    r = r_frac * min(cx, cy)
    theta = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    gx_s = map_coordinates(gx, [y, x], order=1, mode='nearest')
    gy_s = map_coordinates(gy, [y, x], order=1, mode='nearest')
    tx = -r * np.sin(theta)
    ty =  r * np.cos(theta)
    dtheta = 2.0 * np.pi / n
    circ = np.sum(gx_s * tx + gy_s * ty) * dtheta
    return circ

# ----------------------------- GIF demo and bench -----------------------------
def torus_demo_and_gif(
    D=64, H=64, W=64, lam=1e-2, n_frames=80, fps=20,
    out_dir="out", seed=42, visualize_slice=True
):
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    A, B = make_scene(D,H,W, rng=rng)

    # Compute phi once (one-pass). We will animate by scaling phi: s in [0,1].
    t0 = time.perf_counter()
    phi = poisson_phi_3d(A, B, lam=lam)
    t_phi = time.perf_counter() - t0

    # Generate frames: warped_s = warp(A, s*phi)
    zmid = D // 2
    frames = []
    residuals = []
    times = []
    s_vals = np.linspace(0.0, 1.0, n_frames)
    H_last = None

    for i, s in enumerate(s_vals):
        tA = time.perf_counter()
        warped = warp_3d(A, s * phi)
        times.append(time.perf_counter() - tA)

        # Store mid-slice for GIF
        frames.append(warped[zmid])

        # Residual against target B (mid-slice) and per-frame delta
        resid = np.linalg.norm((B - warped)[zmid])
        residuals.append(resid)

        H_last = warped

    # Final metrics
    cons = conservation_score_3d(A, H_last)
    p = psnr(B, H_last)
    circ = circulation_midloop(phi)

    # Logs
    csv_path = os.path.join(out_dir, "torus_anim_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "s", "residual_norm_mid_slice", "frame_time_sec"])
        for i, (s, r, t) in enumerate(zip(s_vals, residuals, times)):
            w.writerow([i, float(s), float(r), float(t)])

    # GIF
    vmin = min(np.min(f) for f in frames)
    vmax = max(np.max(f) for f in frames)

    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(frames[0], cmap="cividis", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title("L/H propagation (mid z-slice) - non-temporal animation")
    ax.axis("off")

    def update(i):
        im.set_data(frames[i])
        ax.set_title(f"L/H propagation - frame {i+1}/{len(frames)}")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(frames), blit=True, interval=1000/fps)
    gif_path = os.path.join(out_dir, "L_H_torus.gif")
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    # Residual plot
    plt.figure(figsize=(6,3))
    plt.plot(residuals, lw=1.5)
    plt.xlabel("Frame")
    plt.ylabel("‖B - warped(s·φ)‖ (mid-slice)")
    plt.title("Residual vs animation frame")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    resid_png = os.path.join(out_dir, "residual_vs_frame.png")
    plt.savefig(resid_png, dpi=150)
    plt.close()

    # Summary
    summary_txt = os.path.join(out_dir, "torus_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"phi_time_sec={t_phi:.4f}\n")
        f.write(f"mean_frame_time_sec={np.mean(times):.6f}\n")
        f.write(f"conservation_score={cons:.6f}\n")
        f.write(f"psnr_db={p:.2f}\n")
        f.write(f"circulation_integral={circ:.6e}\n")

    print(f"phi_time_sec={t_phi:.4f}")
    print(f"mean_frame_time_sec={np.mean(times):.6f}")
    print(f"Conservation={cons:.6f}, PSNR={p:.2f} dB")
    print(f"Circulation integral (∮∇φ·dl) = {circ:.6e}")
    print(f"Wrote: {gif_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {resid_png}")
    print(f"Wrote: {summary_txt}")

# ----------------------------- Entry -----------------------------
if __name__ == "__main__":
    # Adjust D,H,W and lam as needed. 64^3 is quick; 96^3 still reasonable on CPU.
    torus_demo_and_gif(D=64, H=64, W=64, lam=1e-2, n_frames=80, fps=20, out_dir="out")
