## License

This repository is dual-licensed:

- **Open-source GPL-3.0** – for research and non-commercial or open work.
- **Commercial license** – for closed-source, proprietary, or embedded use.

Commercial entities may obtain a proprietary license from:

> Jeff Boylan  
> JeffBoylan@proton.me

If you use this code under the GPL, you must comply with its terms.


## Citation

If you use the L/H Framework in academic work or software, please cite:

> **Boylan, Jeff.** (2025). *L/H Framework: Non-Time-Stepped Laplacian and 3-D Harmonic Solver (v0.2).*  
> Zenodo. https://doi.org/10.5281/zenodo.17589521

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17589521.svg)](https://doi.org/10.5281/zenodo.17589521)


# L/H Framework — 3-D Sphere Test
+ [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17589521.svg)](https://doi.org/10.5281/zenodo.17589521)

A demonstration of the **L/H Framework** extended into full 3-D harmonic form.
This test shows single-pass Laplacian motion alignment and energy conservation
across volumetric arrays (spherical and ellipsoidal examples).

---

## Overview
The L/H Framework operates on arrays `A(x)` in harmonic equilibrium satisfying:

\[
\nabla^2 A = 0
\]

For two arrays \( A, B \), the displacement field is recovered by a single Laplacian inversion:

\[
Δφ = ∇·(ΔI·g)/(‖g‖² + λ)
\]

and the warp is applied as:

\[
u(x) = ∇φ(x), \quad Â(x) = A(x + u(x))
\]

---

## Usage

```bash
python LH_3D_Sphere_Volume_Laplacian_Test.py
