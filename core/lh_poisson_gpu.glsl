# -----------------------------------------------------------
# © 2025 Jeff Boylan
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# DOI: 10.5281/zenodo.17589521
# -----------------------------------------------------------


vec2 g = sobel3x3(src, p);
vec2 dI = texelFetch(target, p, 0).rg - texelFetch(src, p, 0).rg;  // 2-channel mock
float denom = dot(g, g) + 1e-12;
float S = divergence(dI * g) / denom;
float phi = laplacian_inverse_tex(S, p);
vec2 u = gradient_tex(phi, p);
vec2 p_warp = p + u;

vec4 A_warp = texture(src, (p_warp + 0.5) / textureSize(src, 0));
vec2 g_L = sobel3x3(src, p_warp);
vec4 Lx = normalize_gaussian(A_warp, g_L);
vec4 Hx = A_warp - 1.0 * Lx;
outColor = Hx + 1.0 * Lx;

