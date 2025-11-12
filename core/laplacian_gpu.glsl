# -----------------------------------------------------------
# © 2025 Jeff Boylan
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-JeffBoylan-Commercial
# DOI: 10.5281/zenodo.17589521
# -----------------------------------------------------------


float laplacian(sampler2D tex, ivec2 p) {
    float c  = texelFetch(tex, p, 0).r;
    float up = texelFetch(tex, p + ivec2(0,1), 0).r;
    float dn = texelFetch(tex, p + ivec2(0,-1), 0).r;
    float lf = texelFetch(tex, p + ivec2(-1,0), 0).r;
    float rt = texelFetch(tex, p + ivec2(1,0), 0).r;
    return up + dn + lf + rt - 4.0 * c;
}

