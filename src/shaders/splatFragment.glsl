
precision highp float;
precision highp int;

#include <splatDefines>

uniform float near;
uniform float far;
uniform bool encodeLinear;
uniform float time;
uniform bool debugFlag;
uniform float maxStdDev;
uniform float minAlpha;
uniform bool disableFalloff;
uniform float falloff;

out vec4 fragColor;

in vec4 vRgba;
in vec2 vSplatUv;
in vec3 vNdc;
flat in uint vSplatIndex;
flat in float adjustedStdDev;

// ---- 3DGUT per-pixel 3D evaluation (see splatVertex.glsl) ----
// The pixel's ray in SPLAT space, u(t) = A + t·B (Gaussian = N(0, I) there).
flat in vec3 vRayA;              // constant per splat
in vec3 vRayB;                   // interpolated per pixel (exact — see vertex)
flat in float vSplat3D;          // 1.0 = 3D ray test valid, 0.0 = 2D fallback

#include <logdepthbuf_pars_fragment>

void main() {
    vec4 rgba = vRgba;

    // Screen-space radius² within the splat's footprint. Still the primitive's
    // extent test even under 3DGUT — the quad is sized from the 2D covariance,
    // so this is what bounds the rasterised area.
    float z2 = dot(vSplatUv, vSplatUv);
    float stdDev2 = adjustedStdDev * adjustedStdDev;
    if (z2 > stdDev2) {
        discard;
    }

    // ---- Gaussian response ----
    // `response` is exp(-0.5 * d²) for the appropriate distance measure d:
    // the per-pixel ray's closest approach to the Gaussian in 3D (3DGUT), or
    // the flat screen-space footprint (upstream's original behaviour).
    float response;

    if (vSplat3D > 0.5) {
        // Closest approach of the pixel's ray to the centre, in splat space:
        // |A + t*B|² is minimised at t* = −(A·B)/(B·B), leaving
        // ρ² = |A|² − (A·B)²/|B|² (the Mahalanobis distance² there). Scale-
        // free in B, so the interpolated, unnormalised direction is fine.
        vec3 A = vRayA;
        vec3 B = vRayB;
        float BB = dot(B, B);
        if (BB < 1e-20) {
            // Degenerate / near-parallel: nothing meaningful to integrate.
            discard;
        }
        float BA = dot(B, A);
        float rho2 = max(0.0, dot(A, A) - BA * BA / BB);
        response = exp(-0.5 * rho2);

        // Fade to zero at the edge of the rasterised footprint. The 3D
        // response does NOT reach zero at the quad boundary the way the 2D one
        // does, so without this the splat ends in a hard edge where the
        // primitive stops.
        response *= clamp(3.0 - 3.0 * z2 / stdDev2, 0.0, 1.0);
    } else {
        // Covariance splats, 2DGS discs and orthographic cameras: unchanged.
        response = exp(-0.5 * z2);
    }

    // Upstream's alpha curve, with the response swapped in. PRESERVED rather
    // than replaced: `rgba.a` here can exceed 1.0 (the vertex stage stretches
    // 1..2 into 1..5 for LOD-merged splats), and the >1 branch is what keeps
    // those from saturating. Feeding a >1 alpha straight into the 3DGUT
    // formulation would blow past 1.0 and clip.
    if (rgba.a <= 1.0) {
        rgba.a = mix(rgba.a, rgba.a * response, falloff);
    } else {
        float a = exp((rgba.a * rgba.a - 1.0) / 2.718281828459045);
        float alpha = 1.0 - pow(1.0 - response, a);
        rgba.a = mix(1.0, alpha, falloff);
    }

    if (rgba.a < minAlpha) {
        discard;
    }
    if (encodeLinear) {
        rgba.rgb = srgbToLinear(rgba.rgb);
    }

    #ifdef PREMULTIPLIED_ALPHA
        fragColor = vec4(rgba.rgb * rgba.a, rgba.a);
    #else
        fragColor = rgba;
    #endif

    #include <logdepthbuf_fragment>
}
