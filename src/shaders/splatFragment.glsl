
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
flat in mat3 vInvRS;             // (R*S)^-1 = S^-1 * R^T  (view -> splat space)
flat in vec3 vMu;                // view-space centre μ
flat in vec2 vInvFocal;          // (1/px, 1/py)
flat in vec2 vScaledRenderSize;  // framebuffer size used in the vertex stage
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
        // ---- Per-pixel ray in VIEW space (perspective) ----
        // Pixel centre in NDC [-1,1].
        vec2 ndc = vec2(
            (gl_FragCoord.x / vScaledRenderSize.x) * 2.0 - 1.0,
            (gl_FragCoord.y / vScaledRenderSize.y) * 2.0 - 1.0
        );
        // Camera at the origin looking down -Z; no skew, so the direction
        // through this pixel is (x/px, y/py, -1).
        vec3 dView = normalize(vec3(ndc.x * vInvFocal.x, ndc.y * vInvFocal.y, -1.0));

        // ---- Map the ray into SPLAT space, where the Gaussian is N(0, I) ----
        // r(t) = o + t*d with o = 0. Solving RS*u + μ = o + t*d gives
        // u(t) = RS^-1 (o - μ) + t * RS^-1 d.
        vec3 A = -(vInvRS * vMu);
        vec3 B = vInvRS * dView;

        // Closest approach of the ray to the centre, in splat space.
        float BB = dot(B, B);
        if (BB < 1e-20) {
            // Degenerate / near-parallel: nothing meaningful to integrate.
            discard;
        }
        float tStar = -dot(B, A) / BB;
        vec3 uStar = A + tStar * B;

        // Mahalanobis distance² (covariance is identity in this space).
        float rho2 = dot(uStar, uStar);
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
