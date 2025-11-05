
precision highp float;
precision highp int;

#include <splatDefines>
#include <logdepthbuf_pars_fragment>

uniform float near;
uniform float far;
uniform bool encodeLinear;
uniform float time;
uniform bool debugFlag;
uniform float maxStdDev;
uniform float minAlpha;
uniform bool stochastic;
uniform bool disableFalloff;
uniform float falloff;

uniform bool splatTexEnable;
uniform sampler3D splatTexture;
uniform mat2 splatTexMul;
uniform vec2 splatTexAdd;
uniform float splatTexNear;
uniform float splatTexFar;
uniform float splatTexMid;

out vec4 fragColor;

in vec4 vRgba;
in vec2 vSplatUv;
in vec3 vNdc;
flat in uint vSplatIndex;

flat in mat3 vInvRS;           // from vertex
flat in mat3 vRS;              // from vertex (only needed if you want closest-point depth/tex)
flat in vec3 vMu;              // view-space center μ
flat in vec2 vInvFocal;        // (1/px,1/py)
flat in vec2 vScaledRenderSize;// framebuffer size used in vertex

void main() {
    vec4 rgba = vRgba;

    float z = dot(vSplatUv, vSplatUv);
    if (!splatTexEnable) {
        if (z > (maxStdDev * maxStdDev)) {
            discard;
        }
    } else {
        vec2 uv = splatTexMul * vSplatUv + splatTexAdd;
        float ndcZ = vNdc.z;
        float depth = (2.0 * near * far) / (far + near - ndcZ * (far - near));
        float clampedFar = max(splatTexFar, splatTexNear);
        float clampedDepth = clamp(depth, splatTexNear, clampedFar);
        float logDepth = log2(clampedDepth + 1.0);
        float logNear = log2(splatTexNear + 1.0);
        float logFar = log2(clampedFar + 1.0);

        float texZ;
        if (splatTexMid > 0.0) {
            float clampedMid = clamp(splatTexMid, splatTexNear, clampedFar);
            float logMid = log2(clampedMid + 1.0);
            texZ = (clampedDepth <= clampedMid) ?
                (0.5 * ((logDepth - logNear) / (logMid - logNear))) :
                (0.5 * ((logDepth - logMid) / (logFar - logMid)) + 0.5);
        } else {
            texZ = (logDepth - logNear) / (logFar - logNear);
        }

        vec4 modulate = texture(splatTexture, vec3(uv, 1.0 - texZ));
        rgba *= modulate;
    }

    // ---- Per-pixel ray in VIEW space (perspective) ----
    // NDC pixel coords in [-1,1]
    vec2 ndc = vec2(
        (gl_FragCoord.x / vScaledRenderSize.x) * 2.0 - 1.0,
        (gl_FragCoord.y / vScaledRenderSize.y) * 2.0 - 1.0
    );
    // View-space direction through this pixel. Camera at origin, looking -Z.
    // For perspective (no skew): d = normalize( (x/px, y/py, -1) ).
    vec3 d_view = normalize(vec3(ndc.x * vInvFocal.x, ndc.y * vInvFocal.y, -1.0));

    // ---- Map ray into SPLAT space where Gaussian is N(0, I) ----
    // Ray: r(t) = o + t*d in view space, with o=(0,0,0)
    // Solve RS * u + μ = o + t*d  => u(t) = RS^{-1}(o - μ) + t * RS^{-1} d
    vec3 A = -(vInvRS * vMu);   // constant per splat: RS^{-1}(o - μ), o=0
    vec3 B =  (vInvRS * d_view);// per-pixel: RS^{-1} d

    // ---- Closest approach of the ray to the Gaussian center in splat space ----
    float BB = dot(B, B);
    float BA = dot(B, A);
    float AA = dot(A, A);

    // Robust guard for near-parallel (degenerated) cases
    if (BB < 1e-20) discard;

    float tStar = -BA / BB;
    vec3  uStar = A + tStar * B;

    // Mahalanobis distance in splat space (covariance = I)
    float rho2 = dot(uStar, uStar);

    // Opacity at the maximum-response point along the ray
    float alpha = rgba.a * exp(-0.5 * rho2) * max(min(3.0 - 3.0 * z / (maxStdDev * maxStdDev), 1.0), 0.0);

    if (alpha < minAlpha) {
        discard;
    }

    // rgba.a *= 1.0 - mix(1.0, exp(-0.5 * z), falloff);
    // rgba.a *= mix(1.0, exp(-0.5 * z), falloff);
    // rgba.a = 1.0;
    rgba.a = mix(1.0, alpha, falloff);

    if (rgba.a < minAlpha) {
        discard;
    }
    if (encodeLinear) {
        rgba.rgb = srgbToLinear(rgba.rgb);
    }

    if (stochastic) {
        const bool STEADY = false;
        uint uTime = STEADY ? 0u : floatBitsToUint(time);
        uvec2 coord = uvec2(gl_FragCoord.xy);
        uint state = uTime + 0x9e3779b9u * coord.x + 0x85ebca6bu * coord.y + 0xc2b2ae35u * uint(vSplatIndex);
        state = state * 747796405u + 2891336453u;
        uint hash = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        hash = (hash >> 22u) ^ hash;
        float rand = float(hash) / 4294967296.0;
        if (rand < rgba.a) {
            fragColor = vec4(rgba.rgb, 1.0);
        } else {
            discard;
        }
    } else {
        #ifdef PREMULTIPLIED_ALPHA
            fragColor = vec4(rgba.rgb * rgba.a, rgba.a);
        #else
            fragColor = rgba;
        #endif
    }
    #include <logdepthbuf_fragment>
}
