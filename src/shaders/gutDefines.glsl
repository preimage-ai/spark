// 3DGUT-matched rendering (FullCircle / NVIDIA 3DGRUT "3dgut" renderer).
//
// Instead of the 2D EWA splat used by standard 3DGS, every fragment evaluates the
// particle along its own camera ray, exactly like 3DGUT's `gaussianParticle::hit`
// (threedgut_tracer/include/3dgut/kernels/slang/models/gaussianParticles.slang):
//   o_c = S^-1 R^T (o - mu),  d_c = normalize(S^-1 R^T d)
//   d2  = |cross(d_c, o_c)|^2                      (squared ray-to-centre distance)
//   response = exp(-4.5/3^n * d2^(n/2))            (generalized Gaussian of degree n)
//   alpha = min(maxAlpha, response * opacity)
//   accepted iff dot(d_c, o_c) < 0 && response > minResponse && alpha > minAlpha
// All 3DGUT geometry here is expressed in OpenCV camera coordinates
// (x right, y down, z forward) with pixel coordinates measured from the top-left
// image corner (pixel centres at +0.5), like the reference renderer.

uniform bool gutMode;
uniform int gutKernelDegree;
uniform float gutMinResponse;
uniform float gutMinAlpha;
uniform float gutMaxAlpha;
uniform float gutMinSensorDistance;
// Debug: emulate 3DGUT's tile-level footprint culling (unscented-transform projection,
// boundFootprint, 16x16 tiles). Off by default: exact silhouettes.
uniform bool gutTileEmulation;
// Skip particles whose cutoff ellipsoid is entirely outside the pinhole view frustum before their footprint
// work (same image; saves vertex work where it dominates, e.g. phone GPUs). Off by default.
uniform bool gutEarlyCull;
uniform bool gutTileBasedCulling;
uniform bool gutRectBounding;
uniform bool gutTightOpacityBounding;
uniform bool gutMipSplattingScaling;
// UT alpha, beta, kappa, in-image margin factor
uniform vec4 gutUtParams;
uniform float gutCovarianceDilation;
uniform float gutTileSize;
// Camera model: 0 = pinhole (intrinsics from the three.js projection matrix),
// 1 = OpenCV fisheye: projection from gutFisheye* (for culling and footprints), per-pixel ray
// directions (OpenCV camera space, w = valid) from gutRayMap, e.g. FullCircle's dataset rays.
uniform int gutCameraModel;
uniform vec4 gutFisheyeIntr;      // fx, fy, cx, cy (pixels, top-left origin)
uniform vec4 gutFisheyeK;         // k1..k4
uniform float gutFisheyeMaxAngle; // radians

// Generalized Gaussian kernel response for a squared canonical distance,
// mirrors canonicalRayMaxKernelResponse<KernelDegree>.
float gutKernelResponse(int degree, float d2) {
    if (degree == 8) {
        float d4 = d2 * d2;
        return exp(-0.000685871056241 * d4 * d4);
    } else if (degree == 5) {
        return exp(-0.0185185185185 * d2 * d2 * sqrt(d2));
    } else if (degree == 4) {
        return exp(-0.0555555555556 * d2 * d2);
    } else if (degree == 3) {
        return exp(-0.166666666667 * d2 * sqrt(d2));
    } else if (degree == 1) {
        return exp(-1.5 * sqrt(d2));
    } else if (degree == 0) {
        return max(1.0 - 0.329630334487 * sqrt(d2), 0.0);
    }
    return exp(-0.5 * d2);
}

// Canonical radius (in units of the particle scales) at which the response drops to thr.
float gutCutoffRadius(int degree, float thr) {
    if (!(thr < 1.0)) {
        return 0.0;
    }
    if (degree == 0) {
        return (1.0 - thr) / 0.329630334487;
    }
    float L = -log(thr);
    float n = 2.0;
    float s = 0.5;
    if (degree == 1) { n = 1.0; s = 1.5; }
    else if (degree == 3) { n = 3.0; s = 0.166666666667; }
    else if (degree == 4) { n = 4.0; s = 0.0555555555556; }
    else if (degree == 5) { n = 5.0; s = 0.0185185185185; }
    else if (degree == 8) { n = 8.0; s = 0.000685871056241; }
    return pow(L / s, 1.0 / n);
}

// tileMinParticlePowerResponse from 3DGUT's gutProjector.cuh
float gutTileMinPower(vec2 tileCoords, vec3 conic, vec2 mean) {
    vec2 tileSize = vec2(gutTileSize);
    vec2 tileMin = tileSize * tileCoords;
    vec2 tileMax = tileSize + tileMin;
    vec2 minOffset = tileMin - mean;
    vec2 leftAbove = vec2(minOffset.x > 0.0 ? 1.0 : 0.0, minOffset.y > 0.0 ? 1.0 : 0.0);
    vec2 notInRange = vec2(
        leftAbove.x + (mean.x > tileMax.x ? 1.0 : 0.0),
        leftAbove.y + (mean.y > tileMax.y ? 1.0 : 0.0)
    );
    if ((notInRange.x + notInRange.y) > 0.0) {
        vec2 p = mix(tileMax, tileMin, leftAbove);
        vec2 dxy = vec2(minOffset.x < 0.0 ? -tileSize.x : tileSize.x, minOffset.y < 0.0 ? -tileSize.y : tileSize.y);
        vec2 diff = mean - p;
        vec2 rcpv = vec2(1.0 / (tileSize.x * tileSize.x * conic.x), 1.0 / (tileSize.y * tileSize.y * conic.z));
        float tx = notInRange.y * clamp((dxy.x * conic.x * diff.x + dxy.x * conic.y * diff.y) * rcpv.x, 0.0, 1.0);
        float ty = notInRange.x * clamp((dxy.y * conic.y * diff.x + dxy.y * conic.z * diff.y) * rcpv.y, 0.0, 1.0);
        vec2 minPosDiff = mean - vec2(p.x + tx * dxy.x, p.y + ty * dxy.y);
        return 0.5 * (conic.x * minPosDiff.x * minPosDiff.x + conic.z * minPosDiff.y * minPosDiff.y) + conic.y * minPosDiff.x * minPosDiff.y;
    }
    return 0.0;
}
