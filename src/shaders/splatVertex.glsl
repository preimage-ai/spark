
precision highp float;
precision highp int;
precision highp usampler2DArray;

#include <splatDefines>

out vec4 vRgba;
out vec2 vSplatUv;
out vec3 vNdc;
flat out uint vSplatIndex;
flat out float adjustedStdDev;

// ---- 3DGUT (3D Gaussian Unscented Transform) ----
// Reference: "3DGUT: Enabling Distorted Cameras and Secondary Rays in
// Gaussian Splatting".
//
// Instead of projecting the 3D covariance with the EWA Jacobian evaluated at
// the splat centre, the 2D covariance is ESTIMATED from the projection of 6
// sigma points (±principal axes). That stays correct where the Jacobian's
// local-linearity assumption breaks down: wide FOV, strong distortion, and
// splats far off the optical axis.
//
// The fragment stage additionally evaluates each splat against the real
// per-pixel ray in 3D rather than against a flat screen-space footprint, so
// these carry the data it needs to reconstruct that ray.
flat out mat3 vInvRS;            // (R*S)^-1 = S^-1 * R^T  (view -> splat space)
flat out vec3 vMu;               // view-space centre μ
flat out vec2 vInvFocal;         // (1/px, 1/py) from the projection matrix
flat out vec2 vScaledRenderSize; // the scaledRenderSize used for bounds here
/**
 * 1.0 when the 3D ray evaluation above is valid for this splat, 0.0 when the
 * fragment stage must fall back to the classic 2D screen-space Gaussian.
 *
 * Needed because two splat kinds have no (R,S) decomposition to invert:
 *  - COVARIANCE splats (`enableCovSplats`), which arrive as a raw 3x3
 *    covariance with no scales/quaternion;
 *  - 2DGS discs (`enable2DGS` with a zero scale axis), which are flat and
 *    whose RS is singular by construction.
 * Both keep upstream's Jacobian/screen-space path untouched.
 */
flat out float vSplat3D;

// uniform uint numSplats;
uniform vec2 renderSize;
uniform vec4 renderToViewQuat;
uniform vec3 renderToViewPos;
uniform mat3 renderToViewBasis;
uniform float maxStdDev;
uniform float minPixelRadius;
uniform float maxPixelRadius;
uniform bool enableExtSplats;
uniform bool enableCovSplats;
uniform float time;
uniform float deltaTime;
uniform bool debugFlag;
uniform float minAlpha;
uniform bool enable2DGS;
uniform bool lodInflate;
uniform float blurAmount;
uniform float preBlurAmount;
uniform float focalDistance;
uniform float apertureAngle;
uniform float clipXY;
uniform float focalAdjustment;

uniform usampler2D ordering;
uniform usampler2DArray extSplats;
uniform usampler2DArray extSplats2;

// Required by logdepthbuf_pars_vertex (normally defined in three.js #include <common>)
bool isPerspectiveMatrix( mat4 m ) {
    return m[ 2 ][ 3 ] == -1.0;
}

#include <logdepthbuf_pars_vertex>

// ---- Unscented transform weights ----
// Standard UT parameterisation: lambda = a^2 (n + kappa) - n for n = 3 dims.
// `UT_SPREAD` (a) keeps the sigma points close to the mean so the estimate
// stays local; `UT_BETA` is the usual Gaussian-prior correction on the
// centre's covariance weight.
const float UT_SPREAD = 0.2;
const float UT_KAPPA = 0.0;
const float UT_BETA = 2.0;
const float UT_LAMBDA = UT_SPREAD * UT_SPREAD * (3.0 + UT_KAPPA) - 3.0;
/** Distance of each sigma point from the centre, in std devs. */
const float UT_MULTIPLIER = sqrt(3.0 + UT_LAMBDA);
const float UT_CENTER_WEIGHT = UT_LAMBDA / (3.0 + UT_LAMBDA);
const float UT_CENTER_WEIGHT_COV =
    UT_CENTER_WEIGHT + 1.0 - UT_SPREAD * UT_SPREAD + UT_BETA;
const float UT_SIGMA_WEIGHT = 1.0 / (2.0 * (3.0 + UT_LAMBDA));

/**
 * The 6 sigma points of a 3D Gaussian: the centre offset along each principal
 * axis, in both directions, by `multiplier` std devs.
 */
void computeSplatSigmaPoints(
    vec3 center,
    vec3 scales,
    vec4 quaternion,
    float multiplier,
    out vec3 sigmaPts[6]
) {
    vec3 axes[3] = vec3[3](
        quatVec(quaternion, vec3(1.0, 0.0, 0.0)),
        quatVec(quaternion, vec3(0.0, 1.0, 0.0)),
        quatVec(quaternion, vec3(0.0, 0.0, 1.0))
    );
    vec3 offsets = scales * multiplier;
    sigmaPts[0] = center + axes[0] * offsets.x;
    sigmaPts[1] = center - axes[0] * offsets.x;
    sigmaPts[2] = center + axes[1] * offsets.y;
    sigmaPts[3] = center - axes[1] * offsets.y;
    sigmaPts[4] = center + axes[2] * offsets.z;
    sigmaPts[5] = center - axes[2] * offsets.z;
}

void main() {
    // Default to outside the frustum so it's discarded if we return early
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    // Assume the 2D fallback until the full ellipsoid path below qualifies.
    vSplat3D = 0.0;

    ivec2 orderingCoord = ivec2((gl_InstanceID >> 2) & 4095, gl_InstanceID >> 14);
    uint splatIndex = texelFetch(ordering, orderingCoord, 0)[gl_InstanceID & 3];
    if (splatIndex == 0xffffffffu) {
        // Special value reserved for "no splat"
        return;
    }

    ivec3 texCoord = splatTexCoord(int(splatIndex));
    vec3 center, scales, xxyyzz, xyxzyz;
    vec4 quaternion, rgba;
    mat3 cov3D;
    bvec3 zeroScales = bvec3(false);

    if (enableExtSplats) {
        uvec4 ext1 = texelFetch(extSplats, texCoord, 0);
        float alpha = unpackSplatExtAlpha(ext1);
        if ((alpha == 0.0) || (alpha < minAlpha)) {
            return;
        }
        uvec4 ext2 = texelFetch(extSplats2, texCoord, 0);

        if (!enableCovSplats) {
            unpackSplatExt(ext1, ext2, center, scales, quaternion, rgba);
            zeroScales = equal(scales, vec3(0.0));
            if (all(zeroScales)) {
                return;
            }
        } else {
            unpackSplatExtCov(ext1, ext2, center, rgba, xxyyzz, xyxzyz);
            if (all(equal(xxyyzz, vec3(0.0))) && all(equal(xyxzyz, vec3(0.0)))) {
                return;
            }
        }
    } else {
        uvec4 packedData = texelFetch(extSplats, texCoord, 0);
        if (!enableCovSplats) {
            unpackSplatEncoding(packedData, center, scales, quaternion, rgba, vec4(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX));
            zeroScales = equal(scales, vec3(0.0));
            if (all(zeroScales)) {
                return;
            }
        } else {
            unpackSplatCovEncoding(packedData, center, rgba, xxyyzz, xyxzyz, vec4(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX));
            if (all(equal(xxyyzz, vec3(0.0))) && all(equal(xyxzyz, vec3(0.0)))) {
                return;
            }
        }

        rgba.a *= 2.0;
        if ((rgba.a == 0.0) || (rgba.a < minAlpha)) {
            return;
        }
    }

    // Match the reference 3DGS rasterizer (graphdeco-inria/diff-gaussian-rasterization),
    // which clamps the SH-evaluated color to positive
    rgba.rgb = max(rgba.rgb, vec3(0.0));

    adjustedStdDev = maxStdDev;
    if (rgba.a > 1.0) {
        // Stretch 1..2 to 1..5
        rgba.a = min(rgba.a * 4.0 - 3.0, 5.0);

        if (lodInflate) {
            // Adjust size to componsate for loss of opacity
            float opacity = exp((rgba.a * rgba.a - 1.0) / 2.718281828459045);
            float rescale = pow(opacity, 1.0 / 3.0);
            scales *= rescale;
            rgba.a = 1.0;
        }

        // Expand the maximum std dev to approximately cover the larger range
        adjustedStdDev = maxStdDev + 0.7 * (rgba.a - 1.0);
    }

    // Compute the view space center of the splat
    vec3 viewCenter = (!enableCovSplats ? quatVec(renderToViewQuat, center) : (renderToViewBasis * center)) + renderToViewPos;

    // Discard splats behind the camera
    if (viewCenter.z >= 0.0) {
        return;
    }

    // Compute the clip space center of the splat
    vec4 clipCenter = projectionMatrix * vec4(viewCenter, 1.0);

    // Discard splats outside near/far planes
    if (abs(clipCenter.z) >= clipCenter.w) {
        return;
    }

    // 3DGUT applies to the (R,S) splat kinds only — a covariance splat has no
    // principal axes to place sigma points on. `scales` is final here (the
    // lodInflate rescale above already applied), so the points match the
    // geometry that actually gets drawn.
    bool useUT = !enableCovSplats;
    vec4 clipSigmaPts[6];
    if (useUT) {
        vec3 sigmaPts[6];
        computeSplatSigmaPoints(center, scales, quaternion, UT_MULTIPLIER, sigmaPts);
        for (int i = 0; i < 6; ++i) {
            vec3 viewSigmaPt = quatVec(renderToViewQuat, sigmaPts[i]) + renderToViewPos;
            clipSigmaPts[i] = projectionMatrix * vec4(viewSigmaPt, 1.0);
        }
    }

    // Discard splats more than clipXY times outside the XY frustum.
    //
    // 3DGUT widens this from a centre-only test: a splat whose centre is off
    // screen can still have sigma points on screen (large, or near the
    // frustum edge under a wide FOV), and culling it on the centre alone pops
    // it out of frame. Kept as an early-accept so the common on-screen case
    // costs exactly one comparison, as before.
    float clip = clipXY * clipCenter.w;
    bool anyInside = (abs(clipCenter.x) <= clip) && (abs(clipCenter.y) <= clip);
    if (!anyInside && useUT) {
        for (int i = 0; i < 6; ++i) {
            float clipSigma = clipXY * clipSigmaPts[i].w;
            if (abs(clipSigmaPts[i].x) <= clipSigma &&
                abs(clipSigmaPts[i].y) <= clipSigma) {
                anyInside = true;
                break;
            }
        }
    }
    if (!anyInside) {
        return;
    }

    vRgba = rgba;
    vSplatUv = position.xy * adjustedStdDev;

    // Record the splat index for entropy
    vSplatIndex = splatIndex;

    vec2 scaledRenderSize = renderSize * focalAdjustment;

    if (!enableCovSplats) {
        // Compute view space quaternion of splat
        vec4 viewQuaternion = quatQuat(renderToViewQuat, quaternion);

        if (enable2DGS && any(zeroScales)) {
            vec3 offset;
            if (zeroScales.z) {
                offset = vec3(vSplatUv.xy * scales.xy, 0.0);
            } else if (zeroScales.y) {
                offset = vec3(vSplatUv.x * scales.x, 0.0, vSplatUv.y * scales.z);
            } else {
                offset = vec3(0.0, vSplatUv.xy * scales.yz);
            }

            vec3 viewPos = viewCenter + quatVec(viewQuaternion, offset);
            gl_Position = projectionMatrix * vec4(viewPos, 1.0);
            vNdc = gl_Position.xyz / gl_Position.w;

            #include <logdepthbuf_vertex>
            return;
        }

        // Compute the 3D covariance matrix of the splat
        mat3 RS = scaleQuaternionToMatrix(scales, viewQuaternion);
        cov3D = RS * transpose(RS);

        // ---- 3DGUT: data for the fragment stage's per-pixel ray test ----
        // (R*S)^-1 = S^-1 * R^T, built without a general matrix inverse.
        mat3 R = scaleQuaternionToMatrix(vec3(1.0), viewQuaternion);
        vec3 sInv = vec3(
            (scales.x > 0.0) ? 1.0 / scales.x : 0.0,
            (scales.y > 0.0) ? 1.0 / scales.y : 0.0,
            (scales.z > 0.0) ? 1.0 / scales.z : 0.0
        );
        mat3 Sinv = mat3(
            sInv.x, 0.0, 0.0,
            0.0, sInv.y, 0.0,
            0.0, 0.0, sInv.z
        );
        vInvRS = Sinv * transpose(R);
        vMu = viewCenter;
        vScaledRenderSize = scaledRenderSize;
        vInvFocal = vec2(1.0 / projectionMatrix[0][0], 1.0 / projectionMatrix[1][1]);
        // An orthographic camera has no single ray origin, so the view-space
        // ray reconstruction in the fragment stage does not apply.
        vSplat3D = isOrthographic ? 0.0 : 1.0;
    } else {
        cov3D = mat3(
            xxyyzz.x, xyxzyz.x, xyxzyz.y,
            xyxzyz.x, xxyyzz.y, xyxzyz.z,
            xyxzyz.y, xyxzyz.z, xxyyzz.z
        );
        cov3D = renderToViewBasis * cov3D * transpose(renderToViewBasis);
    }

    // Compute the Jacobian of the splat's projection at its center
    vec2 focal = 0.5 * scaledRenderSize * vec2(projectionMatrix[0][0], projectionMatrix[1][1]);

    float a, b, d;
    if (useUT) {
        // ---- 3DGUT: estimate the 2D covariance from the sigma points ----
        // Weighted mean and covariance of the projected points in NDC, then
        // scaled into pixel units (the frame the blur/AA math below works in).
        vec3 ndcCenterUT = clipCenter.xyz / clipCenter.w;
        vec2 ndcSigmaPts[6];
        vec2 ndcMean = ndcCenterUT.xy * UT_CENTER_WEIGHT;
        for (int i = 0; i < 6; ++i) {
            ndcSigmaPts[i] = clipSigmaPts[i].xy / clipSigmaPts[i].w;
            ndcMean += ndcSigmaPts[i] * UT_SIGMA_WEIGHT;
        }

        vec2 centerDev = ndcCenterUT.xy - ndcMean;
        mat2 estCov2D = UT_CENTER_WEIGHT_COV * outerProduct(centerDev, centerDev);
        for (int i = 0; i < 6; ++i) {
            vec2 dev = ndcSigmaPts[i] - ndcMean;
            estCov2D += UT_SIGMA_WEIGHT * outerProduct(dev, dev);
        }

        // NDC spans [-1,1] across the viewport, hence the half-size scale.
        vec2 ndcToPixelScale = 0.5 * scaledRenderSize;
        mat2 scaleMatrix = mat2(
            ndcToPixelScale.x, 0.0,
            0.0, ndcToPixelScale.y
        );
        mat2 cov2DPixels = scaleMatrix * estCov2D * transpose(scaleMatrix);

        a = cov2DPixels[0][0];
        d = cov2DPixels[1][1];
        b = cov2DPixels[0][1];
    } else {
        mat3 J;
        if (isOrthographic) {
            J = mat3(
                focal.x, 0.0, 0.0,
                0.0, focal.y, 0.0,
                0.0, 0.0, 0.0
            );
        } else {
            float invZ = 1.0 / viewCenter.z;
            vec2 J1 = focal * invZ;
            vec2 J2 = -(J1 * viewCenter.xy) * invZ;
            J = mat3(
                J1.x, 0.0, J2.x,
                0.0, J1.y, J2.y,
                0.0, 0.0, 0.0
            );
        }

        // Compute the 2D covariance by projecting the 3D covariance
        // and picking out the XY plane components.
        mat3 cov2D = transpose(J) * cov3D * J;
        a = cov2D[0][0];
        d = cov2D[1][1];
        b = cov2D[0][1];
    }

    // Optionally pre-blur the splat to match non-antialias optimized splats
    a += preBlurAmount;
    d += preBlurAmount;

    float fullBlurAmount = blurAmount;
    if ((focalDistance > 0.0) && (apertureAngle > 0.0)) {
        float focusRadius = maxPixelRadius;
        if (viewCenter.z < 0.0) {
            float focusBlur = abs((-viewCenter.z - focalDistance) / viewCenter.z);
            float apertureRadius = focal.x * tan(0.5 * apertureAngle);
            focusRadius = focusBlur * apertureRadius;
        }
        fullBlurAmount = clamp(sqr(focusRadius), blurAmount, sqr(maxPixelRadius));
    }

    // Do convolution with a 0.5-pixel Gaussian for anti-aliasing: sqrt(0.3) ~= 0.5
    float detOrig = a * d - b * b;
    a += fullBlurAmount;
    d += fullBlurAmount;
    float det = a * d - b * b;

    // Compute anti-aliasing intensity scaling factor
    float blurAdjust = sqrt(max(0.0, detOrig / det));
    rgba.a *= blurAdjust;
    if (rgba.a < minAlpha) {
        return;
    }
    vRgba.a = rgba.a;

    // Compute the eigenvalue and eigenvectors of the 2D covariance matrix
    float eigenAvg = 0.5 * (a + d);
    float eigenDelta = sqrt(max(0.0, eigenAvg * eigenAvg - det));
    float eigen1 = eigenAvg + eigenDelta;
    float eigen2 = eigenAvg - eigenDelta;

    vec2 eigenVec1 = (abs(b) > 0.001) ? normalize(vec2(b, eigen1 - a))
        : ((a >= d) ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    vec2 eigenVec2 = vec2(eigenVec1.y, -eigenVec1.x);

    float scale1 = min(maxPixelRadius, adjustedStdDev * sqrt(eigen1));
    float scale2 = min(maxPixelRadius, adjustedStdDev * sqrt(eigen2));
    if (scale1 < minPixelRadius && scale2 < minPixelRadius) {
        return;
    }

    // Compute the NDC coordinates for the ellipsoid's diagonal axes.
    vec2 pixelOffset = position.x * eigenVec1 * scale1 + position.y * eigenVec2 * scale2;
    vec2 ndcOffset = (2.0 / scaledRenderSize) * pixelOffset;

    // Compute NDC center of the splat
    vec3 ndcCenter = clipCenter.xyz / clipCenter.w;
    vec3 ndc = vec3(ndcCenter.xy + ndcOffset, ndcCenter.z);

    vNdc = ndc;
    gl_Position = vec4(ndc.xy * clipCenter.w, clipCenter.zw);

    #include <logdepthbuf_vertex>
}
