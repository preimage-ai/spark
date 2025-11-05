
precision highp float;
precision highp int;
precision highp usampler2DArray;

#include <splatDefines>
#include <logdepthbuf_pars_vertex>

attribute uint splatIndex;

out vec4 vRgba;
out vec2 vSplatUv;
out vec3 vNdc;
flat out uint vSplatIndex;

uniform vec2 renderSize;
uniform uint numSplats;
uniform vec4 renderToViewQuat;
uniform vec3 renderToViewPos;
uniform float maxStdDev;
uniform float minPixelRadius;
uniform float maxPixelRadius;
uniform float time;
uniform float deltaTime;
uniform bool debugFlag;
uniform float minAlpha;
uniform bool stochastic;
uniform bool enable2DGS;
uniform float blurAmount;
uniform float preBlurAmount;
uniform float focalDistance;
uniform float apertureAngle;
uniform float clipXY;
uniform float focalAdjustment;

uniform usampler2DArray packedSplats;
uniform vec4 rgbMinMaxLnScaleMinMax;

#ifdef USE_LOGDEPTHBUF
    bool isPerspectiveMatrix( mat4 m ) {
      return m[ 2 ][ 3 ] == - 1.0;
    }
#endif

// Compute the 6 sigma points of a 3D gaussian
void computeSplatSigmaPoints(vec3 center, vec3 scales, vec4 quaternion, float multiplier, out vec3 sigmaPts[6]) {
    vec3 axes[3] = vec3[3](
        quatVec(quaternion, vec3(1.0, 0.0, 0.0)),
        quatVec(quaternion, vec3(0.0, 1.0, 0.0)),
        quatVec(quaternion, vec3(0.0, 0.0, 1.0))
    );
    vec3 sqrtScales = scales * multiplier; 
    sigmaPts[0] = center + axes[0] * sqrtScales.x;
    sigmaPts[1] = center - axes[0] * sqrtScales.x;
    sigmaPts[2] = center + axes[1] * sqrtScales.y;
    sigmaPts[3] = center - axes[1] * sqrtScales.y;
    sigmaPts[4] = center + axes[2] * sqrtScales.z;
    sigmaPts[5] = center - axes[2] * sqrtScales.z;
}

void main() {
    // Default to outside the frustum so it's discarded if we return early
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

    if (uint(gl_InstanceID) >= numSplats) {
        return;
    }

    ivec3 texCoord;
    if (stochastic) {
        texCoord = ivec3(
            uint(gl_InstanceID) & SPLAT_TEX_WIDTH_MASK,
            (uint(gl_InstanceID) >> SPLAT_TEX_WIDTH_BITS) & SPLAT_TEX_HEIGHT_MASK,
            (uint(gl_InstanceID) >> SPLAT_TEX_LAYER_BITS)
        );
    } else {
        if (splatIndex == 0xffffffffu) {
            // Special value reserved for "no splat"
            return;
        }
        texCoord = ivec3(
            splatIndex & SPLAT_TEX_WIDTH_MASK,
            (splatIndex >> SPLAT_TEX_WIDTH_BITS) & SPLAT_TEX_HEIGHT_MASK,
            splatIndex >> SPLAT_TEX_LAYER_BITS
        );
    }
    uvec4 packed = texelFetch(packedSplats, texCoord, 0);

    vec3 center, scales;
    vec4 quaternion, rgba;
    unpackSplatEncoding(packed, center, scales, quaternion, rgba, rgbMinMaxLnScaleMinMax);
    vec3 sigmaPts[6];

    // Reference: "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting"
    float spreadFac = 0.2;
    float kappa = 0.0;
    float beta = 2.0;
    float lambda = spreadFac * spreadFac * (3.0 + kappa) - 3.0;
    float multiplier = sqrt(3.0 + lambda);
    computeSplatSigmaPoints(center, scales, quaternion, multiplier, sigmaPts);

    if (rgba.a < minAlpha) {
        return;
    }
    bvec3 zeroScales = equal(scales, vec3(0.0));
    if (all(zeroScales)) {
        return;
    }

    // Compute the view space center of the splat
    vec3 viewCenter = quatVec(renderToViewQuat, center) + renderToViewPos;

    // Compute the view space sigma points of the splat
    vec3 viewSigmaPts[6];
    for (int i = 0; i < 6; ++i) {
        viewSigmaPts[i] = quatVec(renderToViewQuat, sigmaPts[i]) + renderToViewPos;
    }

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

    // Discard splats more than clipXY times outside the XY frustum
    float clip = clipXY * clipCenter.w;
    if (abs(clipCenter.x) > clip || abs(clipCenter.y) > clip) {
        return;
    }

    // Record the splat index for entropy
    vSplatIndex = splatIndex;

    // Compute view space quaternion of splat
    vec4 viewQuaternion = quatQuat(renderToViewQuat, quaternion);

    if (enable2DGS && any(zeroScales)) {
        vRgba = rgba;
        vSplatUv = position.xy * maxStdDev;

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
        return;
    }

    // Compute NDC center of the splat
    vec3 ndcCenter = clipCenter.xyz / clipCenter.w;
    
    // Compute the NDC space sigma points of the splat
    vec4 ndcSigmaPts[6];
    for (int i = 0; i < 6; ++i) {
        vec4 clipSigmaPt = projectionMatrix * vec4(viewSigmaPts[i], 1.0);
        ndcSigmaPts[i] = clipSigmaPt / clipSigmaPt.w;
    }

    float centerWeight = lambda / (3.0 + lambda);
    float centerWeightCov = centerWeight + 1.0 - spreadFac * spreadFac + beta;
    float sigmaWeight = 1.0  / (2.0 * (3.0 + lambda));

    vec2 ndcMean = ndcCenter.xy * centerWeight;
    for (int i = 0; i < 6; ++i) {
        ndcMean += ndcSigmaPts[i].xy * sigmaWeight;
    }
    mat2 estCov2D = mat2(0.0, 0.0, 0.0, 0.0);
    estCov2D += centerWeightCov * outerProduct(ndcCenter.xy - ndcMean, ndcCenter.xy - ndcMean);

    for (int i = 0; i < 6; ++i) {
        estCov2D += sigmaWeight * outerProduct(ndcSigmaPts[i].xy - ndcMean, ndcSigmaPts[i].xy - ndcMean);
    }

    // Compute the 3D covariance matrix of the splat
    mat3 RS = scaleQuaternionToMatrix(scales, viewQuaternion);
    mat3 cov3D = RS * transpose(RS);

    // Compute the Jacobian of the splat's projection at its center
    vec2 scaledRenderSize = renderSize * focalAdjustment;
    vec2 focal = 0.5 * scaledRenderSize * vec2(projectionMatrix[0][0], projectionMatrix[1][1]);

    mat3 J;
    if(isOrthographic) {
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

    vec2 ndcToPixelScale = 0.5 * scaledRenderSize; // scaledRenderSize = renderSize * focalAdjustment
    mat2 scaleMatrix = mat2(ndcToPixelScale.x, 0.0, 0.0, ndcToPixelScale.y);
    mat2 estCov2DPixelSpace = scaleMatrix * estCov2D * transpose(scaleMatrix);

    // Compute the 2D covariance by projecting the 3D covariance
    // and picking out the XY plane components.
    // Keeping below because we may need it in the future
    // for skinning deformations.
    // mat3 W = transpose(mat3(viewMatrix));
    // mat3 T = W * J;
    // mat3 cov2D = transpose(T) * cov3D * T;
    // mat3 cov2D = transpose(J) * cov3D * J;
    mat2 cov2D = estCov2DPixelSpace;
    float a = cov2D[0][0];
    float d = cov2D[1][1];
    float b = cov2D[0][1];

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

    // Compute the eigenvalue and eigenvectors of the 2D covariance matrix
    float eigenAvg = 0.5 * (a + d);
    float eigenDelta = sqrt(max(0.0, eigenAvg * eigenAvg - det));
    float eigen1 = eigenAvg + eigenDelta;
    float eigen2 = eigenAvg - eigenDelta;

    vec2 eigenVec1 = normalize(vec2((abs(b) < 0.001) ? 1.0 : b, eigen1 - a));
    vec2 eigenVec2 = vec2(eigenVec1.y, -eigenVec1.x);

    float scale1 = min(maxPixelRadius, maxStdDev * sqrt(eigen1));
    float scale2 = min(maxPixelRadius, maxStdDev * sqrt(eigen2));
    if (scale1 < minPixelRadius && scale2 < minPixelRadius) {
        return;
    }

    // Compute the NDC coordinates for the ellipsoid's diagonal axes.
    vec2 pixelOffset = position.x * eigenVec1 * scale1 + position.y * eigenVec2 * scale2;
    vec2 ndcOffset = (2.0 / scaledRenderSize) * pixelOffset;
    vec3 ndc = vec3(ndcCenter.xy + ndcOffset, ndcCenter.z);

    vRgba = rgba;
    vSplatUv = position.xy * maxStdDev;
    vNdc = ndc;
    gl_Position = vec4(ndc.xy * clipCenter.w, clipCenter.zw);
    #include <logdepthbuf_vertex>
}
