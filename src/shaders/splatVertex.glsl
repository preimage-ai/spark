
precision highp float;
precision highp int;
precision highp usampler2DArray;

#include <splatDefines>
#include <gutDefines>

out vec4 vRgba;
out vec2 vSplatUv;
out vec3 vNdc;
flat out uint vSplatIndex;
flat out float adjustedStdDev;

// 3DGUT-matched mode: per-splat data for the per-fragment ray evaluation
flat out vec3 vGutRow0;   // rows of S^-1 R^T K^-1 (pixel -> canonical ray direction)
flat out vec3 vGutRow1;
flat out vec3 vGutRow2;
flat out vec3 vGutOrigin; // camera centre in the particle's canonical frame
flat out vec4 vGutColor;  // rgb (SH evaluated, clamped >= 0) + opacity
flat out vec4 vGutConic;  // UT-projected conic (xyz) + max conic opacity power (w)
flat out vec2 vGutCenter; // UT-projected centre (pixels)
flat out vec4 vGutTiles;  // tile bbox [min.xy, max.xy)

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


// ---------------------------------------------------------------------------
// 3DGUT-matched splat setup (see gutDefines.glsl)

// threedgut::projectPoint for the active camera model. Returns the projected pixel position and
// counts it in numValid if it is valid and inside the image (with margin), like the reference.
vec2 gutProjectPoint(vec3 p, vec4 K, vec2 res, float margin, inout int numValid) {
    vec2 uv;
    bool valid;
    if (gutCameraModel == 1) {
        // OpenCV fisheye (equidistant + k1..k4), clamped at max_angle
        float rho = max(length(p.xy), 1.1920929e-07);
        float theta = min(atan(rho, p.z), gutFisheyeMaxAngle);
        float t2 = theta * theta;
        float poly = gutFisheyeK.x + t2 * (gutFisheyeK.y + t2 * (gutFisheyeK.z + t2 * gutFisheyeK.w));
        float delta = (theta * (poly * t2 + 1.0)) / rho;
        uv = K.xy * p.xy * delta + K.zw;
        valid = theta < gutFisheyeMaxAngle;
    } else {
        // OpenCV pinhole without distortion
        if (p.z <= 0.0) {
            return vec2(0.0);
        }
        uv = (p.xy / p.z) * K.xy + K.zw;
        valid = true;
    }
    vec2 tol = res * margin;
    if (valid && uv.x > -tol.x && uv.y > -tol.y && uv.x < res.x + tol.x && uv.y < res.y + tol.y) {
        numValid += 1;
    }
    return uv;
}

// A sigma point whose projection is a placeholder: behind a pinhole's image plane / past a fisheye's
// max field angle (3DGUT's numDegeneratePoints)
bool gutDegeneratePoint(vec3 p) {
    return gutCameraModel == 1 ? atan(length(p.xy), p.z) >= gutFisheyeMaxAngle : p.z <= 0.0;
}

// Normalized fisheye radius r(theta) = theta (1 + k1 theta^2 + ... + k4 theta^8)
float gutFisheyeRadius(float theta) {
    float t2 = theta * theta;
    return theta * (1.0 + t2 * (gutFisheyeK.x + t2 * (gutFisheyeK.y + t2 * (gutFisheyeK.z + t2 * gutFisheyeK.w))));
}

vec4 gutPixelToClip(vec2 p, vec2 res) {
    return vec4(2.0 * p.x / res.x - 1.0, 1.0 - 2.0 * p.y / res.y, 0.0, 1.0);
}

// Roots of a t^2 + 2 h t + k = 0 (numerically stable); returns the number of real roots.
int gutQuadRoots(float a, float h, float k, out float t0, out float t1) {
    t0 = 0.0;
    t1 = 0.0;
    if (abs(a) <= 1.0e-20 * (abs(h) + abs(k))) {
        if (h == 0.0) {
            return 0;
        }
        t0 = -k / (2.0 * h);
        t1 = t0;
        return 1;
    }
    float disc = h * h - a * k;
    if (disc < 0.0) {
        return 0;
    }
    float q = -(h + (h >= 0.0 ? sqrt(disc) : -sqrt(disc)));
    t0 = q / a;
    t1 = (q != 0.0) ? k / q : t0;
    return 2;
}

float gutConicF(vec2 p, vec3 A, vec2 b, float c) {
    return A.x * p.x * p.x + 2.0 * A.y * p.x * p.y + A.z * p.y * p.y + 2.0 * dot(b, p) + c;
}

// Bounding box of {p : f(p) <= 0} within rect, for the point conic
// f(p) = A.x x^2 + 2 A.y x y + A.z y^2 + 2 b.x x + 2 b.y y + c (any type: ellipse, hyperbola, parabola).
// The box is spanned by the rectangle corners inside the region, the conic's intersections with
// the rectangle edges and its axis-extreme points inside the rectangle. Returns false if empty.
bool gutConicRectBBox(vec3 A, vec2 b, float c, vec4 rect, out vec2 lo, out vec2 hi) {
    lo = vec2(1.0e30);
    hi = vec2(-1.0e30);
    vec2 corners[4] = vec2[4](rect.xy, rect.zy, rect.xw, rect.zw);
    for (int i = 0; i < 4; ++i) {
        if (gutConicF(corners[i], A, b, c) <= 0.0) {
            lo = min(lo, corners[i]);
            hi = max(hi, corners[i]);
        }
    }
    float t0;
    float t1;
    for (int i = 0; i < 2; ++i) {
        // vertical edges x = x0
        float x0 = i == 0 ? rect.x : rect.z;
        int n = gutQuadRoots(A.z, A.y * x0 + b.y, (A.x * x0 + 2.0 * b.x) * x0 + c, t0, t1);
        for (int j = 0; j < n; ++j) {
            float y = j == 0 ? t0 : t1;
            if (y >= rect.y && y <= rect.w) {
                lo = min(lo, vec2(x0, y));
                hi = max(hi, vec2(x0, y));
            }
        }
        // horizontal edges y = y0
        float y0 = i == 0 ? rect.y : rect.w;
        n = gutQuadRoots(A.x, A.y * y0 + b.x, (A.z * y0 + 2.0 * b.y) * y0 + c, t0, t1);
        for (int j = 0; j < n; ++j) {
            float x = j == 0 ? t0 : t1;
            if (x >= rect.x && x <= rect.z) {
                lo = min(lo, vec2(x, y0));
                hi = max(hi, vec2(x, y0));
            }
        }
    }
    // x-extremes (df/dy = 0) and y-extremes (df/dx = 0)
    if (A.z != 0.0) {
        int n = gutQuadRoots(A.x - A.y * A.y / A.z, b.x - A.y * b.y / A.z, c - b.y * b.y / A.z, t0, t1);
        for (int j = 0; j < n; ++j) {
            float x = j == 0 ? t0 : t1;
            vec2 pe = vec2(x, -(A.y * x + b.y) / A.z);
            if (all(greaterThanEqual(pe, rect.xy)) && all(lessThanEqual(pe, rect.zw))) {
                lo = min(lo, pe);
                hi = max(hi, pe);
            }
        }
    }
    if (A.x != 0.0) {
        int n = gutQuadRoots(A.z - A.y * A.y / A.x, b.y - A.y * b.x / A.x, c - b.x * b.x / A.x, t0, t1);
        for (int j = 0; j < n; ++j) {
            float y = j == 0 ? t0 : t1;
            vec2 pe = vec2(-(A.y * y + b.x) / A.x, y);
            if (all(greaterThanEqual(pe, rect.xy)) && all(lessThanEqual(pe, rect.zw))) {
                lo = min(lo, pe);
                hi = max(hi, pe);
            }
        }
    }
    return hi.x >= lo.x && hi.y >= lo.y;
}

void gutSplat(vec3 center, vec3 scales, vec4 quaternion, vec4 rgba) {
    float opacity = rgba.a;
    // 3DGUT culls particles whose opacity is below the alpha threshold
    if (!(opacity >= gutMinAlpha)) {
        return;
    }
    vec2 res = renderSize;
    bool fisheye = gutCameraModel == 1;
    // Camera intrinsics in pixels (top-left origin): fisheye uniforms, or the three.js projection matrix
    vec4 K = fisheye ? gutFisheyeIntr : vec4(
        0.5 * res.x * projectionMatrix[0][0],
        0.5 * res.y * projectionMatrix[1][1],
        0.5 * res.x * (1.0 - projectionMatrix[2][0]),
        0.5 * res.y * (1.0 + projectionMatrix[2][1])
    );

    // three.js view space (y up, looking down -z) -> OpenCV camera space (y down, z forward)
    vec3 vc = quatVec(renderToViewQuat, center) + renderToViewPos;
    vec3 c = vec3(vc.x, -vc.y, -vc.z);
    // 3DGUT near culling (all camera models): centre closer than ParticleMinSensorDistance
    if (length(c) < gutMinSensorDistance) {
        return;
    }
    // Early frustum cull (gutEarlyCull, pinhole): an accepted hit lies on a forward pixel ray, i.e. inside the
    // pyramid of the image rectangle, and inside the cutoff ellipsoid, i.e. within r * max(s) of the centre.
    // A particle whose bounding sphere is entirely outside one side plane of that pyramid (one-pixel margin)
    // gets no fragment, so skipping its footprint work leaves the image unchanged.
    if (gutEarlyCull && !fisheye) {
        float cullThr = max(gutMinResponse, gutMinAlpha / opacity);
        vec3 cullS = max(scales, vec3(1.0e-12));
        float radius = (gutCutoffRadius(gutKernelDegree, cullThr) * 1.001 + 1.0e-4) * max(cullS.x, max(cullS.y, cullS.z));
        // side planes through the camera centre: x / z in [xl, xr], y / z in [yt, yb]
        float xl = (-1.0 - K.z) / K.x;
        float xr = (res.x + 1.0 - K.z) / K.x;
        float yt = (-1.0 - K.w) / K.y;
        float yb = (res.y + 1.0 - K.w) / K.y;
        if (c.x - xl * c.z < -radius * sqrt(1.0 + xl * xl) ||
            xr * c.z - c.x < -radius * sqrt(1.0 + xr * xr) ||
            c.y - yt * c.z < -radius * sqrt(1.0 + yt * yt) ||
            yb * c.z - c.y < -radius * sqrt(1.0 + yb * yb)) {
            return;
        }
    }
    mat3 Rv = quaternionToMatrix(quatQuat(renderToViewQuat, quaternion));
    vec3 flipYZ = vec3(1.0, -1.0, -1.0);
    // Columns are the particle's principal axes in OpenCV camera space
    mat3 R = mat3(Rv[0] * flipYZ, Rv[1] * flipYZ, Rv[2] * flipYZ);
    vec3 s = max(scales, vec3(1.0e-12));
    vec3 invS = 1.0 / s;

    // Tile-footprint emulation (debug, off by default): 3DGUT's unscented-transform projection,
    // projected conic, boundFootprint silhouette bound and 16x16 tile culling
    // The UT projection is also needed without tile emulation when mip-splatting scaling is on:
    // 3DGUT culls a projected particle whose scaled culling opacity drops below the alpha threshold.
    vec4 tiles = vec4(0.0, 0.0, 1.0e9, 1.0e9);
    if (gutTileEmulation || gutMipSplattingScaling) {
        float utAlpha = gutUtParams.x;
        float utBeta = gutUtParams.y;
        float utKappa = gutUtParams.z;
        float margin = gutUtParams.w;
        const float D = 3.0;
        float lambda = utAlpha * utAlpha * (D + utKappa) - D;
        float delta = sqrt(utAlpha * utAlpha * (D + utKappa));
        int numValid = 0;
        int numDegenerate = gutDegeneratePoint(c) ? 1 : 0;
        vec2 pts[7];
        pts[0] = gutProjectPoint(c, K, res, margin, numValid);
        vec2 mean = pts[0] * (lambda / (D + lambda));
        float wi = 1.0 / (2.0 * (D + lambda));
        for (int i = 0; i < 3; ++i) {
            vec3 dlt = delta * s[i] * R[i];
            pts[i + 1] = gutProjectPoint(c + dlt, K, res, margin, numValid);
            mean += wi * pts[i + 1];
            pts[i + 4] = gutProjectPoint(c - dlt, K, res, margin, numValid);
            mean += wi * pts[i + 4];
            numDegenerate += (gutDegeneratePoint(c + dlt) ? 1 : 0) + (gutDegeneratePoint(c - dlt) ? 1 : 0);
        }
        // A UT with degenerate sigma points is unreliable (its covariance is float noise): 3DGUT bounds such
        // particles by their silhouette, with the raw opacity and no mip-splatting cull.
        bool utValid = numValid > 0 && numDegenerate == 0;
        vec3 conic = vec3(0.0);
        vec2 extent = vec2(0.0);
        float maxPower = 0.0;
        if (utValid) {
            vec2 d0 = pts[0] - mean;
            float w0 = lambda / (D + lambda) + (1.0 - utAlpha * utAlpha + utBeta);
            vec3 cov = w0 * vec3(d0.x * d0.x, d0.x * d0.y, d0.y * d0.y);
            for (int i = 1; i < 7; ++i) {
                vec2 di = pts[i] - mean;
                cov += wi * vec3(di.x * di.x, di.x * di.y, di.y * di.y);
            }
            // computeProjectedExtentConicOpacity (a projected particle failing it is culled)
            vec3 dil = vec3(cov.x + gutCovarianceDilation, cov.y, cov.z + gutCovarianceDilation);
            float dilDet = dil.x * dil.z - dil.y * dil.y;
            if (dilDet == 0.0) {
                return;
            }
            conic = vec3(dil.z, -dil.y, dil.x) / dilDet;
            float conicOpacity = opacity;
            if (gutMipSplattingScaling) {
                // det of the sigma-point covariance via Cauchy-Binet, sum_{i<j} w_i w_j (d_i x d_j)^2:
                // numerically stable, and exactly ~0 for a rank-1 covariance (e.g. a single sigma point in
                // front of the camera, the others projected to (0,0)), where 3DGUT's a*c - b*b is rounding noise
                float covDet = 0.0;
                for (int i = 0; i < 7; ++i) {
                    vec2 di = pts[i] - mean;
                    float wI = i == 0 ? w0 : wi;
                    for (int j = i + 1; j < 7; ++j) {
                        vec2 dj = pts[j] - mean;
                        float cr = di.x * dj.y - di.y * dj.x;
                        covDet += wI * wi * cr * cr;
                    }
                }
                conicOpacity = opacity * sqrt(max(0.000025, covDet / dilDet));
            }
            if (conicOpacity < gutMinAlpha) {
                return;
            }
            maxPower = log(conicOpacity / gutMinAlpha);
            float extentFactor = gutTightOpacityBounding ? min(3.33, sqrt(2.0 * maxPower)) : 3.33;
            float mid = 0.5 * (dil.x + dil.z);
            float lam = mid + sqrt(max(0.01, mid * mid - dilDet));
            float radius = extentFactor * sqrt(lam);
            if (!(radius > 0.0)) {
                return;
            }
            extent = gutRectBounding ? min(extentFactor * sqrt(vec2(dil.x, dil.z)), vec2(radius)) : vec2(radius);
        }

        if (gutTileEmulation) {
        // boundFootprint: bound the particle's silhouette (also when the UT projection failed)
        const float noCullingConic = 1.0e-12;
        const float bigF = 3.4e38;
        float maxAngle = fisheye ? gutFisheyeMaxAngle : 0.0;
        float cutoff = gutCutoffRadius(gutKernelDegree, gutMinAlpha / opacity);
        if (!utValid) {
            float bradius = cutoff * max(s.x, max(s.y, s.z));
            float distance = length(c);
            float centerZ = c.z;
            if (!fisheye) {
                if (centerZ < -bradius) {
                    return;
                }
            } else if (distance > bradius) {
                float centerAngle = acos(clamp(centerZ / distance, -1.0, 1.0));
                if (centerAngle - asin(bradius / distance) >= maxAngle) {
                    return;
                }
            }
            maxPower = log(opacity / gutMinAlpha);
        }
        vec3 axisScale = cutoff * s;
        vec3 cw = (-c * R) / axisScale;
        vec3 bw = (vec3(0.0, 0.0, -1.0) * R) / axisScale;
        float c2 = dot(cw, cw);
        bool fullImage = c2 <= 1.0;
        if (!fullImage && fisheye) {
            float t = -dot(cw, bw) / dot(bw, bw);
            vec3 q = cw + t * bw;
            fullImage = (t > 0.0) && (dot(q, q) < 1.0);
        }
        vec2 bboxMin = utValid ? mean - extent : vec2(bigF);
        vec2 bboxMax = utValid ? mean + extent : vec2(-bigF);
        float maxSamplePower = 0.0;
        bool crossesImagePlane = false;
        bool keepUt = false;
        if (!fullImage) {
            vec3 nrm = cw * inversesqrt(c2);
            vec3 e1 = normalize(cross(nrm, abs(nrm.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
            vec3 e2 = cross(nrm, e1);
            vec3 circleCenter = cw / c2;
            float circleRadius = sqrt(1.0 - 1.0 / c2);
            int numOutside = 0;
            int dummy = 0;
            vec3 prev = vec3(0.0);
            bool prevClamped = false;
            float prevAzimuth = 0.0;
            for (int j = 0; j <= 16; ++j) {
                float a = float(j) * (2.0 * 3.14159265 / 16.0);
                vec3 pc = axisScale * (circleCenter + circleRadius * (cos(a) * e1 + sin(a) * e2));
                vec3 x = c + R * pc;
                if (!fisheye) {
                    if (j > 0 && ((x.z > 0.0) != (prev.z > 0.0))) {
                        vec2 crossing = mix(prev.xy, x.xy, prev.z / (prev.z - x.z));
                        bboxMin.x = crossing.x < 0.0 ? -bigF : bboxMin.x;
                        bboxMax.x = crossing.x > 0.0 ? bigF : bboxMax.x;
                        bboxMin.y = crossing.y < 0.0 ? -bigF : bboxMin.y;
                        bboxMax.y = crossing.y > 0.0 ? bigF : bboxMax.y;
                        crossesImagePlane = true;
                    }
                    prev = x;
                    if (j == 16) {
                        break;
                    }
                    if (x.z <= 0.0) {
                        numOutside++;
                        continue;
                    }
                } else {
                    // past the max field angle the projection is clamped onto it; between consecutive clamped
                    // samples the footprint follows the max-angle ellipse (boundMaxAngleArc)
                    bool clamped = atan(length(x.xy), x.z) >= maxAngle;
                    float azimuth = atan(x.y, x.x);
                    if (j > 0 && clamped && prevClamped) {
                        const float twoPi = 6.28318530718;
                        float span = (azimuth - prevAzimuth) - twoPi * round((azimuth - prevAzimuth) / twoPi);
                        float arcRadius = gutFisheyeRadius(maxAngle);
                        for (int k = 0; k < 4; ++k) {
                            float d = float(k) * 1.57079632679 - prevAzimuth;
                            d -= twoPi * round(d / twoPi);
                            if (span >= 0.0 ? (d >= 0.0 && d <= span) : (d <= 0.0 && d >= span)) {
                                vec2 dir = vec2(k == 0 ? 1.0 : (k == 2 ? -1.0 : 0.0), k == 1 ? 1.0 : (k == 3 ? -1.0 : 0.0));
                                vec2 pe = K.zw + arcRadius * K.xy * dir;
                                bboxMin = min(bboxMin, pe);
                                bboxMax = max(bboxMax, pe);
                            }
                        }
                    }
                    prevClamped = clamped;
                    prevAzimuth = azimuth;
                    if (j == 16) {
                        break;
                    }
                    numOutside += clamped ? 1 : 0;
                }
                vec2 uv = gutProjectPoint(x, K, res, margin, dummy);
                bboxMin = min(bboxMin, uv);
                bboxMax = max(bboxMax, uv);
                if (utValid) {
                    vec2 dd = uv - mean;
                    maxSamplePower = max(maxSamplePower, 0.5 * (conic.x * dd.x * dd.x + conic.z * dd.y * dd.y) + conic.y * dd.x * dd.y);
                }
            }
            if (numOutside == 16 && !crossesImagePlane) {
                if (!utValid) {
                    return;
                }
                keepUt = true;
            }
        }
        if (keepUt) {
            // silhouette out of view: keep the UT footprint
        } else if (fullImage) {
            mean = 0.5 * res;
            extent = mean + 1.0;
            conic = vec3(noCullingConic, 0.0, noCullingConic);
        } else if (crossesImagePlane) {
            bboxMin = max(bboxMin, -res);
            bboxMax = min(bboxMax, 2.0 * res);
            mean = 0.5 * (bboxMin + bboxMax);
            extent = 0.5 * (bboxMax - bboxMin) + 2.0;
            conic = vec3(noCullingConic, 0.0, noCullingConic);
            if (!(bboxMax.x > 0.0 && bboxMax.y > 0.0 && bboxMin.x < res.x && bboxMin.y < res.y)) {
                return;
            }
        } else if (!(utValid && 1.1 * maxSamplePower <= maxPower)) {
            vec2 bboxExtent = 0.5 * (bboxMax - bboxMin) * 1.05 + 2.0;
            float growth = 1.1 * maxSamplePower / maxPower;
            vec2 grownExtent = extent * sqrt(growth);
            if (utValid && grownExtent.x * grownExtent.y <= bboxExtent.x * bboxExtent.y) {
                extent = grownExtent;
                conic /= growth;
            } else {
                mean = 0.5 * (bboxMin + bboxMax);
                extent = bboxExtent;
                conic = vec3(noCullingConic, 0.0, noCullingConic);
                if (!utValid && !(mean.x + extent.x > 0.0 && mean.y + extent.y > 0.0 &&
                                  mean.x - extent.x < res.x && mean.y - extent.y < res.y)) {
                    return;
                }
            }
        }

        vec2 tileGrid = ceil(res / gutTileSize);
        vec2 tmin = clamp(floor((mean - 0.5 - extent) / gutTileSize), vec2(0.0), tileGrid);
        vec2 tmax = clamp(ceil((mean - 0.5 + extent) / gutTileSize), vec2(0.0), tileGrid);
        if (!(tmax.x > tmin.x && tmax.y > tmin.y)) {
            return;
        }
        tiles = vec4(tmin, tmax);
        vGutConic = vec4(conic, maxPower);
        vGutCenter = mean;
        }
    }
    vGutTiles = tiles;

    // Canonical-frame ray quantities: o_c = S^-1 R^T (0 - c), d_c ~ S^-1 R^T d
    // pinhole: d = K^-1 [px, py, 1] (folded into the rows); fisheye: d comes from the ray map
    vec3 oc = invS * (-(c * R));
    vec3 m0 = R[0] * invS.x;
    vec3 m1 = R[1] * invS.y;
    vec3 m2 = R[2] * invS.z;
    if (fisheye) {
        vGutRow0 = m0;
        vGutRow1 = m1;
        vGutRow2 = m2;
    } else {
        vec3 kinvX = vec3(1.0 / K.x, 0.0, -K.z / K.x);
        vec3 kinvY = vec3(0.0, 1.0 / K.y, -K.w / K.y);
        vGutRow0 = m0.x * kinvX + m0.y * kinvY + vec3(0.0, 0.0, m0.z);
        vGutRow1 = m1.x * kinvX + m1.y * kinvY + vec3(0.0, 0.0, m1.z);
        vGutRow2 = m2.x * kinvX + m2.y * kinvY + vec3(0.0, 0.0, m2.z);
    }
    vGutOrigin = oc;
    vGutColor = vec4(max(rgba.rgb, vec3(0.0)), opacity);
    vSplatIndex = 0u;

    // Exact screen-space footprint: silhouette of the ellipsoid |S^-1 R^T (x - c)| <= r,
    // where r is the canonical radius at which the hit test stops accepting the particle.
    float thr = max(gutMinResponse, gutMinAlpha / opacity);
    float r = gutCutoffRadius(gutKernelDegree, thr) * 1.001 + 1.0e-4;
    if (!(r > 0.0)) {
        return;
    }
    // Pixel rectangle that may receive fragments
    vec4 rect = vec4(0.0, 0.0, res);
    if (gutTileEmulation) {
        rect = vec4(max(rect.xy, tiles.xy * gutTileSize), min(rect.zw, tiles.zw * gutTileSize));
    }

    if (!fisheye) {
        // whole cutoff ellipsoid behind the camera plane: no ray can hit it in front of the camera
        float zExtent = r * length(vec3(R[0].z * s.x, R[1].z * s.y, R[2].z * s.z));
        if (c.z + zExtent <= 0.0) {
            return;
        }
    }

    vec3 e = oc / r;
    float e2 = dot(e, e);
    bool fullScreen = !(e2 > 1.0);
    bool axisAligned = false;
    vec2 quadCenter = vec2(0.0);
    vec2 ax1 = vec2(0.0);
    vec2 ax2 = vec2(0.0);
    if (!fullScreen) {
        // Contour generator (rim) of the unit sphere seen from e: circle centred at e/|e|^2 with
        // radius sqrt(1 - 1/|e|^2) in the plane orthogonal to e. Map it to camera space.
        float invLen = inversesqrt(e2);
        vec3 n = e * invLen;
        vec3 t1 = normalize(cross(n, abs(n.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
        vec3 t2 = cross(n, t1);
        float rho = sqrt(max(0.0, 1.0 - 1.0 / e2));
        mat3 RSr = mat3(R[0] * (s.x * r), R[1] * (s.y * r), R[2] * (s.z * r));
        vec3 Cr = c + RSr * (n * invLen);
        vec3 U = RSr * (t1 * rho);
        vec3 V = RSr * (t2 * rho);
        if (fisheye) {
            // Bound the cone of directions through the rim by a circular cone, then bound the
            // image of that cone (an annular sector around the principal point) by a rectangle.
            vec3 axis = normalize(Cr);
            float cosMin = 1.0;
            for (int j = 0; j < 16; ++j) {
                float a = float(j) * (2.0 * 3.14159265 / 16.0);
                cosMin = min(cosMin, dot(normalize(Cr + U * cos(a) + V * sin(a)), axis));
            }
            float halfAngle = acos(clamp(cosMin, -1.0, 1.0)) * 1.05 + 2.0e-3;
            float thetaC = acos(clamp(axis.z, -1.0, 1.0));
            float thetaMin = thetaC - halfAngle;
            float thetaMax = min(thetaC + halfAngle, gutFisheyeMaxAngle);
            if (thetaMin >= gutFisheyeMaxAngle) {
                return;
            }
            if (halfAngle > 1.4) {
                fullScreen = true;
            } else {
                float rMax = gutFisheyeRadius(thetaMax);
                vec2 lo;
                vec2 hi;
                if (thetaMin <= 0.0) {
                    lo = vec2(-rMax);
                    hi = vec2(rMax);
                } else {
                    float rMin = gutFisheyeRadius(thetaMin);
                    float phiC = atan(axis.y, axis.x);
                    float dPhi = asin(min(1.0, sin(halfAngle) / sin(thetaC)));
                    vec2 a0 = vec2(cos(phiC - dPhi), sin(phiC - dPhi));
                    vec2 a1 = vec2(cos(phiC + dPhi), sin(phiC + dPhi));
                    lo = min(min(rMin * a0, rMin * a1), min(rMax * a0, rMax * a1));
                    hi = max(max(rMin * a0, rMin * a1), max(rMax * a0, rMax * a1));
                    // outer arc crossing the coordinate axes
                    for (int k = 0; k < 4; ++k) {
                        float phiA = float(k) * 1.5707963;
                        float dd = abs(mod(phiA - phiC + 3.14159265, 6.2831853) - 3.14159265);
                        if (dd <= dPhi) {
                            vec2 pa = rMax * vec2(cos(phiA), sin(phiA));
                            lo = min(lo, pa);
                            hi = max(hi, pa);
                        }
                    }
                }
                vec2 plo = K.zw + K.xy * lo - 1.0;
                vec2 phi = K.zw + K.xy * hi + 1.0;
                quadCenter = 0.5 * (plo + phi);
                ax1 = vec2(0.5 * (phi.x - plo.x), 0.0);
                ax2 = vec2(0.0, 0.5 * (phi.y - plo.y));
                axisAligned = true;
            }
        } else {
            float minZ = Cr.z - sqrt(U.z * U.z + V.z * V.z);
            // Image of the rim: p ~ T K [U V Cr] (cos, sin, 1), in pixels relative to pref
            vec2 pref = (Cr.z > 1.0e-3 * length(Cr)) ? (Cr.xy / Cr.z) * K.xy + K.zw : K.zw;
            vec2 kc = K.zw - pref;
            vec3 gU = vec3(K.x * U.x + kc.x * U.z, K.y * U.y + kc.y * U.z, U.z);
            vec3 gV = vec3(K.x * V.x + kc.x * V.z, K.y * V.y + kc.y * V.z, V.z);
            vec3 gC = vec3(K.x * Cr.x + kc.x * Cr.z, K.y * Cr.y + kc.y * Cr.z, Cr.z);
            mat3 Gi = inverse(mat3(gU, gV, gC));
            // point conic C = Gi^T diag(1,1,-1) Gi: pixel rays that intersect the cutoff ellipsoid
            // (in front of or behind the camera) satisfy p^T C p < 0
            vec3 g0 = vec3(Gi[0][0], Gi[1][0], Gi[2][0]);
            vec3 g1 = vec3(Gi[0][1], Gi[1][1], Gi[2][1]);
            vec3 g2 = vec3(Gi[0][2], Gi[1][2], Gi[2][2]);
            // C_ij = g0_i g0_j + g1_i g1_j - g2_i g2_j  (rows of Gi are g0, g1, g2)
            float a11 = g0.x * g0.x + g1.x * g1.x - g2.x * g2.x;
            float a12 = g0.x * g0.y + g1.x * g1.y - g2.x * g2.y;
            float a22 = g0.y * g0.y + g1.y * g1.y - g2.y * g2.y;
            vec2 b = vec2(
                g0.x * g0.z + g1.x * g1.z - g2.x * g2.z,
                g0.y * g0.z + g1.y * g1.z - g2.y * g2.z
            );
            float cc = g0.z * g0.z + g1.z * g1.z - g2.z * g2.z;
            float det = a11 * a22 - a12 * a12;
            bool oriented = false;
            float maxRes = max(res.x, res.y);
            if (minZ > 1.0e-5 && det > 0.0 && a11 > 0.0) {
                // Ellipse: oriented quad from centre and eigen-decomposition
                vec2 p0 = -vec2(a22 * b.x - a12 * b.y, a11 * b.y - a12 * b.x) / det;
                float k = cc + dot(b, p0);
                float halfDiagDiff = 0.5 * (a11 - a22);
                float eigenDelta = sqrt(halfDiagDiff * halfDiagDiff + a12 * a12);
                float mid = 0.5 * (a11 + a22);
                float l1 = (mid + eigenDelta) / -k;
                float l2 = (mid - eigenDelta) / -k;
                if (k < 0.0 && l2 > 0.0) {
                    vec2 v1u = (halfDiagDiff >= 0.0)
                        ? vec2(eigenDelta + halfDiagDiff, a12)
                        : vec2(a12, eigenDelta - halfDiagDiff);
                    float v1l = dot(v1u, v1u);
                    vec2 v1 = v1l > 0.0 ? v1u * inversesqrt(v1l) : vec2(1.0, 0.0);
                    vec2 v2 = vec2(-v1.y, v1.x);
                    // semi-axes plus a small conservative margin
                    float s1 = inversesqrt(l1) * 1.002 + 0.5;
                    float s2 = inversesqrt(l2) * 1.002 + 0.5;
                    if (s2 <= 8.0 * maxRes && all(lessThanEqual(abs(pref + p0), vec2(16.0 * maxRes)))) {
                        quadCenter = pref + p0;
                        ax1 = v1 * s1;
                        ax2 = v2 * s2;
                        oriented = true;
                    }
                }
            }
            if (!oriented) {
                // Hyperbola / parabola (ellipsoid crossing the camera plane) or a huge ellipse:
                // bounding box of the conic region inside the pixel rectangle
                vec2 lo;
                vec2 hi;
                bool hasArea = gutConicRectBBox(vec3(a11, a12, a22), b, cc, rect - vec4(pref, pref), lo, hi);
                if (any(isnan(vec4(lo, hi))) || any(isinf(vec4(lo, hi))) || any(isnan(vec3(a11, a22, cc)))) {
                    fullScreen = true;
                } else if (!hasArea) {
                    return;
                } else {
                    lo = max(lo + pref - 1.0, rect.xy);
                    hi = min(hi + pref + 1.0, rect.zw);
                    quadCenter = 0.5 * (lo + hi);
                    ax1 = vec2(0.5 * (hi.x - lo.x), 0.0);
                    ax2 = vec2(0.0, 0.5 * (hi.y - lo.y));
                    axisAligned = true;
                }
            }
        }
        float maxResAll = max(res.x, res.y);
        if (!fullScreen && !axisAligned && (max(length(ax1), length(ax2)) > 8.0 * maxResAll || any(greaterThan(abs(quadCenter), vec2(16.0 * maxResAll))))) {
            fullScreen = true;
        }
    }

    if (axisAligned) {
        // clip the axis-aligned box to the rectangle
        vec2 lo = max(quadCenter - ax1 - ax2, rect.xy);
        vec2 hi = min(quadCenter + ax1 + ax2, rect.zw);
        if (!(hi.x > lo.x && hi.y > lo.y)) {
            return;
        }
        gl_Position = gutPixelToClip(mix(lo, hi, 0.5 * (position.xy + 1.0)), res);
        return;
    }
    if (fullScreen) {
        gl_Position = gutPixelToClip(mix(rect.xy, rect.zw, 0.5 * (position.xy + 1.0)), res);
        return;
    }
    vec2 ext = sqrt(ax1 * ax1 + ax2 * ax2);
    if (quadCenter.x + ext.x < rect.x || quadCenter.y + ext.y < rect.y ||
        quadCenter.x - ext.x > rect.z || quadCenter.y - ext.y > rect.w) {
        return;
    }
    // Use the ellipse's bounding box clipped to the rectangle when that is smaller than the
    // oriented quad (large splats extending beyond the viewport or the emulated tile bounds).
    vec2 clipLo = max(quadCenter - ext, rect.xy);
    vec2 clipHi = min(quadCenter + ext, rect.zw);
    vec2 clipSize = clipHi - clipLo;
    if (clipSize.x * clipSize.y < 4.0 * length(ax1) * length(ax2)) {
        gl_Position = gutPixelToClip(mix(clipLo, clipHi, 0.5 * (position.xy + 1.0)), res);
        return;
    }
    vec2 corner = quadCenter + position.x * ax1 + position.y * ax2;
    gl_Position = gutPixelToClip(corner, res);
}

void main() {
    // Default to outside the frustum so it's discarded if we return early
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

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

    if (gutMode) {
        // PREIMAGE: decode Spark's >1 "super-opacity" before handing the splat to
        // gutSplat. Not in the upstream patch, which only ever loaded one flat PLY
        // and so never saw it; paged RAD does.
        //
        // Spark's LoD merges several splats into one and records the extra opacity
        // as alpha > 1 (1..2 encoding the 1..5 range decoded below). The non-gut
        // path a few lines down either turns that into a physically larger splat
        // (lodInflate: volume scales with opacity, hence the cube root) or widens
        // adjustedStdDev. gutSplat does neither -- it reads rgba.a straight into
        // `opacity` and uses it for the alpha cull, for gutMinAlpha / opacity in
        // the cutoff radius, and for alpha = min(maxAlpha, response * opacity).
        // Left alone, a merged splat would therefore draw at its un-inflated size
        // and saturate against maxAlpha: visibly too small and too opaque at every
        // LoD level above the leaves.
        //
        // The arithmetic is copied from the block below rather than re-derived, so
        // the two paths inflate identically. 3DGUT has no analogue of the
        // adjustedStdDev widening, so with lodInflate off the only honest option is
        // to clamp -- opacity must be <= 1 for `response * opacity` to mean
        // anything, and maxAlpha (0.99) would clamp it regardless.
        if (rgba.a > 1.0) {
            float lodAlpha = min(rgba.a * 4.0 - 3.0, 5.0);
            if (lodInflate) {
                float lodOpacity = exp((lodAlpha * lodAlpha - 1.0) / 2.718281828459045);
                scales *= pow(lodOpacity, 1.0 / 3.0);
                lodAlpha = 1.0;
            }
            rgba.a = min(lodAlpha, 1.0);
        }
        gutSplat(center, scales, quaternion, rgba);
        return;
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

    // Discard splats more than clipXY times outside the XY frustum
    float clip = clipXY * clipCenter.w;
    if (abs(clipCenter.x) > clip || abs(clipCenter.y) > clip) {
        return;
    }

    vRgba = rgba;
    vSplatUv = position.xy * adjustedStdDev;

    // Record the splat index for entropy
    vSplatIndex = splatIndex;

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
    } else {
        cov3D = mat3(
            xxyyzz.x, xyxzyz.x, xyxzyz.y,
            xyxzyz.x, xxyyzz.y, xyxzyz.z,
            xyxzyz.y, xyxzyz.z, xxyyzz.z
        );
        cov3D = renderToViewBasis * cov3D * transpose(renderToViewBasis);
    }

    // Compute the Jacobian of the splat's projection at its center
    vec2 scaledRenderSize = renderSize * focalAdjustment;
    vec2 focal = 0.5 * scaledRenderSize * vec2(projectionMatrix[0][0], projectionMatrix[1][1]);

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
    vRgba.a = rgba.a;

    // Compute the eigenvalue and eigenvectors of the 2D covariance matrix
    float eigenAvg = 0.5 * (a + d);
    float halfDiagDiff = 0.5 * (a - d);
    float eigenDelta = sqrt(sqr(halfDiagDiff) + sqr(b));
    float eigen1 = eigenAvg + eigenDelta;
    float eigen2 = max(0.0, det / eigen1);

    vec2 eigenVec1Unnorm = (halfDiagDiff >= 0.0)
        ? vec2(eigenDelta + halfDiagDiff, b)
        : vec2(b, eigenDelta - halfDiagDiff);
    float eigenVec1Len2 = dot(eigenVec1Unnorm, eigenVec1Unnorm);
    vec2 eigenVec1 = (eigenVec1Len2 > 0.0)
        ? eigenVec1Unnorm * inversesqrt(eigenVec1Len2) : vec2(1.0, 0.0);
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
