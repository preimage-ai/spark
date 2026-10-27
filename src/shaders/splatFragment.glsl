
precision highp float;
precision highp int;

#include <splatDefines>
#include <gutDefines>

uniform float near;
uniform float far;
uniform bool encodeLinear;
uniform float time;
uniform bool debugFlag;
uniform float maxStdDev;
uniform float minAlpha;
uniform bool disableFalloff;
uniform float falloff;
uniform vec2 renderSize;
uniform highp sampler2D gutRayMap;

out vec4 fragColor;

in vec4 vRgba;
in vec2 vSplatUv;
in vec3 vNdc;
flat in uint vSplatIndex;
flat in float adjustedStdDev;

flat in vec3 vGutRow0;
flat in vec3 vGutRow1;
flat in vec3 vGutRow2;
flat in vec3 vGutOrigin;
flat in vec4 vGutColor;
flat in vec4 vGutConic;
flat in vec2 vGutCenter;
flat in vec4 vGutTiles;

#include <logdepthbuf_pars_fragment>

// 3DGUT-matched fragment: evaluate the particle along this pixel's camera ray
void gutFragment() {
    // pixel coordinates from the top-left corner (pixel centres at +0.5), as in the reference
    vec2 pix = vec2(gl_FragCoord.x, renderSize.y - gl_FragCoord.y);
    if (gutTileEmulation) {
        vec2 tile = floor(pix / gutTileSize);
        if (any(lessThan(tile, vGutTiles.xy)) || any(greaterThanEqual(tile, vGutTiles.zw))) {
            discard;
        }
        if (gutTileBasedCulling && !(gutTileMinPower(tile, vGutConic.xyz, vGutCenter) < vGutConic.w)) {
            discard;
        }
    }
    vec3 q = vec3(pix, 1.0);
    if (gutCameraModel == 1) {
        vec4 ray = texelFetch(gutRayMap, ivec2(pix), 0);
        if (ray.w == 0.0) {
            discard;
        }
        q = ray.xyz;
    }
    vec3 dc = normalize(vec3(dot(vGutRow0, q), dot(vGutRow1, q), dot(vGutRow2, q)));
    vec3 cr = cross(dc, vGutOrigin);
    float response = gutKernelResponse(gutKernelDegree, dot(cr, cr));
    float alpha = min(gutMaxAlpha, response * vGutColor.a);
    bool inFront = dot(dc, vGutOrigin) < 0.0;
    if (!(inFront && (response > gutMinResponse) && (alpha > gutMinAlpha))) {
        discard;
    }
    vec3 rgb = vGutColor.rgb;
    if (encodeLinear) {
        rgb = srgbToLinear(rgb);
    }
    #ifdef PREMULTIPLIED_ALPHA
        fragColor = vec4(rgb * alpha, alpha);
    #else
        fragColor = vec4(rgb, alpha);
    #endif
}

void main() {
    if (gutMode) {
        gutFragment();
        #include <logdepthbuf_fragment>
        return;
    }

    vec4 rgba = vRgba;

    float z2 = dot(vSplatUv, vSplatUv);
    if (z2 > (adjustedStdDev * adjustedStdDev)) {
        discard;
    }

    if (false) {
    // if (debugFlag) {
        float a = rgba.a;
        float shifted = sqrt(z2) - max(0.0, a - 1.0);
        float exponent = -0.5 * max(1.0, a) * sqr(max(0.0, shifted));
        float min1a = min(1.0, a);
        rgba.a = mix(min1a, min1a * exp(exponent), falloff);
    } else {
        // New falloff function, more or less equivalent
        if (rgba.a <= 1.0) {
            rgba.a = mix(rgba.a, rgba.a * exp(-0.5 * z2), falloff);
        } else {
            float a = exp((rgba.a*rgba.a - 1.0) / 2.718281828459045);
            float alpha = 1.0 - pow(1.0 - exp(-0.5 * z2), a);
            rgba.a = mix(1.0, alpha, falloff);
        }
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
