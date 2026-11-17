// Color: luminance and the HSV and Oklab conversions, for any shader that weighs or turns colors.

// Rec. 709 luminance, the weights of sRGB's primaries
float luminance(vec3 rgb) {
    return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

// RGB to hue, saturation, value, all 0..1 (Sam Hocevar's branchless form)
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// sRGB's transfer: its encoded values to linear light and back, negative values kept negative
vec3 srgb_to_linear(vec3 c) {
    vec3 a = abs(c);
    return sign(c) * mix(a / 12.92, pow((a + 0.055) / 1.055, vec3(2.4)), step(0.04045, a));
}

vec3 linear_to_srgb(vec3 c) {
    vec3 a = abs(c);
    return sign(c) * mix(a * 12.92, 1.055 * pow(a, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, a));
}

// sRGB (encoded) to Oklab, L about 0..1 and a, b about ±0.4 (Ottosson 2020), and back
vec3 rgb2oklab(vec3 rgb) {
    vec3 c = srgb_to_linear(rgb);
    vec3 lms = mat3(0.4122214708, 0.2119034982, 0.0883024619,
                    0.5363325363, 0.6806995451, 0.2817188376,
                    0.0514459929, 0.1073969566, 0.6299787005) * c;
    lms = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    return mat3(0.2104542553, 1.9779984951, 0.0259040371,
                0.7936177850, -2.4285922050, 0.7827717662,
                -0.0040720468, 0.4505937099, -0.8086757660) * lms;
}

vec3 oklab2rgb(vec3 lab) {
    vec3 lms = mat3(1.0, 1.0, 1.0,
                    0.3963377774, -0.1055613458, -0.0894841775,
                    0.2158037573, -0.0638541728, -1.2914855480) * lab;
    lms = lms * lms * lms;
    vec3 c = mat3(4.0767416621, -1.2684380046, -0.0041960863,
                  -3.3077115913, 2.6097574011, -0.7034186147,
                  0.2309699292, -0.3413193965, 1.7076147010) * lms;
    return linear_to_srgb(c);
}
