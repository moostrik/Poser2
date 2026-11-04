#version 460 core

// Depth Guide Shader
// The depth sharpened at the image's edges, a joint bilateral filter guided by the luminance: the
// weighted mean of the depth over (2·RADIUS+1)² taps `spacing` texels apart, each weighed by its
// distance (Gaussian, SPACE texels) and by how far its luminance is from the center's (Gaussian,
// RANGE). Stereo depth ramps over a few samples where a limb is before the body; the image has
// its edge there, so a pixel on the limb's side averages the limb's depth and the ramp becomes a
// step at the image's edge. Taps without depth do not count; a pixel without depth keeps none.
// The numpy reference is depth_velocity.guide_depth

uniform sampler2D tex0;  // Depth R (m), 0 where there is none
uniform sampler2D tex1;  // Luminance R (0 to 1), the guide
uniform float spacing;   // Texels between two taps

const int RADIUS = 5;           // taps to each side
const float SPACE = 8.0;        // Gaussian of the distance (texels)
const float RANGE = 0.05;       // Gaussian of the luminance difference

in vec2 texCoord;
out vec4 fragColor;

void main() {
    ivec2 size = textureSize(tex0, 0);
    ivec2 at = clamp(ivec2(floor(texCoord * vec2(size))), ivec2(0), size - 1);
    float center = texelFetch(tex0, at, 0).r;
    if (center <= 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float guide = texture(tex1, texCoord).r;
    float sum = 0.0;
    float total = 0.0;
    for (int j = -RADIUS; j <= RADIUS; j++) {
        for (int i = -RADIUS; i <= RADIUS; i++) {
            vec2 offset = vec2(i, j) * spacing;
            ivec2 tap = clamp(at + ivec2(round(offset)), ivec2(0), size - 1);
            float sample_depth = texelFetch(tex0, tap, 0).r;
            if (sample_depth <= 0.0) continue;
            float difference = texture(tex1, (vec2(tap) + 0.5) / vec2(size)).r - guide;
            float weight = exp(-dot(offset, offset) / (2.0 * SPACE * SPACE) - difference * difference / (2.0 * RANGE * RANGE));
            sum += weight * sample_depth;
            total += weight;
        }
    }
    fragColor = vec4(sum / total, 0.0, 0.0, 1.0);
}
