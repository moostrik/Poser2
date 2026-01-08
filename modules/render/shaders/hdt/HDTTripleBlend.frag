#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;

uniform sampler2D mask0;
uniform sampler2D mask1;
uniform sampler2D mask2;

uniform float blend0;
uniform float blend1;
uniform float blend2;

uniform vec4 color0;
uniform vec4 color1;
uniform vec4 color2;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 c0 = texture(tex0, texCoord);
    vec4 c1 = texture(tex1, texCoord);
    vec4 c2 = texture(tex2, texCoord);

    float m0 = texture(mask0, texCoord).r;
    float m1 = texture(mask1, texCoord).r;
    float m2 = texture(mask2, texCoord).r;

    vec4 f0 = color0 * m0;
    vec4 f1 = color1 * m1 * blend1 * blend0;
    vec4 f2 = color2 * m2 * blend2 * blend0;

    vec4 triple_mask = f0 + f1 + f2;

    // Compute how close triple_mask is to white
    float mask_white = min(min(triple_mask.r, triple_mask.g), triple_mask.b);
    float blend_in = smoothstep(0.33, 0.8, mask_white);

    // Compute brightness of c0
    float brightness = dot(c0.rgb, vec3(0.299, 0.587, 0.114));
    vec4 bright_color0 = vec4(color0.rgb * brightness, color0.a);

    // Crank up contrast for the blend-in color
    float contrast = 6.0; // Increase for more contrast
    vec3 contrasted = clamp((bright_color0.rgb - 0.5) * contrast + 0.5, 0.0, 1.0);
    vec4 high_contrast_color = vec4(contrasted, bright_color0.a);

    vec4 baseColor = f0 + f1 + f2;
    baseColor = mix(baseColor, high_contrast_color, blend_in);

    fragColor = baseColor;
}