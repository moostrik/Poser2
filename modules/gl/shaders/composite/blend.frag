#version 460 core

#include "blend_modes.glsl"

layout(binding = 0) uniform sampler2D src;      // the layer
layout(binding = 1) uniform sampler2D dst;     // the base
uniform int mode;           // BlendShaderMode
uniform float amount;       // the layer's opacity

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 layer = texture(src, texCoord);
    vec4 base = texture(dst, texCoord);
    float opacity = clamp(layer.a * amount, 0.0, 1.0);
    vec3 blended = blendMode(mode, clamp(base.rgb, 0.0, 1.0), clamp(layer.rgb, 0.0, 1.0));
    fragColor = vec4(mix(base.rgb, blended, opacity), opacity + base.a * (1.0 - opacity));
}
