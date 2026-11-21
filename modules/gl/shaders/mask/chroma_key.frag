#version 460 core

layout(binding = 0) uniform sampler2D src;
uniform vec3 key;
uniform float threshold;
uniform float softness;

in vec2 texCoord;
out vec4 fragColor;

// BT.709 chroma, Cb and Cr, -0.5..0.5
vec2 chroma(vec3 rgb) {
    float y = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    return vec2((rgb.b - y) / 1.8556, (rgb.r - y) / 1.5748);
}

void main() {
    vec4 color = texture(src, texCoord);
    float distance = length(chroma(color.rgb) - chroma(key));
    float alpha = smoothstep(threshold, threshold + softness, distance);
    fragColor = vec4(color.rgb, color.a * alpha);
}
