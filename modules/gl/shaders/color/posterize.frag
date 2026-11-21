#version 460 core

layout(binding = 0) uniform sampler2D src;
uniform float levels;       // 2 or more

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(src, texCoord);
    float steps = levels - 1.0;
    fragColor = vec4(floor(clamp(color.rgb, 0.0, 1.0) * steps + 0.5) / steps, color.a);
}
