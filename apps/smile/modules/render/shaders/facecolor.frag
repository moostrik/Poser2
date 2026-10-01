#version 460 core

uniform sampler2D tex0;
uniform vec4 rect;          // x, y, width, height (normalized 0..1, top-left origin, Y-down)
uniform float saturation;   // colour inside the rect: 0 black and white, 1 full colour

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(tex0, texCoord);
    // black and white as the Hsv shader makes it at saturation 0: the brightest channel
    float gray = max(color.r, max(color.g, color.b));

    vec2 p = vec2(texCoord.x, 1.0 - texCoord.y);
    vec2 inside = step(rect.xy, p) * step(p, rect.xy + rect.zw);
    float amount = saturation * inside.x * inside.y;

    fragColor = vec4(mix(vec3(gray), color.rgb, amount), color.a);
}
