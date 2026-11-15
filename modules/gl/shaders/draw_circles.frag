#version 460 core

#include "position.glsl"

uniform int numCircles;
uniform vec4 circles[8];  // [x, y, size, smooth], x and y in 0..1 of the target, origin top-left
uniform vec4 colors[8];   // [r, g, b, a]
uniform float aspectRatio = 1.0;  // width / height

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = vec4(0.0);

    for (int i = 0; i < numCircles; i++) {
        vec4 circle = circles[i];
        vec2 pos = circle.xy;
        float size = circle.z;
        float smth = circle.w;
        vec4 circleColor = colors[i];

        // Calculate distance with aspect ratio correction
        vec2 delta = topLeft(texCoord) - pos;
        delta.x *= aspectRatio;
        float dist = length(delta);

        // Create smooth circle with antialiasing
        float alpha = 1.0 - smoothstep(size - smth, size + smth, dist);
        alpha *= circleColor.a;

        // Premultiplied alpha blending
        vec3 rgb = circleColor.rgb * alpha;
        color.rgb = color.rgb + rgb * (1.0 - color.a);
        color.a = color.a + alpha * (1.0 - color.a);
    }

    fragColor = color;
}
