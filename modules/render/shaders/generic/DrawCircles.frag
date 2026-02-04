#version 460 core

uniform int num_circles;
uniform vec4 circles[8];  // [x, y, size, smooth]
uniform vec4 colors[8];   // [r, g, b, a]
uniform float aspect_ratio = 1.0;  // width / height

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = vec4(0.0);

    for (int i = 0; i < num_circles; i++) {
        vec4 circle = circles[i];
        vec2 pos = circle.xy;
        float size = circle.z;
        float smth = circle.w;
        vec4 circle_color = colors[i];

        // Calculate distance with aspect ratio correction
        vec2 delta = texCoord - pos;
        delta.x *= aspect_ratio;
        float dist = length(delta);

        // Create smooth circle with antialiasing
        float alpha = 1.0 - smoothstep(size - smth, size + smth, dist);
        alpha *= circle_color.a;

        // Premultiplied alpha blending
        vec3 rgb = circle_color.rgb * alpha;
        color.rgb = color.rgb + rgb * (1.0 - color.a);
        color.a = color.a + alpha * (1.0 - color.a);
    }

    fragColor = color;
}
