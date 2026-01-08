#version 460 core

uniform int num_values;
uniform float line_thickness = 0.001;
uniform float line_smooth = 0.001;

uniform samplerBuffer values_buffer;
uniform samplerBuffer colors_buffer;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float step_width = 1.0 / float(num_values);
    int step_index = int(texCoord.x / step_width);

    float value = texelFetch(values_buffer, step_index).r;
    vec4 color = texelFetch(colors_buffer, step_index);

    // Draw a thick horizontal line at normalized_value with smooth edges inward

    float dist = abs(texCoord.y - value);
    float alpha = 1.0 - smoothstep(line_thickness, line_thickness + line_smooth, dist);
    vec4 bg_color = vec4(0.0, 0.0, 0.0, 0.0);

    if (alpha > 0.01) {
        fragColor = vec4(color.rgb, color.a * alpha);
    } else {
        fragColor = bg_color;
    }

}