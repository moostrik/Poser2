#version 460 core

uniform int num_joints;
uniform float value_min;
uniform float value_max;

uniform samplerBuffer combined_buffer;
uniform vec4 color = vec4(1.0);

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float joint_width = 1.0 / float(num_joints);
    int joint_index = int(texCoord.x / joint_width);

    if (joint_index >= num_joints) {
        fragColor = vec4(0.0); // Transparent if out of range
        return;
    }

    vec3 data = texelFetch(combined_buffer, joint_index).rgb;
    float angle = data.r;
    float delta = max(data.g - 0.2, 0.001);
    float score = 1.0; //data.b;

    float normalized_value = (angle - value_min) / (value_max - value_min);
    normalized_value = clamp(normalized_value, 0.0, 1.0);

    // Use delta for thickness, score for alpha
    float dist = abs(texCoord.y - normalized_value);
    float alpha = (1.0 - smoothstep(delta, delta + 0.001, dist)) * score;

    fragColor = vec4(color.rgb, color.a * alpha);
}