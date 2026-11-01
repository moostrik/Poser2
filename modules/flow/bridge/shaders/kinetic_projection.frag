#version 460 core

// Kinetic Projection Shader
// Projects fast - slow on the direction of the 3D motion: acceleration along it (+ onset, - release)
// or turn across it (signed 2D cross), or takes its z alone. The numpy reference is kinetics.project

uniform sampler2D tex0;  // Fast trail RGB
uniform sampler2D tex1;  // Slow trail RGB
uniform int mode;        // 0 acceleration, 1 turn, 2 depth acceleration

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;

void main() {
    vec3 fast = texture(tex0, texCoord).xyz;
    vec3 slow = texture(tex1, texCoord).xyz;
    vec3 d = fast - slow;

    // A stopped fast trail has no direction left; the slow trail still holds it
    float fast_length = length(fast);
    float slow_length = length(slow);
    vec3 unit = fast_length > EPSILON ? fast / fast_length : slow_length > EPSILON ? slow / slow_length : vec3(0.0);

    float value = mode == 0 ? dot(d, unit) : mode == 1 ? unit.x * d.y - unit.y * d.x : d.z;
    fragColor = vec4(value, 0.0, 0.0, 1.0);
}
