#version 460 core

// Kinetic Projection Shader
// Projects fast - slow on the direction of the motion: acceleration along it (+ onset, - release)
// or turn across it (signed 2D cross). The numpy reference is kinetics.project

uniform sampler2D tex0;  // Fast trail RG
uniform sampler2D tex1;  // Slow trail RG
uniform bool turn;       // Output the turn instead of the acceleration

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;

void main() {
    vec2 fast = texture(tex0, texCoord).xy;
    vec2 slow = texture(tex1, texCoord).xy;
    vec2 d = fast - slow;

    // A stopped fast trail has no direction left; the slow trail still holds it
    float fast_length = length(fast);
    float slow_length = length(slow);
    vec2 unit = fast_length > EPSILON ? fast / fast_length : slow_length > EPSILON ? slow / slow_length : vec2(0.0);

    float value = turn ? unit.x * d.y - unit.y * d.x : dot(d, unit);
    fragColor = vec4(value, 0.0, 0.0, 1.0);
}
