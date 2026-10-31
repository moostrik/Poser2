#version 460 core

// Kinetic Hold Shader
// Peak-hold of the release, max(0, -acceleration): a stop reads as one blob, not a flicker

uniform sampler2D tex0;  // Acceleration R
uniform sampler2D tex1;  // Previous held release R
uniform float decay;     // What remains of the held release this update

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float release = max(0.0, -texture(tex0, texCoord).r);
    float held = texture(tex1, texCoord).r * decay;
    fragColor = vec4(max(release, held), 0.0, 0.0, 1.0);
}
