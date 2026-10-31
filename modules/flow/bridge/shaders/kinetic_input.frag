#version 460 core

// Kinetic Input Shader
// Copies the optical flow into the bridge, zeroing what is not finite: the trails feed back,
// so one such pixel would stay and spread through the blur until a reset

uniform sampler2D tex0;  // Optical flow RG

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 flow = texture(tex0, texCoord).xy;
    bool finite = !any(isnan(flow)) && !any(isinf(flow));
    fragColor = vec4(finite ? flow : vec2(0.0), 0.0, 1.0);
}
