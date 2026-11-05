#version 460 core

// Density Bridge Shader
// Ported from ofxFlowTools ftDensityBridgeShader.h
// Combines RGB density with an amount (the speed, or the kinetic bridge's emission) as alpha

uniform sampler2D tex0;  // Density RGB input
uniform sampler2D tex1;  // Amount R
uniform float speed;     // Speed multiplier for alpha

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 den = texture(tex0, texCoord);

    den.w = abs(texture(tex1, texCoord).r) * speed;

    // Premultiply RGB by alpha for proper blending
    den.xyz *= den.w;

    fragColor = den;
}
