#version 460 core

// Density Bridge Shader
// Ported from ofxFlowTools ftDensityBridgeShader.h
// Combines RGB density with its presence as alpha: the amount (the readings' speed, widths/s), capped at 1

uniform sampler2D tex0;  // Density RGB input
uniform sampler2D tex1;  // Amount R
uniform float speed;     // the presence per width/s: 1

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 den = texture(tex0, texCoord);

    den.w = min(abs(texture(tex1, texCoord).r) * speed, 1.0);

    // Premultiply RGB by alpha for proper blending
    den.xyz *= den.w;

    fragColor = den;
}
