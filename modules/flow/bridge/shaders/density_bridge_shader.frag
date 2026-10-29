#version 460 core

// Density Bridge Shader
// Ported from ofxFlowTools ftDensityBridgeShader.h
// Combines RGB density with velocity magnitude to drive alpha channel

uniform sampler2D tex0;  // Density RGB input
uniform sampler2D tex1;  // Velocity RG input
uniform float speed;     // Speed multiplier for alpha

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 den = texture(tex0, texCoord);
    vec4 vel = texture(tex1, texCoord);

    // Calculate velocity magnitude and use as alpha
    float alpha = length(vel.xy) * speed;
    den.w = alpha;

    // Premultiply RGB by alpha for proper blending
    den.xyz *= den.w;

    fragColor = den;
}
