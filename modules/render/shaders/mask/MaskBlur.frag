#version 460 core

uniform sampler2D tex0;      // Input texture
uniform vec2 direction;      // Blur direction: (1,0) for horizontal, (0,1) for vertical
uniform vec2 texelSize;      // 1.0 / texture_dimensions
uniform float blurRadius;    // Blur radius multiplier

in vec2 texCoord;
out vec4 fragColor;

// 9-tap Gaussian weights (sigma ≈ 2.0)
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec2 offset = direction * texelSize * blurRadius;
    
    // Center sample
    float result = texture(tex0, texCoord).r * weights[0];
    
    // Symmetric samples
    for(int i = 1; i < 5; i++) {
        vec2 sampleOffset = offset * float(i);
        result += texture(tex0, texCoord + sampleOffset).r * weights[i];
        result += texture(tex0, texCoord - sampleOffset).r * weights[i];
    }
    
    fragColor = vec4(result, result, result, 1.0);
}
