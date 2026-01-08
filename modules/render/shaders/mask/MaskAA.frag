#version 460 core

uniform sampler2D tex0;      // Input mask texture
uniform vec2 texelSize;      // 1.0 / texture_dimensions
uniform float blurRadius;    // Blur radius multiplier
uniform int aaMode = 1;          // Antialiasing mode selector (0-5)

in vec2 texCoord;
out vec4 fragColor;

// ============================================================================
// MODE 0: Simple Smoothstep (Basic AA)
// ============================================================================
float simpleSmooth(float value) {
    return smoothstep(0.5, 0.5, value);
}

// ============================================================================
// MODE 1: Adaptive Smoothstep (Screen-space AA)
// ============================================================================
float adaptiveSmooth(float value) {
    float aa = fwidth(value) * 0.5;
    return smoothstep(0.5 - aa, 0.5 + aa, value);
}

// ============================================================================
// MODE 2: Multi-tap Gaussian Blur (High Quality)
// ============================================================================
float gaussianBlur(sampler2D tex, vec2 uv, vec2 direction) {
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    float result = texture(tex, uv).r * weights[0];
    for(int i = 1; i < 5; i++) {
        vec2 offset = direction * float(i) * blurRadius;
        result += texture(tex, uv + offset * texelSize).r * weights[i];
        result += texture(tex, uv - offset * texelSize).r * weights[i];
    }
    return result;
}

// ============================================================================
// MODE 3: Separable Gaussian (Two-pass quality)
// ============================================================================
float separableGaussian(sampler2D tex, vec2 uv) {
    // Horizontal pass
    float horizontal = gaussianBlur(tex, uv, vec2(1.0, 0.0));
    // Note: For true separable blur, you'd need a two-pass system
    // This is a simplified version
    return horizontal;
}

// ============================================================================
// MODE 4: Edge-Aware Smoothing (Adaptive)
// ============================================================================
float edgeAwareSmooth(sampler2D tex, vec2 uv) {
    float center = texture(tex, uv).r;

    // Detect edge strength using derivatives
    float dx = abs(dFdx(center));
    float dy = abs(dFdy(center));
    float edgeStrength = length(vec2(dx, dy));

    // Apply more blur at edges
    float radius = mix(1.0, 3.0, clamp(edgeStrength * 10.0, 0.0, 1.0));

    // Simple box blur with adaptive radius
    float result = 0.0;
    float samples = 0.0;
    for(float x = -1.0; x <= 1.0; x += 1.0) {
        for(float y = -1.0; y <= 1.0; y += 1.0) {
            vec2 offset = vec2(x, y) * radius * texelSize;
            result += texture(tex, uv + offset).r;
            samples += 1.0;
        }
    }
    return result / samples;
}

// ============================================================================
// MODE 5: Super-sampling (4x MSAA pattern)
// ============================================================================
float supersample(sampler2D tex, vec2 uv) {
    vec2 offset = texelSize * 0.25;

    float result = 0.0;
    result += texture(tex, uv + vec2(-offset.x, -offset.y)).r;
    result += texture(tex, uv + vec2( offset.x, -offset.y)).r;
    result += texture(tex, uv + vec2(-offset.x,  offset.y)).r;
    result += texture(tex, uv + vec2( offset.x,  offset.y)).r;

    return result * 0.25;
}

// ============================================================================
// MODE 6: Distance Field AA
// ============================================================================
float distanceFieldAA(float value) {
    float width = fwidth(value) * 0.5;
    return smoothstep(0.5 - width, 0.5 + width, value);
}

// ============================================================================
// MODE 7: Box Blur (Fast)
// ============================================================================
float boxBlur(sampler2D tex, vec2 uv) {
    float result = 0.0;
    float r = blurRadius;

    result += texture(tex, uv).r * 0.25;
    result += texture(tex, uv + vec2(-1.0 * texelSize.x * r, 0.0)).r * 0.15;
    result += texture(tex, uv + vec2( 1.0 * texelSize.x * r, 0.0)).r * 0.15;
    result += texture(tex, uv + vec2(0.0, -1.0 * texelSize.y * r)).r * 0.15;
    result += texture(tex, uv + vec2(0.0,  1.0 * texelSize.y * r)).r * 0.15;
    result += texture(tex, uv + vec2(-2.0 * texelSize.x * r, 0.0)).r * 0.075;
    result += texture(tex, uv + vec2( 2.0 * texelSize.x * r, 0.0)).r * 0.075;
    result += texture(tex, uv + vec2(0.0, -2.0 * texelSize.y * r)).r * 0.075;
    result += texture(tex, uv + vec2(0.0,  2.0 * texelSize.y * r)).r * 0.075;

    return result;
}

// ============================================================================
// Main
// ============================================================================
void main() {
    float texel0 = texture(tex0, texCoord).r;
    float result = texel0;

    // Select antialiasing mode
    if (aaMode == 0) {
        // Simple smoothstep
        result = simpleSmooth(texel0);
    }
    else if (aaMode == 1) {
        // Adaptive screen-space AA
        result = adaptiveSmooth(texel0);
    }
    else if (aaMode == 2) {
        // Multi-tap Gaussian blur (horizontal + vertical)
        float horizontal = gaussianBlur(tex0, texCoord, vec2(1.0, 0.0));
        result = gaussianBlur(tex0, texCoord, vec2(0.0, 1.0));
        result = (horizontal + result) * 0.5; // Combine both passes
    }
    else if (aaMode == 3) {
        // Separable Gaussian (simplified)
        result = adaptiveSmooth(texel0);
        result = separableGaussian(tex0, texCoord);
        // result = smoothstep(0.3, 0.7, result);
    }
    else if (aaMode == 4) {
        // Edge-aware smoothing
        result = edgeAwareSmooth(tex0, texCoord);
    }
    else if (aaMode == 5) {
        // Super-sampling
        result = supersample(tex0, texCoord);
    }
    else if (aaMode == 6) {
        // Distance field AA
        result = distanceFieldAA(texel0);
    }
    else if (aaMode == 7) {
        // Box blur
        result = boxBlur(tex0, texCoord);
        result = smoothstep(0.2, 0.6, result);
    }

    result = clamp(result, 0.0, 1.0);
    fragColor = vec4(result, result, result, 1.0);
}