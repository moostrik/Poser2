#version 460 core

/*
 * Dampen - Exponential drag on magnitude excess above a threshold.
 *
 * Values below the threshold pass through unchanged.
 * Values above are smoothly pulled back: excess *= dampenFactor per frame.
 * dampenFactor is precomputed in Python as pow(0.01, dt / dampen_time).
 *
 * uIncludeAlpha:
 *   false → magnitude from .rgb only (velocity, temperature — avoids
 *           phantom alpha=1 from OpenGL sampling of R/RG formats)
 *   true  → magnitude from .rgba (density — alpha is real data)
 */

uniform sampler2D src;
uniform float uThreshold;
uniform float uDampenFactor;
uniform bool uIncludeAlpha;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 val = texture(src, texCoord);

    float mag = uIncludeAlpha ? length(val) : length(val.rgb);

    if (mag > uThreshold && mag > 0.0) {
        float newMag = uThreshold + (mag - uThreshold) * uDampenFactor;
        val *= newMag / mag;
    }

    fragColor = val;
}
