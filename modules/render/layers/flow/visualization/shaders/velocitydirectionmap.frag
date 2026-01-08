#version 460 core

// Velocity Field Color Visualization
// Direction encoded as Hue, Magnitude as Saturation/Value

uniform sampler2D tex0;
uniform float scale;

in vec2 texCoord;
out vec4 fragColor;

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 velocity = texture(tex0, texCoord).xy * scale;

    // Calculate magnitude and angle
    float magnitude = length(velocity);
    float angle = atan(velocity.y, velocity.x);

    // Normalize angle to 0-1 range for hue
    float hue = (angle + 3.14159265) / (2.0 * 3.14159265);

    // Clamp magnitude for saturation/value
    float sat = clamp(magnitude, 0.0, 1.0);
    float val = clamp(magnitude * 0.5 + 0.5, 0.0, 1.0);

    // Convert HSV to RGB
    vec3 color = hsv2rgb(vec3(hue, sat, val));

    // Fade out when no motion
    float alpha = smoothstep(0.0, 0.05, magnitude);

    fragColor = vec4(color, alpha);
}
