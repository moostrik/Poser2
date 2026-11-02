#version 460 core

#define PI 3.14159265
#define TWO_PI 6.2831853

uniform sampler2D tex0;
uniform float scale;
uniform bool xyz;       // A 3D velocity: x, y, z as red, green, blue around grey

in vec2 texCoord;
out vec4 fragColor;

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    if (xyz) {
        vec3 velocity = texture(tex0, texCoord).xyz * scale;
        fragColor = vec4(clamp(0.5 + 0.5 * velocity, 0.0, 1.0), 1.0);
        return;
    }

    vec2 velocity = texture(tex0, texCoord).xy * scale;

    // Calculate magnitude and angle
    float mag =     clamp(length(velocity), 0.0, 1.0);
    float angle =   atan(velocity.y, velocity.x);

    // Normalize angle to 0-1 range for hue
    float hue =     (angle + PI) / (TWO_PI);
    float sat =     1.0;
    float val =     mag;
    float alpha =   1.0;

    // Convert HSV to RGB
    fragColor = vec4(hsv2rgb(vec3(hue, sat, val)), alpha);
}
