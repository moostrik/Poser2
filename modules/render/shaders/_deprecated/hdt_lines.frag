#version 460 core

uniform float time;
uniform float speed;
uniform float phase;
uniform float anchor;
uniform float amount;
uniform float thickness;
uniform float sharpness;
uniform float stretch = 0.5;
uniform float mess = 0.0;
uniform float param01 = 0.0;
uniform float param02 = 0.0;
uniform float param03 = 0.0;
uniform float param04 = 0.0;

in vec2 texCoord;
out vec4 fragColor;

#define PI 3.14159265359
#define TWOPI 6.28318530718
#define HALFPI 1.57079632679


void main() {
    // Calculate wave position based on time and parameters
    float t = time * speed + phase;
    float relativePos = texCoord.y - anchor;

    float distanceFromAnchor = abs(relativePos);
    float dynamicExponent = mix(0.5, stretch, pow(distanceFromAnchor, param01));

    float stretchedPos = sign(relativePos) * pow(abs(relativePos), dynamicExponent);

    // float stretchedPos = sign(relativePos) * pow(abs(relativePos), stretch);
    // float S = 0.8;
    // float stretchMix = 4.0;

    // float curved = sign(relativePos) * pow(abs(relativePos), S);
    // float linear = relativePos;
    // float stretchedPos = mix(linear, curved, stretchMix);

    // Calculate wave position with stretched coordinates
    float wave_pos = stretchedPos * amount + t;

    // Calculate distance to nearest line
    float distToLine = mod(wave_pos, 1.0);
    distToLine = min(distToLine, 1.0 - distToLine) * 2.0;

    // Apply thickness and sharpness to create lines

    // float distanceFromAnchor = sin(relativePos + TWOPI * (time * 0.5 + phase));
    float distanceFromAnchor2 = sin(relativePos * TWOPI * mess);
    float T = thickness * (1.0 - distanceFromAnchor2 * 0.5); // Increase thickness away from anchor

    float lineIntensity = smoothstep(T, T * (sharpness), distToLine);

    // Output the final color
    fragColor = vec4(lineIntensity, lineIntensity, lineIntensity, 1.0);
}
