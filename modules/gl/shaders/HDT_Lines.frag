#version 460 core

uniform float time;
uniform float speed;
uniform float phase;
uniform float anchor;
uniform float amount;
uniform float thickness;
uniform float sharpness;

in vec2 texCoord;
out vec4 fragColor;

#define PI 3.14159265359
#define TWOPI 6.28318530718
#define HALFPI 1.57079632679


void main() {
    // Calculate wave position based on time and parameters
    float t = time * speed + phase;
    float relativePos = texCoord.y - anchor;
    // float wave_pos = relativePos * amount + t;
    // wave_pos = sin(wave_pos);
    float stretchFactor = 0.5; // Adjust this value for more/less stretching
    float stretchedPos = sign(relativePos) * pow(abs(relativePos), stretchFactor);
    
    // Calculate wave position with stretched coordinates
    float wave_pos = stretchedPos * amount + t;
    

    // Calculate distance to nearest line
    float distToLine = mod(wave_pos, 1.0);
    distToLine = min(distToLine, 1.0 - distToLine) * 2.0;
    
    // Apply thickness and sharpness to create lines

    // float distanceFromAnchor = sin(relativePos + TWOPI * (time * 0.5 + phase));
    float messiness = 0.0;
    float distanceFromAnchor = sin(relativePos * TWOPI * messiness);
    float T = thickness * (1.0 - distanceFromAnchor * 0.5); // Increase thickness away from anchor
    // Alternative: float variableThickness = thickness * (1.0 / (1.0 + distanceFromAnchor * 3.0)); // Decrease thickness away from anchor
    // T = thickness * 0.5 + (sin(t) * 0.5 + 0.5) * thickness;
    // Apply variable thickness and sharpness to create lines
    // float lineIntensity = smoothstep(variableThickness, variableThickness * sharpness, distToLine);
    
    float lineIntensity = smoothstep(T, T * (sharpness), distToLine);



    // Output the final color
    fragColor = vec4(lineIntensity, lineIntensity, lineIntensity, 1.0);
}
