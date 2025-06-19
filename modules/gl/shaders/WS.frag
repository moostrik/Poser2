#version 460 core

uniform sampler2D tex0;         // Nx1 texture, N lines for 360 degrees, default is 4000 lines
uniform float beamAngle = 0.09;  // e.g., 3 degrees
uniform float texWidth;         // Width of the texture in pixels (N)

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Calculate how many degrees each line in the texture represents
    float degreesPerLine = 360.0 / texWidth;    // e.g., 0.09 degrees per line for 4000 lines

    // Convert the x texture coordinate to an angle in degrees (0-360)
    float xAngle = texCoord.x * 360.0;

    // Half of the beam angle, used for falloff calculation
    float halfBeam = beamAngle * 0.5;

    // Find the index in the texture corresponding to the current angle and calculate the range of lines to sample
    float lineIdx = xAngle / degreesPerLine;
    int minIdx = int(max(0.0, floor(lineIdx - beamAngle / degreesPerLine)));
    int maxIdx = int(min(texWidth - 1.0, ceil(lineIdx + beamAngle / degreesPerLine)));

    // Calculate the intensity by sampling the texture lines within the beam angle
    float intensity = 0.0;
    for (int i = minIdx; i <= maxIdx; ++i) {
        float lineAngle = float(i) * degreesPerLine;
        float delta = abs(xAngle - lineAngle);

        if (delta < halfBeam) {
            float falloff = smoothstep(halfBeam, 0.0, delta);
            float lineValue = texture(tex0, vec2(float(i) / (texWidth - 1.0), 0.5)).b;
            intensity += lineValue * falloff;
        }
    }
    float overlap = ((texWidth * beamAngle) / 360.0) * 0.5;
    float outIntensity = intensity / overlap;

    fragColor = vec4(vec3(clamp(outIntensity, 0.0, 1.0)), 1.0);
}

void main2() {
    vec2 lineTexCoord = vec2(texCoord.x, 0.0);
    vec4 texel0 = texture(tex0, lineTexCoord);
    fragColor = texel0;
}