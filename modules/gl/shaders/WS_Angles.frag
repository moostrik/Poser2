#version 460 core

uniform sampler2D tex0;             // Nx1 texture, N lines for 360 degrees, default is 4000 lines
uniform float beamAngle = 10;        // e.g., 3 degrees
uniform float texWidth = 4000;      // Width of the texture in pixels (N)

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Calculate how many degrees each line in the texture represents
    float degreesPerLine = 360.0 / texWidth;    // e.g., 0.09 degrees per line for 4000 lines

    // Convert the x texture coordinate to an angle in degrees (0-360)
    float xAngle = texCoord.x * 360.0;

    // Half of the beam angle, used for falloff calculation
    float angle = clamp(beamAngle, degreesPerLine * 2.0, 10.0);
    float halfAngle = angle * 0.5;

    // Find the index in the texture corresponding to the current angle and calculate the range of lines to sample
    float lineIdx = xAngle / degreesPerLine;
    int minIdx = int(floor(lineIdx - angle / degreesPerLine));
    int maxIdx = int(ceil(lineIdx + angle / degreesPerLine));

    // Calculate the intensity by sampling the texture lines within the beam angle
    float white = 0.0;
    float blue = 0.0;
    for (int j = minIdx; j <= maxIdx; ++j) {
        // Wrap index using modulo for circular sampling
        int i = int(mod(float(j + int(texWidth)), texWidth));
        float lineAngle = float(i) * degreesPerLine;
        float delta = abs(xAngle - lineAngle);
        delta = min(delta, 360.0 - delta); // <-- wrap angular distance

        if (delta < halfAngle) {
            float falloff = smoothstep(halfAngle, 0.0, delta);
            float whiteValue = texture(tex0, vec2(float(i) / (texWidth - 1.0), 0.5)).b;
            white += whiteValue * falloff;
            float blueValue = texture(tex0, vec2(float(i) / (texWidth - 1.0), 0.5)).g;
            blue += blueValue * falloff;

        }
    }

    float overlap = ((texWidth * angle) / 360.0) * 0.5;
    white = white / overlap;
    blue = blue / overlap;

    vec3 color = vec3(white, white, white);
    color += vec3(blue * 0.05,  0.0, blue);

    fragColor = vec4(vec3(clamp(color, 0.0, 1.0)), 1.0);
}