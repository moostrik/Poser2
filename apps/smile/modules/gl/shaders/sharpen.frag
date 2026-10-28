#version 460 core

uniform sampler2D tex0;
uniform float amount;
uniform vec2 texelSize;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Sample center and 4 neighbors (cross pattern)
    vec4 center = texture(tex0, texCoord);
    vec4 top = texture(tex0, texCoord + vec2(0.0, texelSize.y));
    vec4 bottom = texture(tex0, texCoord - vec2(0.0, texelSize.y));
    vec4 left = texture(tex0, texCoord - vec2(texelSize.x, 0.0));
    vec4 right = texture(tex0, texCoord + vec2(texelSize.x, 0.0));

    // Compute blur (average of neighbors)
    vec4 blur = (top + bottom + left + right) * 0.25;

    // Unsharp mask: original + amount * (original - blur)
    vec3 sharpened = center.rgb + amount * (center.rgb - blur.rgb);

    // Clamp to valid range
    sharpened = clamp(sharpened, 0.0, 1.0);

    fragColor = vec4(sharpened, center.a);
}
