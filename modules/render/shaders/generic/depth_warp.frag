#version 460 core

// Depth in metres as a level camera would have measured it. The pixels move as the color frame's do
// (`Uploader`), then the mirror; the value is scaled by the third coordinate the homography gives
// its source pixel, which turns a distance along the tilted axis into one along the level axis.
// The homography runs in image uv, y down; texture v runs up. Nearest only: an edge must never
// read as a distance in between.

uniform sampler2D depth;
uniform mat3 toCorrected;   // source uv to corrected uv
uniform mat3 toSource;      // corrected uv to source uv
uniform bool flipH;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 correctedUv = vec2(flipH ? 1.0 - texCoord.x : texCoord.x, 1.0 - texCoord.y);
    vec3 source = toSource * vec3(correctedUv, 1.0);
    vec2 sourceUv = source.xy / source.z;
    float scale = dot(transpose(toCorrected)[2], vec3(sourceUv, 1.0));

    bool inside = all(greaterThanEqual(sourceUv, vec2(0.0))) && all(lessThanEqual(sourceUv, vec2(1.0)));
    float metres = inside && scale > 0.0 ? texture(depth, vec2(sourceUv.x, 1.0 - sourceUv.y)).r * scale : 0.0;
    fragColor = vec4(metres, 0.0, 0.0, 1.0);
}
