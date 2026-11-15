// A 2D position from 0 to 1 with (0, 0) at the top left (gl.md), turned into OpenGL's coordinates here and nowhere
// else. GL's own: clip space -1..1 and texture coordinates 0..1, both with y up from the bottom left.

// A position in the target as GL's clip-space position
vec4 clipPosition(vec2 position) {
    return vec4(position.x * 2.0 - 1.0, 1.0 - position.y * 2.0, 0.0, 1.0);
}

// A texture coordinate as a position from the top left, and a position as a texture coordinate: the same turn
vec2 topLeft(vec2 coord) {
    return vec2(coord.x, 1.0 - coord.y);
}
