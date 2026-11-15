#version 460 core

// The quad placed at `dst` in the target and textured from `src`; both x, y, width, height in 0..1, origin top-left.
// A negative width or height of `src` draws it mirrored on that axis.

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform vec4 dst;
uniform vec4 src;

out vec2 texCoord;

void main() {
    // texcoord (0, 0) is the quad's bottom-left corner; down from the top is 1 - texcoord.y
    vec2 place = vec2(dst.x + texcoord.x * dst.z, dst.y + (1.0 - texcoord.y) * dst.w);
    gl_Position = vec4(place.x * 2.0 - 1.0, 1.0 - place.y * 2.0, 0.0, 1.0);
    texCoord = vec2(src.x + texcoord.x * src.z, 1.0 - (src.y + (1.0 - texcoord.y) * src.w));
}
