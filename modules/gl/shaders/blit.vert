#version 460 core

// The quad placed at `dstRect` in the target and textured from `srcRect`; both x, y, width, height in 0..1, origin
// top-left. A negative width or height of `srcRect` draws it mirrored on that axis.

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform vec4 dstRect;
uniform vec4 srcRect;

out vec2 texCoord;

void main() {
    // texcoord (0, 0) is the quad's bottom-left corner; down from the top is 1 - texcoord.y
    vec2 place = vec2(dstRect.x + texcoord.x * dstRect.z, dstRect.y + (1.0 - texcoord.y) * dstRect.w);
    gl_Position = vec4(place.x * 2.0 - 1.0, 1.0 - place.y * 2.0, 0.0, 1.0);
    texCoord = vec2(srcRect.x + texcoord.x * srcRect.z, 1.0 - (srcRect.y + (1.0 - texcoord.y) * srcRect.w));
}
