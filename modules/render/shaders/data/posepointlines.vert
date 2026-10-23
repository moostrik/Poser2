#version 460 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

// Where the skeleton is drawn: x, y, width, height, normalized 0..1 in the quad's own
// orientation (origin bottom-left, as GL's). The whole target by default, which draws exactly as
// the generic vertex shader does, so a layer that draws into its own FBO needs no rect.
uniform vec4 rect = vec4(0.0, 0.0, 1.0, 1.0);

out vec2 texCoord;

void main() {
    vec2 rectPos = position * 0.5 + 0.5;       // -1..1 to 0..1
    vec2 pos01 = rect.xy + rectPos * rect.zw;  // position in 0..1
    gl_Position = vec4(pos01 * 2.0 - 1.0, 0.0, 1.0);
    texCoord = texcoord;
}
