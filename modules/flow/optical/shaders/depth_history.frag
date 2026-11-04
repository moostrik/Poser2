#version 460 core

// Depth History Shader
// The depth of the last frames at each pixel, in place: the depth now in red, and the history the
// frame before moved one channel on, so red to alpha hold the depth 1 to 4 frames before the next
// one. The numpy reference is depth_velocity.shift_history

uniform sampler2D tex0;  // Depth now R (m), at the history's size
uniform sampler2D tex1;  // The history the frame before (m)

in vec2 texCoord;
out vec4 fragColor;

void main() {
    ivec2 at = ivec2(floor(texCoord * vec2(textureSize(tex1, 0))));
    vec4 before = texelFetch(tex1, at, 0);
    fragColor = vec4(texelFetch(tex0, at, 0).r, before.rgb);
}
