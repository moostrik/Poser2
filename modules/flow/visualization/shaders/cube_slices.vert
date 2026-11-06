#version 460 core

// Cube Slices: one instance per layer of a volume, each the fullscreen quad placed at its depth in
// the box and drawn from the far side to the near one; the selected layer stands out of the stack,
// its neighbors pushed away from it by a gap that fades over the next layers (an exploded stack)

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform mat4 view_projection;   // the box's space to clip space
uniform vec3 box;               // the box's size, its width 1
uniform int layers;             // the volume's layers, one instance each
uniform bool back_first;        // the camera is on the front's side, so the back layers are the far ones
uniform int selected;           // the layer pulled out of the stack
uniform float gap;              // the gap on each side of it, in layer spacings

out vec2 texCoord;
out float w;                    // the layer's place through the volume, its texture coordinate
flat out int is_selected;

void main() {
    int layer = back_first ? layers - 1 - gl_InstanceID : gl_InstanceID;
    w = (float(layer) + 0.5) / float(layers);
    float spacing = box.z / float(layers);
    float apart = float(layer - selected);
    float pushed = sign(apart) * gap * spacing * (1.0 - pow(0.5, abs(apart)));     // away from the selected one
    float z = box.z * (0.5 - w) - pushed;           // layer 0, the front, toward +z
    texCoord = texcoord;
    is_selected = layer == selected ? 1 : 0;
    gl_Position = view_projection * vec4(position.x * 0.5 * box.x, position.y * 0.5 * box.y, z, 1.0);
}
