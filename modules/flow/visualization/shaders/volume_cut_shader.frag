#version 460 core

// Volume Cut Shader
// A volume drawn flat: the slice at one layer, or the projection along z (front), y (top) or x
// (side), the max or the mean of the texels on the way. The volume's w runs from the front face
// (0) to the back (1). The top has x across and the back at the top; the side has the front on
// the right and y up. A velocity's components follow the picture's axes, so arrows on a cut point
// as the motion does in it: in the top x and away from the viewer (-z), in the side z and y.

uniform sampler3D tex0;  // The volume
uniform int mode;        // 0 slice, 1 front, 2 top, 3 side
uniform float position;  // The slice's place, 0 (front) to 1 (back); the nearest layer is drawn
uniform bool use_max;    // The projection's max, else its mean
uniform bool velocity;   // A velocity: its in-plane components into x and y

in vec2 texCoord;
out vec4 fragColor;

const int SLICE = 0;
const int FRONT = 1;
const int TOP = 2;
const int SIDE = 3;

vec4 in_plane(vec4 v) {
    if (!velocity) return v;
    if (mode == TOP) return vec4(v.x, -v.z, v.y, v.w);
    if (mode == SIDE) return vec4(v.z, v.y, v.x, v.w);
    return v;
}

void main() {
    ivec3 size = textureSize(tex0, 0);
    if (mode == SLICE) {
        int layer = clamp(int(position * float(size.z)), 0, size.z - 1);
        fragColor = in_plane(texture(tex0, vec3(texCoord, (float(layer) + 0.5) / float(size.z))));
        return;
    }
    int steps = mode == FRONT ? size.z : (mode == TOP ? size.y : size.x);
    vec4 total = use_max ? vec4(-3.0e38) : vec4(0.0);
    for (int i = 0; i < steps; i++) {
        float t = (float(i) + 0.5) / float(steps);
        vec3 p = mode == FRONT ? vec3(texCoord, t)
               : mode == TOP   ? vec3(texCoord.x, t, texCoord.y)
               :                 vec3(t, texCoord.y, 1.0 - texCoord.x);
        vec4 value = texture(tex0, p);
        total = use_max ? max(total, value) : total + value;
    }
    fragColor = in_plane(use_max ? total : total / float(steps));
}
