#version 460 core

#include "position.glsl"

uniform sampler2D src;
uniform vec4 roi;               // the part of src drawn: x, y, width, height in 0..1, origin top-left
uniform float rotation;         // radians, about rotationCenter
uniform vec2 rotationCenter;    // in 0..1 of src, origin top-left
uniform float aspectRatio;      // width / height of src

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // The target's point in src, from the top left
    vec2 offset = roi.xy + topLeft(texCoord) * roi.zw - rotationCenter;
    offset.x *= aspectRatio;            // round turns in pixels

    // Turned about the center, the angle in the same top-left frame as the image's points
    float cosA = cos(rotation);
    float sinA = sin(rotation);
    vec2 turned = mat2(cosA, -sinA, sinA, cosA) * offset;
    turned.x /= aspectRatio;

    fragColor = texture(src, topLeft(turned + rotationCenter));
}
