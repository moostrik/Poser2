#version 460 core

uniform sampler2D tex0;
uniform vec4 roi; // x, y, width, height in texture coordinates [0,1]
uniform float rotation; // rotation angle in radians
uniform vec2 rotationCenter; // rotation center in texture coordinates [0,1]
uniform float aspectRatio; // width / height of the texture

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Map from quad space [0,1] to ROI bounds in texture space
    vec2 roiCoord = roi.xy + texCoord * roi.zw;

    // Translate to rotation center
    vec2 offsetFromCenter = roiCoord - rotationCenter;

    // Apply aspect ratio correction for uniform rotation
    offsetFromCenter.x *= aspectRatio;

    // Rotate around center
    float cosA = cos(rotation);
    float sinA = sin(rotation);
    mat2 rotMatrix = mat2(cosA, -sinA, sinA, cosA);
    vec2 rotatedOffset = rotMatrix * offsetFromCenter;

    // Remove aspect ratio correction
    rotatedOffset.x /= aspectRatio;

    // Translate back from rotation center
    vec2 textureCoord = rotatedOffset + rotationCenter;

    // Sample texture
    fragColor = texture(tex0, textureCoord);
}