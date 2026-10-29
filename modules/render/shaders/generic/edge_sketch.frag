#version 460 core

uniform sampler2D tex;
uniform float threshold;
uniform float strength;
uniform bool invert;
uniform vec2 texelSize;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Sample 3x3 neighborhood and convert to luminance
    float tl = dot(texture(tex, texCoord + vec2(-texelSize.x,  texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    float t  = dot(texture(tex, texCoord + vec2( 0.0,          texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    float tr = dot(texture(tex, texCoord + vec2( texelSize.x,  texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    float l  = dot(texture(tex, texCoord + vec2(-texelSize.x,  0.0)).rgb,         vec3(0.299, 0.587, 0.114));
    float r  = dot(texture(tex, texCoord + vec2( texelSize.x,  0.0)).rgb,         vec3(0.299, 0.587, 0.114));
    float bl = dot(texture(tex, texCoord + vec2(-texelSize.x, -texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    float b  = dot(texture(tex, texCoord + vec2( 0.0,         -texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    float br = dot(texture(tex, texCoord + vec2( texelSize.x, -texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));

    // Sobel kernels
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;

    // Edge magnitude
    float edge = sqrt(gx*gx + gy*gy) * strength;

    // Threshold with smoothstep for anti-aliased edges
    float line = smoothstep(threshold - 0.05, threshold + 0.05, edge);

    // Invert: black edges on white, or white edges on black
    if (invert) {
        line = 1.0 - line;
    }

    fragColor = vec4(vec3(line), 1.0);
}
