#version 460 core

// The velocity's curl at each confinement's radius, one channel each (r, g, b), in one pass, as VorticityCurl3D on
// the plane: the curl a scalar, ω = ∂v/∂x − ∂u/∂y, each derivative a difference of the velocity `uRadius[k]` cells to
// either side. A neighbor beyond the field is the wall's ghost (walls_2d.glsl), mirrored from this cell.

in vec2 texCoord;
out vec4 fragColor;

const int MAX_CURLS = 3;     // VorticityCurl.MAX_CURLS

layout(binding = 0) uniform sampler2D uVelocity; // Velocity field (RG16F), widths/s
layout(binding = 1) uniform sampler2D uObstacle; // Obstacle mask (1.0 = obstacle, 0.0 = fluid)
uniform bool uHasObstacles;  // Skip obstacle checks when false

uniform int  uCount;                     // the curls, 1 to MAX_CURLS
uniform vec2 uHalfRdxInv[MAX_CURLS];     // 0.5 / (the cell × the radius in cells), widths, on x and y
uniform vec2 uRadius[MAX_CURLS];         // each sampling radius in cells on x and y, at least one: the radius in widths over each cell's side

#include "walls_2d.glsl"

// The curl at this cell (`cell`, its velocity `vC`) from the velocity `radius` cells to either side
float curl_at(vec2 st, ivec2 cell, ivec2 size, vec2 vC, vec2 radius, vec2 half_rdx_inv) {
    vec2 texel = 1.0 / vec2(size);
    vec2 vT = texture(uVelocity, st + vec2(0.0, radius.y * texel.y)).xy;
    vec2 vB = texture(uVelocity, st - vec2(0.0, radius.y * texel.y)).xy;
    vec2 vR = texture(uVelocity, st + vec2(radius.x * texel.x, 0.0)).xy;
    vec2 vL = texture(uVelocity, st - vec2(radius.x * texel.x, 0.0)).xy;
    ivec2 reach = max(ivec2(round(radius)), ivec2(1));
    if (cell.y + reach.y >= size.y) vT = wall_velocity(vC, 1);
    if (cell.y - reach.y < 0)       vB = wall_velocity(vC, 1);
    if (cell.x + reach.x >= size.x) vR = wall_velocity(vC, 0);
    if (cell.x - reach.x < 0)       vL = wall_velocity(vC, 0);
    return half_rdx_inv.x * (vR.y - vL.y) - half_rdx_inv.y * (vT.x - vB.x);
}

void main() {
    vec2 st = texCoord;

    if (uHasObstacles && texture(uObstacle, st).r > 0.5) {
        fragColor = vec4(0.0);
        return;
    }

    ivec2 cell = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(uVelocity, 0);
    vec2 vC = texture(uVelocity, st).xy;
    vec4 curls = vec4(0.0);
    for (int k = 0; k < uCount; k++) {
        curls[k] = curl_at(st, cell, size, vC, uRadius[k], uHalfRdxInv[k]);
    }
    fragColor = curls;
}
