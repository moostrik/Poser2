#version 460 core

// Semi-Lagrangian advection with dissipation
// Backward trace along velocity field and sample source
//
// Advection formula:  st_back = st - timestep * rdx * velocity
//   timestep = dt * speed   (advection distance in UV-space per frame)
//   rdx = vec2(1.0, 1.0/aspect)  for aspect-correct tracing on non-square grids

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D uVelocity;   // Velocity field (RG32F)
uniform sampler2D uSource;     // Field to advect (any format)
uniform sampler2D uObstacle;   // Obstacle mask (R8 or R32F)

uniform float uTimestep;       // dt * speed — advection rate
uniform vec2  uRdx;            // (1.0, 1.0/aspect) — aspect-corrected inverse grid scale
uniform float uDissipation;    // Energy loss multiplier per frame

void main() {
    vec2 st = texCoord;

    float obstacle = texture(uObstacle, st).x;
    if (obstacle > 0.5) {
        fragColor = vec4(0.0);
        return;
    }

    // Backward trace along velocity (aspect-corrected)
    vec2 velocity = texture(uVelocity, st).xy;
    vec2 st_back = st - uTimestep * uRdx * velocity;
    // No clamp — GL_CLAMP_TO_BORDER returns 0 for out-of-bounds samples,
    // providing natural boundary decay without explicit obstacle checks.

    // Sample and dissipate
    fragColor = uDissipation * texture(uSource, st_back);
}
