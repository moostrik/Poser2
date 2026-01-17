#version 460 core

// Semi-Lagrangian advection with dissipation
// Backward trace along velocity field and sample source

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uVelocity;   // Velocity field (RG32F)
uniform sampler2D uSource;     // Field to advect (any format)
uniform sampler2D uObstacle;   // Obstacle mask (R8 or R32F)

uniform float uTimestep;       // Time step
uniform float uRdx;            // 1.0 / gridScale
uniform float uDissipation;    // Energy loss multiplier
uniform vec2 uScale;           // Texture resolution ratio

void main() {
    vec2 st = vTexCoord;
    vec2 st2 = st * uScale;

    float obstacle = texture(uObstacle, st2).x;
    if (obstacle > 0.5) {
        fragColor = vec4(0.0);
        return;
    }

    // Backward trace along velocity
    vec2 velocity = texture(uVelocity, st2).xy;
    vec2 st_back = st - uTimestep * uRdx * velocity / uScale;

    // Sample and dissipate
    fragColor = uDissipation * texture(uSource, st_back);
}
