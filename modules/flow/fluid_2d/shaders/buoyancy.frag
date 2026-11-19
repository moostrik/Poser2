#version 460 core

// Buoyancy on the 2D velocity grid, an acceleration (widths/s²) the fluid adds × dt, as the 3D's:
// F = uGravity * (κ·density − σ·T), the density its amount (alpha), T the warmth, the difference from the room's
// (Boussinesq), 0 the room's: dense density sinks along gravity; warm water rises against it. The weight acts
// wherever density is, not gated by the warmth. The density sits on a finer grid than the force; it is sampled by uv.

in vec2 texCoord;
out vec2 fragColor;

uniform sampler2D uTemperature; // Temperature (R16F)
uniform sampler2D uDensity;     // Density (RGBA16F, premultiplied; its amount in alpha)
uniform sampler2D uObstacle;    // Obstacle mask (R8, CLAMP_TO_BORDER=1)
uniform bool uHasObstacles;     // Skip obstacle checks when false

uniform float uSigma;           // Thermal buoyancy (widths/s² per unit of warmth)
uniform float uKappa;           // Density weight (widths/s² per unit of density)
uniform vec2  uGravity;         // Gravity's direction, the grid's frame, as the 3D's: (0, -1) down the screen

void main() {
    vec2 st = texCoord;

    if (uHasObstacles && texture(uObstacle, st).r > 0.5) {
        fragColor = vec2(0.0);
        return;
    }

    float warmth = texture(uTemperature, st).r;
    float lift = (abs(warmth) > 0.0001) ? warmth * uSigma : 0.0;
    float density = texture(uDensity, st).a;       // its amount, in RGB and in Oklab alike
    float force = density * uKappa - lift;

    fragColor = (abs(force) > 0.00001) ? uGravity * force : vec2(0.0);
}
