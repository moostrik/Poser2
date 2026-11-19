#version 460 core

// Buoyancy on the 2D velocity grid, an acceleration (widths/s²), as the 3D's:
// F = uGravity * (κ·density − σ·T), the density its amount (alpha), T the warmth, the difference from the room's
// (Boussinesq), 0 the room's: dense density sinks along gravity; warm water rises against it. The weight acts
// wherever density is, not gated by the warmth; the warmth is read only while σ is above 0. The density sits on a
// finer grid than the force; it is sampled by uv. With `uAdd` the velocity plus F × uStrength (dt), in one pass;
// without, F itself, for its monitor.

in vec2 texCoord;
out vec2 fragColor;

uniform sampler2D uTemperature; // Temperature (R16F)
uniform sampler2D uDensity;     // Density (RGBA16F, premultiplied; its amount in alpha)
uniform sampler2D uObstacle;    // Obstacle mask (R8, CLAMP_TO_BORDER=1)
uniform sampler2D uVelocity;    // With uAdd: the velocity (RG16F) the force is added to
uniform bool uHasObstacles;     // Skip obstacle checks when false
uniform bool uAdd;              // The velocity plus the force × uStrength, else the force

uniform float uSigma;           // Thermal buoyancy (widths/s² per unit of warmth)
uniform float uKappa;           // Density weight (widths/s² per unit of density)
uniform vec2  uGravity;         // Gravity's direction, the grid's frame, as the 3D's: (0, -1) down the screen
uniform float uStrength;        // With uAdd: dt (s)

void main() {
    vec2 st = texCoord;
    vec2 velocity = uAdd ? texture(uVelocity, st).xy : vec2(0.0);

    if (uHasObstacles && texture(uObstacle, st).r > 0.5) {
        fragColor = velocity;
        return;
    }

    float lift = 0.0;
    if (uSigma > 0.0) {
        float warmth = texture(uTemperature, st).r;
        lift = (abs(warmth) > 0.0001) ? warmth * uSigma : 0.0;
    }
    float density = texture(uDensity, st).a;       // its amount, in RGB and in Oklab alike
    float force = density * uKappa - lift;

    vec2 buoyancy = (abs(force) > 0.00001) ? uGravity * force : vec2(0.0);
    fragColor = uAdd ? velocity + buoyancy * uStrength : buoyancy;
}
