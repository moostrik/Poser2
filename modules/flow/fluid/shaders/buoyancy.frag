#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out vec2 fragColor;

// Input textures
uniform sampler2D uVelocity;    // Current velocity field (RG32F) - not used but passed for consistency
uniform sampler2D uTemperature; // Temperature field (R32F)
uniform sampler2D uDensity;     // Density field (RGBA32F)

// Parameters
uniform float uSigma;              // Buoyancy strength (temperature effect)
uniform float uWeight;             // Density weight (gravity effect)
uniform float uAmbientTemperature; // Ambient/reference temperature

void main() {
    vec2 st = texCoord;

    // Sample temperature
    float temperature = texture(uTemperature, st).r;

    // Compute temperature difference from ambient
    float dtemp = temperature - uAmbientTemperature;

    // Initialize buoyancy force
    vec2 buoyancy = vec2(0.0);

    if (abs(dtemp) > 0.0001) {
        // Sample density (use alpha channel)
        float density = texture(uDensity, st).a;

        // Buoyancy force: hot air rises (positive dtemp), dense falls (positive density)
        // Force = temperature_effect - density_effect
        float buoyancy_force = dtemp * uSigma - density * uWeight;

        // Apply force in vertical direction (negative Y = up in screen space)
        buoyancy = vec2(0.0, -1.0) * buoyancy_force;
    }

    fragColor = buoyancy;
}
