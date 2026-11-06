#version 460 core

// ParticleSurface: each particle a sphere sprite, for screen-space fluid rendering (van der Laan
// 2009). Its radius follows its birth mass (the cube root against the reference mass) and widens
// with its age as a blob's, at most twice; faded over the ring's last share.

struct Particle { vec4 position; vec4 previous; };
layout(std430, binding = 0) readonly buffer Particles { Particle particles[]; };
layout(std430, binding = 1) readonly buffer Ring { uint head; };

uniform uint  uCount;
uniform float uScale;           // the output's pixels per emitted texel
uniform float uRingFade;
uniform float uRadius;          // a sphere's radius in pixels at the reference mass
uniform float uReferenceMass;
uniform float uSpread;          // the widening: radius² grows by 2 × spread × age (pixels²)
uniform float uFadeTime;
uniform float uMaxPointSize;

flat out vec4 vColor;           // rgb, value
flat out float vRadius;         // pixels
flat out float vDepth;          // the center's z, 0 to 1

void main() {
    uint index = uint(gl_VertexID);
    Particle particle = particles[index];
    vec2 mass_age = unpackHalf2x16(floatBitsToUint(particle.position.w));
    if (mass_age.x <= 0.0 || index >= uCount) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 1.0;
        vColor = vec4(0.0);
        return;
    }
    uint ring_age = (head + uCount - 1u - index) % uCount;
    float fade = clamp((1.0 - float(ring_age) / float(uCount)) / max(uRingFade, 1e-6), 0.0, 1.0);
    float birth = mass_age.x * pow(100.0, mass_age.y / max(uFadeTime, 1e-3));
    float r0 = uRadius * min(pow(max(birth / max(uReferenceMass, 1e-9), 0.0), 1.0 / 3.0), 2.0);   // at most twice radius
    float radius = max(min(sqrt(r0 * r0 + 2.0 * uSpread * mass_age.y), 2.0 * r0), 0.75);
    vRadius = radius;
    vDepth = particle.position.z;
    vColor = vec4(unpackUnorm4x8(floatBitsToUint(particle.previous.w)).rgb, mass_age.x * uScale * fade);
    gl_PointSize = min(2.0 * ceil(radius), uMaxPointSize);
    gl_Position = vec4(particle.position.xy * 2.0 - 1.0, 0.0, 1.0);
}
