#version 460 core
#extension GL_ARB_shader_viewport_layer_array : require

// ParticleSprites: the particles pulled from their buffer by gl_VertexID, no copy, as points (one a
// vertex), streaks (two a particle, the previous and the current position) or blobs (a gaussian
// sprite a particle). Each into its depth slice (gl_Layer), or all into one for the additive look.
// Its value is its mass × the output's pixels per emitted texel, so a texel's particles sum to the
// grid's density there; faded over the ring's last share before the ring overwrites it.

struct Particle { vec4 position; vec4 previous; };
layout(std430, binding = 0) readonly buffer Particles { Particle particles[]; };
layout(std430, binding = 1) readonly buffer Ring { uint head; };

uniform int   uMode;            // 0 points, 1 streaks, 2 blobs
uniform uint  uCount;           // the ring's size
uniform int   uSlices;          // the slices drawn into; 1 is the additive look
uniform float uScale;           // the output's pixels per emitted texel
uniform float uRingFade;        // the ring's last share the oldest fade over
uniform vec2  uOutputSize;      // pixels
uniform float uSize;            // a point's pixels
uniform float uStreak;          // a streak's length as a share of its step
uniform float uRadius;          // a blob's radius (2 sigma) in pixels at the reference mass
uniform float uReferenceMass;   // the mass a texel of amount 1 gives a particle
uniform float uSpread;          // a blob's widening: sigma² grows by 2 × spread × age (pixels²)
uniform float uFadeTime;        // the density's fade (s): the birth mass from the mass and the age
uniform float uMaxPointSize;    // the implementation's largest point
uniform vec3  uFog;             // the additive look's fog by depth, per particle: the water's extinction over the depth, per channel

flat out vec4 vColor;           // rgb × value, value
flat out float vSigma;          // a blob's sigma in pixels
flat out float vPointSize;
flat out float vDepth;          // the particle's z, 0 at the front, for the weighted look


const float CUT = 2.5;          // a blob is drawn out to this many sigma, 95.6% of its gaussian

void main() {
    uint index = uMode == 1 ? uint(gl_VertexID) >> 1 : uint(gl_VertexID);
    Particle particle = particles[index];
    vec2 mass_age = unpackHalf2x16(floatBitsToUint(particle.position.w));
    if (mass_age.x <= 0.0 || index >= uCount) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);     // outside the clip space
        gl_PointSize = 1.0;
        vColor = vec4(0.0);
        return;
    }

    uint ring_age = (head + uCount - 1u - index) % uCount;
    float fade = clamp((1.0 - float(ring_age) / float(uCount)) / max(uRingFade, 1e-6), 0.0, 1.0);
    vec3 color = unpackUnorm4x8(floatBitsToUint(particle.previous.w)).rgb;
    float value = mass_age.x * uScale * fade;
    vec3 at = particle.position.xyz;
    float point = 1.0;
    vSigma = 1.0;

    if (uMode == 0) {
        point = max(uSize, 1.0);
        value /= point * point;
    } else if (uMode == 1) {
        vec3 tail = mix(at, particle.previous.xyz, uStreak);
        value /= max(length((at.xy - tail.xy) * uOutputSize), 1.0);
        if ((gl_VertexID & 1) == 1) at = tail;
    } else {
        float birth = mass_age.x * pow(100.0, mass_age.y / max(uFadeTime, 1e-3));
        // the cube root of the birth mass, at most twice `radius`: a strong drop is big, never a sprite
        // over the whole frame
        float sigma0 = 0.5 * uRadius * min(pow(max(birth / max(uReferenceMass, 1e-9), 0.0), 1.0 / 3.0), 2.0);
        float sigma = min(sqrt(sigma0 * sigma0 + 2.0 * uSpread * mass_age.y), 2.0 * sigma0);
        vSigma = max(sigma, 0.5);
        point = min(ceil(2.0 * CUT * vSigma), uMaxPointSize);     // the gaussian to CUT sigma, the rest of it not drawn
    }

    if (uSlices == 1) {
        color *= exp(-uFog * particle.position.z);
    }
    vColor = vec4(color * value, value);
    vPointSize = point;
    vDepth = particle.position.z;
    gl_PointSize = point;
    gl_Layer = uSlices == 1 ? 0 : min(int(particle.position.z * float(uSlices)), uSlices - 1);
    gl_Position = vec4(at.xy * 2.0 - 1.0, 0.0, 1.0);
}
