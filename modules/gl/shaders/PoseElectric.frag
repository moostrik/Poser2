#version 460 core

uniform vec4 points[17];
uniform vec2 resolution;
uniform float time;

const int N_JOINTS = 17;
const int N_CONN = 16;
const int NUM_ARCS = 6;  // Number of arcs per connection
const float CURVE_SPEED = 3.0;  // Speed of curve oscillation
const float WAVE_SPEED = 3.0;   // Speed of traveling waves

// Connectivity pairs (COCO 17)
const ivec2 CONN[N_CONN] = ivec2[](
    ivec2(0,1), ivec2(0,2), ivec2(1,3), ivec2(2,4),
    ivec2(5,6), ivec2(5,7), ivec2(7,9), ivec2(6,8),
    ivec2(8,10), ivec2(11,12), ivec2(5,11), ivec2(6,12),
    ivec2(11,13), ivec2(13,15), ivec2(12,14), ivec2(14,16)
);

in vec2 texCoord;
out vec4 fragColor;

// ----------------- cheap hash / noise -----------------
float hash21(vec2 p){
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p){
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i + vec2(0.0,0.0));
    float b = hash21(i + vec2(1.0,0.0));
    float c = hash21(i + vec2(0.0,1.0));
    float d = hash21(i + vec2(1.0,1.0));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

float fbm(vec2 p){
    float v = 0.0;
    float a = 0.5;
    for(int i=0;i<4;i++){
        v += a * noise(p);
        p *= 2.0;
        a *= 0.5;
    }
    return v;
}

// Quadratic bezier
vec2 bezierQuad(vec2 A, vec2 B, vec2 C, float t){
    float u = 1.0 - t;
    return u*u*A + 2.0*u*t*B + t*t*C;
}

// Fast approximate distance to quadratic bezier curve
float distToBezier(vec2 p, vec2 A, vec2 B, vec2 C, out float bestT){
    // Convert to polynomial form
    vec2 a = A - 2.0*B + C;
    vec2 b = 2.0*(B - A);
    vec2 c = A - p;

    // Cubic coefficients for closest point
    float qa = 4.0*dot(a, a);
    float qb = 6.0*dot(a, b);
    float qc = 2.0*(dot(b, b) + 2.0*dot(a, c));
    float qd = 2.0*dot(b, c);

    float minDist = 1e10;
    float minT = 0.0;

    // Check endpoints
    float d0 = dot(A - p, A - p);
    float d1 = dot(C - p, C - p);
    if(d0 < minDist){ minDist = d0; minT = 0.0; }
    if(d1 < minDist){ minDist = d1; minT = 1.0; }

    // 3 starting points with 2 Newton iterations each
    for(int i = 0; i < 3; i++){
        float t = float(i) * 0.5;

        // 2 Newton-Raphson iterations
        float t2 = t*t;
        float f = qa*t*t2 + qb*t2 + qc*t + qd;
        float fp = 3.0*qa*t2 + 2.0*qb*t + qc;
        t = t - f / max(fp, 1e-6);

        t2 = t*t;
        f = qa*t*t2 + qb*t2 + qc*t + qd;
        fp = 3.0*qa*t2 + 2.0*qb*t + qc;
        t = clamp(t - f / max(fp, 1e-6), 0.0, 1.0);

        vec2 pt = bezierQuad(A, B, C, t);
        float d = dot(pt - p, pt - p);
        if(d < minDist){
            minDist = d;
            minT = t;
        }
    }

    bestT = minT;
    return sqrt(minDist);
}

// Get joint position from uniform array
vec2 getJoint(int idx){
    // points[idx].xy contains normalized coordinates (0..1)
    // points[idx].y is already flipped in Python
    vec2 pos = points[idx].xy;
    // Check for invalid points (marked as -1)
    if(pos.x < 0.0 || pos.y < 0.0) return vec2(-1.0);
    return pos;
}

void main(){
    vec2 uv = texCoord;

    // Correct for aspect ratio - work in screen-space pixels
    vec2 p = uv * resolution;

    // Render params (in pixels)
    float thickness_px = 2.5;
    float glow_px = 12.0;
    float maxDist = thickness_px + glow_px;

    vec3 col = vec3(0.0);

    // Iterate connections
    for(int i=0; i<N_CONN; i++){
        ivec2 c = CONN[i];
        vec2 A_raw = getJoint(c.x);
        vec2 C_raw = getJoint(c.y);

        // Skip invalid joints
        if(A_raw.x < 0.0 || C_raw.x < 0.0) continue;

        // Convert to pixel space
        vec2 A = A_raw * resolution;
        vec2 C = C_raw * resolution;

        // Skip degenerate pairs
        float seglen = length(C - A);
        if(seglen < 1.0) continue;

        vec2 mid = (A + C) * 0.5;

        // Early bounding box culling - skip if pixel is too far from segment
        float halfLen = seglen * 0.5 + seglen * 0.4 + maxDist; // segment + max curve offset + glow
        if(abs(p.x - mid.x) > halfLen || abs(p.y - mid.y) > halfLen) continue;

        vec2 dir = normalize(C - A);
        vec2 perp = vec2(-dir.y, dir.x);
        float baseOff = seglen * 0.25;

        // Pre-compute shared noise for this connection (slow-moving)
        float connNoise = noise((mid / resolution + float(i) * 0.13 + time * 0.3) * 2.0);

        // Draw multiple arcs per connection
        for(int arc = 0; arc < NUM_ARCS; arc++){
            float arcSeed = float(i) * 7.3 + float(arc) * 3.7;

            // Per-arc random phase offset (constant per arc, varies by connection)
            float phaseOffset = hash21(vec2(arcSeed, float(i) * 2.3)) * 6.28;
            float speedVar = 0.8 + hash21(vec2(float(arc) * 5.1, float(i) * 3.2)) * 0.8;  // 0.8 to 1.6

            // Slow drifting phase offset per arc (adds ±25% worth of phase drift over ~20 sec cycle)
            float curveDriftPhase = hash21(vec2(arcSeed * 1.3, 7.7)) * 6.28;
            float curveDrift = 0.25 * sin(time * 0.05 * CURVE_SPEED + curveDriftPhase);

            // Simplified noise - use single sample + variation per arc
            float tnoise = connNoise + float(arc) * 0.15 + hash21(vec2(arcSeed, 0.0)) * 0.2;
            float sideOsc = sin(time * speedVar * CURVE_SPEED + phaseOffset + curveDrift + tnoise * 2.0);
            float off = baseOff * (0.15 + 0.55 * tnoise) * sideOsc;

            vec2 ctrl = mid + perp * off;

            // Smooth jitter using noise instead of hash
            float jitterPhase = hash21(vec2(arcSeed * 0.7, float(i))) * 10.0;
            float jitter = 0.1 * seglen * (noise(vec2(arcSeed * 0.5 + jitterPhase, time * 0.8)) - 0.5);
            ctrl += perp * jitter;

            // Distance to curve
            float bestT;
            float d_px = distToBezier(p, A, ctrl, C, bestT);

            // Add sinusoidal wave traveling along the curve
            // Get curve tangent at bestT for perpendicular offset direction
            vec2 curvePoint = bezierQuad(A, ctrl, C, bestT);
            vec2 tangent = normalize(2.0 * (1.0 - bestT) * (ctrl - A) + 2.0 * bestT * (C - ctrl));
            vec2 wavePerp = vec2(-tangent.y, tangent.x);

            // Slow drifting phase for waves per arc
            float waveDriftPhase = hash21(vec2(arcSeed * 2.9, 3.3)) * 6.28;
            float waveDrift = 0.25 * sin(time * 0.04 * WAVE_SPEED + waveDriftPhase);

            // Multiple wave frequencies traveling along curve
            float waveSpd = (hash21(vec2(arcSeed, 1.23)) * 2.0 + 3.0) * WAVE_SPEED;  // 3-5 speed variation
            float waveFreq = 4.0 + hash21(vec2(arcSeed, 4.56)) * 3.0;   // 4-7 frequency variation
            float waveAmp = seglen * 0.02 * (0.5 + 0.5 * sin(bestT * 3.14159));  // Amplitude peaks in middle

            // Randomize wave direction per arc (some go A->B, others B->A)
            float waveDir = (hash21(vec2(arcSeed * 2.1, float(i) * 1.7)) > 0.5) ? 1.0 : -1.0;

            float wave1 = sin(bestT * waveFreq * 6.28 - time * waveSpd * waveDir + waveDrift) * waveAmp;
            float wave2 = sin(bestT * waveFreq * 1.5 * 6.28 - time * waveSpd * 1.3 * waveDir + waveDrift + 1.0) * waveAmp * 0.5;
            float totalWave = wave1 + wave2;

            // Offset the distance check by the wave
            vec2 waveOffset = wavePerp * totalWave;
            d_px = length(p - curvePoint - waveOffset);

            // Early out if too far
            if(d_px > maxDist) continue;

            float core = smoothstep(thickness_px, 0.0, d_px);
            float g = smoothstep(maxDist, 0.0, d_px) - core;

            float flick = 0.5 + 0.5 * sin(time * 20.0 + arcSeed * 3.14 + bestT * 20.0);
            float noiseAmp = 0.5 + 0.5 * noise(vec2(bestT * 10.0, time * 1.5 + arcSeed));

            vec3 baseCol = vec3(0.15, 0.85, 1.0);
            vec3 innerCol = vec3(1.0, 1.0, 1.0);

            vec3 contrib = innerCol * core * (0.8 + 0.5 * noiseAmp * flick)
                         + baseCol * g * (0.5 + 0.5 * noiseAmp);

            // Spark branches
            float branchNoise = hash21(vec2(arcSeed * 7.0 + bestT * 50.0, time * 5.0));
            float spike = pow(branchNoise, 6.0) * core * 0.6;
            contrib += vec3(1.0, 0.9, 0.7) * spike;

            col += contrib;
        }
    }

    // Tone-map and clamp
    vec3 outc = 1.0 - exp(-col * 1.6);
    fragColor = vec4(outc, 1.0);
}
