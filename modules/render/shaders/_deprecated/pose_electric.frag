#version 460 core

uniform vec4 points[17];
uniform vec2 resolution;
uniform float time;

const int N_JOINTS = 17;
const int N_CONN = 16;
const int NUM_ARCS = 15;                // Number of arcs per connection

const float CURVE_SPEED = 1.0;          // Speed of curve oscillation
const float CURVE_OFFSET = 80.0;        // Control point offset in pixels

const float WAVE_SPEED = 6.0;           // Speed of traveling waves
const float WAVE_DENSITY = 0.008;       // Waves per pixel of segment length
const float WAVE_AMPLITUDE = 15.0;      // Wave offset amplitude in pixels
const float CURVE_THICKNESS = 10;       // Extra thickness at curved parts (pixels)

const float BOLT_SPEED = 5000.0;        // Bolt travel speed in pixels per second
const float BOLT_INTERVAL = 1.5;        // Base interval between bolts in seconds
const float BOLT_HOLD = 0.2;            // Time bolt stays fully visible before fading (seconds)
const float BOLT_FADE = 0.3;            // Time for bolt to fade out (seconds)

const float MIN_BRIGHTNESS = 0.81;      // Minimum arc brightness (0.0 = invisible, 1.0 = full)

// Connectivity pairs (COCO 17)
const ivec2 CONN[N_CONN] = ivec2[](
    ivec2(0,1), ivec2(0,2), ivec2(1,3), ivec2(2,4),
    ivec2(5,6), ivec2(6,5), ivec2(5,7), ivec2(7,9), ivec2(6,8),
    ivec2(8,10),  ivec2(5,11), ivec2(6,12),
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
        float halfLen = seglen * 0.5 + CURVE_OFFSET + maxDist; // segment + max curve offset + glow
        if(abs(p.x - mid.x) > halfLen || abs(p.y - mid.y) > halfLen) continue;

        vec2 dir = normalize(C - A);
        vec2 perp = vec2(-dir.y, dir.x);
        float baseOff = CURVE_OFFSET;

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
            float baseWaveCount = seglen * WAVE_DENSITY;

            // Wave direction: always A->B
            float waveDir = 1.0;

            // Time-varying amplitude with noise modulation
            float noiseModT = noise(vec2(bestT * 5.0 + arcSeed, time * 0.3));
            float noiseModTime = noise(vec2(arcSeed * 3.0, time * 0.7));
            float taper = sin(bestT * 3.14159);  // Taper at endpoints
            float waveAmp = WAVE_AMPLITUDE * taper * (0.4 + 0.6 * noiseModT) * (0.7 + 0.3 * noiseModTime);

            // Five independent sinusoids with varying frequencies and speeds
            float freq1 = baseWaveCount * (0.5 + hash21(vec2(arcSeed, 4.56)) * 0.5);
            float freq2 = baseWaveCount * (1.0 + hash21(vec2(arcSeed, 7.89)) * 1.0);
            float freq3 = baseWaveCount * (1.5 + hash21(vec2(arcSeed, 3.14)) * 1.0);
            float freq4 = baseWaveCount * (2.5 + hash21(vec2(arcSeed, 1.41)) * 1.5);
            float freq5 = baseWaveCount * (0.3 + hash21(vec2(arcSeed, 2.71)) * 0.4);  // Slow bend

            float speed1 = (1.5 + hash21(vec2(arcSeed, 1.23)) * 2.0) * WAVE_SPEED;
            float speed2 = (3.0 + hash21(vec2(arcSeed, 2.34)) * 3.0) * WAVE_SPEED;
            float speed3 = (2.0 + hash21(vec2(arcSeed, 9.87)) * 2.5) * WAVE_SPEED;
            float speed4 = (4.0 + hash21(vec2(arcSeed, 5.55)) * 4.0) * WAVE_SPEED;
            float speed5 = (0.3 + hash21(vec2(arcSeed, 6.66)) * 0.4) * WAVE_SPEED;  // Slow

            float phase1 = hash21(vec2(arcSeed, 5.67)) * 6.28;
            float phase2 = hash21(vec2(arcSeed, 8.90)) * 6.28;
            float phase3 = hash21(vec2(arcSeed, 6.54)) * 6.28;
            float phase4 = hash21(vec2(arcSeed, 7.77)) * 6.28;
            float phase5 = hash21(vec2(arcSeed, 4.44)) * 6.28;

            // Noise-modulated frequency variation over time
            float freqMod = 1.0 + 0.3 * noise(vec2(arcSeed * 2.0, time * 0.2));

            // Wave arguments (reused for derivative calculation)
            float arg1 = bestT * freq1 * freqMod * 6.28 - time * speed1 * waveDir + phase1 + waveDrift;
            float arg2 = bestT * freq2 * freqMod * 6.28 - time * speed2 * waveDir + phase2 + waveDrift;
            float arg3 = bestT * freq3 * 6.28 - time * speed3 * waveDir + phase3;
            float arg4 = bestT * freq4 * 6.28 - time * speed4 * waveDir + phase4;
            float arg5 = bestT * freq5 * 6.28 - time * speed5 * waveDir + phase5;

            float wave1 = sin(arg1) * 0.35;
            float wave2 = sin(arg2) * 0.25;
            float wave3 = sin(arg3) * 0.15;
            float wave4 = sin(arg4) * 0.1;
            float wave5 = sin(arg5) * 0.15;  // Slow bend

            // Second derivative (curvature) - negative sin gives bending intensity
            // Only count where wave is at peak/valley (where it actually bends)
            float bend1 = abs(sin(arg1)) * 0.35;
            float bend2 = abs(sin(arg2)) * 0.25;
            float bend3 = abs(sin(arg3)) * 0.15;
            float bend4 = abs(sin(arg4)) * 0.1;
            float bend5 = abs(sin(arg5)) * 0.15;
            // Curvature is high only near peaks (sin close to ±1)
            float curvature = max(0.0, (bend1 + bend2 + bend3 + bend4 + bend5) - 0.5);

            // Add FBM noise for organic variation
            float noiseWave = (fbm(vec2(bestT * 3.0 + arcSeed, time * 0.5 * waveDir)) - 0.5) * 0.2;

            float totalWave = (wave1 + wave2 + wave3 + wave4 + wave5 + noiseWave) * waveAmp;

            // Thickness increases with curvature (only at sharp bends)
            float curveThickness = thickness_px + CURVE_THICKNESS * clamp(curvature, 0.0, 1.0);

            // Offset the distance check by the wave
            vec2 waveOffset = wavePerp * totalWave;
            d_px = length(p - curvePoint - waveOffset);

            // Early out if too far
            if(d_px > maxDist) continue;

            // Lightning bolt system (branch-free)
            // Normalize position along curve (0 to 1)
            float t = bestT;
            float tDir = t;  // Always A->B direction

            // Random interval per arc (70-130% of base)
            float boltInterval = BOLT_INTERVAL * (0.7 + hash21(vec2(arcSeed, 0.99)) * 0.6);
            float boltPhase = hash21(vec2(arcSeed, 0.88)) * boltInterval;

            // Time within current cycle
            float cycleT = mod(time + boltPhase, boltInterval);

            // Bolt front position (0 to 1+, normalized to segment)
            float travelTime = seglen / BOLT_SPEED;
            float boltFront = cycleT / travelTime;  // Can exceed 1.0

            // Time since arc was fully filled
            float timeSinceFull = max(0.0, cycleT - travelTime);

            // Phase detection
            float isTraveling = step(cycleT, travelTime);  // 1 during travel, 0 after
            float isHolding = step(travelTime, cycleT) * step(timeSinceFull, BOLT_HOLD);  // 1 during hold
            float isFading = step(travelTime + BOLT_HOLD, cycleT);  // 1 during fade

            // Traveling phase: pixel visible if behind the front, with soft leading edge
            float softEdge = 0.1;  // Softness of leading edge (fraction of segment)
            float behindFront = smoothstep(boltFront + softEdge, boltFront - softEdge, tDir);
            float travelVisible = behindFront * isTraveling;

            // Hold phase: full brightness
            float holdVisible = isHolding;

            // Fade phase: exponential decay
            float fadeTime = max(0.0, timeSinceFull - BOLT_HOLD);
            float fadeVisible = isFading * exp(-fadeTime * 3.0 / BOLT_FADE);

            // Combine all phases
            float boltVisible = travelVisible + holdVisible + fadeVisible;

            // Zero out if we're past the active phase (in dark/wait period)
            float activeEnd = travelTime + BOLT_HOLD + BOLT_FADE * 2.0;
            boltVisible *= step(cycleT, activeEnd);

            float core = smoothstep(curveThickness, 0.0, d_px);
            float g = smoothstep(maxDist, 0.0, d_px) - core;

            float flick = 0.5 + 0.5 * sin(time * 20.0 + arcSeed * 3.14 + bestT * 20.0);
            float noiseAmp = 0.5 + 0.5 * noise(vec2(bestT * 10.0, time * 1.5 + arcSeed));

            vec3 baseCol = vec3(0.15, 0.85, 1.0);
            vec3 innerCol = vec3(1.0, 1.0, 1.0);

            // Apply bolt visibility to brightness (with minimum)
            float effectiveVisible = mix(MIN_BRIGHTNESS, 1.0, boltVisible);
            float brightness = effectiveVisible * (0.8 + 0.5 * noiseAmp * flick);
            vec3 contrib = innerCol * core * brightness
                         + baseCol * g * effectiveVisible * (0.5 + 0.5 * noiseAmp);

            // Spark branches - brighter at bolt front
            float branchNoise = hash21(vec2(arcSeed * 7.0 + bestT * 50.0, time * 5.0));
            float spike = pow(branchNoise, 6.0) * core * boltVisible * 0.8;
            contrib += vec3(1.0, 0.9, 0.7) * spike;

            col += contrib;
        }
    }

    // Tone-map and clamp
    vec3 outc = 1.0 - exp(-col * 1.6);
    fragColor = vec4(outc, 1.0);
}
