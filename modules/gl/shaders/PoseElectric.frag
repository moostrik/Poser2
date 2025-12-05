#version 460 core

uniform vec4 points[17];
uniform vec2 resolution;
uniform float time;

const int N_JOINTS = 17;
const int N_CONN = 16;
const int SAMPLES = 24;  // Increased for better curve sampling

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

// Sample distance to curve by discrete sampling
float distToCurve(vec2 p, vec2 A, vec2 B, vec2 C, float seglen, out float bestT){
    float best = 1e10;
    float bt = 0.0;
    // Scale samples by segment length (1 sample per ~8 pixels)
    int numSamples = int(clamp(seglen / 8.0, 12.0, 64.0));
    for(int i=0; i<=numSamples; i++){
        float t = float(i)/float(numSamples);
        vec2 q = bezierQuad(A, B, C, t);
        float d = distance(p, q);
        if(d < best){
            best = d;
            bt = t;
        }
    }
    bestT = bt;
    return best;
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
    float thickness_px = 4.0;
    float glow_px = 20.0;

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

        // Compute control point
        vec2 mid = (A + C) * 0.5;
        vec2 dir = normalize(C - A);
        vec2 perp = vec2(-dir.y, dir.x);
        float baseOff = seglen * 0.25;

        // Time+index driven noise for dynamic arcs
        float tnoise = fbm((mid / resolution + float(i)*0.13 + time*0.5) * 3.0);
        float sideSign = (hash21(vec2(float(i), 3.21)) > 0.5) ? 1.0 : -1.0;
        float off = baseOff * (0.5 + 0.8 * tnoise) * sideSign;

        vec2 ctrl = mid + perp * off;

        // Fine jitter
        float jitter = 0.12 * seglen * (noise((A_raw + C_raw) * 10.0 + time*2.0 + float(i)) - 0.5);
        ctrl += perp * jitter;

        // Distance to curve (in pixel space)
        float bestT;
        float d_px = distToCurve(p, A, ctrl, C, seglen, bestT);

        float core = smoothstep(thickness_px, 0.0, d_px);
        float g = smoothstep(thickness_px + glow_px, 0.0, d_px) - core;

        float flick = 0.5 + 0.5 * sin(time*20.0 + float(i)*3.14 + bestT*20.0);
        float noiseAmp = 0.5 + 0.5*fbm(vec2(bestT*10.0, time*1.5 + float(i)));

        vec3 baseCol = vec3(0.15, 0.85, 1.0);
        vec3 innerCol = vec3(1.0, 1.0, 1.0);

        vec3 contrib = innerCol * core * (1.0 + 0.6 * noiseAmp * flick)
                     + baseCol * g * (0.7 + 0.6 * noiseAmp);

        // Small spark branches near core
        if(core > 0.01){
            float spikeChance = smoothstep(thickness_px * 2.0, 0.0, d_px);
            float branchNoise = noise(vec2(float(i)*7.0 + bestT*50.0, time*5.0));
            float spike = pow(branchNoise, 6.0) * spikeChance * 0.8;
            contrib += vec3(1.0, 0.9, 0.7) * spike;
        }

        col += contrib;
    }

    // Tone-map and clamp
    vec3 outc = 1.0 - exp(-col * 1.6);
    fragColor = vec4(outc, 1.0);
}
