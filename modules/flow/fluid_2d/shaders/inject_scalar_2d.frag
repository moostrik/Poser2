#version 460 core

// Inject Scalar 2D Shader
// A scalar input (red, signed) into a 2D field, in place of the field before (uField):
//   0 ADD     the input × uStrength added
//   1 FILL    the field dragged toward the input × uGain (capped at ±1) by uStrength, never past it, only where the
//             target is further from 0 than the field on its side: a warm target only warms, a cold one only cools,
//             so the field never falls toward a fainter body; only where the input is not 0

uniform sampler2D uField;      // the field before (red)
uniform sampler2D uInput;      // the scalar, red
uniform int   uMode;           // 0 ADD, 1 FILL
uniform float uStrength;       // ADD: the weight; FILL: the drag's share × the weight
uniform float uGain;           // FILL: the input times this, at most ±1, is the target

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float current = texture(uField, texCoord).r;
    float source = texture(uInput, texCoord).r;
    if (source != 0.0) {
        if (uMode == 1) {
            float target = clamp(source * uGain, -1.0, 1.0);
            bool further = target > 0.0 ? target > current : target < current;
            if (further) current += (target - current) * min(uStrength, 1.0);
        } else {
            current += source * uStrength;
        }
    }
    fragColor = vec4(current, 0.0, 0.0, 0.0);
}
