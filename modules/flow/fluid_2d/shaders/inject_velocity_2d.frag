#version 460 core

// Inject Velocity 2D Shader
// What the body gives the 2D velocity (widths/s, RG), in place of the velocity before (uVelocity):
//   0 DRAG   the boundary: the water pulled toward the body's velocity (uSource, the bridge's) × uGain, by uShare × the
//            body's presence × the injection's weight, never past it; the presence is the body's speed above a floor,
//            min(|body| / uFloor, 1), so a dead trail drags nothing toward 0
//   1 ADD    the body force: the bridge's acceleration × uStrength (weight × gain × dt) added
// Both read the bridge's texture at the cell's place on the image plane.

uniform sampler2D uVelocity;   // the velocity before (RG)
uniform sampler2D uSource;       // the bridge's velocity or acceleration (RG)
uniform int   uMode;           // 0 DRAG, 1 ADD
uniform float uGain;           // DRAG
uniform float uShare;          // DRAG: 1 - exp(-dt / velocity_drag)
uniform float uWeight;         // DRAG
uniform float uFloor;          // DRAG: widths/s below which the body drags nothing
uniform float uStrength;       // ADD: weight × gain × dt

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 v = texture(uVelocity, texCoord).xy;
    vec2 source = texture(uSource, texCoord).xy;
    if (uMode == 0) {
        float presence = min(length(source) / max(uFloor, 1e-6), 1.0);
        v += (source * uGain - v) * min(uShare * presence * uWeight, 1.0);
    } else {
        v += source * uStrength;
    }
    fragColor = vec4(v, 0.0, 0.0);
}
