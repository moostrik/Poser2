#version 460 core

// The four camera frames unwrapped into one 360-degree strip: x is azimuth, 0 at the left edge.
//
// A transcription of modules/tracker/panoramic/panorama_map.py, which is round-tripped against
// the tracker's own Geometry in modules/tracker/tests/test_panorama_map.py. Keep the two in step:
// the whole value of this display is that it draws with the numbers the tracker tracks with, so
// a wrong camera constant shows up as a ghost instead of being silently absorbed.
//
// All cameras in one pass rather than one draw each, so coverage is counted per fragment and the
// averaging divisor is exact even where the parallax correction has narrowed the overlap band.

#define MAX_CAMS 8

#define BLEND_MAX     0
#define BLEND_AVERAGE 1

uniform sampler2D tex[MAX_CAMS];
uniform int   numCams;
uniform float camFov;      // one camera's horizontal field (degrees)
uniform float targetFov;   // the sector one camera owns, 360 / numCams (degrees)
uniform float ringRadius;  // camera distance from the rig centre (m)
uniform float focusRadius; // half the focus diameter: the cylinder the image is aligned for (m)
uniform int   blendMode;

in vec2 texCoord;
out vec4 fragColor;

// The normalized column of camera `cam` showing this azimuth, or -1 outside its field.
float cameraColumn(int cam, float azimuth) {
    // Camera axes sit between the seams: targetFov * (cam + 0.5).
    float phi = azimuth - targetFov * (float(cam) + 0.5);
    phi = mod(phi + 180.0, 360.0) - 180.0;      // fold to [-180, 180)
    float p = radians(phi);

    // Distance from this camera to the focus cylinder at this bearing (law of cosines).
    float d = sqrt(max(0.0, ringRadius * ringRadius + focusRadius * focusRadius
                            - 2.0 * ringRadius * focusRadius * cos(p)));

    // The camera sits ringRadius behind the centre, so its own bearing to the same point is
    // wider than the centre's: theta = phi + asin(r * sin(phi) / d). Exact, not an expansion.
    float theta = phi;
    if (ringRadius > 0.0 && d > 1e-6) {
        theta = phi + degrees(asin(clamp(ringRadius * sin(p) / d, -1.0, 1.0)));
    }

    float local = theta + camFov * 0.5;
    if (local < 0.0 || local > camFov) return -1.0;
    return local / camFov;
}

void main() {
    float azimuth = texCoord.x * 360.0;

    vec3  peak  = vec3(0.0);
    vec3  total = vec3(0.0);
    float count = 0.0;

    for (int cam = 0; cam < MAX_CAMS; ++cam) {
        if (cam >= numCams) break;

        float column = cameraColumn(cam, azimuth);
        if (column < 0.0) continue;

        vec3 rgb = texture(tex[cam], vec2(column, texCoord.y)).rgb;
        peak  = max(peak, rgb);
        total += rgb;
        count += 1.0;
    }

    if (count == 0.0) {
        fragColor = vec4(0.0);
        return;
    }

    vec3 rgb = (blendMode == BLEND_AVERAGE) ? total / count : peak;
    fragColor = vec4(rgb, 1.0);
}
