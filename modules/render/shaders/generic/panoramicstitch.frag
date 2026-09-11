#version 460 core

// The four camera frames unwrapped into one 360-degree strip as the RIG CENTRE would see them:
// x is azimuth, y is elevation, both measured at the centre, both linear.
//
// A transcription of modules/tracker/panoramic/panorama_map.py, which is round-tripped against
// the tracker's own Geometry in modules/tracker/tests/test_panorama_map.py. Keep the two in step:
// the whole value of this display is that it draws with the numbers the tracker tracks with, so
// a wrong camera constant shows up as a ghost instead of being silently absorbed.
//
// BOTH AXES ARE RE-PROJECTED, and they have to be. A camera sits `ringRadius` out from the
// centre, so it sees a point on the focus cylinder at a wider bearing AND a higher elevation
// than the centre does — by the same factor, R/d, which is 1.19 straight ahead at Ø 4.5 and
// 1.08 at a seam. Re-projecting only the azimuth (which is all the tracker needs) would leave
// everything in the strip 19% too tall for its width, and no single row aspect can undo that
// because the factor varies across the frame.
//
// All cameras in one pass rather than one draw each, so coverage is counted per fragment and the
// averaging divisor is exact even where the parallax correction has narrowed the overlap band.

#define MAX_CAMS 8

#define BLEND_MAX     0
#define BLEND_AVERAGE 1

uniform sampler2D tex[MAX_CAMS];
uniform int   numCams;
uniform float camFov;      // one camera's horizontal field (degrees)
uniform float vfov;        // one camera's vertical field (degrees)
uniform float targetFov;   // the sector one camera owns, 360 / numCams (degrees)
uniform float ringRadius;  // camera distance from the rig centre (m)
uniform float focusRadius; // half the focus diameter: the cylinder the image is aligned for (m)
uniform float elevTop;     // elevation of this strip's top row, at the centre (degrees)
uniform float elevBottom;  // elevation of its bottom row, at the centre (degrees)
uniform float camElevLo;   // the frames' POPULATED elevation band, at the camera (degrees).
uniform float camElevHi;   // Tilt empties the rest: an up-aimed camera never imaged the floor.
uniform int   blendMode;

in vec2 texCoord;
out vec4 fragColor;

// Where camera `cam` shows this azimuth and elevation, in its own normalized frame.
// Returns x < 0 when this camera does not cover the point.
vec2 cameraUV(int cam, float azimuth, float elevation) {
    // --- azimuth: the inverse of Geometry's forward chain -------------------------------
    // Camera axes sit between the seams: targetFov * (cam + 0.5).
    float phi = azimuth - targetFov * (float(cam) + 0.5);
    phi = mod(phi + 180.0, 360.0) - 180.0;      // fold to [-180, 180)
    float p = radians(phi);

    // Distance from this camera to the focus cylinder at this bearing (law of cosines).
    float d = sqrt(max(0.0, ringRadius * ringRadius + focusRadius * focusRadius
                            - 2.0 * ringRadius * focusRadius * cos(p)));
    if (d < 1e-6) return vec2(-1.0);

    // The camera sits ringRadius behind the centre, so its own bearing to the same point is
    // wider than the centre's: theta = phi + asin(r * sin(phi) / d). Exact, not an expansion.
    float theta = phi;
    if (ringRadius > 0.0) {
        theta = phi + degrees(asin(clamp(ringRadius * sin(p) / d, -1.0, 1.0)));
    }

    float local = theta + camFov * 0.5;
    if (local < 0.0 || local > camFov) return vec2(-1.0);

    // --- elevation: the same triangle, in the vertical plane ----------------------------
    // A point on the cylinder stands `h` above the lens plane at horizontal distance
    // focusRadius from the centre and d from the camera, so tan(e_cam) = tan(e_centre) * R/d.
    // The height itself cancels: only the ratio of the two distances survives.
    float eCam = degrees(atan(tan(radians(elevation)) * focusRadius / d));
    if (eCam < camElevLo || eCam > camElevHi) return vec2(-1.0);

    // The delivered frame is equirectangular, so a row IS an elevation, linearly.
    return vec2(local / camFov, 0.5 + eCam / vfov);
}

void main() {
    float azimuth   = texCoord.x * 360.0;
    float elevation = mix(elevBottom, elevTop, texCoord.y);

    vec3  peak  = vec3(0.0);
    vec3  total = vec3(0.0);
    float count = 0.0;

    for (int cam = 0; cam < MAX_CAMS; ++cam) {
        if (cam >= numCams) break;

        vec2 uv = cameraUV(cam, azimuth, elevation);
        if (uv.x < 0.0 || uv.y < 0.0 || uv.y > 1.0) continue;

        vec3 rgb = texture(tex[cam], uv).rgb;
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
