#version 460 core

// The four camera frames unwrapped into one 360-degree strip as the RIG CENTRE would see them:
// x is azimuth, linear in degrees; y is the TANGENT of elevation (strip.strip_y), so the
// strip's vertical is a photograph's, the same shape as the camera frames' rows.
//
// A transcription of modules/tracker/panoramic/projection.py — round-tripped against the tracker's
// own Rig in modules/tracker/tests/test_projection.py — and of the strip's rows, vertical re-projection
// and coverage in modules/render/layers/panorama/strip.py (tests in test_strip.py). Keep them in step:
// the whole value of this display is that it draws with the numbers the tracker tracks with, so
// a wrong camera constant shows up as a ghost instead of being silently absorbed.
//
// BOTH AXES ARE RE-PROJECTED, and they have to be. A camera sits `cameraRadius` out from the
// centre, so it sees a point on the focus cylinder at a wider bearing AND a higher elevation
// than the centre does — by the same factor, R/d, which is 1.19 straight ahead at R 2.25 and
// 1.08 at a seam. Re-projecting only the azimuth (which is all the tracker needs) would leave
// everything in the strip 19% too tall for its width, and no single row aspect can undo that
// because the factor varies across the frame.
//
// All cameras in one pass rather than one draw each, so coverage is counted per fragment and the
// averaging divisor is exact even where the parallax correction has narrowed the overlap band.

#define MAX_CAMS 8

// Must stay in step with PanoramaBlend in modules/render/layers/panorama/settings.py:
// the enum's value IS this uniform.
#define BLEND_MAX        0
#define BLEND_AVERAGE    1
#define BLEND_MIN        2
#define BLEND_DIFFERENCE 3
#define BLEND_SPLIT      4
#define BLEND_STRIPE     5

// Column width (output pixels) of one BLEND_STRIPE band.
#define STRIPE_PX 8.0

uniform sampler2D tex[MAX_CAMS];
uniform int   numCams;
uniform float camFov;      // one camera's horizontal field (degrees)
uniform float horizonRow;  // the frames' rows are TANGENTS of elevation: row = horizonRow -
uniform float focalRows;   //   focalRows * tan(e), normalised, 0 = top (projection.row_from_elevation)
uniform float targetFov;   // the sector one camera owns, 360 / numCams (degrees)
uniform float cameraRadius;  // camera distance from the rig centre (m)
uniform float focusRadius; // radius of the cylinder the image is aligned for (m), from the fixture axis
uniform float elevTop;     // elevation of this strip's top row, at the centre (degrees)
uniform float elevBottom;  // elevation of its bottom row, at the centre (degrees)
uniform float camElevLo;   // the frames' window, at the camera (degrees): the bottom row is the
uniform float camElevHi;   //   sensor's lowest reach, the top row whatever the rows reach
uniform int   blendMode;

in vec2 texCoord;
out vec4 fragColor;

// Where camera `cam` shows this azimuth and elevation, in its own normalized frame.
// Returns x < 0 when this camera does not cover the point.
vec2 cameraUV(int cam, float azimuth, float elevation) {
    // --- azimuth: the inverse of the Rig's forward chain -------------------------------
    // Camera axes sit between the seams: targetFov * (cam + 0.5).
    float phi = azimuth - targetFov * (float(cam) + 0.5);
    phi = mod(phi + 180.0, 360.0) - 180.0;      // fold to [-180, 180)
    float p = radians(phi);

    // Distance from this camera to the focus cylinder at this bearing (law of cosines).
    float d = sqrt(max(0.0, cameraRadius * cameraRadius + focusRadius * focusRadius
                            - 2.0 * cameraRadius * focusRadius * cos(p)));
    if (d < 1e-6) return vec2(-1.0);

    // The camera sits cameraRadius behind the centre, so its own bearing to the same point is
    // wider than the centre's: theta = phi + asin(r * sin(phi) / d). Exact, not an expansion.
    float theta = phi;
    if (cameraRadius > 0.0) {
        theta = phi + degrees(asin(clamp(cameraRadius * sin(p) / d, -1.0, 1.0)));
    }

    float local = theta + camFov * 0.5;
    if (local < 0.0 || local > camFov) return vec2(-1.0);

    // --- elevation: the same triangle, in the vertical plane ----------------------------
    // A point on the cylinder stands `h` above the lens plane at horizontal distance
    // focusRadius from the centre and d from the camera, so tan(e_cam) = tan(e_centre) * R/d.
    // The height itself cancels: only the ratio of the two distances survives.
    float eCam = degrees(atan(tan(radians(elevation)) * focusRadius / d));
    if (eCam < camElevLo || eCam > camElevHi) return vec2(-1.0);

    // The delivered frame is cylindrical: a row is the tangent of its elevation below the
    // horizon row (`row_from_elevation`, top-down, 0 = the top row). The texture's v runs
    // bottom-up, so the row is flipped into it. The black arch at the top of the side columns
    // is in the pixels themselves.
    float row = horizonRow - focalRows * tan(radians(eCam));
    return vec2(local / camFov, 1.0 - row);
}

void main() {
    float azimuth   = texCoord.x * 360.0;
    // Rows are tangents of centre elevation: linear between tan(bottom) and tan(top).
    float elevation = degrees(atan(mix(tan(radians(elevBottom)), tan(radians(elevTop)), texCoord.y)));

    vec3  peak   = vec3(0.0);
    vec3  low    = vec3(0.0);
    vec3  total  = vec3(0.0);
    // The first two covering cameras, in id order. Every mode that compares the two views rather
    // than merging them needs them kept apart, which peak/total have already thrown away.
    vec3  first  = vec3(0.0);
    vec3  second = vec3(0.0);
    float count  = 0.0;

    for (int cam = 0; cam < MAX_CAMS; ++cam) {
        if (cam >= numCams) break;

        vec2 uv = cameraUV(cam, azimuth, elevation);
        if (uv.x < 0.0 || uv.y < 0.0 || uv.y > 1.0) continue;

        vec3 rgb = texture(tex[cam], uv).rgb;
        peak  = max(peak, rgb);
        low   = (count == 0.0) ? rgb : min(low, rgb);
        total += rgb;
        if      (count == 0.0) first  = rgb;
        else if (count == 1.0) second = rgb;
        count += 1.0;
    }

    if (count == 0.0) {
        fragColor = vec4(0.0);
        return;
    }

    // Outside an overlap there is only one view, so the comparing modes have nothing to say: they
    // fall back to the plain picture, except DIFFERENCE which goes black so only overlaps light up.
    vec3 rgb = peak;
    if (blendMode == BLEND_AVERAGE) {
        rgb = total / count;
    } else if (blendMode == BLEND_MIN) {
        rgb = low;
    } else if (blendMode == BLEND_DIFFERENCE) {
        rgb = (count < 2.0) ? vec3(0.0) : abs(first - second);
    } else if (blendMode == BLEND_SPLIT) {
        // The frames are mono, so the channels are free: one camera into red, its neighbour into
        // green. Which fringe leads says which camera is left of the other.
        rgb = (count < 2.0) ? peak : vec3(first.r, second.r, 0.0);
    } else if (blendMode == BLEND_STRIPE) {
        bool even = mod(floor(gl_FragCoord.x / STRIPE_PX), 2.0) < 1.0;
        rgb = (count < 2.0) ? first : (even ? first : second);
    }
    fragColor = vec4(rgb, 1.0);
}
