#version 460 core

uniform sampler2D   src;
uniform int         numStreams;
uniform float       streamStep;
uniform float       lineWidth;
uniform float       outputAspectRatio;  // output buffer aspect ratio (width/height)
uniform vec2        displayRange;       // (min, max) value range for display
uniform vec4        colors[8];           // RGBA color array to cycle through
uniform int         numColors;          // number of colors in array

in vec2     texCoord;
out vec4    fragColor;


bool isBetweenStreamLine(vec2 uv, float streamStep, float lineWidth, int numStreams, float used_height) {
    // Check top boundary (y = 0)
    if (abs(uv.y) < lineWidth) return true;

    // Check bottom boundary (y = used_height)
    if (abs(uv.y - used_height) < lineWidth) return true;

    // Check intermediate stream boundaries
    int stream_id = int(floor(uv.y / streamStep));
    if (stream_id >= numStreams - 1) return false;

    float boundary_y = (float(stream_id + 1)) * streamStep;
    return abs(uv.y - boundary_y) < lineWidth;
}

void main() {
    // Flip Y coordinate so top of screen = 0, bottom = 1
    float flipped_y = 1.0 - texCoord.y;

    // Calculate used height for all streams
    float used_height = streamStep * float(numStreams);

    // Check if outside used vertical space (streams are at top)
    if (flipped_y < 0.0 || flipped_y > used_height) {
        fragColor = vec4(0.0);
        return;
    }

    vec2 stream_uv = vec2(texCoord.x, flipped_y);
    vec2 frag_pos = vec2(texCoord.x, flipped_y);

    // Calculate stream ID and bounds
    int stream_id = int(floor(stream_uv.y / streamStep));
    float stream_center = (float(stream_id) + 0.5) * streamStep;
    float stream_top = stream_center - 0.5 * streamStep;

    // Sample from texture - flip v coordinate so first stream (index 0) is at top
    float texture_v = (float(numStreams - stream_id) - 0.5) / float(numStreams);
    vec2 value_uv = vec2(stream_uv.x, texture_v);
    float value = texture(src, value_uv).r;
    float mask = texture(src, value_uv).g;

    // Draw the data dot FIRST so it sits in front of the white separator lines.
    // (Otherwise a value landing on a band boundary — e.g. 0 or 1 — is hidden by the line.)
    if (mask >= 0.001) {
        float norm = clamp((value - displayRange.x) / (displayRange.y - displayRange.x), 0.0, 1.0);
        // Invert norm so high values are at top, low values at bottom
        vec2 point = vec2(value_uv.x, stream_top + (1.0 - norm) * streamStep);

        // Draw continuous dots (no spacing)
        float dot_radius = lineWidth * 2.0;
        vec2 diff = frag_pos - point;
        // Correct for aspect ratio by dividing Y
        diff.y /= outputAspectRatio;
        if (length(diff) < dot_radius) {
            int color_index = stream_id % numColors;
            fragColor = colors[color_index];
            return;
        }
    }

    // Stream separator lines behind the data
    if (isBetweenStreamLine(stream_uv, streamStep, lineWidth / outputAspectRatio * 0.5, numStreams, used_height)) {
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);  // White separator lines
        return;
    }

    fragColor = vec4(0.0);
}
