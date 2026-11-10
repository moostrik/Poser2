// The light, as the composite and the rays read it: the phase toward the viewer and the caustic net's
// place. Places are uvw, the box's texture coordinates; distances in widths over the box `uBox`.

uniform bool  uPoint;         // a lamp at uVector's place, else a sun toward uVector
uniform vec3  uVector;        // toward the sun, or the lamp's place, ±1 the box's sides
uniform vec3  uBox;           // the box in widths: 1, height over width, depth over width
uniform float uAnisotropy;    // Henyey–Greenstein g, 0 even, toward 1 forward
uniform float uCaustics;      // the caustic net's contrast, 0 off to 1
uniform float uCausticSize;   // the net's cell, widths
uniform float uCausticDepth;  // widths down the light's way over which the net fades to 37%
uniform sampler2D uCausticNet; // the net, tileable, its mean 1

// Toward the light from a place, unit
vec3 to_light(vec3 uvw) {
    vec3 toward = uPoint ? (0.5 + 0.5 * uVector) * uBox - uvw * uBox : uVector;
    return normalize(length(toward) > 1e-6 ? toward : vec3(0.0, -1.0, 0.0));
}

// The share of light a place sends toward the viewer (along -z), 1 for g = 0: Henyey–Greenstein, its
// cosine between the light's travel and the way to the eye, which is the way to the light's z
float phase(vec3 uvw) {
    if (uAnisotropy <= 0.0) return 1.0;
    float cosine = to_light(uvw).z;
    float g = uAnisotropy;
    return (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosine, 1.5);
}

// The caustic net at a place, the same all along each of the light's lines, so it is carried down them as
// shafts: a sun's net laid flat across its way, a lamp's spread from it (where its line crosses a plane one
// lamp-to-center distance out); 1 where off
float caustic(vec3 uvw) {
    if (uCaustics <= 0.0) return 1.0;
    vec3 lamp = (0.5 + 0.5 * uVector) * uBox;
    vec3 axis = uPoint ? lamp - 0.5 * uBox : uVector;
    axis = normalize(length(axis) > 1e-6 ? axis : vec3(0.0, -1.0, 0.0));
    vec3 side = normalize(abs(axis.y) < 0.9 ? cross(axis, vec3(0.0, 1.0, 0.0)) : cross(axis, vec3(1.0, 0.0, 0.0)));
    vec3 up = cross(axis, side);
    vec3 p = uvw * uBox;
    vec2 across;
    float down;         // how far the light has come into the water, widths
    if (uPoint) {
        vec3 from_lamp = p - lamp;
        float out_along = max(dot(from_lamp, -axis), 1e-3);
        across = vec2(dot(from_lamp, side), dot(from_lamp, up)) / out_along * length(lamp - 0.5 * uBox);
        down = length(from_lamp);
    } else {
        across = vec2(dot(p, side), dot(p, up));
        vec3 a = abs(axis);
        int major = a.x > a.y ? (a.x > a.z ? 0 : 2) : (a.y > a.z ? 1 : 2);     // the face the sun comes through
        down = (axis[major] < 0.0 ? p[major] : uBox[major] - p[major]) / a[major];
    }
    float contrast = uCaustics * exp(-down / max(uCausticDepth, 1e-4));
    return mix(1.0, texture(uCausticNet, across / (8.0 * max(uCausticSize, 1e-4))).r, contrast);   // a tile holds 8 cells
}
