// The light, as the layer light and the composite read it: the phase toward the viewer and the caustic
// net. Places are uvw, the box's texture coordinates; distances in widths over the box `uBox`.

uniform bool  uPoint;         // a lamp at uVector's place, else a sun toward uVector
uniform vec3  uVector;        // toward the sun, or the lamp's place, ±1 the box's sides
uniform vec3  uBox;           // the box in widths: 1, height over width, depth over width
uniform float uAnisotropy;    // Henyey–Greenstein g, 0 even, toward 1 forward
uniform float uBackScattering; // the share of a backward lobe as strong
uniform float uSunPhase;      // a sun's phase, the same everywhere
uniform float uCaustics;      // the caustic net's contrast, 0 off to 1
uniform float uCausticSize;   // the net's cell, widths
uniform float uCausticDepth;  // widths down the light's way over which the net fades to 37%
uniform vec2  uNetX;          // a sun's net: its tile coordinates per uvw x, y and z, the net laid across its way
uniform vec2  uNetY;
uniform vec2  uNetZ;
uniform sampler2D uCausticNet; // the net, tileable, its mean 1

// Toward the light from a place, unit
vec3 to_light(vec3 uvw) {
    vec3 toward = uPoint ? (0.5 + 0.5 * uVector) * uBox - uvw * uBox : uVector;
    return normalize(length(toward) > 1e-6 ? toward : vec3(0.0, -1.0, 0.0));
}

// Henyey–Greenstein, 1 for g = 0, at the cosine between the light's travel and the way to the eye
float henyey_greenstein(float g, float cosine) {
    return (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosine, 1.5);
}

// The share of light a place sends toward the viewer (along -z): a forward lobe of g uAnisotropy and a
// backward one as strong, uBackScattering its share; the cosine is the way to the light's z. A sun's is the
// same everywhere, worked out once (uSunPhase)
float phase(vec3 uvw) {
    if (!uPoint) return uSunPhase;
    if (uAnisotropy <= 0.0) return 1.0;
    float cosine = to_light(uvw).z;
    return mix(henyey_greenstein(uAnisotropy, cosine), henyey_greenstein(-uAnisotropy, cosine), uBackScattering);
}

// The axis a lamp's net spreads along, from the box's center to the lamp; a sun's is its direction
vec3 net_axis() {
    vec3 axis = uPoint ? (0.5 + 0.5 * uVector) * uBox - 0.5 * uBox : uVector;
    return normalize(length(axis) > 1e-6 ? axis : vec3(0.0, -1.0, 0.0));
}

// The net's contrast at a place: uCaustics, fading with how far the light has come into the water (from
// the face a sun comes through, or from the lamp)
float caustic_contrast(vec3 uvw) {
    if (uCaustics <= 0.0) return 0.0;
    vec3 p = uvw * uBox;
    float down;
    if (uPoint) {
        down = length(p - (0.5 + 0.5 * uVector) * uBox);
    } else {
        vec3 axis = net_axis();
        vec3 a = abs(axis);
        int major = a.x > a.y ? (a.x > a.z ? 0 : 2) : (a.y > a.z ? 1 : 2);
        down = (axis[major] < 0.0 ? p[major] : uBox[major] - p[major]) / a[major];
    }
    return uCaustics * exp(-down / max(uCausticDepth, 1e-4));
}

// The net at a place, the same all along each of the light's lines, so it is carried down them as shafts:
// a sun's net laid flat across its way (uNetX, uNetY, uNetZ, linear in uvw), a lamp's spread from it, where
// its line crosses a plane one lamp-to-center distance out
float caustic_net(vec3 uvw) {
    vec2 tile;
    if (uPoint) {
        vec3 lamp = (0.5 + 0.5 * uVector) * uBox;
        vec3 axis = net_axis();
        vec3 side = normalize(abs(axis.y) < 0.9 ? cross(axis, vec3(0.0, 1.0, 0.0)) : cross(axis, vec3(1.0, 0.0, 0.0)));
        vec3 up = cross(axis, side);
        vec3 from_lamp = uvw * uBox - lamp;
        float out_along = max(dot(from_lamp, -axis), 1e-3);
        tile = vec2(dot(from_lamp, side), dot(from_lamp, up)) / out_along * length(lamp - 0.5 * uBox)
             / (8.0 * max(uCausticSize, 1e-4));     // a tile holds 8 cells
    } else {
        tile = uvw.x * uNetX + uvw.y * uNetY + uvw.z * uNetZ;
    }
    return texture(uCausticNet, tile).r;
}

// The caustic's factor on the light at a place: 1 where off
float caustic(vec3 uvw) {
    float contrast = caustic_contrast(uvw);
    return contrast > 0.0 ? mix(1.0, caustic_net(uvw), contrast) : 1.0;
}
