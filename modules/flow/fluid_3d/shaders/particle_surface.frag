#version 460 core

// ParticleSurface: a sphere sprite's pixel writes its front's depth (the targets' MIN blend keeps the
// nearest), its thickness and its color weighted (the SUM blend adds them). The thickness is the
// value times the sphere's height there over the disc's integral of it, (2/3) π r², so a sphere's
// thickness sums to its value, as a point's and a blob's do.

uniform vec2  uOutputSize;      // pixels
uniform float uZPerPixel;       // the volume's z (0 to 1) one pixel deep is: 1 / (width × z_extent)

flat in vec4 vColor;
flat in float vRadius;
flat in float vDepth;

layout(location = 0) out vec4 outDepth;       // R32F, MIN
layout(location = 1) out vec4 outThickness;   // R32F, SUM
layout(location = 2) out vec4 outColor;       // RGBA16F, SUM: rgb × weight, weight

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float height = sqrt(1.0 - r2);
    outDepth = vec4(vDepth - height * vRadius * uZPerPixel);
    float thickness = vColor.a * height * 3.0 / (6.2831853 * vRadius * vRadius);
    outThickness = vec4(thickness);
    outColor = vec4(vColor.rgb * thickness, thickness);
}
