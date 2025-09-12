#version 460 core

uniform sampler2D tex0;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 lineTexCoord = vec2(texCoord.x, 0.0);
    float left = texture(tex0, lineTexCoord).b;
    float right = texture(tex0, lineTexCoord).g;
    float blue = texture(tex0, lineTexCoord).r;
    float vd = texture(tex0, lineTexCoord).a;

    float yNorm = texCoord.y;

    float edge = 0.02; // Adjust for desired smoothness

    float leftBar = smoothstep(left, left - edge, yNorm);
    float rightBar = smoothstep(right, right - edge, yNorm);
    float blueBar = smoothstep(blue, blue - edge, yNorm);
    float voidBar = smoothstep(vd, vd - edge, yNorm);

    // draw leftbar in orange, rightbar in cyan, bluebar in blue
    vec3 leftColor = vec3(1.0, 0.5, 0.0); // Orange
    vec3 rightColor = vec3(0.0, 1.0, 1.0); // Cyan
    vec3 blueColor = vec3(0.0, 0.0, 1.0); // Blue
    vec3 voidColor = vec3(0.5, 0.5, 0.5); // Gray
    vec3 color = (leftBar * leftColor + rightBar * rightColor + blueBar * blueColor);

    color = mix(color, voidColor, voidBar);

    // vec3 color = vec3(leftBar, rightBar, blueBar);
    fragColor = vec4(color, 1.0);



}