#version 460 core

layout(binding = 0) uniform sampler2D src;
uniform float brightness;
uniform float contrast;

in vec2 texCoord;
out vec4 fragColor;

void main(){
    vec4 originalColor = texture(src, texCoord);
    vec3 modifiedColor = ((originalColor.rgb - 0.5f) * max(contrast, 0)) + 0.5f;
    modifiedColor.rgb *= brightness;
    modifiedColor = clamp(modifiedColor, 0.0, 1.0);
    fragColor = vec4(modifiedColor, originalColor.a);
}