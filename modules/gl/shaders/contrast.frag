#version 460 core

uniform sampler2D tex0;
uniform float brightness;
uniform float contrast;

in vec2 texCoord;
out vec4 fragColor;

void main(){
    vec4 originalColor = texture(tex0, texCoord);
    vec3 modifiedColor = ((originalColor.rgb - 0.5f) * max(contrast, 0)) + 0.5f;
    modifiedColor.rgb *= brightness;
    modifiedColor = clamp(modifiedColor, 0.0, 1.0);
    fragColor = vec4(modifiedColor, originalColor.a);
}