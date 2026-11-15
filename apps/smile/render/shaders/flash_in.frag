#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform float flash;

in vec2 texCoord;
out vec4 fragColor;

void main(){
    vec4 originalColor = texture(tex0, texCoord);
    vec4 logoColor = texture(tex1, texCoord);
    originalColor.rgb += flash * 2.0;
    // fragColor = vec4(originalColor.rgb, originalColor.a);

    fragColor = mix(originalColor, logoColor, max(flash * 2.0 - 1.0, 0.0));
}