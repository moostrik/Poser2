#version 460 core

uniform sampler2D src;
uniform float exposure;
uniform float offset;
uniform float gamma;

in vec2 texCoord;
out vec4 fragColor;

void main(){
    vec4 color = texture(src, texCoord);
    vec3 eog = pow((color.xyz * vec3(exposure)) + vec3(offset), vec3(1.0/gamma));
    fragColor = vec4(eog, color.w);
}