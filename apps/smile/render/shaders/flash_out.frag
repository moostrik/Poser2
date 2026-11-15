#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform float flash;

in vec2 texCoord;
out vec4 fragColor;

void main(){
    vec4 originalColor = texture(tex0, texCoord);
    vec4 logoColor = texture(tex1, texCoord);
    float lum = 1.0 - texture(tex2, texCoord).x;
    lum = clamp(lum * 2.5 - 1.3, 0.0, 1.0);
    lum = pow(lum, 2.0);
    lum = clamp((flash * 2.0 - 1.0) + lum, 0.0, 1.0);
    vec3 color = vec3(lum) * logoColor.rgb;
    // originalColor.rgb += color;
    // // // originalColor.rgb *= 1.0 + (1.0 - flash) * vec3(lum);
    // // fragColor = mix(originalColor, logoColor, lum);
    // fragColor = originalColor;
    
    fragColor = mix(originalColor, logoColor, lum);
}