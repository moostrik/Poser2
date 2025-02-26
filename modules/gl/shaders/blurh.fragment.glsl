#version 460 core

uniform sampler2D tex0;
uniform vec2 resolution;
uniform float radius;

in vec2 texCoord;
out vec4 fragColor;

void main(void) {
    vec2 uv = texCoord;

    // horizontal blur (since fragCoord samples at pixel centers it has a 0.5 added to it)
    // hence, i added an extra 0.5 to the texel coordinates to sample not at texel centers
    // but right between texels. the bilinear filtering hardware will average two texels
    // in each sample for me).

    float r = radius;
    vec3 blr  = vec3(0.0);
    blr += 0.013658 * texture(tex0, (uv + vec2(-19.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.019227 * texture(tex0, (uv + vec2(-17.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.026109 * texture(tex0, (uv + vec2(-15.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.034202 * texture(tex0, (uv + vec2(-13.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.043219 * texture(tex0, (uv + vec2(-11.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.052683 * texture(tex0, (uv + vec2( -9.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.061948 * texture(tex0, (uv + vec2( -7.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.070266 * texture(tex0, (uv + vec2( -5.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.076883 * texture(tex0, (uv + vec2( -3.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.081149 * texture(tex0, (uv + vec2( -1.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.041312 * texture(tex0, (uv + vec2(  0.0, 0.0) / resolution.xy* r)).xyz;
    blr += 0.081149 * texture(tex0, (uv + vec2(  1.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.076883 * texture(tex0, (uv + vec2(  3.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.070266 * texture(tex0, (uv + vec2(  5.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.061948 * texture(tex0, (uv + vec2(  7.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.052683 * texture(tex0, (uv + vec2(  9.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.043219 * texture(tex0, (uv + vec2( 11.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.034202 * texture(tex0, (uv + vec2( 13.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.026109 * texture(tex0, (uv + vec2( 15.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.019227 * texture(tex0, (uv + vec2( 17.5, 0.0) / resolution.xy* r)).xyz;
    blr += 0.013658 * texture(tex0, (uv + vec2( 19.5, 0.0) / resolution.xy* r)).xyz;

    fragColor = vec4( blr, 1.0 );
}
