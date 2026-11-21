#version 460 core

layout(binding = 0) uniform sampler2D src;
layout(binding = 1) uniform sampler2D dst;
layout(binding = 2) uniform sampler2D mask;
uniform float bloom;
uniform float omission;
uniform vec2 resolution;

in vec2 texCoord;
out vec4 fragColor;

vec4 BlurColor (in vec2 Coord, in sampler2D Tex, in float MipBias)
{
	vec2 TexelSize = MipBias/resolution.xy;

    vec4  Color = texture(Tex, Coord, MipBias);
    Color += texture(Tex, Coord + vec2(TexelSize.x,0.0), MipBias);
    Color += texture(Tex, Coord + vec2(-TexelSize.x,0.0), MipBias);
    Color += texture(Tex, Coord + vec2(0.0,TexelSize.y), MipBias);
    Color += texture(Tex, Coord + vec2(0.0,-TexelSize.y), MipBias);
    Color += texture(Tex, Coord + vec2(TexelSize.x,TexelSize.y), MipBias);
    Color += texture(Tex, Coord + vec2(-TexelSize.x,TexelSize.y), MipBias);
    Color += texture(Tex, Coord + vec2(TexelSize.x,-TexelSize.y), MipBias);
    Color += texture(Tex, Coord + vec2(-TexelSize.x,-TexelSize.y), MipBias);

    return Color/9.0;
}

void main() {
    vec4 src_color = texture(src, texCoord);
    vec4 dst_color = texture(dst, texCoord);
    float mask_value = texture(mask, texCoord).x;
    mask_value = clamp(mask_value * 2.5 - 1.3, 0.0, 1.0);
    mask_value = pow(mask_value, 1.0);
    mask_value = clamp((omission * 2.0 - 1.0) + mask_value, 0.0, 1.0);

    // src_color *= 1.0 + bloom * 3.0;
    // src_color = clamp(src_color, 0.0, 1.0);
    // fragColor = mix(src_color, dst_color, omission);

    float b = bloom * 2;
    if (b > 1) b = 2.0 - b;

    float Threshold = omission;
    float Intensity = 30.5 * bloom;
    float BlurSize = 8.0;
    vec4 Highlight = clamp(BlurColor(texCoord, src, BlurSize)-Threshold,0.0,1.0)*1.0/(1.0-Threshold);
    vec4 bloomColor = 1.0-(1.0-src_color)*(1.0-Highlight*Intensity); //Screen Blend Mode

    fragColor = bloomColor;

    float lum = (bloomColor.r + bloomColor.r + bloomColor.g + bloomColor.g, + bloomColor.g + bloomColor.b) / 7.0;
    // lum = clamp(lum * 2.5 - 1.3, 0.0, 1.0);
    lum = pow(lum, 1.0);
    lum = clamp((omission * 2.0 - 1.0) + lum, 0.0, 1.0);

    bloomColor = clamp(bloomColor, 0.0, 1.0);
    fragColor = mix(bloomColor, dst_color, mask_value);
}