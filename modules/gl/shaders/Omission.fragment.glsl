#version 460 core

uniform sampler2D src;
uniform sampler2D dst;
uniform sampler2D mask;
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
vec3 blendExclusion(vec3 base, vec3 blend) { return base+blend-2.0*base*blend; }
vec3 blendExclusion(vec3 base, vec3 blend, float opacity) { return (blendExclusion(base, blend) * opacity + base * (1.0 - opacity));}

float blendReflect(float base, float blend) { return (blend==1.0)?blend:min(base*base/(1.0-blend),1.0); }
vec3 blendReflect(vec3 base, vec3 blend) { return vec3(blendReflect(base.r,blend.r),blendReflect(base.g,blend.g),blendReflect(base.b,blend.b)); }
vec3 blendReflect(vec3 base, vec3 blend, float opacity) { return (blendReflect(base, blend) * opacity + base * (1.0 - opacity)); }

vec3 blendGlow(vec3 base, vec3 blend) { return blendReflect(blend,base); }
vec3 blendGlow(vec3 base, vec3 blend, float opacity) { return (blendGlow(base, blend) * opacity + base * (1.0 - opacity));}

vec3 blendDifference(vec3 base, vec3 blend) { return abs(base-blend); }
vec3 blendDifference(vec3 base, vec3 blend, float opacity) { return (blendDifference(base, blend) * opacity + base * (1.0 - opacity));}

float blendOverlay(float base, float blend) { return base<0.5?(2.0*base*blend):(1.0-2.0*(1.0-base)*(1.0-blend)); }
vec3 blendOverlay(vec3 base, vec3 blend) { return vec3(blendOverlay(base.r,blend.r),blendOverlay(base.g,blend.g),blendOverlay(base.b,blend.b));}
vec3 blendOverlay(vec3 base, vec3 blend, float opacity) { return (blendOverlay(base, blend) * opacity + base * (1.0 - opacity));}

float blendLinearBurn(float base, float blend) { return max(base+blend-1.0,0.0); }
vec3 blendLinearBurn(vec3 base, vec3 blend) { return max(base+blend-vec3(1.0),vec3(0.0));}
vec3 blendLinearBurn(vec3 base, vec3 blend, float opacity) { return (blendLinearBurn(base, blend) * opacity + base * (1.0 - opacity));}

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