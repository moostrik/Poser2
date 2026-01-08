#version 460 core

uniform sampler2D tex0;
uniform sampler2D line_tex;
uniform vec4 target_color;
uniform float visibility;
uniform float param0;
uniform float param1;

// Control parameters with defaults matching original behavior
uniform float blend_strength = 4.0;     // Overall effect intensity (0.0-1.0)
uniform float saturation_influence = 1.0; // How much saturation affects blending (0.0-2.0)
uniform float shadow_preserve = 0.5;    // How much to preserve shadows (0.0-1.0)
uniform float highlight_blend = 0.7;    // Where highlights begin taking target color (0.0-1.0)
uniform float midtone_contrast = 2.9;   // Contrast enhancement in midtones (0.0-1.0)
uniform float color_preservation = 1.0; // How much original color detail to preserve (0.0-1.0)

in vec2 texCoord;
out vec4 fragColor;

void main(){
    vec4 tex_color = texture(tex0, texCoord);

    float line = texture(line_tex, texCoord).r;

    // Extract luminance value of original texture
    float lum = dot(tex_color.rgb, vec3(0.299, 0.587, 0.114));

    // Calculate color distance from grayscale
    float saturation = length(tex_color.rgb - vec3(lum)) * 3.0;

    // Get color "temperature" (rudimentary but effective)
    float warmth = (tex_color.r - tex_color.b) * 2.0;

    // Create blend factor based on color properties and user control
    float blend_factor = mix(0.4, 0.8, saturation * saturation_influence) * blend_strength;

    // More saturated areas take more of target color's hue
    // Dark areas preserve more original color
    // Bright areas take more of target color
    vec3 result = mix(
        tex_color.rgb,
        mix(
            target_color.rgb * lum,
            target_color.rgb + (tex_color.rgb - vec3(lum)) * color_preservation,
            smoothstep(shadow_preserve, highlight_blend, lum)
        ),
        blend_factor
    );

    // Enhance contrast in mid-tones
    float midtone = 1.0 - 2.0 * abs(lum - 0.5);
    result = mix(result, result * (1.0 + midtone * midtone_contrast), 0.5);



    result = mix(target_color.rgb, result, visibility);

    // fragColor = vec4(0.5, 0.5, 0.0, line);
    fragColor = vec4(result, min(line, param1));
}