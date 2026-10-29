#version 460 core

in vec2 tex_coord;

uniform sampler2D atlas;
uniform vec4 text_color;

out vec4 frag_color;

void main() {
    float alpha = texture(atlas, tex_coord).r;
    frag_color = vec4(text_color.rgb, text_color.a * alpha);
}
