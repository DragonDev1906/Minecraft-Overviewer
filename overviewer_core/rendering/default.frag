#version 330

uniform sampler2DArray Textures;
in vec4 color;
in vec3 texCoord;
out vec4 fragColor;

void main() {
    //fragColor = color;
    fragColor = vec4(texture(Textures, texCoord));
    // fragColor = vec4(color.xyz, 1) + 0.00001 * vec4(texture(Texture, texCoordV));
    //fragColor = vec4(texCoordV, 0, 1);
}
