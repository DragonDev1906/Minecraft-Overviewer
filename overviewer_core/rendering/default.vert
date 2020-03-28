#version 330

uniform mat4 Mvp;
uniform vec3 pos;
uniform vec3 scale;
uniform uint face_texture_ids[6];
uniform vec4 face_uvs[6];
uniform vec2 model_rot;
//uniform vec3 dir_light;
in vec3 in_vert;
in vec3 in_normal;
in vec2 in_texcoord_0;
in uint in_faceid;
out vec3 texCoord;
out vec4 color;
//out float lum;

void main() {
    texCoord = vec3(in_texcoord_0 * face_uvs[in_faceid].xy + (1-in_texcoord_0) * face_uvs[in_faceid].zw + vec2(0, 0.0001) * in_normal.xy, face_texture_ids[in_faceid]);
    //color = vec4(0.0001 * texCoordV.x, in_faceid * 0.125, 0, 1);
    // face_texture_ids[in_faceid] * 0.124

    // mat4 translate_mat;
    // translate_mat[0] = vec4(1.0, 0.0, 0.0, pos.x);
    // translate_mat[1] = vec4(0.0, 1.0, 0.0, pos.y);
    // translate_mat[2] = vec4(0.0, 0.0, 1.0, pos.z);
    // translate_mat[3] = vec4(0.0, 0.0, 0.0, 1.0);

    // mat4 scale_mat;
    // scale_mat[0] = vec4(scale.x, 0, 0, 0);
    // scale_mat[1] = vec4(0, scale.y, 0, 0);
    // scale_mat[2] = vec4(0, 0, scale.z, 0);
    // scale_mat[3] = vec4(0, 0, 0, 1);

    mat4 model_rot_matrix;
    model_rot_matrix[0] = vec4(cos(model_rot.y), 0.0, sin(model_rot.y), 0.0);
    model_rot_matrix[1] = vec4(sin(model_rot.y)*sin(model_rot.x), cos(model_rot.x), -sin(model_rot.x)*cos(model_rot.y), 0.0);
    model_rot_matrix[2] = vec4(-sin(model_rot.y)*cos(model_rot.x), sin(model_rot.x), cos(model_rot.x)*cos(model_rot.y), 0.0);
    model_rot_matrix[3] = vec4(0.0, 0.0, 0.0, 1.0);

    //gl_Position = Mvp * x_rot_matrix * y_rot_matrix * (scale_mat * vec4(in_vert, 1.0) + vec4(pos, 0.0));
    gl_Position = Mvp * model_rot_matrix * (vec4(in_vert * scale, 1.0) + vec4(pos, 0.0));

}
