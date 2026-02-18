#version 450

// Входные атрибуты вершины
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;


// Выходные переменные (в фрагментный шейдер)
layout(location = 0) out vec3 f_position;
layout(location = 1) out vec3 f_normal;


// SceneUniforms — binding = 0
layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_pos;
    float _padding0;

        float _padding1[44];
};

// ModelUniforms — binding = 2 (динамический, но в вершинном шейдере это не имеет значения)
layout(binding = 2, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    float _pad0[3];
    vec3 specular_color;
    float _pad1;
    float _pad_to_256[37];
};

void main() {
    vec4 world_pos = model * vec4(position, 1.0);
    f_position = world_pos.xyz;
    f_normal = mat3(model) * normal;
    gl_Position = view_projection * world_pos;
}