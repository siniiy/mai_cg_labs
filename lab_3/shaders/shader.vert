#version 450

// Входные атрибуты вершины
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// Выходные переменные (в фрагментный шейдер)
layout(location = 0) out vec3 f_position;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec2 f_uv;

// SceneUniforms — binding = 0
layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_pos;
    float _padding0;
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
    // Преобразуем позицию вершины в мировые координаты
    vec4 world_pos = model * vec4(position, 1.0);

    // Передаём мировую позицию (для точечного и прожекторного света)
    f_position = world_pos.xyz;

    // Преобразуем нормаль: только вращение и масштаб (без перемещения)

    f_normal = mat3(model) * normal;


    f_uv = uv;

    gl_Position = view_projection * world_pos;
}