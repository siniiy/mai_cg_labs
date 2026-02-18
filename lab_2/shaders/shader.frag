#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

// Scene uniforms (binding 0)
layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_pos;
    float _padding0;
};

// Lighting uniforms (binding 1)
layout (binding = 1, std140) uniform LightingUniforms {
    vec3 directional_light_dir;
    float directional_light_intensity;
    vec3 directional_light_color;
    float _padding1;

    vec3 point_light_pos;
    float point_light_intensity;
    vec3 point_light_color;
    float _padding2;

    float point_light_constant;
    float point_light_linear;
    float point_light_quadratic;
    float _padding3;

    vec3 ambient_light_color;
    float ambient_light_intensity;
    float _padding4[32];
};

// Model uniforms (binding 2)
layout (binding = 2, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    vec3 specular_color;
    float shininess;
    float _padding[12];
};

vec3 blinnPhongDirectional(vec3 normal, vec3 view_dir) {
    vec3 light_dir = normalize(-directional_light_dir);
    float diff = max(dot(normal, light_dir), 0.0);
    // Диффузная составляющая Ламберт
    vec3 diffuse = directional_light_color * diff * directional_light_intensity;

    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway), 0.0), shininess);
    vec3 specular = specular_color * spec;

    return diffuse * albedo_color + specular;
}

vec3 blinnPhongPoint(vec3 normal, vec3 view_dir) {
    vec3 light_dir = normalize(point_light_pos - f_position);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = point_light_color * diff * point_light_intensity;

    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway), 0.0), shininess);
    vec3 specular = specular_color * spec;

    float dist = length(point_light_pos - f_position);
    float attenuation = 1.0 / (point_light_constant +
                              point_light_linear * dist +
                              point_light_quadratic * dist * dist);

    return (diffuse * albedo_color + specular) * attenuation;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(camera_pos - f_position);

    // Рассеянный (ambient)
    vec3 ambient = ambient_light_color * ambient_light_intensity * albedo_color;

    vec3 directional = blinnPhongDirectional(normal, view_dir);
    vec3 point = blinnPhongPoint(normal, view_dir);

    vec3 result = ambient + directional + point;

    final_color = vec4(result, 1.0);
}
