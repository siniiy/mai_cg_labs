#version 450

layout(location = 0) in vec3 f_position;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec2 f_uv;

layout(location = 0) out vec4 final_color;

// Scene uniforms (binding = 0)
layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_pos;
    float time;

};

// Lighting uniforms (binding = 1)
layout(binding = 1, std140) uniform LightingUniforms {
    vec3 directional_light_dir;
    float directional_light_intensity;
    vec3 directional_light_color;
    float _padding1;

    vec3 point_light_pos;
    float point_light_intensity;
    vec3 point_light_color;
    float _padding2;

    vec3 spot_light_dir;
    float spot_light_cutoff;
    float spot_light_outer_cutoff;
    float spot_light_constant;
    float spot_light_linear;
    float spot_light_quadratic;

    vec3 ambient_light_color;
    float ambient_light_intensity;
    float _padding3[2]; // до 256 байт
};

// Model uniforms (binding = 2)
layout(binding = 2, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    float _pad0[3];
    vec3 specular_color;
    float time_m;
    float _pad_to_256[37];
};

// Текстура
layout(binding = 3) uniform sampler2D albedoTexture;

vec3 blinnPhongDirectional(vec3 normal, vec3 view_dir, vec3 albedo) {
    vec3 light_dir = normalize(-directional_light_dir);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = directional_light_color * directional_light_intensity * diff * albedo;

    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway), 0.0), shininess);
    vec3 specular = specular_color * spec;

    return diffuse + specular;
}

vec3 blinnPhongPoint(vec3 normal, vec3 view_dir, vec3 albedo) {
    vec3 light_dir = normalize(point_light_pos - f_position);
    float dist = length(point_light_pos - f_position);
    float attenuation = 1.0 / (
        spot_light_constant +
        spot_light_linear * dist +
        spot_light_quadratic * dist * dist
    );

    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = point_light_color * point_light_intensity * diff * albedo * attenuation;

    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway), 0.0), shininess);
    vec3 specular = specular_color * spec * attenuation;

    return diffuse + specular;
}

vec3 blinnPhongSpot(vec3 normal, vec3 view_dir, vec3 albedo) {
    vec3 light_dir = normalize(point_light_pos - f_position);
    vec3 spot_dir = normalize(-spot_light_dir);
    float theta = dot(light_dir, spot_dir);

    if (theta <= spot_light_outer_cutoff) return vec3(0.0);

    float dist = length(point_light_pos - f_position);
    float attenuation = 1.0 / (
        spot_light_constant +
        spot_light_linear * dist +
        spot_light_quadratic * dist * dist
    );


    float epsilon = spot_light_cutoff - spot_light_outer_cutoff;
    float intensity = clamp((theta - spot_light_outer_cutoff) / epsilon, 0.0, 1.0);

    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = point_light_color * point_light_intensity * diff * albedo * attenuation * intensity;

    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway), 0.0), shininess);
    vec3 specular = specular_color * spec * attenuation * intensity;

    return diffuse + specular;
}

void main() {
    vec3 albedo = texture(albedoTexture, f_uv).rgb * albedo_color;
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(camera_pos - f_position);

    vec3 ambient = ambient_light_color * ambient_light_intensity * albedo;
    vec3 directional = blinnPhongDirectional(normal, view_dir, albedo);
    vec3 point = blinnPhongPoint(normal, view_dir, albedo);
    vec3 spot = blinnPhongSpot(normal, view_dir, albedo);

    vec3 result = ambient + directional + point + spot;

    final_color = vec4(result, 1.0);
}