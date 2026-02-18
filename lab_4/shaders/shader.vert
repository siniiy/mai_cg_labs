#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec4 f_shadow_position;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	mat4 shadow_projection;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
    vec3 albedo_color;
    vec3 specular_color;
    float shininess;
    vec2 tilling;
};

void main() {
	vec4 position = model * vec4(v_position, 1.0f);
	vec4 normal = model * vec4(v_normal, 0.0f);

	gl_Position = view_projection * position;

	f_position = position.xyz;
	f_normal = normal.xyz;
	f_uv = v_uv * tilling;
	f_shadow_position = shadow_projection * position;
}

