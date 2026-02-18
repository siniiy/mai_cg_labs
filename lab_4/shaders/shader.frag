#version 450

struct PointLight {
	vec4 position_radius;
	vec3 color;
};

struct SpotLight {
    vec4 position_radius;
    vec4 direction_angle;
    vec3 color;
};

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_shadow_position;

layout (location = 0) out vec4 final_color;

layout(set = 0, binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	mat4 shadow_projection;
    vec3 view_position;
    float time;
	vec3 ambient_light_intensity;
	vec3 sun_light_direction;
	vec3 sun_light_color;
	uint point_light_count;
	uint spot_light_count;
};

layout (set = 0, binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	vec3 specular_color;
    float shininess;
};

layout(set = 0, binding = 2, std430) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(set = 0, binding = 3, std430) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

layout(set = 0, binding = 4) uniform sampler2D shadow_texture;

layout (set = 1, binding = 0) uniform sampler2D albedo_texture;
layout (set = 1, binding = 1) uniform sampler2D specular_texture;
layout (set = 1, binding = 2) uniform sampler2D emissive_texture;


vec2 waveDistortion(vec2 uv, float time) {
    float wave1 = sin(uv.x * 10.0 + time) * 0.05;
    float wave2 = cos(uv.y * 8.0 + time * 1.5) * 0.03;
    vec2 distorted = uv + vec2(wave1, wave2);
    return uv;
}

float calcShadow(vec4 shadow_position, sampler2D shadow_map) {
    vec3 coords = shadow_position.xyz / shadow_position.w;

    coords.xy = coords.xy * 0.5 + 0.5;

    if (coords.z > 1.0 || coords.x < 0.0 || coords.x > 1.0 ||
        coords.y < 0.0 || coords.y > 1.0) {
        return 1.0;
    }

    float shadow_depth = texture(shadow_map, coords.xy).r;

    float cur_depth = coords.z;

    float bias = 0.005;
    return (cur_depth - bias) > shadow_depth ? 0.0 : 1.0;
}

void main() {
    vec2 uv = waveDistortion(f_uv, time);
    vec3 albedo_tex = texture(albedo_texture, uv).rgb;
    vec3 specular_tex = texture(specular_texture, uv).rgb;
    vec3 emissive_tex = texture(emissive_texture, uv).rgb;

    vec3 final_albedo = albedo_tex * albedo_color;
    vec3 final_specular = specular_color * specular_tex;
    vec3 final_emissive = emissive_tex;

	vec3 normal = normalize(f_normal);
	vec3 sun_dir = normalize(-sun_light_direction);
    float sun_shade = max(0.0f, dot(normal, sun_dir));
    vec3 view_dir = normalize(view_position - f_position);
    vec3 half_vector = normalize(view_dir + sun_dir);

    vec3 sun_specular = final_specular *
                        pow(max(0.0f, dot(normal, half_vector)),
                            shininess);
    vec3 sun_light_intensity = sun_shade * sun_light_color *
                               (sun_specular + final_albedo);
    vec3 color = ambient_light_intensity + sun_light_intensity * calcShadow(f_shadow_position, shadow_texture) + final_emissive;


    for (uint i = 0; i < point_light_count; ++i) {
        PointLight light = point_lights[i];

        vec3 light_position = light.position_radius.xyz;
        float light_radius = light.position_radius.w;


        vec3 light_vec = light_position - f_position;
        float distance = length(light_vec);

        if (distance > light_radius){
            continue;
        }
        vec3 light_dir = normalize(light_vec);

        // Light attenuation using inverse square law
        float light_falloff = 1.0 / (distance * distance);
        float light_shade = max(0.0f, dot(light_dir, normal));

        vec3 half_vec = normalize(light_dir + view_dir);
        vec3 light_specular = final_specular * pow(max(0.0, dot(normal, half_vec)), shininess);

        vec3 light_contribution = light_falloff * light_shade * light.color *
                                 (final_albedo + light_specular);
        color += light_contribution;
    }

    for (uint i = 0; i < spot_light_count; ++i) {
        SpotLight light = spot_lights[i];

        vec3 light_position = light.position_radius.xyz;
        float light_radius = light.position_radius.w;
        vec3 light_direction = light.direction_angle.xyz;
        float light_angle = light.direction_angle.w;

        vec3 spot_vec = light_position - f_position;
        float distance = length(spot_vec);
        if (distance > light_radius) {
            continue;
        }
        vec3 spot_dir = normalize(spot_vec);
        float spot_angle = -dot(light_direction, spot_dir);
        float outer_angle = light_angle - 0.1;
        float factor = smoothstep(outer_angle, light_angle, spot_angle);
        if (factor > 0.0) {
            float light_falloff = (factor / (distance * distance + 0.01));
            float light_shade = max(0.0f, dot(spot_dir, normal));

            vec3 half_vec = normalize(spot_dir + view_dir);
            vec3 light_specular = final_specular * pow(max(0.0, dot(normal, half_vec)), shininess);

            vec3 light_contribution = light_falloff * light_shade * light.color *
                                     (final_albedo + light_specular);
            color += light_contribution;
        }
    }
    final_color = vec4(color, 1.0f);
}
