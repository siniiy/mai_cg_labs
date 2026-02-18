#include <vulkan/vulkan_core.h>
#include <veekay/veekay.hpp>
#include "veekay/input.hpp"

#include <imgui.h>
#include <lodepng.h>
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <set>

#define _USE_MATH_DEFINES
#include <math.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 16;

struct Vertex {
	veekay::vec3 position; // Позиция точки на объекте
	veekay::vec3 normal;
	veekay::vec2 uv; // текстура
	// NOTE: You can add more attributes
};

struct PointLight {
	veekay::vec3 position;
	float intensity;  // Для закона обратных квадратов
	veekay::vec3 color;
	float _pad0;  // Выравнивание для std140
};

struct SceneUniforms {
	veekay::mat4 view_projection;
		veekay::mat4 shadow_projection;  // Матрица проекции теней для направленного света (должна быть сразу после view_projection!)
	veekay::vec3 view_position;  // Позиция камеры для расчета view direction
	float _pad0;  // Выравнивание для std140
	
	veekay::vec3 ambient_light_intensity;  // Рассеянное освещение
	float _pad1;  // Выравнивание для std140
	
	veekay::vec3 sun_light_direction;  // Направление направленного света
	float _pad2;  // Выравнивание для std140
	
	veekay::vec3 sun_light_color;  // Цвет направленного света
	float _pad3;  // Выравнивание для std140
	
	uint32_t point_light_count;  // Количество активных точечных источников
	float time;  // Для анимации цвета через sin(time)
	
	uint32_t active_shadow_sources;  // Битовая маска активных источников теней
	float shadow_bias;  // Смещение для расчета теней
	float _pad4;  // Выравнивание для std140
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color;
	float disable_uv_distortion;  // 0.0 = применять искажение UV, 1.0 = не применять
	veekay::vec3 specular_color;
	float enable_color_tinting;  // 1.0 = применять цвет к текстуре, 0.0 = не применять
	float shininess;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Material {
	veekay::graphics::Texture* albedo_texture;
	veekay::graphics::Texture* specular_texture;
	veekay::graphics::Texture* emissive_texture;
	VkSampler albedo_sampler;
	VkSampler specular_sampler;
	VkSampler emissive_sampler;
	uint32_t descriptor_set_index;  // Индекс в массиве material_descriptor_sets
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
	veekay::vec3 specular_color;
	veekay::vec3 shininess;
	uint32_t material_index;  // Индекс материала в массиве materials

};

struct Light {
  veekay::vec3 color;
  veekay::vec3 direction;

  veekay::vec3 ambient;
  veekay::vec3 diffuse;
  veekay::vec3 specular;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};
	veekay::vec3 target = {0.0f, 0.0f, 0.0f};  // Куда смотрит камера
	veekay::vec3 up = {0.0f, 1.0f, 0.0f};  // Вектор вверх

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: Orthographic projection matrix
	veekay::mat4 orthographic(float aspect_ratio) const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;

	// NOTE: Look-At matrix calculation
	veekay::mat4 look_at(const veekay::vec3 at) const;
};

// NOTE: Scene objects
inline namespace {
	//===============================================
	Camera camera{
		.position = {0.0f, 0.0f, 15.0f},
		.target = {0.0f, -5.0f, 0.0f},  // Центр сцены (где находятся объекты)
		.up = {0.0f, 1.0f, 0.0f},
		.fov = 60.0f,
		.near_plane = 0.1f,
		.far_plane = 1000.0f  // Увеличено для большей дальности прорисовки
	};


	// Орбитальные параметры камеры
	float orbit_radius = 0.0f;      // Радиус орбиты (будет вычислен из начальной позиции)
	float orbit_yaw = 0.0f;         // Горизонтальный угол (в радианах)
	float orbit_pitch = 0.0f;       // Вертикальный угол (в радианах)
	//===============================================

	std::vector<Model> models;
	std::vector<PointLight> point_lights;  // Точечные источники света

	// UI variables for additional features
	bool is_animation_paused = false;
	bool is_rotation_reversed = false;
	bool enable_color_tinting = true;  // Включить/выключить наложение цвета на текстуру
	float rotation_speed = 1.0f;      // UI: скорость вращения
	float animation_time = 0.0f;       // Накопленное время анимации
	
	// Lighting parameters
	veekay::vec3 ambient_light_intensity = {0.03f, 0.03f, 0.03f};
	veekay::vec3 sun_light_direction = {0.0f, -1.0f, -0.5f};  // Будет нормализовано при использовании
	veekay::vec3 sun_light_color = {1.0f, 1.0f, 1.0f};
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;
	VkShaderModule light_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;
	std::vector<VkDescriptorSet> material_descriptor_sets;
	std::vector<Material> materials;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;

	Mesh plane_mesh;
	Mesh cube_mesh;

	// Для настройки цвета кубика в прямом эфире
	veekay::vec3 cube_color = {0.0f, 1.0f, 0.735f};
	veekay::vec3 torus_color = {0.0f, 1.0f, 0.35f};

	veekay::graphics::Texture* missing_texture;
	veekay::graphics::Texture* white_texture;
	veekay::graphics::Texture* black_texture;
	VkSampler missing_texture_sampler;

	// Выравнивание размеров для устранения ошибки Vulkan
	uint32_t min_uniform_buffer_offset_alignment;
	uint32_t aligned_model_uniforms_size;


	Mesh sphere_mesh;  // Сфера
	
	veekay::graphics::Buffer* point_lights_buffer;  // Shader-storage буфер для точечных источников
	
	Mesh skybox_mesh;  // Skybox (общий mesh для всех граней)
	Mesh skybox_faces[6];  // Отдельные меши для каждой грани skybox
	uint32_t skybox_model_indices[6] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};  // Индексы моделей skybox
	
	// Shadow mapping
	constexpr uint32_t shadow_map_size = 4096;
	
	// Dynamic Rendering function pointers
	PFN_vkCmdBeginRendering vkCmdBeginRenderingKHR = nullptr;
	PFN_vkCmdEndRendering vkCmdEndRenderingKHR = nullptr;
	struct {
		VkFormat depth_image_format;
		VkImage depth_image;
		VkDeviceMemory depth_image_memory;
		VkImageView depth_image_view;
		VkShaderModule vertex_shader;
		VkDescriptorSetLayout descriptor_set_layout;
		VkDescriptorSet descriptor_set;
		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;
		veekay::graphics::Buffer* uniform_buffer;
		VkSampler sampler;
		veekay::mat4 matrix;
		VkImageLayout current_layout;  // Track current image layout
	} shadow = {};  // Initialize to zero
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::graphics::Texture* load_texture_from_png(VkCommandBuffer cmd, const char* path) {
	std::cerr << "Loading texture: " << path << std::endl;
	std::vector<unsigned char> image;
	unsigned width, height;
	
	std::cerr << "Decoding PNG..." << std::endl;
	unsigned error = lodepng::decode(image, width, height, path);
	if (error != 0) {
		std::cerr << "Failed to load texture from " << path << ": " << lodepng_error_text(error) << std::endl;
		return nullptr;
	}
	
	std::cerr << "PNG decoded: " << width << "x" << height << ", image size: " << image.size() << std::endl;
	
	// Проверяем, что изображение не пустое
	if (image.empty() || width == 0 || height == 0) {
		std::cerr << "Invalid texture dimensions: " << width << "x" << height << std::endl;
		return nullptr;
	}
	
	// Конвертируем RGBA в BGRA для Vulkan формата B8G8R8A8_UNORM
	// В формате B8G8R8A8_UNORM порядок байтов в памяти (little-endian): B, G, R, A
	// В uint32_t (little-endian): младший байт = B, затем G, R, A
	std::cerr << "Converting RGBA to BGRA..." << std::endl;
	std::cerr << "Allocating pixels vector: " << width << " * " << height << " = " << (width * height) << " elements" << std::endl;
	std::vector<uint32_t> pixels(width * height);
	std::cerr << "Pixels vector allocated successfully" << std::endl;
	for (size_t i = 0; i < width * height; ++i) {
		unsigned char r = image[i * 4 + 0];
		unsigned char g = image[i * 4 + 1];
		unsigned char b = image[i * 4 + 2];
		unsigned char a = image[i * 4 + 3];
		// Формат BGRA: B (младший байт), G, R, A (старший байт)
		pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
	}
	
	std::cerr << "Creating texture..." << std::endl;
	try {
		return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_B8G8R8A8_UNORM, pixels.data());
	} catch (const std::exception& e) {
		std::cerr << "Failed to create texture from " << path << ": " << e.what() << std::endl;
		return nullptr;
	}
}

//------------------------------------------------------------
// Функция для создания сэмплера текстуры
VkSampler create_texture_sampler() {
	VkDevice& device = veekay::app.vk_device;
	
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(veekay::app.vk_physical_device, &properties);
	
	VkSamplerCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_TRUE,
		.maxAnisotropy = properties.limits.maxSamplerAnisotropy,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = VK_LOD_CLAMP_NONE,
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE,
	};
	
	VkSampler sampler;
	if (vkCreateSampler(device, &info, nullptr, &sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create texture sampler\n";
		return VK_NULL_HANDLE;
	}
	
	return sampler;
}


Mesh generateSphereMesh(float radius, uint32_t segments) {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	
	// Генерация вершин сферы
	for (uint32_t i = 0; i <= segments; ++i) {
		float theta = float(i) / float(segments) * float(M_PI);  // От 0 до π (вертикальный угол)
		float sin_theta = sinf(theta);
		float cos_theta = cosf(theta);
		
		for (uint32_t j = 0; j <= segments; ++j) {
			float phi = float(j) / float(segments) * 2.0f * float(M_PI);  // От 0 до 2π (горизонтальный угол)
			float sin_phi = sinf(phi);
			float cos_phi = cosf(phi);
			
			// Параметрические уравнения сферы
			float x = radius * sin_theta * cos_phi;
			float y = radius * cos_theta;
			float z = radius * sin_theta * sin_phi;
			
			veekay::vec3 position{x, y, z};
			
			// Нормаль для сферы - это нормализованный вектор от центра к точке
			veekay::vec3 normal = veekay::vec3::normalized(position);
			
			// UV координаты
			veekay::vec2 uv{float(j) / float(segments), float(i) / float(segments)};
			
			vertices.push_back(Vertex{position, normal, uv});
		}
	}
	
	// Генерация индексов для треугольников
	for (uint32_t i = 0; i < segments; ++i) {
		for (uint32_t j = 0; j < segments; ++j) {
			uint32_t current = i * (segments + 1) + j;
			uint32_t next = current + segments + 1;
			
			// Первый треугольник (против часовой стрелки для правильных нормалей)
			indices.push_back(current);
			indices.push_back(next);
			indices.push_back(current + 1);
			
			// Второй треугольник (против часовой стрелки для правильных нормалей)
			indices.push_back(current + 1);
			indices.push_back(next);
			indices.push_back(next + 1);
		}
	}
	
	Mesh mesh;
	mesh.vertex_buffer = new veekay::graphics::Buffer(
		vertices.size() * sizeof(Vertex), vertices.data(),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	
	mesh.index_buffer = new veekay::graphics::Buffer(
		indices.size() * sizeof(uint32_t), indices.data(),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	
	mesh.indices = uint32_t(indices.size());
	
	return mesh;
}

veekay::mat4 Transform::matrix() const {
	// Полная матрица преобразования (scaling + rotation + translation)
	
	auto s = veekay::mat4::scaling(scale);
	
	// Вращение вокруг осей X, Y, Z (в радианах)
	auto rx = veekay::mat4::rotation(veekay::vec3{1.0f, 0.0f, 0.0f}, rotation.x);
	auto ry = veekay::mat4::rotation(veekay::vec3{0.0f, 1.0f, 0.0f}, rotation.y);
	auto rz = veekay::mat4::rotation(veekay::vec3{0.0f, 0.0f, 1.0f}, rotation.z);
	
	auto r = ry * rx * rz;  // Порядок: сначала X, потом Y, потом Z
	
	auto t = veekay::mat4::translation(position);
	
	// Порядок применения: Rotate -> Scale -> Translate
	return t * s * r;
}


// [ r.x  u.x  -f.x  0 ]
// [ r.y  u.y  -f.y  0 ]
// [ r.z  u.z  -f.z  0 ]
// [  0    0     0   1 ]

veekay::mat4 Camera::look_at(const veekay::vec3 at) const {
  const veekay::vec3 forward = veekay::vec3::normalized(position - at);
  constexpr veekay::vec3 world_up = {0, 1, 0};
  const veekay::vec3 right =
      veekay::vec3::normalized(veekay::vec3::cross(forward, world_up));
  const veekay::vec3 up =
      veekay::vec3::normalized(veekay::vec3::cross(right, forward));

  const veekay::mat4 basis = {
      right.x, up.x, -forward.x, 0, right.y, up.y, -forward.y, 0,
      right.z, up.z, -forward.z, 0, 0,       0,    0,          1};

  return veekay::mat4::translation(-position) * basis;
}

veekay::mat4 Camera::view() const {
	return look_at({0, 0, 0});
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto proj = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * proj;
}

// Orthographic projection function for shadow mapping
veekay::mat4 orthographic_projection(float size, float aspect_ratio, float near_plane, float far_plane) {
	veekay::mat4 result{};
	
	float height = size;
	float width = height * aspect_ratio;
	
	float l = -width;   // left
	float r = width;     // right
	float b = -height;   // bottom
	float t = height;    // top
	float n = near_plane; // near
	float f = far_plane;  // far
	
	// Orthographic projection matrix for Vulkan (Z in [0, 1])
	result[0][0] = 2.0f / (r - l);
	result[0][1] = 0.0f;
	result[0][2] = 0.0f;
	result[0][3] = 0.0f;
	
	result[1][0] = 0.0f;
	result[1][1] = 2.0f / (t - b);
	result[1][2] = 0.0f;
	result[1][3] = 0.0f;
	
	result[2][0] = 0.0f;
	result[2][1] = 0.0f;
	result[2][2] = 1.0f / (f - n);  // For Z in [0, 1]
	result[2][3] = 0.0f;
	
	result[3][0] = -(r + l) / (r - l);
	result[3][1] = -(t + b) / (t - b);
	result[3][2] = -n / (f - n);  // Offset for Vulkan
	result[3][3] = 1.0f;
	
	return result;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

// Обновляет позицию камеры на основе орбитальных параметров
void updateCameraPosition();

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device; // Интерфейс для общения с видеокартой
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device; // физические параметры видюхи
	
	// Центр сцены
	camera.target = {0.0f, -5.0f, 0.0f};

	// Инициализация орбитальных параметров камеры из начальной позиции
	//===========================================
	veekay::vec3 offset = camera.position - camera.target;
	orbit_radius = veekay::vec3::length(offset);

	if (orbit_radius < 0.5f) {
		orbit_radius = 15.0f;
		orbit_yaw = 0.0f;
		orbit_pitch = 0.0f;
	} else {
		// Вычисляем углы из начальной позиции камеры
		// Нормализуем offset
		veekay::vec3 normalized_offset = veekay::vec3::normalized(offset);
		orbit_pitch = asinf(normalized_offset.y);
		orbit_yaw = atan2f(normalized_offset.x, normalized_offset.z);
	}
	//===========================================


	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX, // Частота подключения к буферу
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			// Обрезка окна
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 32,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 32,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 32,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 128,  // 32 материалов * 3 текстуры + shadow texture для всех
				}
			};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 32,  // До 32 различных материалов
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				//scene uniforms buffer
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				//model uniforms buffer
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				//point lights buffer
				{
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				//albedo texture - основная текстура
				{
					.binding = 3,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				//specular texture - текстура бликов
				{
					.binding = 4,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				//emissive texture - текстура свечения
				{
					.binding = 5,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				//shadow texture - текстура теней
				{
					.binding = 6,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}


	VkPhysicalDeviceProperties physical_device_properties;
	vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);
	min_uniform_buffer_offset_alignment = static_cast<uint32_t>(
		physical_device_properties.limits.minUniformBufferOffsetAlignment);


	uint32_t model_uniforms_size = sizeof(ModelUniforms);
	aligned_model_uniforms_size = (model_uniforms_size + min_uniform_buffer_offset_alignment - 1) 
		& ~(min_uniform_buffer_offset_alignment - 1);


	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * aligned_model_uniforms_size,
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	// Создаем shader-storage буфер для точечных источников света
	point_lights_buffer = new veekay::graphics::Buffer(
		max_point_lights * sizeof(PointLight),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
		
		uint32_t white_pixels[] = {
			0xffffffff, 0xffffffff,
			0xffffffff, 0xffffffff,
		};
		white_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                             VK_FORMAT_B8G8R8A8_UNORM,
		                                             white_pixels);
		
		uint32_t black_pixels[] = {
			0xff000000, 0xff000000,
			0xff000000, 0xff000000,
		};
		black_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                             VK_FORMAT_B8G8R8A8_UNORM,
		                                             black_pixels);
	}

	// Создаем базовый сэмплер для текстур
	VkSampler default_sampler = create_texture_sampler();
	if (default_sampler == VK_NULL_HANDLE) {
		veekay::app.running = false;
		return;
	}

	// Загружаем текстуры и создаем материалы
	materials.clear();
	
	// Материал 0: 
	veekay::graphics::Texture* geralt_texture = load_texture_from_png(cmd, "./images/tumannosti.png");
	if (!geralt_texture) {
		geralt_texture = load_texture_from_png(cmd, "../testbed/images/geralt.png");
	}
	if (!geralt_texture) {
		std::cerr << "Warning: Could not load geralt.png, using missing texture\n";
		geralt_texture = missing_texture;
	}

	veekay::graphics::Texture* gold_texture = load_texture_from_png(cmd, "./images/gold.png");
	if (!gold_texture) {
		gold_texture = load_texture_from_png(cmd, "images/gold.png");
	}
	if (!gold_texture) {
		gold_texture = load_texture_from_png(cmd, "../testbed/images/gold.png");
	}
	if (!gold_texture) {
		std::cerr << "Warning: Could not load gold.png, using missing texture\n";
		gold_texture = missing_texture;
	}

	veekay::graphics::Texture* wood_texture = load_texture_from_png(cmd, "./images/wood.png");
	if (!wood_texture) {
		wood_texture = load_texture_from_png(cmd, "images/wood.png");
	}
	if (!wood_texture) {
		wood_texture = load_texture_from_png(cmd, "../testbed/images/wood.png");
	}
	if (!wood_texture) {
		std::cerr << "Warning: Could not load wood.png, using missing texture\n";
		wood_texture = missing_texture;
	}


	veekay::graphics::Texture* zoltan_texture = load_texture_from_png(cmd, "./images/tum_yel.png");

	if (!zoltan_texture) {
		std::cerr << "Warning: Could not load zoltan.png, using missing texture\n";
		zoltan_texture = missing_texture;
	}

	// Загружаем текстуры skybox (6 граней куба)
	veekay::graphics::Texture* skybox_px = load_texture_from_png(cmd, "./images/skybox_cube/px.png");
	if (!skybox_px) skybox_px = load_texture_from_png(cmd, "images/skybox_cube/px.png");
	if (!skybox_px) skybox_px = load_texture_from_png(cmd, "../testbed/images/skybox_cube/px.png");
	if (!skybox_px) {
		std::cerr << "Warning: Could not load px.png\n";
		skybox_px = missing_texture;
	}

	veekay::graphics::Texture* skybox_nx = load_texture_from_png(cmd, "./images/skybox_cube/nx.png");
	if (!skybox_nx) skybox_nx = load_texture_from_png(cmd, "images/skybox_cube/nx.png");
	if (!skybox_nx) skybox_nx = load_texture_from_png(cmd, "../testbed/images/skybox_cube/nx.png");
	if (!skybox_nx) {
		skybox_nx = missing_texture;
	}

	veekay::graphics::Texture* skybox_py = load_texture_from_png(cmd, "./images/skybox_cube/py.png");
	if (!skybox_py) skybox_py = load_texture_from_png(cmd, "images/skybox_cube/py.png");
	if (!skybox_py) skybox_py = load_texture_from_png(cmd, "../testbed/images/skybox_cube/py.png");
	if (!skybox_py) {
		std::cerr << "Warning: Could not load py.png\n";
		skybox_py = missing_texture;
	}

	veekay::graphics::Texture* skybox_ny = load_texture_from_png(cmd, "./images/skybox_cube/ny.png");
	if (!skybox_ny) skybox_ny = load_texture_from_png(cmd, "images/skybox_cube/ny.png");
	if (!skybox_ny) skybox_ny = load_texture_from_png(cmd, "../testbed/images/skybox_cube/ny.png");
	if (!skybox_ny) {
		std::cerr << "Warning: Could not load ny.png\n";
		skybox_ny = missing_texture;
	}

	veekay::graphics::Texture* skybox_pz = load_texture_from_png(cmd, "./images/skybox_cube/pz.png");
	if (!skybox_pz) skybox_pz = load_texture_from_png(cmd, "images/skybox_cube/pz.png");
	if (!skybox_pz) skybox_pz = load_texture_from_png(cmd, "../testbed/images/skybox_cube/pz.png");
	if (!skybox_pz) {
		std::cerr << "Warning: Could not load pz.png\n";
		skybox_pz = missing_texture;
	}

	veekay::graphics::Texture* skybox_nz = load_texture_from_png(cmd, "./images/skybox_cube/nz.png");
	if (!skybox_nz) skybox_nz = load_texture_from_png(cmd, "images/skybox_cube/nz.png");
	if (!skybox_nz) skybox_nz = load_texture_from_png(cmd, "../testbed/images/skybox_cube/nz.png");
	if (!skybox_nz) {
		std::cerr << "Warning: Could not load nz.png\n";
		skybox_nz = missing_texture;
	}

	// Материал 0: gold_texture
	materials.emplace_back(Material{
		.albedo_texture = gold_texture,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = default_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 0
	});

	// Материал 1: geralt_texture
	materials.emplace_back(Material{
		.albedo_texture = geralt_texture,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = default_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 1
	});

	// Материал 2: wood_texture
	materials.emplace_back(Material{
		.albedo_texture = wood_texture,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = default_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 2
	});

	// Материал 3: gold_texture (для множества торов)
	materials.emplace_back(Material{
		.albedo_texture = gold_texture,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = default_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 3
	});

	materials.emplace_back(Material{
		.albedo_texture = zoltan_texture,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = default_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 1
	});

	// Материалы для skybox (6 граней куба)
	// Создаем сэмплер с CLAMP_TO_EDGE для skybox
	VkSampler skybox_sampler = create_texture_sampler();
		vkDestroySampler(device, skybox_sampler, nullptr);
	VkSamplerCreateInfo skybox_sampler_info{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0f,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = 0.0f,
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE,
	};
	if (vkCreateSampler(device, &skybox_sampler_info, nullptr, &skybox_sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create skybox sampler\n";
		skybox_sampler = default_sampler;
	}

	// Материал 5: skybox_px (правая грань, X+)
	materials.emplace_back(Material{
		.albedo_texture = skybox_px,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = skybox_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 5
	});

	// Материал 6: skybox_nx (левая грань, X-)
	materials.emplace_back(Material{
		.albedo_texture = skybox_nx,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = skybox_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 6
	});

	// Материал 7: skybox_py (верхняя грань, Y+)
	materials.emplace_back(Material{
		.albedo_texture = skybox_py,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = skybox_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 7
	});

	// Материал 8: skybox_ny (нижняя грань, Y-)
	materials.emplace_back(Material{
		.albedo_texture = skybox_ny,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = skybox_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 8
	});

	// Материал 9: skybox_pz (передняя грань, Z+)
	materials.emplace_back(Material{
		.albedo_texture = skybox_pz,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = skybox_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 9
	});

	// Материал 10: skybox_nz (задняя грань, Z-)
	materials.emplace_back(Material{
		.albedo_texture = skybox_nz,
		.specular_texture = nullptr,
		.emissive_texture = nullptr,
		.albedo_sampler = skybox_sampler,
		.specular_sampler = VK_NULL_HANDLE,
		.emissive_sampler = VK_NULL_HANDLE,
		.descriptor_set_index = 10
	});

	// Создаем дескрипторные наборы для каждого материала
	material_descriptor_sets.resize(materials.size());
	
	for (size_t i = 0; i < materials.size(); ++i) {
		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		if (vkAllocateDescriptorSets(device, &alloc_info, &material_descriptor_sets[i]) != VK_SUCCESS) {
			std::cerr << "Failed to allocate material descriptor set " << i << "\n";
			veekay::app.running = false;
			return;
		}

		// Записываем дескрипторы для этого материала
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = aligned_model_uniforms_size,
			},
			{
				.buffer = point_lights_buffer->buffer,
				.offset = 0,
				.range = max_point_lights * sizeof(PointLight),
			},
		};

		// Image info для текстур (используем missing_texture если текстура не задана)
		VkImageView albedo_view;
		if (!materials[i].albedo_texture) {
			albedo_view = missing_texture->view;
		} else if (materials[i].albedo_texture == missing_texture) {
			albedo_view = missing_texture->view;
		} else {
			albedo_view = materials[i].albedo_texture->view;
		}
		
		VkDescriptorImageInfo albedo_image_info{
			.sampler = materials[i].albedo_sampler,
			.imageView = albedo_view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		VkDescriptorImageInfo specular_image_info{
			.sampler = materials[i].specular_sampler != VK_NULL_HANDLE ? materials[i].specular_sampler : default_sampler,
			.imageView = materials[i].specular_texture ? materials[i].specular_texture->view : white_texture->view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		VkDescriptorImageInfo emissive_image_info{
			.sampler = materials[i].emissive_sampler != VK_NULL_HANDLE ? materials[i].emissive_sampler : default_sampler,
			.imageView = materials[i].emissive_texture ? materials[i].emissive_texture->view : black_texture->view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		// Shadow texture will be updated after shadow resources are created
		// Skip shadow texture binding if resources not ready (will be updated later)
		bool shadow_ready = shadow.depth_image_view != VK_NULL_HANDLE && shadow.sampler != VK_NULL_HANDLE;
		
		VkDescriptorImageInfo shadow_image_info{};
		if (shadow_ready) {
			shadow_image_info = {
				.sampler = shadow.sampler,
				.imageView = shadow.depth_image_view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
		}

		// Only update shadow texture binding if shadow resources are ready
		size_t write_count = shadow_ready ? 7 : 6; // With or without shadow texture

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_descriptor_sets[i],
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_descriptor_sets[i],
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_descriptor_sets[i],
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_descriptor_sets[i],
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &albedo_image_info,
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_descriptor_sets[i],
				.dstBinding = 4,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &specular_image_info,
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_descriptor_sets[i],
				.dstBinding = 5,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &emissive_image_info,
			},
		};

		// Add shadow texture binding only if ready
		if (shadow_ready) {
			VkWriteDescriptorSet shadow_write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material_descriptor_sets[i],
				.dstBinding = 6,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &shadow_image_info,
			};
			write_infos[6] = shadow_write;
		}

		vkUpdateDescriptorSets(device, write_count, write_infos, 0, nullptr);
	}

{
	VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = aligned_model_uniforms_size,
			},
			{
				.buffer = point_lights_buffer->buffer,
				.offset = 0,
				.range = max_point_lights * sizeof(PointLight),
			},
		};

		bool shadow_ready = shadow.depth_image_view != VK_NULL_HANDLE && shadow.sampler != VK_NULL_HANDLE;
		
		VkDescriptorImageInfo shadow_image_info{};
		if (shadow_ready) {
			shadow_image_info = {
				.sampler = shadow.sampler,
				.imageView = shadow.depth_image_view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
		}
		size_t write_count = shadow_ready ? 4 : 3; // With or without shadow texture

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
		};

		// Add shadow texture binding only if ready
		if (shadow_ready) {
			VkWriteDescriptorSet shadow_write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 6,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &shadow_image_info,
			};
			write_infos[3] = shadow_write;
			std::cout<<"shadow ready";
		}

		vkUpdateDescriptorSets(device, write_count, write_infos, 0, nullptr);
	}

	// Note: Material descriptor sets with shadow texture are updated after shadow sampler is created
	// (see code after shadow sampler creation in initialize())

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20,
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}


	sphere_mesh = generateSphereMesh(1.0f, 32);  // Радиус 1.0, 32 сегмента

	{
		std::vector<Vertex> vertices = {
	    // Front face (Z+) - pz.png
	    {{-1.0f, -1.0f, +1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},   // нижний левый
	    {{+1.0f, -1.0f, +1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},   // нижний правый
	    {{+1.0f, +1.0f, +1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},   // верхний правый
	    {{-1.0f, +1.0f, +1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},   // верхний левый

	    // Back face (Z-) - nz.png
	    {{+1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},  // нижний левый
	    {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},  // нижний правый
	    {{-1.0f, +1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},  // верхний правый
	    {{+1.0f, +1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},  // верхний левый

	    // Right face (X+) - px.png
	    {{+1.0f, -1.0f, +1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},   // нижний левый
	    {{+1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},   // нижний правый
	    {{+1.0f, +1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},   // верхний правый
	    {{+1.0f, +1.0f, +1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},   // верхний левый

	    // Left face (X-) - nx.png
	    {{-1.0f, -1.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},  // нижний левый
	    {{-1.0f, -1.0f, +1.0f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},  // нижний правый
	    {{-1.0f, +1.0f, +1.0f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},  // верхний правый
	    {{-1.0f, +1.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}}, // верхний левый

	    // Top face (Y+) - py.png
	    {{-1.0f, +1.0f, +1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},   // нижний левый
	    {{+1.0f, +1.0f, +1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},   // нижний правый
	    {{+1.0f, +1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},   // верхний правый
	    {{-1.0f, +1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},  // верхний левый

	    // Bottom face (Y-) - ny.png
	    {{-1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}}, // нижний левый
	    {{+1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}}, // нижний правый
	    {{+1.0f, -1.0f, +1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},  // верхний правый
	    {{-1.0f, -1.0f, +1.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},  // верхний левый
	};


		std::vector<uint32_t> indices = {
			// Front
			0, 1, 2, 2, 3, 0,
			// Back
			4, 5, 6, 6, 7, 4,
			// Right
			8, 9, 10, 10, 11, 8,
			// Left
			12, 13, 14, 14, 15, 12,
			// Bottom
			16, 17, 18, 18, 19, 16,
			// Top
			20, 21, 22, 22, 23, 20,
		};

		skybox_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		skybox_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		skybox_mesh.indices = uint32_t(indices.size());

		// Создаем отдельные меши для каждой грани (для использования разных материалов)
		// Каждая грань имеет локальные индексы 0,1,2,3
		// Для skybox (камера внутри) используем обратный порядок индексов, чтобы грани были видны изнутри
		uint32_t face_indices[6] = {0, 1, 2, 2, 3, 0};  // Обратный порядок для внутренних граней

		// Front face (Z+)
		skybox_faces[0].vertex_buffer = new veekay::graphics::Buffer(
			4 * sizeof(Vertex), &vertices[0],
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		skybox_faces[0].index_buffer = new veekay::graphics::Buffer(
			6 * sizeof(uint32_t), face_indices,
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		skybox_faces[0].indices = 6;

		// Back face (Z-)
		skybox_faces[1].vertex_buffer = new veekay::graphics::Buffer(
			4 * sizeof(Vertex), &vertices[4],
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		skybox_faces[1].index_buffer = new veekay::graphics::Buffer(
			6 * sizeof(uint32_t), face_indices,
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		skybox_faces[1].indices = 6;

		// Right face (X+)
		skybox_faces[2].vertex_buffer = new veekay::graphics::Buffer(
			4 * sizeof(Vertex), &vertices[8],
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		skybox_faces[2].index_buffer = new veekay::graphics::Buffer(
			6 * sizeof(uint32_t), face_indices,
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		skybox_faces[2].indices = 6;

		// Left face (X-)
		skybox_faces[3].vertex_buffer = new veekay::graphics::Buffer(
			4 * sizeof(Vertex), &vertices[12],
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		skybox_faces[3].index_buffer = new veekay::graphics::Buffer(
			6 * sizeof(uint32_t), face_indices,
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		skybox_faces[3].indices = 6;

		// Bottom face (Y-)
		skybox_faces[4].vertex_buffer = new veekay::graphics::Buffer(
			4 * sizeof(Vertex), &vertices[16],
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		skybox_faces[4].index_buffer = new veekay::graphics::Buffer(
			6 * sizeof(uint32_t), face_indices,
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		skybox_faces[4].indices = 6;

		// Top face (Y+)
		skybox_faces[5].vertex_buffer = new veekay::graphics::Buffer(
			4 * sizeof(Vertex), &vertices[20],
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		skybox_faces[5].index_buffer = new veekay::graphics::Buffer(
			6 * sizeof(uint32_t), face_indices,
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		skybox_faces[5].indices = 6;
	}

	// NOTE: Add models to scene
	models.clear();



	// Куб
	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {1.0f, -0.5f, 0.0f},
			.scale = {2.0f, 3.0f, 1.0f},
			.rotation = {0.0f, -1.0f, 0.0f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},  // Белый цвет для текстуры
		.specular_color = veekay::vec3{0.8f, 0.8f, 0.8f},
		.shininess = veekay::vec3{64.0f, 64.0f, 64.0f},
		.material_index = 1
	});

	// Куб
	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {-1.0f,  -0.5f, 0.0f},
			.scale = {2.0f, 2.0f, 2.0f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.specular_color = veekay::vec3{0.8f, 0.8f, 0.8f},
		.shininess = veekay::vec3{64.0f, 64.0f, 64.0f},
		.material_index = 4
	});

	// Пол (сетка из досок) - деревянная текстура
	// Создаем сетку из множества маленьких досок для пиксельного вида
	float floor_size_x = 25.0f;  // Размер пола по X
	float floor_size_z = 25.0f;  // Размер пола по Z
	float board_width = 2.0f;     // Ширина одной доски
	float board_length = 2.0f;   // Длина одной доски
	float board_height = 1.0f;    // Высота доски (тонкая)

	float pos_board_height = 0.5f;
	
	// Вычисляем количество досок
	int boards_x = static_cast<int>(floorf(floor_size_x / board_width));
	int boards_z = static_cast<int>(floorf(floor_size_z / board_length));

	float start_x = -board_width / 2.0f * static_cast<float>(boards_x - 1) / 2.0f;
	float start_z = -board_length / 2.0f * static_cast<float>(boards_z - 1) / 2.0f;

	for (int x = 0; x < boards_x; ++x) {
		for (int z = 0; z < boards_z; ++z) {
			// Смещаем каждую доску на board_width/2 для правильного соприкосновения
			float pos_x = start_x + board_width / 2.0f * static_cast<float>(x);
			float pos_z = start_z + board_length / 2.0f * static_cast<float>(z);

			models.emplace_back(Model{
				.mesh = cube_mesh,
				.transform = Transform{
					.position = {pos_x, pos_board_height, pos_z},
					.scale = {board_width, board_height, board_length},
				},
				.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},  // Белый цвет для текстуры (wood_texture)
				.specular_color = veekay::vec3{0.5f, 0.5f, 0.5f},
				.shininess = veekay::vec3{16.0f, 16.0f, 16.0f},
				.material_index = 2
			});
		}
	}



	// Две сферы с текстурой gold
	models.emplace_back(Model{
		.mesh = sphere_mesh,
		.transform = Transform{
			.position = {1.0f, -3.0f, 0.0f},
			.scale = {20.0f, 20.0f, 20.0f},
			.rotation = {0.0f, 0.0f, 0.0f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.specular_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.shininess = veekay::vec3{32.0f, 32.0f, 32.0f},
		.material_index = 0
	});

	models.emplace_back(Model{
		.mesh = sphere_mesh,
		.transform = Transform{
			.position = {-1.0f, -3.0f, 0.0f},
			.scale = {20.0f, 20.0f, 20.0f},
			.rotation = {0.0f, 0.0f, 0.0f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},  // Белый цвет для текстуры
		.specular_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.shininess = veekay::vec3{32.0f, 32.0f, 32.0f},
		.material_index = 0  // Материал с gold_texture
	});

	// Добавляем skybox (6 граней)
	// Для look-at камеры skybox должен быть в (0,0,0), так как view matrix уже компенсирует translation
	// Материалы: 5=px, 6=nx, 7=py, 8=ny, 9=pz, 10=nz
	Transform skybox_transform{
		.position = {0.0f, 0.0f, 0.0f},  // Skybox в начале координат
		.scale = {500.0f, 500.0f, 500.0f},  // Большой масштаб для skybox
		.rotation = {0.0f, 0.0f, 0.0f},  // Без поворота
	};

	// Front (Z+) - pz.png, материал 9
	skybox_model_indices[0] = models.size();
	models.emplace_back(Model{
		.mesh = skybox_faces[0],
		.transform = skybox_transform,
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.specular_color = veekay::vec3{0.0f, 0.0f, 0.0f},
		.shininess = veekay::vec3{0.0f, 0.0f, 0.0f},
		.material_index = 9  // pz
	});

	// Back (Z-) - nz.png, материал 10
	skybox_model_indices[1] = models.size();
	models.emplace_back(Model{
		.mesh = skybox_faces[1],
		.transform = skybox_transform,
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.specular_color = veekay::vec3{0.0f, 0.0f, 0.0f},
		.shininess = veekay::vec3{0.0f, 0.0f, 0.0f},
		.material_index = 10  // nz
	});

	// Right (X+) - px.png, материал 5
	skybox_model_indices[2] = models.size();
	models.emplace_back(Model{
		.mesh = skybox_faces[2],
		.transform = skybox_transform,
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.specular_color = veekay::vec3{0.0f, 0.0f, 0.0f},
		.shininess = veekay::vec3{0.0f, 0.0f, 0.0f},
		.material_index = 5  // px
	});

	// Left (X-) - nx.png, материал 6
	skybox_model_indices[3] = models.size();
	models.emplace_back(Model{
		.mesh = skybox_faces[3],
		.transform = skybox_transform,
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.specular_color = veekay::vec3{0.0f, 0.0f, 0.0f},
		.shininess = veekay::vec3{0.0f, 0.0f, 0.0f},
		.material_index = 6  // nx
	});

	// Bottom (Y-) - ny.png, материал 8
	skybox_model_indices[4] = models.size();
	models.emplace_back(Model{
		.mesh = skybox_faces[4],
		.transform = skybox_transform,
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.specular_color = veekay::vec3{0.0f, 0.0f, 0.0f},
		.shininess = veekay::vec3{0.0f, 0.0f, 0.0f},
		.material_index = 8  // ny
	});

	// Top (Y+) - py.png, материал 7
	skybox_model_indices[5] = models.size();
	models.emplace_back(Model{
		.mesh = skybox_faces[5],
		.transform = skybox_transform,
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.specular_color = veekay::vec3{0.0f, 0.0f, 0.0f},
		.shininess = veekay::vec3{0.0f, 0.0f, 0.0f},
		.material_index = 7  // py
	});


	// Инициализация 3 точечных источников света по умолчанию
	point_lights.clear();
	point_lights.emplace_back(PointLight{
		.position = {-5.0f, -5.0f, 5.0f},
		.intensity = 30.0f,
		.color = {1.0f, 1.0f, 1.0f},  // Белый свет
		._pad0 = 0.0f
	});
	point_lights.emplace_back(PointLight{
		.position = {5.0f, -5.0f, 5.0f},
		.intensity = 30.0f,
		.color = {1.0f, 1.0f, 1.0f},  // Белый свет
		._pad0 = 0.0f
	});
	point_lights.emplace_back(PointLight{
		.position = {0.0f, 5.0f, 0.0f},
		.intensity = 30.0f,
		.color = {1.0f, 1.0f, 1.0f},  // Белый свет
		._pad0 = 0.0f
	});

	// ============================================================
	// Shadow mapping initialization
	// ============================================================
	{
		// Select optimal depth format
		VkFormat candidates[] = {
			VK_FORMAT_D32_SFLOAT,
			VK_FORMAT_D32_SFLOAT_S8_UINT,
			VK_FORMAT_D24_UNORM_S8_UINT,
			VK_FORMAT_D16_UNORM,
		};

		shadow.depth_image_format = VK_FORMAT_UNDEFINED;

		for (const auto& f : candidates) {
			VkFormatProperties properties;
			vkGetPhysicalDeviceFormatProperties(physical_device, f, &properties);

			if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT &&
			    properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
				shadow.depth_image_format = f;
				break;
			}
		}

		if (shadow.depth_image_format == VK_FORMAT_UNDEFINED) {
			std::cerr << "Failed to find suitable depth format for shadow map\n";
			veekay::app.running = false;
			return;
		}
		
		// Debug: выводим выбранный формат
		std::cout << "Shadow map format: ";
		if (shadow.depth_image_format == VK_FORMAT_D32_SFLOAT) {
			std::cout << "D32_SFLOAT\n";
		} else if (shadow.depth_image_format == VK_FORMAT_D24_UNORM_S8_UINT) {
			std::cout << "D24_UNORM_S8_UINT\n";
		} else if (shadow.depth_image_format == VK_FORMAT_D16_UNORM) {
			std::cout << "D16_UNORM\n";
		} else if (shadow.depth_image_format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
			std::cout << "D32_SFLOAT_S8_UINT\n";
		} else {
			std::cout << "Unknown format\n";
		}

		// Create depth image
		VkImageCreateInfo image_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = shadow.depth_image_format,
			.extent = {shadow_map_size, shadow_map_size, 1},
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		};

		if (vkCreateImage(device, &image_info, nullptr, &shadow.depth_image) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow depth image\n";
			veekay::app.running = false;
			return;
		}

		// Allocate memory
		VkMemoryRequirements mem_requirements;
		vkGetImageMemoryRequirements(device, shadow.depth_image, &mem_requirements);

		VkPhysicalDeviceMemoryProperties mem_properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

		uint32_t memory_type_index = UINT32_MAX;
		for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
			if ((mem_requirements.memoryTypeBits & (1 << i)) &&
			    (mem_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				memory_type_index = i;
				break;
			}
		}

		if (memory_type_index == UINT32_MAX) {
			std::cerr << "Failed to find suitable memory type for shadow depth image\n";
			veekay::app.running = false;
			return;
		}

		VkMemoryAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = mem_requirements.size,
			.memoryTypeIndex = memory_type_index,
		};

		if (vkAllocateMemory(device, &alloc_info, nullptr, &shadow.depth_image_memory) != VK_SUCCESS) {
			std::cerr << "Failed to allocate memory for shadow depth image\n";
			veekay::app.running = false;
			return;
		}

		if (vkBindImageMemory(device, shadow.depth_image, shadow.depth_image_memory, 0) != VK_SUCCESS) {
			std::cerr << "Failed to bind memory to shadow depth image\n";
			veekay::app.running = false;
			return;
		}

		// Create image view
		VkImageViewCreateInfo view_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = shadow.depth_image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = shadow.depth_image_format,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

		if (vkCreateImageView(device, &view_info, nullptr, &shadow.depth_image_view) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow depth image view\n";
			veekay::app.running = false;
			return;
		}

		// Initialize layout tracking
		shadow.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	// Create shadow pipeline
	{
		// Load shadow vertex shader
		shadow.vertex_shader = loadShaderModule("./shaders/shadow.vert.spv");
		if (!shadow.vertex_shader) {
			std::cerr << "Failed to load shadow vertex shader\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = shadow.vertex_shader,
			.pName = "main",
		};

		// Vertex input - only position
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position),
			},
		};

		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_FRONT_BIT,  // Front-face culling
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = true,
			.lineWidth = 1.0f,
		};

		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport shadow_viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(shadow_map_size),
			.height = static_cast<float>(shadow_map_size),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D shadow_scissor{
			.offset = {0, 0},
			.extent = {shadow_map_size, shadow_map_size},
		};

		VkPipelineViewportStateCreateInfo shadow_viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &shadow_viewport,
			.scissorCount = 1,
			.pScissors = &shadow_scissor,
		};

		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		};

		VkDynamicState dyn_states[] = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
			VK_DYNAMIC_STATE_DEPTH_BIAS,
		};

		VkPipelineDynamicStateCreateInfo dyn_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = sizeof(dyn_states) / sizeof(dyn_states[0]),
			.pDynamicStates = dyn_states,
		};

		VkPipelineRenderingCreateInfo format_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.depthAttachmentFormat = shadow.depth_image_format,
		};

		// Create descriptor set layout
		VkDescriptorSetLayoutBinding bindings[] = {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			},
			{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			},
		};

		VkDescriptorSetLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
			.pBindings = bindings,
		};

		if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &shadow.descriptor_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow descriptor set layout\n";
			veekay::app.running = false;
			return;
		}

		// Allocate descriptor set
		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &shadow.descriptor_set_layout,
		};

		if (vkAllocateDescriptorSets(device, &alloc_info, &shadow.descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate shadow descriptor set\n";
			veekay::app.running = false;
			return;
		}

		// Create pipeline layout
		VkPipelineLayoutCreateInfo pipeline_layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &shadow.descriptor_set_layout,
		};

		if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &shadow.pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow pipeline layout\n";
			veekay::app.running = false;
			return;
		}

		// Create graphics pipeline
		VkGraphicsPipelineCreateInfo pipeline_info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.pNext = &format_info,
			.stageCount = 1,
			.pStages = &stage_info,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &shadow_viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.pDynamicState = &dyn_state_info,
			.layout = shadow.pipeline_layout,
		};

		if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info, nullptr, &shadow.pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	// Create shadow sampler
	{
		VkSamplerCreateInfo sampler_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.compareEnable = VK_FALSE,  // Отключено для ручного сравнения глубины
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.0f,
			.maxLod = VK_LOD_CLAMP_NONE,
			.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		};

		if (vkCreateSampler(device, &sampler_info, nullptr, &shadow.sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow sampler\n";
			veekay::app.running = false;
			return;
		}
	}

	// Update material descriptor sets with shadow texture now that shadow resources are ready
	// This MUST be done after shadow.depth_image_view and shadow.sampler are created
	if (shadow.depth_image_view != VK_NULL_HANDLE && shadow.sampler != VK_NULL_HANDLE) {
		VkDescriptorImageInfo shadow_image_info{
			.sampler = shadow.sampler,
			.imageView = shadow.depth_image_view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		// Update all material descriptor sets
		if (!material_descriptor_sets.empty()) {
			for (size_t i = 0; i < material_descriptor_sets.size(); ++i) {
				if (material_descriptor_sets[i] != VK_NULL_HANDLE) {
					VkWriteDescriptorSet write_info{
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = material_descriptor_sets[i],
						.dstBinding = 6,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						.pImageInfo = &shadow_image_info,
					};

					vkUpdateDescriptorSets(device, 1, &write_info, 0, nullptr);
				}
			}
		}

		// Also update the main descriptor set if it exists
		if (descriptor_set != VK_NULL_HANDLE) {
			VkWriteDescriptorSet write_info{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 6,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &shadow_image_info,
			};

			vkUpdateDescriptorSets(device, 1, &write_info, 0, nullptr);
		}
	}

	// Create shadow uniform buffer
	{
		shadow.uniform_buffer = new veekay::graphics::Buffer(
			sizeof(veekay::mat4), nullptr,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

		// Update descriptor set with buffers
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = shadow.uniform_buffer->buffer,
				.offset = 0,
				.range = sizeof(veekay::mat4),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = aligned_model_uniforms_size,
			},
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = shadow.descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = shadow.descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// Load Dynamic Rendering functions
	{
		// Try KHR versions first (for Vulkan 1.2 with extension)
		vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRendering>(
			vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR"));
		vkCmdEndRenderingKHR = reinterpret_cast<PFN_vkCmdEndRendering>(
			vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR"));
		
		if (!vkCmdBeginRenderingKHR || !vkCmdEndRenderingKHR) {
			// Try core versions (Vulkan 1.3+)
			vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRendering>(
				vkGetDeviceProcAddr(device, "vkCmdBeginRendering"));
			vkCmdEndRenderingKHR = reinterpret_cast<PFN_vkCmdEndRendering>(
				vkGetDeviceProcAddr(device, "vkCmdEndRendering"));
		}
		
		if (!vkCmdBeginRenderingKHR || !vkCmdEndRenderingKHR) {
			std::cerr << "Failed to load Dynamic Rendering functions\n";
			veekay::app.running = false;
			return;
		}
	}

	// Обновляем позицию камеры на основе вычисленных орбитальных параметров
	updateCameraPosition();

}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;
	delete white_texture;
	delete black_texture;

	// Удаляем текстуры материалов
	// Собираем уникальные текстуры и сэмплеры для удаления
	// Используем set для избежания дубликатов (несколько материалов могут использовать одну текстуру)
	std::set<veekay::graphics::Texture*> textures_to_delete;
	std::set<VkSampler> samplers_to_destroy;
	
	for (auto& material : materials) {
		if (material.albedo_texture && material.albedo_texture != missing_texture) {
			textures_to_delete.insert(material.albedo_texture);
		}
		if (material.specular_texture && material.specular_texture != missing_texture) {
			textures_to_delete.insert(material.specular_texture);
		}
		if (material.emissive_texture && material.emissive_texture != missing_texture) {
			textures_to_delete.insert(material.emissive_texture);
		}
		if (material.albedo_sampler != VK_NULL_HANDLE && material.albedo_sampler != missing_texture_sampler) {
			samplers_to_destroy.insert(material.albedo_sampler);
		}
		if (material.specular_sampler != VK_NULL_HANDLE && material.specular_sampler != missing_texture_sampler) {
			samplers_to_destroy.insert(material.specular_sampler);
		}
		if (material.emissive_sampler != VK_NULL_HANDLE && material.emissive_sampler != missing_texture_sampler) {
			samplers_to_destroy.insert(material.emissive_sampler);
		}
	}
	
	// Удаляем уникальные текстуры
	for (auto* tex : textures_to_delete) {
		delete tex;
	}
	
	// Удаляем уникальные сэмплеры
	for (auto sampler : samplers_to_destroy) {
		vkDestroySampler(device, sampler, nullptr);
	}
	
	materials.clear();

	delete plane_mesh.vertex_buffer;
	delete plane_mesh.index_buffer;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete skybox_mesh.index_buffer;
	delete skybox_mesh.vertex_buffer;

	// Удаляем меши для каждой грани skybox
	for (int i = 0; i < 6; ++i) {
		delete skybox_faces[i].index_buffer;
		delete skybox_faces[i].vertex_buffer;
	}

	delete point_lights_buffer;
	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	// Дескрипторные наборы освобождаются автоматически при уничтожении pool
	material_descriptor_sets.clear();
	
	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);

	// Cleanup shadow resources
	vkDestroyPipeline(device, shadow.pipeline, nullptr);
	vkDestroyPipelineLayout(device, shadow.pipeline_layout, nullptr);
	vkDestroyShaderModule(device, shadow.vertex_shader, nullptr);
	vkDestroyDescriptorSetLayout(device, shadow.descriptor_set_layout, nullptr);
	delete shadow.uniform_buffer;
	vkDestroySampler(device, shadow.sampler, nullptr);
	vkDestroyImageView(device, shadow.depth_image_view, nullptr);
	vkDestroyImage(device, shadow.depth_image, nullptr);
	vkFreeMemory(device, shadow.depth_image_memory, nullptr);
}

void updateCameraPosition() {
	// Ограничиваем pitch, чтобы камера не переворачивалась
	constexpr float min_pitch = -float(M_PI) / 2.0f + 0.1f;  // -90 градусов
	constexpr float max_pitch = float(M_PI) / 2.0f - 0.1f;   //  +90 градусов
	orbit_pitch = std::max(min_pitch, std::min(max_pitch, orbit_pitch));
	
	// Сферические координаты: x = r * cos(pitch) * cos(y	aw)
	//                        y = r * sin(pitch)
	//                        z = r * cos(pitch) * sin(yaw)

	float cos_pitch = cosf(orbit_pitch);
	camera.position.x = camera.target.x + orbit_radius * cosf(orbit_yaw) * cos_pitch;
	camera.position.y = camera.target.y + orbit_radius * sinf(orbit_pitch);
	camera.position.z = camera.target.z + orbit_radius * sinf(orbit_yaw) * cos_pitch;
}


void update(double time) {
	ImGui::Begin("Controls:");
	ImGui::ColorEdit3("Cube Color", &cube_color.x);

	
	// Checkbox для включения/выключения наложения цвета на текстуру
	if (ImGui::Checkbox("Enable Color Tinting", &enable_color_tinting)) {
		// Флаг обновляется автоматически
	}

	// Slider для FOV камеры

	if (ImGui::SliderFloat("FOV", &camera.fov, 30.0f, 120.0f)) {
	}
	
	// Slider для скорости вращения
	if (ImGui::SliderFloat("Rotation Speed", &rotation_speed, 0.0f, 100.0f)) {
	}
	ImGui::SameLine();

	// Управление анимацией
	if (ImGui::Checkbox("Pause Animation", &is_animation_paused)) {
		// Пауза/возобновление анимации
	}
	
	if (ImGui::Checkbox("Reverse Rotation", &is_rotation_reversed)) {
		// Реверс направления вращения
	}
	
	ImGui::End();

	// UI для управления точечными источниками света
	ImGui::Begin("Point Lights");
	ImGui::Text("Active lights: %zu", point_lights.size());

	for (size_t i = 0; i < point_lights.size(); ++i) {
		ImGui::PushID(static_cast<int>(i));
		ImGui::Text("Light %zu", i + 1);
		
		ImGui::SliderFloat3("Position", &point_lights[i].position.x, -10.0f, 10.0f);
		ImGui::ColorEdit3("Color", &point_lights[i].color.x);
		ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 100.0f);
		
		if (ImGui::Button("Remove") && point_lights.size() > 1) {
			point_lights.erase(point_lights.begin() + i);
			ImGui::PopID();
			break;
		}
		
		ImGui::Separator();
		ImGui::PopID();
	}
	
	if (ImGui::Button("Add Light") && point_lights.size() < max_point_lights) {
		point_lights.emplace_back(PointLight{
			.position = {0.0f, 2.0f, 0.0f},
			.intensity = 50.0f,
			.color = {1.0f, 1.0f, 1.0f},
			._pad0 = 0.0f
		});
	}

	ImGui::End();

	// UI для управления рассеянным и направленным освещением
	ImGui::Begin("Lighting");
	ImGui::Text("Ambient Light");
	ImGui::ColorEdit3("Ambient Intensity", &ambient_light_intensity.x);
	
	ImGui::Separator();
	ImGui::Text("Directional Light (Sun)");
	ImGui::SliderFloat3("Direction", &sun_light_direction.x, -200.0f, 200.0f);
	ImGui::ColorEdit3("Sun Color", &sun_light_color.x);
	
	// Кнопка для сброса направления к значению по умолчанию
	if (ImGui::Button("Reset Direction")) {
		sun_light_direction = {0.0f, -1.0f, -0.5f};
	}
	
	ImGui::End();

	// Анимация вращения
	// Обновляем накопленное время анимации (если не на паузе)
	if (!is_animation_paused) {
		animation_time = time * rotation_speed * (is_rotation_reversed ? -1.0f : 1.0f);
	}

	//АНИМАЦИЯ
	if (!models.empty()) {
		//тор
		models[0].transform.rotation.y = animation_time;
		// //куб
		// models[4].transform.rotation.y = animation_time * -1.0f;

		// Маленькие торы по кругу на полу
		float floor_size_x = 25.0f;
		float floor_size_z = 25.0f;
		float board_width = 2.0f;
		float board_length = 2.0f;
		int boards_x = static_cast<int>(floorf(floor_size_x / board_width));
		int boards_z = static_cast<int>(floorf(floor_size_z / board_length));
		int num_floor_boards = boards_x * boards_z;
		int first_small_torus_index = 3 + num_floor_boards;  // После тора (0), куба (1) и всех досок (2 + num_floor_boards)
		
		// Анимируем все маленькие торы
		for (int i = first_small_torus_index; i < static_cast<int>(models.size()); ++i) {
			// Пропускаем skybox грани, чтобы они не вращались
			bool is_skybox_face = false;
			for (int j = 0; j < 6; ++j) {
				if (skybox_model_indices[j] != UINT32_MAX && i == static_cast<int>(skybox_model_indices[j])) {
					is_skybox_face = true;
					break;
				}
			}
			if (is_skybox_face) continue;

			models[i].transform.rotation.y = animation_time * 2.0f;  // Вращаем в 2 раза быстрее
			// models[i].transform.rotation.x = animation_time * 0.5f;  // Добавляем небольшое вращение по X
		}
	}

	if (!ImGui::IsWindowHovered()) {
		using namespace veekay::input;

		constexpr float rotation_speed = 0.02f;  // Скорость вращения камеры
		constexpr float zoom_speed = 0.5f;        // Скорость приближения/отдаления

		// W/S - изменение pitch (вертикальный угол)
		if (keyboard::isKeyDown(keyboard::Key::w))
			orbit_pitch += rotation_speed;

		if (keyboard::isKeyDown(keyboard::Key::s))
			orbit_pitch -= rotation_speed;

		// A/D - изменение yaw (горизонтальный угол)
		if (keyboard::isKeyDown(keyboard::Key::a))
			orbit_yaw -= rotation_speed;

		if (keyboard::isKeyDown(keyboard::Key::d))
			orbit_yaw += rotation_speed;

		// Q/Z - изменение радиуса (приближение/отдаление)
		if (keyboard::isKeyDown(keyboard::Key::q)) {
			orbit_radius -= zoom_speed;
			if (orbit_radius < 1.0f) orbit_radius = 1.0f;  // Минимальный радиус
		}

		if (keyboard::isKeyDown(keyboard::Key::z)) {
			orbit_radius += zoom_speed;
			if (orbit_radius > 500.0f) orbit_radius = 500.0f;  // Максимальный радиус увеличен
		}
	}

	// Камера обновляется каждый кадр
	updateCameraPosition();

	const float aspect_ratio = static_cast<float>(veekay::app.window_width) / 
	                            static_cast<float>(veekay::app.window_height);
	
	// Обновление shader-storage буфера с данными точечных источников
	uint32_t active_light_count = static_cast<uint32_t>(point_lights.size());
	if (active_light_count > max_point_lights) {
		active_light_count = max_point_lights;
	}
	
	// Копирование данных источников в буфер
	PointLight* lights_buffer = static_cast<PointLight*>(point_lights_buffer->mapped_region);
	for (size_t i = 0; i < active_light_count; ++i) {
		lights_buffer[i] = point_lights[i];
	}
	
	// Compute shadow projection matrix
	if (shadow.uniform_buffer) {
		
		Camera light_camera;
		veekay::vec3 sun_direction_normalized = veekay::vec3::normalized(sun_light_direction);
		light_camera.position = -sun_direction_normalized * 15.0f;
		light_camera.target = {0.0f, 0.0f, 0.0f};
		
		shadow.matrix = light_camera.look_at({0, 0, 0}) * orthographic_projection(4.0f, aspect_ratio, 0.0001f, 50.0f);

		// Write to shadow uniform buffer
		if (shadow.uniform_buffer->mapped_region) {
			*reinterpret_cast<veekay::mat4*>(shadow.uniform_buffer->mapped_region) = shadow.matrix;
		}
	}

	// Добавление времени в uniform
	veekay::mat4 shadow_proj_matrix = shadow.uniform_buffer ? shadow.matrix : veekay::mat4::identity();
	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
		.shadow_projection = shadow_proj_matrix,  // Должна быть сразу после view_projection!
		.view_position = camera.position,
		._pad0 = 0.0f,
		.ambient_light_intensity = ambient_light_intensity,
		._pad1 = 0.0f,
		.sun_light_direction = veekay::vec3::normalized(sun_light_direction),
		._pad2 = 0.0f,
		.sun_light_color = sun_light_color,
		._pad3 = 0.0f,
		.point_light_count = active_light_count,
		.time = static_cast<float>(time),  // Передача времени для анимации цвета
		.active_shadow_sources = 1u,  // Включаем тени для направленного света (бит 0)
		.shadow_bias = 0.004f,
		._pad4 = 0.0f
	};

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];

		uniforms.model = model.transform.matrix();
		
		// Проверяем, есть ли у материала текстура (не missing_texture)
		uint32_t material_idx = model.material_index < materials.size() ? model.material_index : 0;
		bool has_texture = (materials[material_idx].albedo_texture && 
		                   materials[material_idx].albedo_texture != missing_texture);
		
		if (i == 1) {
			uniforms.albedo_color = cube_color;
		} else if (i == 0) {
			// Тор - используем torus_color
			uniforms.albedo_color = torus_color;
		} else {
			// Остальные объекты - используем цвет модели
			uniforms.albedo_color = model.albedo_color;
		}
		
		// Если наложение цвета выключено и есть текстура, используем белый цвет
		// (чтобы текстура была видна полностью без тонирования)
		if (!enable_color_tinting && has_texture) {
			uniforms.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f};
		}
		
		// Передача материалов
		uniforms.specular_color = model.specular_color;
		uniforms.shininess = model.shininess.x;  // Используем x компонент как shininess
		
		// Управление наложением цвета на текстуру
		if (model.material_index >= 5 && model.material_index <= 10) {
			// Для skybox всегда отключаем наложение цвета
			uniforms.enable_color_tinting = 0.0f;
		} else {
			// Для остальных объектов используем глобальный флаг
			uniforms.enable_color_tinting = enable_color_tinting ? 1.0f : 0.0f;
		}
		
		// Отключаем искажение UV для материала доски (material_index = 2) и skybox (5-10)
		if (model.material_index == 2 || (model.material_index >= 5 && model.material_index <= 10)) {
			uniforms.disable_uv_distortion = 1.0f;  // Отключаем искажение UV
		} else {
			uniforms.disable_uv_distortion = 0.0f;  // Включаем искажение UV
		}
	}

	// Для skybox позиция всегда (0,0,0) - view matrix look-at уже компенсирует translation камеры
	// Обновляем uniforms для всех граней skybox
	for (int j = 0; j < 6; ++j) {
		if (skybox_model_indices[j] != UINT32_MAX && skybox_model_indices[j] < models.size()) {
			uint32_t idx = skybox_model_indices[j];
		// Обновляем uniform для skybox грани
		model_uniforms[idx].model = models[idx].transform.matrix();
		// Для skybox отключаем искажение UV и тонирование
		model_uniforms[idx].disable_uv_distortion = 1.0f;
		model_uniforms[idx].enable_color_tinting = 0.0f;  // Отключаем наложение цвета для skybox
		// Убеждаемся, что albedo_color белый для skybox
		model_uniforms[idx].albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f};
		}
	}

	// Исправление ошибки Vulkan о выравнивании
	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;
	// NOTE: Copy model uniforms with proper alignment
	uint8_t* buffer_ptr = static_cast<uint8_t*>(model_uniforms_buffer->mapped_region);
	for (size_t i = 0; i < model_uniforms.size(); ++i) {
		std::memcpy(buffer_ptr + i * aligned_model_uniforms_size, 
		           &model_uniforms[i], sizeof(ModelUniforms));
	}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);
     Light light;

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	// Render shadow map
	if (vkCmdBeginRenderingKHR && vkCmdEndRenderingKHR && 
	    shadow.depth_image != VK_NULL_HANDLE && 
	    shadow.depth_image_view != VK_NULL_HANDLE &&
	    shadow.pipeline != VK_NULL_HANDLE) {
		// Debug: проверяем, что shadow pass выполняется
		static bool first_shadow_pass = true;
		if (first_shadow_pass) {
			std::cout << "Shadow pass started\n";
			first_shadow_pass = false;
		}
		// Barrier: transition shadow depth image to DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		VkAccessFlags src_access = 0;
		
		if (shadow.current_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			src_access = VK_ACCESS_SHADER_READ_BIT;
		}
		
		VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = src_access,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = shadow.current_layout,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.image = shadow.depth_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

		vkCmdPipelineBarrier(cmd,
			src_stage,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);

		// Begin Dynamic Rendering
		VkRenderingAttachmentInfo depth_attachment{
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = shadow.depth_image_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = { .depthStencil = {1.0f, 0} },
		};

		VkRenderingInfo rendering_info{
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
			.renderArea = {0, 0, shadow_map_size, shadow_map_size},
			.layerCount = 1,
			.pDepthAttachment = &depth_attachment,
		};

		vkCmdBeginRenderingKHR(cmd, &rendering_info);

		// Set viewport and scissor
		VkViewport viewport{
			.x = 0.0f, .y = 0.0f,
			.width = float(shadow_map_size),
			.height = float(shadow_map_size),
			.minDepth = 0.0f, .maxDepth = 1.0f,
		};

		vkCmdSetViewport(cmd, 0, 1, &viewport);

		VkRect2D scissor = {0, 0, shadow_map_size, shadow_map_size};
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		// Set depth bias
		vkCmdSetDepthBias(cmd, 2.5f, 0.0f, 3.0f);

		// Bind shadow pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline);

		VkDeviceSize zero_offset = 0;
		VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
		VkBuffer current_index_buffer = VK_NULL_HANDLE;

		// Draw all models except skybox
		for (size_t i = 0, n = models.size(); i < n; ++i) {
			// Skip skybox faces
			bool is_skybox_face = false;
			for (int j = 0; j < 6; ++j) {
				if (skybox_model_indices[j] != UINT32_MAX && i == skybox_model_indices[j]) {
					is_skybox_face = true;
					break;
				}
			}
			if (is_skybox_face) continue;

			const Model& model = models[i];
			const Mesh& mesh = model.mesh;

			if (!mesh.vertex_buffer || !mesh.index_buffer) continue;

			if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}

			if (current_index_buffer != mesh.index_buffer->buffer) {
				current_index_buffer = mesh.index_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}

			uint32_t offset = i * aligned_model_uniforms_size;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
			                        shadow.pipeline_layout, 0, 1,
			                        &shadow.descriptor_set, 1, &offset);

			vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
		}

		vkCmdEndRenderingKHR(cmd);

		// Barrier: transition shadow depth image to SHADER_READ_ONLY_OPTIMAL
		barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);

		// Update layout tracking
		shadow.current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

        // veekay::vec3 lightDir = normalize(-light.direction);
		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	// Сначала рендерим skybox (все 6 граней)
	for (int j = 0; j < 6; ++j) {
		if (skybox_model_indices[j] != UINT32_MAX && skybox_model_indices[j] < models.size()) {
			uint32_t idx = skybox_model_indices[j];
			const Model& model = models[idx];
			const Mesh& mesh = model.mesh;

			if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}

			if (current_index_buffer != mesh.index_buffer->buffer) {
				current_index_buffer = mesh.index_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}

			uint32_t offset = idx * aligned_model_uniforms_size;
			uint32_t material_idx = model.material_index < materials.size() ? model.material_index : 0;
			VkDescriptorSet material_set = material_descriptor_sets[material_idx];
			
			if (material_set != VK_NULL_HANDLE) {
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
				                        0, 1, &material_set, 1, &offset);
				vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
			}
		}
	}

	// Затем рендерим все остальные модели
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		// Пропускаем skybox грани, так как они уже отрендерены
		bool is_skybox_face = false;
		for (int j = 0; j < 6; ++j) {
			if (skybox_model_indices[j] != UINT32_MAX && i == skybox_model_indices[j]) {
				is_skybox_face = true;
				break;
			}
		}
		if (is_skybox_face) continue;
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		// Исправление ошибки Vulkan о выравнивании
		uint32_t offset = i * aligned_model_uniforms_size;

		// Используем дескрипторный набор материала модели
		uint32_t material_idx = model.material_index < materials.size() ? model.material_index : 0;
		VkDescriptorSet material_set = material_descriptor_sets[material_idx];
		
		// Проверяем, что дескрипторный набор не нулевой
		if (material_set == VK_NULL_HANDLE) {
			std::cerr << "ERROR: material_set is VK_NULL_HANDLE for model " << i << ", material_idx = " << material_idx << "\n";
		}

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &material_set, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,

	});
}
