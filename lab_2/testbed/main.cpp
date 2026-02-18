#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <glm/gtc/constants.hpp>


#define _USE_MATH_DEFINES
#include <algorithm>
#include <math.h>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

#include <lodepng.h>
#include <ranges>
//((N - 1) % 4) + 1    (15-1)%4+1 = 1
/*
 Матрица камеры рассчитывается с помощью матрицы трансформации камеры
 (положения и ориентации/поворота). Должны быть реализованы следующие
 компоненты освещения: рассеянное, направленное и точечные источники света.
 Точечные источники света должны терять свою интенсивность по закону обратных квадратов
 */


namespace {
	constexpr uint32_t max_models = 1024;

	struct Vertex {
		veekay::vec3 position;//// Позиция в 3D пространстве
		veekay::vec3 normal;//для освещения
		veekay::vec2 uv;//для текстур
		// NOTE: You can add more attributes
	};
	struct SceneUniforms {
		veekay::mat4 view_projection;        // 64 bytes
		veekay::vec3 camera_pos;             // 12 bytes
		float _padding0;             // 4 bytes (vec3 → 16 bytes)
		// ДОБАВЛЯЕМ padding до 256 байт для базового выравнивания
		float _padding1[44];         // 176 bytes дополнительного padding
	};

	// ИТОГО: 64 + 16 + 176 = 256 bytes
	// LightingUniforms - Variant 1: point lights with inverse square law
	struct LightingUniforms {
		veekay::vec3 directional_light_dir;
		float directional_light_intensity;

		veekay::vec3 directional_light_color;
		float _padding1;

		veekay::vec3 point_light_pos;
		float point_light_intensity;
		veekay::vec3 point_light_color;
		float _padding2;

		float point_light_constant;
		float point_light_linear;
		float point_light_quadratic;
		float _padding3;

		veekay::vec3 ambient_light_color;
		float ambient_light_intensity;

		float _padding4[32];  // Padding to 256 bytes
	};
	struct ModelUniforms {
		veekay::mat4 model;                // 0–63
		veekay::vec3 albedo_color;         // 64–75
		float shininess;
		float _pad0[3];                    // 80–91 → выравнивание до 96
		veekay::vec3 specular_color;       // 96–107
		float _pad1;                       // 108–111 → завершаем 16-байт блок

		// Padding до 256
		float _pad_to_256[37];
	};

	//Хранит геометрию объекта (вершины + индексы).
	struct Mesh {
		veekay::graphics::Buffer* vertex_buffer;
		veekay::graphics::Buffer* index_buffer; //Массив индексов, указывающих на вершины:
		uint32_t indices;
	};
	//положение, масштаб и вращение объекта.
	struct Transform {
		veekay::vec3 position = {};
		veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
		veekay::vec3 rotation = {};

		// NOTE: Model matrix (translation, rotation and scaling)
		veekay::mat4 matrix() const;

		// Обратная матрица: S⁻¹ * R⁻¹ * T⁻¹
		veekay::mat4 inverse_matrix() const;
	};

	struct Model {
		Mesh mesh;
		Transform transform;
		veekay::vec3 albedo_color;
		veekay::vec3 specular_color;
		float shininess;
	};

	struct Camera {
		constexpr static float default_fov = 60.0f;
		constexpr static float default_near_plane = 0.01f;
		constexpr static float default_far_plane = 100.0f;

		veekay::vec3 position = {};
		veekay::vec3 rotation = {};

		float fov = default_fov;
		float near_plane = default_near_plane;
		float far_plane = default_far_plane;

		// NOTE: View matrix of camera (inverse of a transform)
		veekay::mat4 view() const;

		// NOTE: View and projection composition
		veekay::mat4 view_projection(float aspect_ratio) const;
	};

	void createConeGeometry1111(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
					   int segments = 16, float radius = 0.5f, float height = 1.0f) {
		vertices.clear();
		indices.clear();

		if (segments < 3) segments = 3;//segments - количество треугольников для аппроксимации круга (чем больше, тем круглее)

		vertices.reserve(segments + 2);
		indices.reserve(segments * 6);

		std::cout << "Creating cone with " << segments << " segments" << std::endl;

		// Вершины основания (по кругу в плоскости XZ)
		for (int i = 0; i < segments; ++i) {
			float angle = 2.0f * float(M_PI) * i / segments;
			float x = radius * cosf(angle);
			float z = radius * sinf(angle);

			// Нормаль основания направлена вниз
			vertices.push_back({{x, -height/2.0f, z}, {0.0f, -1.0f, 0.0f},
							   {0.5f + 0.5f * cosf(angle), 0.5f + 0.5f * sinf(angle)}});
		}

		//Вершина конуса (центр вверху)
		vertices.push_back({{0.0f, height/2.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.5f, 1.0f}});

		//Центр основания (для треугольников основания)
		vertices.push_back({{0.0f, -height/2.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.5f, 0.5f}});

		//Боковые грани (треугольники от вершины к основанию)
		for (int i = 0; i < segments; ++i) {
			int next_i = (i + 1) % segments;
			indices.push_back(segments);     // Вершина конуса
			indices.push_back(i);            // Текущая вершина основания
			indices.push_back(next_i);       // Следующая вершина основания
		}

		//Основание (треугольники от центра к краям)
		for (int i = 0; i < segments; ++i) {
			int next_i = (i + 1) % segments;
			indices.push_back(segments + 1); // Центр основания
			indices.push_back(next_i);       // Следующая вершина
			indices.push_back(i);            // Текущая вершина
		}

		std::cout << "Cone geometry created: " << vertices.size() << " vertices, "
				  << indices.size() << " indices" << std::endl;
	}



	void createTorusGeometry(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
		// Generate torus vertices and indices
		constexpr int major_segments = 20;  // u segments
		constexpr int minor_segments = 10;  // v segments
		constexpr float major_radius = 1.0f; // R
		constexpr float minor_radius = 0.4f; // r

		vertices.clear();
		indices.clear();

		// Generate vertices (double loop)
		for (int i = 0; i < major_segments; ++i) {
			float u = 2.0f * M_PI * i / major_segments;
			for (int j = 0; j < minor_segments; ++j) {
				float v = 2.0f * M_PI * j / minor_segments;

				float cos_u = cosf(u);
				float sin_u = sinf(u);
				float cos_v = cosf(v);
				float sin_v = sinf(v);

				Vertex vertex;
				vertex.position.x = (major_radius + minor_radius * cos_v) * cos_u;
				vertex.position.y = (major_radius + minor_radius * cos_v) * sin_u;
				vertex.position.z = minor_radius * sin_v;

				// Calculate normal for torus
				vertex.normal.x = cos_v * cos_u;
				vertex.normal.y = cos_v * sin_u;
				vertex.normal.z = sin_v;

				// UV coordinates
				vertex.uv.x = float(i) / major_segments;
				vertex.uv.y = float(j) / minor_segments;

				vertices.push_back(vertex);
			}
		}

		// Generate indices for triangles
		for (int i = 0; i < major_segments; ++i) {
			int i_next = (i + 1) % major_segments;
			for (int j = 0; j < minor_segments; ++j) {
				int j_next = (j + 1) % minor_segments;

				int v0 = i * minor_segments + j;
				int v1 = i_next * minor_segments + j;
				int v2 = i_next * minor_segments + j_next;
				int v3 = i * minor_segments + j_next;

				// First triangle
				indices.push_back(v0);
				indices.push_back(v1);
				indices.push_back(v2);

				// Second triangle
				indices.push_back(v0);
				indices.push_back(v2);
				indices.push_back(v3);
			}
		}

		std::cout << "Generated torus with " << vertices.size() << " vertices and "
				  << indices.size() << " indices" << std::endl;
	}

	//масштабирование
	veekay::mat4 scaling_matrix(const veekay::vec3& scale) {
		return {
			scale.x, 0.0f,    0.0f,    0.0f,
			0.0f,    scale.y, 0.0f,    0.0f,
			0.0f,    0.0f,    scale.z, 0.0f,
			0.0f,    0.0f,    0.0f,    1.0f
		};
	}
	//поворот по y
	veekay::mat4 rotation_y_matrix(float angle) {
		float sina = sinf(angle);
		float cosa = cosf(angle);

		return {
			cosa,  0.0f, sina,  0.0f,
			0.0f,  1.0f, 0.0f,  0.0f,
			-sina, 0.0f, cosa,  0.0f,
			0.0f,  0.0f, 0.0f,  1.0f
		};
	}


	// NOTE: Scene objects
	inline namespace {
		Camera camera{
			.position = {0.0f, 0.0f, -6.0f},

		};

		std::vector<Model> models;
		Mesh cone_mesh;
	}

	// NOTE: Vulkan objects
	inline namespace {
		VkShaderModule vertex_shader_module;
		VkShaderModule fragment_shader_module;

		VkDescriptorPool descriptor_pool;//// Пул памяти для дескрипторов
		VkDescriptorSetLayout descriptor_set_layout;//// Макет расположения данных
		VkDescriptorSet descriptor_set;// // Набор конкретных данных для шейдеров

		VkPipelineLayout pipeline_layout;//Макет конвейера
		VkPipeline pipeline;//конвейер рендеринга

		veekay::graphics::Buffer* scene_uniforms_buffer;// Данные сцены (камера)
		veekay::graphics::Buffer* model_uniforms_buffer;// Данные моделей (трансформации)
		veekay::graphics::Buffer* lighting_uniforms_buffer;
		//veekay::graphics::Texture* missing_texture;
		VkSampler missing_texture_sampler;
	}

	veekay::mat4 Transform::matrix() const {
		// Матрица масштабирования
		auto scale_mat = scaling_matrix(scale);
		// Матрица вращения (вокруг оси Y)
		auto rot_mat = rotation_y_matrix(rotation.y);
		// Матрица перемещения
		auto trans_mat = veekay::mat4::translation(position);
		// Комбинируем: T * R * S
		return  scale_mat * rot_mat * trans_mat;
	}
	veekay::mat4 inv_scaling_matrix(const veekay::vec3& scale) {
		return {
			1/scale.x, 0.0f,    0.0f,    0.0f,
			0.0f,    1/scale.y, 0.0f,    0.0f,
			0.0f,    0.0f,    1/scale.z, 0.0f,
			0.0f,    0.0f,    0.0f,    1.0f
		};
	}
	veekay::mat4 Transform::inverse_matrix() const {
		// Обратное масштабирование: S⁻¹
		auto inv_scale_mat = inv_scaling_matrix(veekay::vec3(scale));

		// Обратное вращение через транспонирование: R⁻¹ = Rᵀ
		auto rot_x = veekay::mat4::rotation(veekay::vec3{1.0, 0.0, 0.0}, rotation.x);
		auto rot_y = veekay::mat4::rotation(veekay::vec3{0.0, 1.0, 0.0}, rotation.y);
		auto rot_z = veekay::mat4::rotation(veekay::vec3{0.0, 0.0, 1.0}, rotation.z);
		auto rot_mat = rot_z * rot_y * rot_x;
		auto inv_rot_mat = veekay::mat4::transpose(rot_mat);

		// Обратное перемещение: T⁻¹
		auto inv_trans_mat = veekay::mat4::translation(-position);
		// Комбинируем: S⁻¹ * R⁻¹ * T⁻¹, но матрицы перемножаются справа налево
		return inv_trans_mat * inv_rot_mat  * inv_scale_mat;
	}

	veekay::mat4 Camera::view() const{
		Transform temp_transform;
		temp_transform.position = position;
		temp_transform.rotation = rotation;
		return temp_transform.inverse_matrix();
	}
	//Матрица вида преобразует мировые координаты в координаты камеры. По сути, она "перемещает" весь мир так, чтобы камера оказалась в начале координат
	veekay::mat4 Camera::view_projection(float aspect_ratio) const {
		auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
		return view() * projection;
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

		//Загружает и настраивает шейдеры (вершинный и фрагментный).
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

	void initialize(VkCommandBuffer cmd) {
		VkDevice& device = veekay::app.vk_device;
		VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
		{ // NOTE: Build graphics pipeline
			vertex_shader_module = loadShaderModule("../shaders/shader.vert.spv");
			if (!vertex_shader_module) {
				std::cerr << "Failed to load Vulkan vertex shader from file\n";
				veekay::app.running = false;
				return;
			}

			fragment_shader_module = loadShaderModule("../shaders/shader.frag.spv");
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
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
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

			// СОЗДАНИЕ DESCRIPTOR POOL
			{
				VkDescriptorPoolSize pools[] = {
					{
						.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.descriptorCount = 12,
					},
					{
						.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
						.descriptorCount = 8,
					}
				};

				VkDescriptorPoolCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
					.maxSets = 1,
					.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
					.pPoolSizes = pools,
				};

				if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan descriptor pool\n";
					veekay::app.running = false;
					return;
				}
			}

			{
				VkDescriptorSetLayoutBinding bindings[] = {
					{
						.binding = 0, // Scene uniforms
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, // И vertex И fragment
					},
					{
						.binding = 1, // Lighting uniforms
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, // Только fragment
					},
					{
						.binding = 2, // Model uniforms (DYNAMIC)
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, // И vertex И fragment!
					},
				};

				VkDescriptorSetLayoutCreateInfo info{
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
					.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
					.pBindings = bindings,
				};

				if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan descriptor set layout\n";
					veekay::app.running = false;
					return;
				}
			}
			// 5. ВЫДЕЛЕНИЕ DESCRIPTOR SET (ЭТОГО ФРАГМЕНТА НЕ БЫЛО!)
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
			// 6. СОЗДАНИЕ UNIFORM БУФЕРОВ (ОБНОВЛЕНО)
			// В разделе создания uniform buffers ИСПРАВЬТЕ:
			{
				// Scene buffer (камера + проекция)
				scene_uniforms_buffer = new veekay::graphics::Buffer(
					sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

				// Lighting buffer
				lighting_uniforms_buffer = new veekay::graphics::Buffer(
					sizeof(LightingUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

				// Model buffer (dynamic) - ИСПРАВЛЕНО: добавлен DYNAMIC_BIT
				model_uniforms_buffer = new veekay::graphics::Buffer(
					max_models * sizeof(ModelUniforms),
					nullptr,
					VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); // ← ДОБАВЬТЕ DYNAMIC_BIT

				std::cout << "Uniform buffers created successfully\n";
			}
			std::cout << "=== ALIGNMENT CHECK ===" << std::endl;
			std::cout << "SceneUniforms: " << sizeof(SceneUniforms) << " bytes, aligned: "
					  << (sizeof(SceneUniforms) % 256 == 0 ? "YES" : "NO") << std::endl;
			std::cout << "LightingUniforms: " << sizeof(LightingUniforms) << " bytes, aligned: "
					  << (sizeof(LightingUniforms) % 256 == 0 ? "YES" : "NO") << std::endl;
			std::cout << "ModelUniforms: " << sizeof(ModelUniforms) << " bytes, aligned: "
					  << (sizeof(ModelUniforms) % 64 == 0 ? "YES" : "NO") << std::endl;

			// Проверка dynamic offsets
			for (size_t i = 0; i < models.size(); ++i) {
				uint32_t offset = static_cast<uint32_t>(i * sizeof(ModelUniforms));
				if (offset % 64 != 0) {
					std::cerr << "WARNING: Model offset " << i << " not aligned: " << offset << std::endl;
				}
			}

			// 7. ИНИЦИАЛИЗАЦИЯ UNIFORM ДАННЫХ (ОБНОВЛЕНО)
			{
				// 7a. Scene Uniforms (камера + проекция)
				SceneUniforms scene_uniforms = {};
				scene_uniforms.view_projection = camera.view_projection(
					float(veekay::app.window_width) / float(veekay::app.window_height));
				scene_uniforms.camera_pos = camera.position;

				memcpy(scene_uniforms_buffer->mapped_region, &scene_uniforms, sizeof(SceneUniforms));
				std::cout << "Scene uniforms initialized\n";

				// 7b. Lighting Uniforms (ОТДЕЛЬНАЯ инициализация освещения)
				LightingUniforms lighting_uniforms = {};

				// Directional Light (солнечный свет)
				lighting_uniforms.directional_light_dir = veekay::vec3{-0.5f, -1.0f, -0.5f};
				lighting_uniforms.directional_light_intensity = 0.8f;
				lighting_uniforms.directional_light_color = veekay::vec3{1.0f, 1.0f, 0.9f};

				// Point Light (точечный свет)
				lighting_uniforms.point_light_pos = veekay::vec3{2.0f, 3.0f, 2.0f};
				lighting_uniforms.point_light_intensity = 1.0f;
				lighting_uniforms.point_light_color = veekay::vec3{1.0f, 0.8f, 0.6f};

				// Point Light Attenuation (inverse square law)
				lighting_uniforms.point_light_constant = 1.0f;
				lighting_uniforms.point_light_linear = 0.09f;
				lighting_uniforms.point_light_quadratic = 0.032f;

				// Ambient Light (рассеянный свет)
				lighting_uniforms.ambient_light_color = veekay::vec3{1.0f, 1.0f, 1.0f};
				lighting_uniforms.ambient_light_intensity = 0.8f;

				memcpy(lighting_uniforms_buffer->mapped_region, &lighting_uniforms, sizeof(LightingUniforms));
				std::cout << "Lighting uniforms initialized\n";

				// 7c. Model Uniforms (инициализация по умолчанию)
				ModelUniforms default_model = {};
				default_model.model = veekay::mat4::identity(); // Identity matrix
				default_model.albedo_color = veekay::vec3{0.8f, 0.2f, 0.2f}; // Красный

				// Заполняем буфер моделей значениями по умолчанию
				for (int i = 0; i < max_models; ++i) {
					memcpy((char*)model_uniforms_buffer->mapped_region + i * sizeof(ModelUniforms),
						   &default_model, sizeof(ModelUniforms));
				}
				std::cout << "Model uniforms initialized\n";
			}

			// 8. ОБНОВЛЕНИЕ DESCRIPTOR SET (ОБНОВЛЕНО)
			{
				VkDescriptorBufferInfo buffer_infos[] = {
					{ // binding 0 - Scene uniforms
						.buffer = scene_uniforms_buffer->buffer,
						.offset = 0,
						.range = sizeof(SceneUniforms),
					},
					{ // binding 1 - Lighting uniforms (НОВЫЙ!)
						.buffer = lighting_uniforms_buffer->buffer,
						.offset = 0,
						.range = sizeof(LightingUniforms),
					},
					{ // binding 2 - Model uniforms (dynamic)
						.buffer = model_uniforms_buffer->buffer,
						.offset = 0,
						.range = sizeof(ModelUniforms),
					},
				};

				VkWriteDescriptorSet write_infos[] = {
					{ // Scene
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = descriptor_set,
						.dstBinding = 0,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.pBufferInfo = &buffer_infos[0],
					},
					{ // Lighting (НОВЫЙ!)
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = descriptor_set,
						.dstBinding = 1,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.pBufferInfo = &buffer_infos[1],
					},
					{ // Models
						.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
						.dstSet = descriptor_set,
						.dstBinding = 2,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
						.pBufferInfo = &buffer_infos[2],
					},
				};
				vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
										   write_infos, 0, nullptr);
				std::cout << "Descriptor sets updated - " << sizeof(write_infos) / sizeof(write_infos[0])
						  << " write operations" << std::endl;

			}

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

		// 10. СОЗДАНИЕ ГЕОМЕТРИИ ТОРУСА
		{
			std::vector<Vertex> cone_vertices;
			std::vector<uint32_t> cone_indices;
			createTorusGeometry(cone_vertices, cone_indices);

			cone_mesh.vertex_buffer = new veekay::graphics::Buffer(
				cone_vertices.size() * sizeof(Vertex), cone_vertices.data(),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

			cone_mesh.index_buffer = new veekay::graphics::Buffer(
				cone_indices.size() * sizeof(uint32_t), cone_indices.data(),
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

			cone_mesh.indices = uint32_t(cone_indices.size());
			std::cout << "Torus geometry created: " << cone_vertices.size()
					  << " vertices, " << cone_indices.size() << " indices\n";
		}

		// 11. СОЗДАНИЕ МОДЕЛЕЙ КОНУСОВ
		{
			// Create static cone models
			std::vector<veekay::vec3> cone_positions = {
				{-3.0f, 0.0f, 0.0f},
				{0.0f, 0.0f, 0.0f},
				{3.0f, 0.0f, 0.0f},
				{-1.5f, 0.0f, 2.0f},
				{1.5f, 0.0f, 2.0f}
			};

			std::vector<veekay::vec3> cone_scales = {
				{0.8f, 1.2f, 0.8f},
				{1.0f, 1.5f, 1.0f},
				{0.6f, 0.9f, 0.6f},
				{0.7f, 1.1f, 0.7f},
				{0.9f, 1.3f, 0.9f}
			};

			for (size_t i = 0; i < cone_positions.size(); ++i) {
				Transform cone_transform;
				cone_transform.position = cone_positions[i];
				cone_transform.scale = cone_scales[i];
				cone_transform.rotation = {0.0f, 0.0f, 0.0f};

				models.emplace_back(Model{
					.mesh = cone_mesh,
					.transform = cone_transform,
					.albedo_color = veekay::vec3{0.8f, 0.2f, 0.2f},
					.specular_color = veekay::vec3{1.0f, 1.0f, 1.0f},
					.shininess = 256.0f
				});
			}

			std::cout << "Created " << cone_positions.size() << " static cones\n";
			std::cout << "Total models in scene: " << models.size() << std::endl;
		}

		std::cout << "=== INITIALIZATION COMPLETE ===" << std::endl;
		std::cout << "Scene uniforms size: " << sizeof(SceneUniforms) << " bytes" << std::endl;
		std::cout << "Lighting uniforms size: " << sizeof(LightingUniforms) << " bytes" << std::endl;
		std::cout << "Model uniforms size: " << sizeof(ModelUniforms) << " bytes" << std::endl;
	}

	// NOTE: Destroy resources here, do not cause leaks in your program!
	void shutdown() {
		VkDevice& device = veekay::app.vk_device;

		vkDestroySampler(device, missing_texture_sampler, nullptr);
		//delete missing_texture;

		delete cone_mesh.vertex_buffer;
		delete cone_mesh.index_buffer;

		delete model_uniforms_buffer;
		delete scene_uniforms_buffer;
		delete lighting_uniforms_buffer;

		vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
		vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	}

	void update(double time) {

		// УПРАВЛЕНИЕ КАМЕРОЙ
		if (!ImGui::GetIO().WantCaptureMouse) {
			// Вращение камеры при зажатой правой кнопке мыши
			if (veekay::input::mouse::isButtonDown(veekay::input::mouse::Button::right)) {
				auto mouse_delta = veekay::input::mouse::cursorDelta();

				// Вращение камеры
				camera.rotation.y += mouse_delta.x * 0.005f;
				camera.rotation.x += mouse_delta.y * 0.005f;

				// Ограничение угла наклона
				camera.rotation.x = std::clamp(camera.rotation.x, -1.5f, 1.5f);
			}
		}

		// Движение камеры WASD (всегда работает)
		// Calculate forward and right vectors from rotation
		veekay::vec3 forward_direction;
		forward_direction.x = sin(camera.rotation.y) * cos(camera.rotation.x);
		forward_direction.y = sin(camera.rotation.x);
		forward_direction.z = cos(camera.rotation.y) * cos(camera.rotation.x);
		forward_direction = veekay::vec3::normalized(forward_direction);

		const veekay::vec3 world_up = {0.0f, 1.0f, 0.0f};
		veekay::vec3 right_direction = veekay::vec3::normalized(veekay::vec3::cross(forward_direction, world_up));

		float speed = 0.0045 * static_cast<float>(time);

		// Same movement controls for both modes
		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::w))
			camera.position += forward_direction * speed;

		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::s))
			camera.position -= forward_direction * speed;

		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::d))
			camera.position -= right_direction * speed;

		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::a))
			camera.position += right_direction * speed;

		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::q))
			camera.position -= world_up * speed;

		if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::e))
			camera.position += world_up * speed;

		// UI ДЛЯ УПРАВЛЕНИЯ ОСВЕЩЕНИЕМ
		ImGui::Begin("Lighting Control");

		// Отдельная структура для освещения
		static LightingUniforms lighting_settings = {
			.directional_light_dir = {-0.5f, -1.0f, -0.5f},
			.directional_light_intensity = 0.8f,
			.directional_light_color = {1.0f, 1.0f, 0.9f},
			.point_light_pos = {2.0f, 3.0f, 2.0f},
			.point_light_intensity = 1.0f,
			.point_light_color = {1.0f, 0.8f, 0.6f},
			.point_light_constant = 1.0f,
			.point_light_linear = 0.09f,
			.point_light_quadratic = 0.032f,
			.ambient_light_color = {1.0f, 1.0f, 1.0f},
			.ambient_light_intensity = 0.8f
		};

		if (ImGui::CollapsingHeader("Directional Light")) {
			ImGui::SliderFloat("Dir Intensity", &lighting_settings.directional_light_intensity, 0.0f, 2.0f);
			ImGui::ColorEdit3("Dir Color", &lighting_settings.directional_light_color.x);
			ImGui::SliderFloat3("Dir Direction", &lighting_settings.directional_light_dir.x, -1.0f, 1.0f);
		}

		if (ImGui::CollapsingHeader("Point Light")) {
			ImGui::SliderFloat("Point Intensity", &lighting_settings.point_light_intensity, 0.0f, 3.0f);
			ImGui::ColorEdit3("Point Color", &lighting_settings.point_light_color.x);
			ImGui::SliderFloat3("Point Position", &lighting_settings.point_light_pos.x, -5.0f, 5.0f);
			ImGui::SliderFloat("Point Constant", &lighting_settings.point_light_constant, 0.1f, 2.0f);
			ImGui::SliderFloat("Point Linear", &lighting_settings.point_light_linear, 0.01f, 0.5f);
			ImGui::SliderFloat("Point Quadratic", &lighting_settings.point_light_quadratic, 0.001f, 0.1f);
		}

		if (ImGui::CollapsingHeader("Ambient Light")) {
			ImGui::SliderFloat("Ambient Intensity", &lighting_settings.ambient_light_intensity, 0.0f, 1.0f);
			ImGui::ColorEdit3("Ambient Color", &lighting_settings.ambient_light_color.x);
		}

		ImGui::End();

		// UI ДЛЯ ОТЛАДКИ КАМЕРЫ
		ImGui::Begin("Camera Debug");

		ImGui::Text("Position: (%.2f, %.2f, %.2f)",
			camera.position.x, camera.position.y, camera.position.z);

		ImGui::Text("Rotation: (%.2f, %.2f, %.2f)",
			camera.rotation.x, camera.rotation.y, camera.rotation.z);

		if (ImGui::Button("Reset Camera")) {
			camera.position = {0.0f, 0.0f, -6.0f};
			camera.rotation = {0.0f, 0.0f, 0.0f};
		}

		ImGui::End();

		// ИСПРАВЛЕНИЕ: ОБНОВЛЯЕМ UNIFORM BUFFERS ДЛЯ ОТДЕЛЬНЫХ БУФЕРОВ

		float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

		// 1. Scene Uniforms (ТОЛЬКО камера и проекция)
		SceneUniforms scene_uniforms = {};
		scene_uniforms.view_projection = camera.view_projection(aspect_ratio);
		scene_uniforms.camera_pos = camera.position;

		// 2. Lighting Uniforms (ОТДЕЛЬНЫЙ буфер)
		// lighting_settings уже обновляется через UI выше

		// 3. Model Uniforms
		std::vector<ModelUniforms> model_uniforms(models.size());
		for (size_t i = 0, n = models.size(); i < n; ++i) {
			const Model& model = models[i];
			model_uniforms[i].model = model.transform.matrix();
			model_uniforms[i].albedo_color = model.albedo_color;
			model_uniforms[i].specular_color = model.specular_color; // ←
			model_uniforms[i].shininess = model.shininess;           // ←
		}

		// Копируем данные в РАЗНЫЕ GPU буферы
		memcpy(scene_uniforms_buffer->mapped_region, &scene_uniforms, sizeof(SceneUniforms));
		memcpy(lighting_uniforms_buffer->mapped_region, &lighting_settings, sizeof(LightingUniforms));
		memcpy(model_uniforms_buffer->mapped_region, model_uniforms.data(),
			   model_uniforms.size() * sizeof(ModelUniforms));
	}
	void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
		vkResetCommandBuffer(cmd, 0);

		{ // Start recording rendering commands
			VkCommandBufferBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			};
			vkBeginCommandBuffer(cmd, &info);
		}

		{ // Render pass begin
			VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
			VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
			VkClearValue clear_values[] = {clear_color, clear_depth};

			VkRenderPassBeginInfo info{
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = veekay::app.vk_render_pass,
				.framebuffer = framebuffer,
				.renderArea = {
					.offset = {0, 0},
					.extent = {veekay::app.window_width, veekay::app.window_height},
				},
				.clearValueCount = 2,
				.pClearValues = clear_values,
			};
			vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		// Bind pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		VkDeviceSize zero_offset = 0;

		// Bind vertex and index buffers
		VkBuffer vertex_buffer = cone_mesh.vertex_buffer->buffer;
		VkBuffer index_buffer = cone_mesh.index_buffer->buffer;

		vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &zero_offset);
		vkCmdBindIndexBuffer(cmd, index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);

		// Render all models
		for (size_t i = 0, n = models.size(); i < n; ++i) {
			// Dynamic offset для ModelUniforms
			uint32_t dynamic_offsets[] = {
				static_cast<uint32_t>(i * sizeof(ModelUniforms))
			};

			// Bind descriptor sets с dynamic offset
			vkCmdBindDescriptorSets(cmd,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				pipeline_layout,
				0, // firstSet
				1, // descriptorSetCount
				&descriptor_set,
				1, // dynamicOffsetCount
				dynamic_offsets
			);

			// Draw indexed
			vkCmdDrawIndexed(cmd, cone_mesh.indices, 1, 0, 0, 0);
		}

		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);
	}
}

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}