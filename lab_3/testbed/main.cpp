#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <array>
#include <glm/gtc/constants.hpp>
#define _USE_MATH_DEFINES
#include <algorithm>
#include <math.h>
#include <veekay/veekay.hpp>
#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include <lodepng.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <ranges>

namespace {
    constexpr uint32_t max_models = 1024;

    struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
    };

    struct SceneUniforms {
        veekay::mat4 view_projection;
        veekay::vec3 camera_pos;
        float time;

        float _padding1[44]; // до 256 байт
    };

    struct LightingUniforms {
        veekay::vec3 directional_light_dir;
        float directional_light_intensity;
        veekay::vec3 directional_light_color;
        float _padding1;

        veekay::vec3 point_light_pos;
        float point_light_intensity;
        veekay::vec3 point_light_color;
        float _padding2;

        veekay::vec3 spot_light_dir;
        float spot_light_cutoff;
        float spot_light_outer_cutoff;
        float spot_light_constant;
        float spot_light_linear;
        float spot_light_quadratic;

        veekay::vec3 ambient_light_color;
        float ambient_light_intensity;
        float _padding4[20]; // до 256 байт
    };

    struct ModelUniforms {
        veekay::mat4 model;
        veekay::vec3 albedo_color;
        float shininess;
        float _pad0[3];
        veekay::vec3 specular_color;
        float time_m;
        float _pad_to_256[37];
    };

    struct Mesh {
        veekay::graphics::Buffer* vertex_buffer = nullptr;
        veekay::graphics::Buffer* index_buffer = nullptr;
        uint32_t indices = 0;
    };

    struct Transform {
        veekay::vec3 position = {};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        veekay::vec3 rotation = {};
        veekay::mat4 matrix() const;
        veekay::mat4 inverse_matrix() const;
    };

    struct Model {
        Mesh mesh;
        Transform transform;
        veekay::vec3 albedo_color;
        veekay::vec3 specular_color;
        float shininess;
        size_t material_index = 0;
    };

    struct Material {
        veekay::graphics::Texture* albedo_texture = nullptr;
        VkSampler sampler = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    };

    struct Camera {
        constexpr static float default_fov = 60.0f;
        constexpr static float default_near_plane = 0.01f;
        constexpr static float default_far_plane = 100.0f;
        veekay::vec3 position = {0.0f, 0.0f, -6.0f};
        veekay::vec3 rotation = {};
        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;
        veekay::mat4 view() const;
        veekay::mat4 view_projection(float aspect_ratio) const;
    };

    inline namespace {
        Camera camera;
        std::vector<Model> models;
        size_t existing_models_count = 0;

        Mesh plane_mesh;
        Mesh cube_mesh;

        std::vector<Material> materials;
        VkSampler default_sampler = VK_NULL_HANDLE;
    }

    inline namespace {
        VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
        VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        veekay::graphics::Buffer* scene_uniforms_buffer = nullptr;
        veekay::graphics::Buffer* lighting_uniforms_buffer = nullptr;
        veekay::graphics::Buffer* model_uniforms_buffer = nullptr;
    }
    veekay::mat4 scaling_matrix(const veekay::vec3& scale) {
        return {
            scale.x, 0.0f,    0.0f,    0.0f,
            0.0f,    scale.y, 0.0f,    0.0f,
            0.0f,    0.0f,    scale.z, 0.0f,
            0.0f,    0.0f,    0.0f,    1.0f
        };
    }
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
    veekay::mat4 Transform::matrix() const {
		auto scale_mat = scaling_matrix(scale);
		auto rot_mat = rotation_y_matrix(rotation.y);
		auto trans_mat = veekay::mat4::translation(position);
		// T * R * S
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
		auto inv_scale_mat = inv_scaling_matrix(veekay::vec3(scale));

		auto rot_x = veekay::mat4::rotation(veekay::vec3{1.0, 0.0, 0.0}, rotation.x);
		auto rot_y = veekay::mat4::rotation(veekay::vec3{0.0, 1.0, 0.0}, rotation.y);
		auto rot_z = veekay::mat4::rotation(veekay::vec3{0.0, 0.0, 1.0}, rotation.z);
		auto rot_mat = rot_z * rot_y * rot_x;
		auto inv_rot_mat = veekay::mat4::transpose(rot_mat);

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

VkShaderModule loadShaderModule(const char* path) {
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) return VK_NULL_HANDLE;
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
		if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
		    return VK_NULL_HANDLE;
		}
		return result;
}

void createPlaneGeometry(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                         float width = 2.0f, float height = 2.0f) {
    float w2 = width / 2.0f, h2 = height / 2.0f;
    // Vertical plane (wall) facing the camera (negative Z direction)
    vertices = {
        {{-w2, -h2, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
        {{ w2, -h2, 0.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
        {{ w2,  h2, 0.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
        {{-w2,  h2, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
    };
    indices = {0, 1, 2, 0, 2, 3};
}

    void createCubeGeometry(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, float size = 1.0f) {
        float s = size / 2.0f;
        std::vector<veekay::vec3> corners = {
            {-s,-s,-s}, {s,-s,-s}, {s,s,-s}, {-s,s,-s},
            {-s,-s,s}, {s,-s,s}, {s,s,s}, {-s,s,s}
        };
        struct Face {
            std::array<int,4> idx; veekay::vec3 normal;
        };
        std::vector<Face> faces = {
            {{0,1,2,3}, {0,0,-1}}, {{4,7,6,5}, {0,0,1}},
            {{0,3,7,4}, {-1,0,0}}, {{1,5,6,2}, {1,0,0}},
            {{3,2,6,7}, {0,1,0}}, {{0,4,5,1}, {0,-1,0}}
        };
        vertices.reserve(24); indices.reserve(36);
        for (const auto& f : faces) {
            size_t base = vertices.size();
            // UV coordinates: (0,0) bottom-left, (1,0) bottom-right, (1,1) top-right, (0,1) top-left
            vertices.push_back({corners[f.idx[0]], f.normal, {0.0f, 0.0f}});
            vertices.push_back({corners[f.idx[1]], f.normal, {1.0f, 0.0f}});
            vertices.push_back({corners[f.idx[2]], f.normal, {1.0f, 1.0f}});
            vertices.push_back({corners[f.idx[3]], f.normal, {0.0f, 1.0f}});
            indices.insert(indices.end(), {
                uint32_t(base+0), uint32_t(base+1), uint32_t(base+2),
                uint32_t(base+0), uint32_t(base+2), uint32_t(base+3)
            });
        }
    }
    void initialize(VkCommandBuffer cmd) {
        VkDevice& device = veekay::app.vk_device;

        vertex_shader_module = loadShaderModule("../shaders/shader.vert.spv");
        fragment_shader_module = loadShaderModule("../shaders/shader.frag.spv");
        if (!vertex_shader_module || !fragment_shader_module) {
            std::cerr << "Failed to load shaders\n";
            veekay::app.running = false;
            return;
        }

        //Загружаем текстуры
        auto loadTexture = [](VkCommandBuffer cmd, const char* path) -> veekay::graphics::Texture* {
        int width, height, channels;
        unsigned char* data = stbi_load(path, &width, &height, &channels, 4); // Force RGBA
        if (!data) {
            std::cerr << "Failed to load texture: " << path << "\n";
            return nullptr;
        }
        uint32_t w = static_cast<uint32_t>(width);
        uint32_t h = static_cast<uint32_t>(height);
        std::vector<unsigned char> pixels(data, data + w * h * 4);
        stbi_image_free(data);
        return new veekay::graphics::Texture(cmd, w, h, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
        };

        veekay::graphics::Texture* land_tex = loadTexture(cmd, "land.jpg");
        veekay::graphics::Texture* mugshot_tex = loadTexture(cmd, "Epstein_2013_mugshot.jpg");
        veekay::graphics::Texture* house_tex = loadTexture(cmd, "house.jpg");
        
        if (!land_tex) {
            land_tex = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_R8G8B8A8_UNORM, (uint8_t*)"\xff\x00\x00\xff\xff\xff\x00\xff\xff\xff\x00\xff\xff\x00\x00\xff");
            std::cout << "Failed to load land.jpg, using fallback texture\n";
        }
        if (!mugshot_tex) {
            mugshot_tex = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_R8G8B8A8_UNORM, (uint8_t*)"\x00\xff\x00\xff\x00\x00\xff\xff\x00\xff\xff\xff\xff\x00\x00\x00\xff");
            std::cout << "Failed to load Epstein_2013_mugshot.jpg, using fallback texture\n";
        }
        if (!house_tex) {
            house_tex = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_R8G8B8A8_UNORM, (uint8_t*)"\x00\x00\xff\xff\xff\xff\x00\xff\xff\x00\xff\xff\x00\x00\x00\xff");
            std::cout << "Failed to load house.jpg, using fallback texture\n";
        }

        // Сэмплер для текстур
        VkSamplerCreateInfo sampler_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = 16.0f,
        };
        if (vkCreateSampler(device, &sampler_info, nullptr, &default_sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create sampler\n";
            veekay::app.running = false;
            return;
        }

        // структуру (layout) дескрипторного сета — то, какие ресурсы и как будут связаны (привязаны) к шейдерам
        VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
            {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
            {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT}
        };
        VkDescriptorSetLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4,
            .pBindings = bindings,
        };
        if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create descriptor set layout\n";
            veekay::app.running = false;
            return;
        }


        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 12},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
        };
        VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = max_models,
            .poolSizeCount = 3,
            .pPoolSizes = pool_sizes,
        };
        if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
            std::cerr << "Failed to create descriptor pool\n";
            veekay::app.running = false;
            return;
        }

        // ПАЙПЛАЙН
        VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[] = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
            {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
        };
        VkPipelineVertexInputStateCreateInfo vi_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
            .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attrs,
        };
        VkPipelineInputAssemblyStateCreateInfo ia_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        VkPipelineRasterizationStateCreateInfo rs_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .lineWidth = 1.0f,
        };
        VkPipelineMultisampleStateCreateInfo ms_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };


        VkViewport viewport{0, 0, float(veekay::app.window_width), float(veekay::app.window_height), 0, 1};
        VkRect2D scissor{{0,0}, {veekay::app.window_width, veekay::app.window_height}};
        VkPipelineViewportStateCreateInfo vp_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, .pViewports = &viewport,
            .scissorCount = 1, .pScissors = &scissor,
        };
        VkPipelineDepthStencilStateCreateInfo ds_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        };

        VkPipelineColorBlendAttachmentState blend_attach{
            .colorWriteMask = 0xF,
        };
        VkPipelineColorBlendStateCreateInfo blend_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &blend_attach,
        };
        VkPipelineShaderStageCreateInfo stages[2] = {
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertex_shader_module, .pName = "main"},
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment_shader_module, .pName = "main"},
        };
        VkPipelineLayoutCreateInfo pl_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &descriptor_set_layout,
        };
        if (vkCreatePipelineLayout(device, &pl_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create pipeline layout\n";
            veekay::app.running = false;
            return;
        }
        VkGraphicsPipelineCreateInfo pipeline_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2, .pStages = stages,
            .pVertexInputState = &vi_info,
            .pInputAssemblyState = &ia_info,
            .pViewportState = &vp_info,
            .pRasterizationState = &rs_info,
            .pMultisampleState = &ms_info,
            .pDepthStencilState = &ds_info,
            .pColorBlendState = &blend_info,
            .layout = pipeline_layout,
            .renderPass = veekay::app.vk_render_pass,
        };
        if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create pipeline\n";
            veekay::app.running = false;
            return;
        }

        // UNIFORM БУФЕРЫ
        scene_uniforms_buffer = new veekay::graphics::Buffer(sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        lighting_uniforms_buffer = new veekay::graphics::Buffer(sizeof(LightingUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        model_uniforms_buffer = new veekay::graphics::Buffer(max_models * sizeof(ModelUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        {
            std::vector<Vertex> v; std::vector<uint32_t> i;
            createPlaneGeometry(v, i, 5.0f, 5.0f);
            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(v.size() * sizeof(Vertex), v.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            plane_mesh.index_buffer = new veekay::graphics::Buffer(i.size() * sizeof(uint32_t), i.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            plane_mesh.indices = uint32_t(i.size());
        }
        {
            std::vector<Vertex> v; std::vector<uint32_t> i;
            createCubeGeometry(v, i, 1.0f);
            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(v.size() * sizeof(Vertex), v.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            cube_mesh.index_buffer = new veekay::graphics::Buffer(i.size() * sizeof(uint32_t), i.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            cube_mesh.indices = uint32_t(i.size());
        }


        auto createMaterial = [&](veekay::graphics::Texture* tex) -> Material {
            Material mat;
            mat.albedo_texture = tex;
            mat.sampler = default_sampler;

            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &descriptor_set_layout,
            };
            //Аллокация памяти под
            if (vkAllocateDescriptorSets(device, &alloc_info, &mat.descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to allocate descriptor set\n";
                veekay::app.running = false;
                return mat;
            }

            VkDescriptorBufferInfo scene_info{scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms)};
            VkDescriptorBufferInfo lighting_info{lighting_uniforms_buffer->buffer, 0, sizeof(LightingUniforms)};
            VkDescriptorBufferInfo model_info{model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms)};
            VkDescriptorImageInfo image_info{mat.sampler, tex->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

            VkWriteDescriptorSet writes[4] = {};
            writes[0] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = mat.descriptor_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &scene_info};
            writes[1] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = mat.descriptor_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &lighting_info};
            writes[2] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = mat.descriptor_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &model_info};
            writes[3] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = mat.descriptor_set, .dstBinding = 3, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image_info};

            vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
            return mat;
        };

        materials.push_back(createMaterial(land_tex));     // 0 — плоскость
        materials.push_back(createMaterial(mugshot_tex));  // 1 — первый куб
        materials.push_back(createMaterial(house_tex));    // 2 — второй куб

        // === МОДЕЛИ ===
        // Первый куб (перед камерой)
        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = {.position = {-1.5f, 0.0f, 0.0f}, .scale = {1.0f, 1.0f, 1.0f}},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {1.0f, 1.0f, 1.0f},
            .shininess = 64.0f,
            .material_index = 1,
        });
        // Второй куб (перед камерой)
        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = {.position = {1.5f, 0.0f, 0.0f}, .scale = {1.0f, 1.0f, 1.0f}},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {1.0f, 1.0f, 1.0f},
            .shininess = 64.0f,
            .material_index = 2,
        });
        // Вертикальная плоскость (позади кубов, лицом к камере)
        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = {.position = {0.0f, 0.0f, 3.0f}, .scale = {6.0f, 4.0f, 1.0f}},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {0.1f, 0.1f, 0.1f},
            .shininess = 8.0f,
            .material_index = 0,
        });
        existing_models_count = models.size();

        // ИНИЦИАЛИЗАЦИЯ UNIFORM-БУФЕРОВ
        SceneUniforms scene_u;
        scene_u.view_projection = camera.view_projection(float(veekay::app.window_width) / veekay::app.window_height);
        scene_u.camera_pos = camera.position;
        memcpy(scene_uniforms_buffer->mapped_region, &scene_u, sizeof(SceneUniforms));

        LightingUniforms lighting_u = {};
        lighting_u.directional_light_dir = {-0.5f, -1.0f, -0.5f};
        lighting_u.directional_light_intensity = 0.8f;
        lighting_u.directional_light_color = {1.0f, 1.0f, 0.9f};
        lighting_u.point_light_pos = {2.0f, 3.0f, 2.0f};
        lighting_u.point_light_intensity = 1.0f;
        lighting_u.point_light_color = {1.0f, 0.8f, 0.6f};
        lighting_u.spot_light_dir = {0.0f, -1.0f, 0.0f};
        lighting_u.spot_light_cutoff = cosf(12.5f * 3.14159f / 180.0f);
        lighting_u.spot_light_outer_cutoff = cosf(17.5f * 3.14159f / 180.0f);
        lighting_u.spot_light_constant = 1.0f;
        lighting_u.spot_light_linear = 0.09f;
        lighting_u.spot_light_quadratic = 0.032f;
        lighting_u.ambient_light_color = {1.0f, 1.0f, 1.0f};
        lighting_u.ambient_light_intensity = 0.8f;
        memcpy(lighting_uniforms_buffer->mapped_region, &lighting_u, sizeof(LightingUniforms));

        ModelUniforms default_u = {};
        default_u.model = veekay::mat4::identity();
        default_u.albedo_color = {0.8f, 0.2f, 0.2f};
        for (int i = 0; i < max_models; ++i) {
            memcpy((char*)model_uniforms_buffer->mapped_region + i * sizeof(ModelUniforms), &default_u, sizeof(ModelUniforms));
        }

        std::cout << "Initialization complete. Total models: " << models.size() << "\n";
    }

    // РЕНДЕР
    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &begin_info);

        VkClearValue clears[2] = {
            {.color = {{0.1f, 0.1f, 0.1f, 1.0f}}},
            {.depthStencil = {1.0f, 0}}
        };
        VkRenderPassBeginInfo rp_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = veekay::app.vk_render_pass,
            .framebuffer = framebuffer,
            .renderArea = {{0,0}, {veekay::app.window_width, veekay::app.window_height}},
            .clearValueCount = 2,
            .pClearValues = clears,
        };
        vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkDeviceSize offset = 0;
        for (size_t i = 0; i < models.size(); ++i) {
            const auto& m = models[i];
            vkCmdBindVertexBuffers(cmd, 0, 1, &m.mesh.vertex_buffer->buffer, &offset);
            vkCmdBindIndexBuffer(cmd, m.mesh.index_buffer->buffer, offset, VK_INDEX_TYPE_UINT32);

            uint32_t dyn_offset = uint32_t(i * sizeof(ModelUniforms));
            // Привязка дескрипторного сета материала модели
            VkDescriptorSet desc_set = materials[m.material_index].descriptor_set;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &desc_set, 1, &dyn_offset);

            vkCmdDrawIndexed(cmd, m.mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
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

        // Обновление uniform-буферов
        float aspect = float(veekay::app.window_width) / veekay::app.window_height;
        SceneUniforms scene_u;
        scene_u.view_projection = camera.view_projection(aspect);
        scene_u.camera_pos = camera.position;
        scene_u.time = static_cast<float>(time);

        memcpy(scene_uniforms_buffer->mapped_region, &scene_u, sizeof(SceneUniforms));

        std::vector<ModelUniforms> model_u(models.size());
        for (size_t i = 0; i < models.size(); ++i) {
            model_u[i].model = models[i].transform.matrix();
            model_u[i].albedo_color = models[i].albedo_color;
            model_u[i].specular_color = models[i].specular_color;
            model_u[i].shininess = models[i].shininess;


        }
        memcpy(model_uniforms_buffer->mapped_region, model_u.data(), model_u.size() * sizeof(ModelUniforms));
    }

    void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        vkDestroySampler(device, default_sampler, nullptr);
        for (auto& mat : materials) {
            delete mat.albedo_texture;
        }

        delete plane_mesh.vertex_buffer;
        delete plane_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;
        delete cube_mesh.index_buffer;

        delete scene_uniforms_buffer;
        delete lighting_uniforms_buffer;
        delete model_uniforms_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
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