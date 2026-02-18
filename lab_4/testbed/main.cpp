#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <unordered_map>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 16;
constexpr uint32_t max_spot_lights = 16;


struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
};

struct SceneUniforms {
	veekay::mat4 view_projection;
    veekay::mat4 shadow_projection;
    veekay::vec3 view_position;
    float time;
    veekay::vec3 ambient_light_intensity; float _pad1;
    veekay::vec3 sun_light_direction; float _pad2;
    veekay::vec3 sun_light_color;
    uint32_t point_light_count;
    uint32_t spot_light_count;
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color; float _pad0;
    veekay::vec3 specular_color;
    float shininess;
    veekay::vec2 tilling;
    veekay::vec2 _pad1;
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

	// NOTE: Model matrix
	veekay::mat4 matrix() const;
};

struct Material {
    VkDescriptorSet descriptor_set;
    veekay::graphics::Texture* albedo_texture;
    veekay::graphics::Texture* specular_texture;
    veekay::graphics::Texture* emissive_texture;
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
    veekay::vec3 specular_color;
    float shininess;
    std::string material_name = "default";
    veekay::vec2 tilling = {1.0f, 1.0f};
};

struct PointLight {
    veekay::vec3 position;
    float radius; // Together with position forms vec4 position_radius
    veekay::vec3 color;
    float _pad0; // For alignment
};

struct SpotLight {
    veekay::vec3 position;
    float radius; // Together with position forms vec4 position_radius
    veekay::vec3 direction;
    float angle; // Together with direction forms vec4 direction_angle
    veekay::vec3 color;
    float _pad0; // For alignment
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 1000.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;
    bool use_lookat = true;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;

    veekay::mat4 lookat_view(const veekay::vec3 &at) const;
private:
    veekay::mat4 transform_view() const;
    veekay::mat4 lookat_view() const;
};

// NOTE: Scene objects
inline namespace {
	Camera camera{
		.position = {0.0f, -0.5f, -3.0f},
	    .rotation = {},
	};

	std::vector<Model> models;
    std::unordered_map<std::string, Material> materials;

    std::vector<PointLight> point_lights;
    std::vector<SpotLight> spot_lights;
    size_t count_lamps{};
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSetLayout material_descriptor_set_layout;

	VkDescriptorSet descriptor_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;
    veekay::graphics::Buffer* point_lights_buffer;
    veekay::graphics::Buffer* spot_lights_buffer;


	Mesh plane_mesh;
	Mesh cube_mesh;
    Mesh hemisphere_mesh;

	veekay::graphics::Texture* missing_texture;
    veekay::graphics::Texture* white_texture;
    veekay::graphics::Texture* black_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::vec3 angles_to_vec3(float pitch, float yaw) {
    float pitch_rad = toRadians(pitch);
    float yaw_rad = toRadians(yaw);
    veekay::vec3 front{};
    front.x = sin(yaw_rad) * cos(pitch_rad);
    front.y = sin(pitch_rad);
    front.z = cos(yaw_rad) * cos(pitch_rad);
    return veekay::vec3::normalized(front);
}

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

    veekay::graphics::Buffer *uniform_buffer; // Shadow projection matrix buffer
    VkSampler sampler;   // Sampler for shadow map

    veekay::mat4 matrix; // Shadow projection matrix
    bool first_frame = true;
} shadow;

constexpr uint32_t shadow_map_size = 8192;

uint32_t findMemoryType(VkPhysicalDevice physical_device, uint32_t type_filter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) ==
                properties) {
            return i;
                }
    }

    return UINT32_MAX;
}

void createShadowMapImage(VkDevice device, VkPhysicalDevice physical_device) {
    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;
    shadow.depth_image_format = depth_format;

    VkImageCreateInfo image_info = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
          .imageType = VK_IMAGE_TYPE_2D,
          .format = depth_format,
          .extent = {shadow_map_size, shadow_map_size, 1},
          .mipLevels = 1,
          .arrayLayers = 1,
          .samples = VK_SAMPLE_COUNT_1_BIT,
          .tiling = VK_IMAGE_TILING_OPTIMAL,
          .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                   VK_IMAGE_USAGE_SAMPLED_BIT,
          .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

      if (vkCreateImage(device, &image_info, nullptr, &shadow.depth_image) !=
          VK_SUCCESS) {
        std::cerr << "Failed to create shadow depth image\n";
        veekay::app.running = false;
        return;
      }

    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(device, shadow.depth_image, &mem_requirements);
    uint32_t memoryTypeIndex = findMemoryType(physical_device, mem_requirements.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memoryTypeIndex == UINT32_MAX) {
    std::cerr << "Failed to find suitable memory type for shadow map!\n";
    veekay::app.running = false;
    return;
}
    VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mem_requirements.size,
      .memoryTypeIndex = memoryTypeIndex
    };

    if (vkAllocateMemory(device, &alloc_info, nullptr, &shadow.depth_image_memory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate shadow depth image memory\n";
        veekay::app.running = false;
        return;
    }

    vkBindImageMemory(device, shadow.depth_image, shadow.depth_image_memory, 0);

      VkImageViewCreateInfo view_info = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
          .image = shadow.depth_image,
          .viewType = VK_IMAGE_VIEW_TYPE_2D,
          .format = depth_format,
          .subresourceRange =
              {
                  .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                  .baseMipLevel = 0,
                  .levelCount = 1,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
      };
    if (vkCreateImageView(device, &view_info, nullptr,
                            &shadow.depth_image_view) != VK_SUCCESS) {
        std::cerr << "Failed to create shadow depth image view\n";
        veekay::app.running = false;
        return;
      }
}

veekay::mat4 Transform::matrix() const {
    auto scale_mat = veekay::mat4::scaling(scale);
    auto rot_x = veekay::mat4::rotation({1, 0, 0}, toRadians(rotation[0]));
    auto rot_y = veekay::mat4::rotation({0, 1, 0}, toRadians(rotation[1]));
    auto rot_z = veekay::mat4::rotation({0, 0, 1}, toRadians(rotation[2]));
    auto translation_mat = veekay::mat4::translation(position);

    return scale_mat * rot_x * rot_y * rot_z * translation_mat;
}

veekay::mat4 Camera::transform_view() const {
    auto translation_mat = veekay::mat4::translation(-position);
    auto rotation_mat = veekay::mat4::rotation({0, 1, 0}, -toRadians(rotation.y)) *
                       veekay::mat4::rotation({1, 0, 0}, toRadians(rotation.x)) *
                           veekay::mat4::rotation({0, 0, 1}, -toRadians(rotation.z));

    return translation_mat * rotation_mat;
}

veekay::mat4 Camera::lookat_view() const {
    auto front = angles_to_vec3(rotation.x, rotation.y);
    veekay::vec3 target = position + front;
    veekay::vec3 world_up = {0.0f, 1.0f, 0.0f};

    veekay::vec3 zaxis = veekay::vec3::normalized(target - position);
    veekay::vec3 xaxis = veekay::vec3::normalized(veekay::vec3::cross(world_up, zaxis));
    veekay::vec3 yaxis = veekay::vec3::cross(zaxis, xaxis);

    veekay::mat4 result;
    result[0][0] = xaxis.x; result[0][1] = yaxis.x; result[0][2] = zaxis.x; result[0][3] = 0.0f;
    result[1][0] = xaxis.y; result[1][1] = yaxis.y; result[1][2] = zaxis.y; result[1][3] = 0.0f;
    result[2][0] = xaxis.z; result[2][1] = yaxis.z; result[2][2] = zaxis.z; result[2][3] = 0.0f;

    result[3][0] = veekay::vec3::dot(-xaxis, position);
    result[3][1] = veekay::vec3::dot(-yaxis, position);
    result[3][2] = veekay::vec3::dot(-zaxis, position);
    result[3][3] = 1.0f;

    return result;
}

veekay::mat4 Camera::lookat_view(const veekay::vec3 &at) const {
    veekay::vec3 world_up = {0.0f, 1.0f, 0.0f};

    veekay::vec3 zaxis = veekay::vec3::normalized(at - position);
    veekay::vec3 xaxis = veekay::vec3::normalized(veekay::vec3::cross(world_up, zaxis));
    veekay::vec3 yaxis = veekay::vec3::cross(zaxis, xaxis);

    veekay::mat4 result;
    result[0][0] = xaxis.x; result[0][1] = yaxis.x; result[0][2] = zaxis.x; result[0][3] = 0.0f;
    result[1][0] = xaxis.y; result[1][1] = yaxis.y; result[1][2] = zaxis.y; result[1][3] = 0.0f;
    result[2][0] = xaxis.z; result[2][1] = yaxis.z; result[2][2] = zaxis.z; result[2][3] = 0.0f;

    result[3][0] = veekay::vec3::dot(-xaxis, position);
    result[3][1] = veekay::vec3::dot(-yaxis, position);
    result[3][2] = veekay::vec3::dot(-zaxis, position);
    result[3][3] = 1.0f;

    return result;
}


veekay::mat4 Camera::view() const {
    return use_lookat ? lookat_view() : transform_view();
}


veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << path << std::endl;
		return nullptr;
	}
	
	size_t size = file.tellg();
	if (size == 0 || size % sizeof(uint32_t) != 0) {
		std::cerr << "Invalid shader file size: " << size << std::endl;
		return nullptr;
	}
	
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
		return nullptr;
	}

	return result;
}

bool createMaterial(VkCommandBuffer cmd,
                   const std::string& name,
                   const std::string& albedo_path = "",
                   const std::string& specular_path = "",
                   const std::string& emissive_path = "") {

    VkDevice& device = veekay::app.vk_device;
    if (materials.find(name) != materials.end()) {
        std::cerr << "Material '" << name << "' already exists!\n";
        return false;
    }
    Material material;
    auto *sampler = &texture_sampler;
    
    // Special handling for light_indicator
    if (name == "light_indicator") {
        // Create white texture for albedo
        veekay::vec4 white = {1.0f, 1.0f, 1.0f, 1.0f};
        material.albedo_texture = new veekay::graphics::Texture(
            cmd, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, &white);
        
        // Create bright emissive texture
        veekay::vec4 emissive = {2.0f, 2.0f, 2.0f, 1.0f};  // Bright emissive
        material.emissive_texture = new veekay::graphics::Texture(
            cmd, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, &emissive);
        material.specular_texture = black_texture;
    } else if (!albedo_path.empty()) {
        uint32_t width, height;
        std::vector<uint8_t> pixels;
        if (lodepng::decode(pixels, width, height, albedo_path) != 0) {
            std::cerr << "Failed to load albedo texture: " << albedo_path << "\n";
            material.albedo_texture = missing_texture;
        } else {
            if (width == 0 || height == 0 || width > 16384 || height > 16384) {
                std::cerr << "Invalid texture dimensions: " << width << "x" << height << std::endl;
                material.albedo_texture = missing_texture;
            } else {
                material.albedo_texture = new veekay::graphics::Texture(
                    cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
            }
        }
    } else {
        material.albedo_texture = missing_texture;
        sampler = &missing_texture_sampler;
    }

    if (!specular_path.empty()) {
        uint32_t width, height;
        std::vector<uint8_t> pixels;
        if (lodepng::decode(pixels, width, height, specular_path) != 0) {
            std::cerr << "Failed to load specular texture: " << specular_path << "\n";
            material.specular_texture = missing_texture;
        } else {
            if (width == 0 || height == 0 || width > 16384 || height > 16384) {
                std::cerr << "Invalid texture dimensions: " << width << "x" << height << std::endl;
                material.specular_texture = missing_texture;
            } else {
                material.specular_texture = new veekay::graphics::Texture(
                    cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
            }
        }
    } else {
        material.specular_texture = white_texture;
    }

    if (!emissive_path.empty()) {
        uint32_t width, height;
        std::vector<uint8_t> pixels;
        if (lodepng::decode(pixels, width, height, emissive_path) != 0) {
            std::cerr << "Failed to load emissive texture: " << emissive_path << "\n";
            material.emissive_texture = missing_texture;
        } else {
            if (width == 0 || height == 0 || width > 16384 || height > 16384) {
                std::cerr << "Invalid texture dimensions: " << width << "x" << height << std::endl;
                material.emissive_texture = missing_texture;
            } else {
                material.emissive_texture = new veekay::graphics::Texture(
                    cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
            }
        }
    } else {
        material.emissive_texture = black_texture;
    }

    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &material_descriptor_set_layout,
    };
    if (vkAllocateDescriptorSets(device, &alloc_info, &material.descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set for material: " << name << "\n";
        return false;
    }

    std::vector<VkDescriptorImageInfo> image_infos = {
        {
            .sampler = *sampler,
            .imageView = material.albedo_texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        },
        {
            .sampler = *sampler,
            .imageView = material.specular_texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        },
        {
            .sampler = *sampler,
            .imageView = material.emissive_texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        }
    };

    std::vector<VkWriteDescriptorSet> writes = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = material.descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_infos[0]
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = material.descriptor_set,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_infos[1]
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = material.descriptor_set,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_infos[2]
        }
    };

    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);

    materials[name] = material;
    std::cout << "Created material: " << name << "\n";
    return true;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

	    shadow.vertex_shader = loadShaderModule("shaders/shadow.vert.spv");
	    if (!shadow.vertex_shader) {
	        std::cerr << "Failed to load shadow shader from file\n";
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
				.offset = static_cast<uint32_t>(offsetof(Vertex, position)), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = static_cast<uint32_t>(offsetof(Vertex, normal)),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = static_cast<uint32_t>(offsetof(Vertex, uv)),
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

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 128,  // Increase for more materials
				},
			    {
			        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			        .descriptorCount = 8,
			    }
			};

			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 16,  // Increase maximum number of sets
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
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
                {
                    .binding = 2,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags =  VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 3,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags =  VK_SHADER_STAGE_FRAGMENT_BIT,
                },
			    {
			        .binding = 4,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			    }
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
		    VkDescriptorSetLayoutBinding bindings[] = {
		        {
		            .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 2,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                }
		    };

		    VkDescriptorSetLayoutCreateInfo info{
		        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                .pBindings = bindings,
            };

		    if (vkCreateDescriptorSetLayout(device, &info, nullptr,
                                            &material_descriptor_set_layout) != VK_SUCCESS) {
		        std::cerr << "Failed to create Vulkan material descriptor set layout\n";
		        veekay::app.running = false;
		        return;
		    }
	    }
		{
		    VkDescriptorSetLayoutBinding shadow_bindings[] = {
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

		    VkDescriptorSetLayoutCreateInfo shadow_layout_info{
		    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = sizeof(shadow_bindings) / sizeof(shadow_bindings[0]),
            .pBindings = shadow_bindings,
            };

		    if (vkCreateDescriptorSetLayout(device, &shadow_layout_info, nullptr,
                                            &shadow.descriptor_set_layout) !=
                VK_SUCCESS) {
		        std::cerr << "Failed to create shadow descriptor set layout\n";
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
	    VkDescriptorSetLayout set_layouts[] = {
		    descriptor_set_layout,
            material_descriptor_set_layout
        };

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 2,
			.pSetLayouts = set_layouts,
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

	    createShadowMapImage(device, physical_device);

	    VkAttachmentDescription depth_attachment{
	        .format = shadow.depth_image_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        };

	    VkAttachmentReference depth_attachment_ref{
	        .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

	    VkSubpassDescription subpass{
	        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 0,
            .pColorAttachments = nullptr,
            .pDepthStencilAttachment = &depth_attachment_ref,
        };

	    VkRenderPassCreateInfo render_pass_info{
	        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &depth_attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
        };

	    VkPipelineShaderStageCreateInfo shadow_stage_info = {
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = shadow.vertex_shader,
            .pName = "main",
        };

	    VkVertexInputAttributeDescription shadow_attributes[] = {
	        {
	            .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = static_cast<uint32_t>(offsetof(Vertex, position)),
            },
        };

	    VkPipelineVertexInputStateCreateInfo shadow_input_state_info{
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &buffer_binding,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = shadow_attributes,
        };

	    VkPipelineRasterizationStateCreateInfo shadow_raster_info{
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_FRONT_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_TRUE,
            .depthBiasConstantFactor = 2.5f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 3.0f,
            .lineWidth = 1.0f,
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

        VkPipelineDepthStencilStateCreateInfo shadow_depth_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
        };

        VkPipelineColorBlendAttachmentState shadow_color_blend_attachment = {};
        shadow_color_blend_attachment.colorWriteMask = 0;

        VkPipelineColorBlendStateCreateInfo shadow_blend_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 0,
            .pAttachments = nullptr,
            .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
        };

        VkDynamicState dyn_states[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_DEPTH_BIAS,
        };

        VkPipelineDynamicStateCreateInfo dyn_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = std::size(dyn_states),
            .pDynamicStates = dyn_states,
        };

        VkPipelineLayoutCreateInfo shadow_pipeline_layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &shadow.descriptor_set_layout,
        };

        if (vkCreatePipelineLayout(device, &shadow_pipeline_layout_info, nullptr,
                                   &shadow.pipeline_layout) != VK_SUCCESS) {
              std::cerr << "Failed to create shadow pipeline layout\n";
              veekay::app.running = false;
              return;
        }

        VkPipelineRenderingCreateInfoKHR rendering_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
            .depthAttachmentFormat = shadow.depth_image_format,
        };

        VkGraphicsPipelineCreateInfo shadow_pipeline_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering_info,
            .stageCount = 1,
            .pStages = &shadow_stage_info,
            .pVertexInputState = &shadow_input_state_info,
            .pInputAssemblyState = &assembly_state_info,
            .pTessellationState = nullptr,
            .pViewportState = &shadow_viewport_info,
            .pRasterizationState = &shadow_raster_info,
            .pMultisampleState = &sample_info,
            .pDepthStencilState = &shadow_depth_info,
            .pColorBlendState = &shadow_blend_info,
            .pDynamicState = &dyn_state_info,
            .layout = shadow.pipeline_layout,
            .renderPass = VK_NULL_HANDLE,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                      &shadow_pipeline_info, nullptr,
                                      &shadow.pipeline) != VK_SUCCESS) {
          std::cerr << "Failed to create shadow pipeline\n";
          veekay::app.running = false;
          return;
        }

        VkSamplerCreateInfo sampler_info{
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
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
            .unnormalizedCoordinates = VK_FALSE,
        };

        if (vkCreateSampler(device, &sampler_info, nullptr, &shadow.sampler) !=
            VK_SUCCESS) {
          std::cerr << "Failed to create shadow sampler\n";
          veekay::app.running = false;
          return;
        }
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	size_t model_uniform_size = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
	size_t model_buffer_size = max_models * model_uniform_size;
	
	// Overflow check
	if (model_uniform_size == 0 || model_buffer_size / model_uniform_size != max_models) {
		std::cerr << "Invalid model uniforms buffer size calculation\n";
		veekay::app.running = false;
		return;
	}
	
	model_uniforms_buffer = new veekay::graphics::Buffer(
		model_buffer_size,
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    size_t point_light_size = veekay::graphics::Buffer::structureAlignment(sizeof(PointLight));
    size_t point_light_buffer_size = max_point_lights * point_light_size;
    
    if (point_light_size == 0 || point_light_buffer_size / point_light_size != max_point_lights) {
        std::cerr << "Invalid point lights buffer size calculation\n";
        veekay::app.running = false;
        return;
    }
    
    point_lights_buffer = new veekay::graphics::Buffer(
        point_light_buffer_size,
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    size_t spot_light_size = veekay::graphics::Buffer::structureAlignment(sizeof(SpotLight));
    size_t spot_light_buffer_size = max_spot_lights * spot_light_size;
    
    if (spot_light_size == 0 || spot_light_buffer_size / spot_light_size != max_spot_lights) {
        std::cerr << "Invalid spot lights buffer size calculation\n";
        veekay::app.running = false;
        return;
    }
    
    spot_lights_buffer = new veekay::graphics::Buffer(
        spot_light_buffer_size,
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    if (sizeof(veekay::mat4) == 0) {
        std::cerr << "Invalid shadow uniform buffer size\n";
        veekay::app.running = false;
        return;
    }
    
    shadow.uniform_buffer = new veekay::graphics::Buffer(
        sizeof(veekay::mat4), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

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

	    veekay::vec4 white = {1.0f, 1.0f, 1.0f, 1.0f};
	    white_texture = new veekay::graphics::Texture(
            cmd, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, &white);
	    veekay::vec4 black = {0.0f, 0.0f, 0.0f, 1.0f};
	    black_texture = new veekay::graphics::Texture(
            cmd, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, &black);

	}
	{
	    VkSamplerCreateInfo info{
	        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .anisotropyEnable = true,
            .maxAnisotropy = 16.0f,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
        };
	    if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
	        std::cerr << "Failed to create Vulkan texture sampler\n";
	        veekay::app.running = false;
	        return;
	    }
	}
	{
	    VkDescriptorSetAllocateInfo shadow_alloc_info{
	        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &shadow.descriptor_set_layout,
        };

	    if (vkAllocateDescriptorSets(device, &shadow_alloc_info,
                                     &shadow.descriptor_set) != VK_SUCCESS) {
	        std::cerr << "Failed to allocate shadow descriptor set\n";
	        veekay::app.running = false;
	        return;
        }
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
				.range = sizeof(ModelUniforms),
			},
            {
                .buffer = point_lights_buffer->buffer,
                .offset = 0,
                .range = max_point_lights * sizeof(PointLight),
            },
            {
                .buffer = spot_lights_buffer->buffer,
                .offset = 0,
                .range = max_spot_lights * sizeof(SpotLight),
            },
		};

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
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 3,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[3],
            },
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	{
	    VkDescriptorBufferInfo shadow_buffer_infos[] = {
	        {
	            .buffer = shadow.uniform_buffer->buffer,
                .offset = 0,
                .range = sizeof(veekay::mat4),
            },
            {
            .buffer = model_uniforms_buffer->buffer,
            .offset = 0,
            .range = sizeof(ModelUniforms),
            },
        };

	    VkWriteDescriptorSet shadow_write_infos[] = {
	        {
	            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = shadow.descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &shadow_buffer_infos[0],
            },
                {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = shadow.descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo = &shadow_buffer_infos[1],
            },
        };

	    vkUpdateDescriptorSets(device, 2, shadow_write_infos, 0, nullptr);
	}

	{
	    VkDescriptorImageInfo shadow_image_info{
	        .sampler = shadow.sampler,
            .imageView = shadow.depth_image_view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

	    VkWriteDescriptorSet shadow_write{
	        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 4,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &shadow_image_info,
        };

	    vkUpdateDescriptorSets(device, 1, &shadow_write, 0, nullptr);
	}

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
	{
	    std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        const uint32_t segments = 50;
        const float radius = 0.5f;
        for (uint32_t i = 0; i <= segments / 2; ++i) {
            float theta = float(i) * float(M_PI/2) / float(segments / 2);
            for (uint32_t j = 0; j <= segments; ++j) {
                float phi = float(j) * 2.0f * float(M_PI) / float(segments);

                Vertex vertex{};
                vertex.position.x = radius * sin(theta) * cos(phi);
                vertex.position.y = radius * cos(theta);
                vertex.position.z = radius * sin(theta) * sin(phi);

                vertex.normal = veekay::vec3::normalized(vertex.position);
                vertex.uv.x = float(j) / float(segments);
                vertex.uv.y = float(i) / (float(segments) / 2);
                vertices.push_back(vertex);
            }
        }

        uint32_t base_vertex_start = vertices.size();
	    vertices.push_back(Vertex{
            .position = {0.0f, 0.0f, 0.0f},
            .normal = {0.0f, -1.0f, 0.0f},
            .uv = {0.5f, 0.5f}
        });

        for (uint32_t j = 0; j <= segments; ++j) {
            float phi = float(j) * 2.0f * float(M_PI) / float(segments);
            Vertex vertex{};
            vertex.position.x = radius * cos(phi);
            vertex.position.y = 0.0f;
            vertex.position.z = radius * sin(phi);
            vertex.normal = {0.0f, -1.0f, 0.0f};
            vertex.uv.x = float(j) / float(segments);
            vertex.uv.y = 0.0f;
            vertices.push_back(vertex);
        }
        for (uint32_t i = 0; i < segments / 2; ++i) {
            for (uint32_t j = 0; j < segments; ++j) {
                uint32_t first = i * (segments + 1) + j;
                uint32_t second = i * (segments + 1) + (j + 1);
                uint32_t third = (i + 1) * (segments + 1) + j;
                uint32_t fourth = (i + 1) * (segments + 1) + (j + 1);
                indices.push_back(first);
                indices.push_back(third);
                indices.push_back(second);

                indices.push_back(second);
                indices.push_back(third);
                indices.push_back(fourth);
            }
        }
        for (uint32_t j = 0; j < segments; ++j) {
            uint32_t center = base_vertex_start;
            uint32_t edge1 = base_vertex_start + 1 + j;
            uint32_t edge2 = base_vertex_start + 1 + ((j + 1) % (segments + 1));

            indices.push_back(center);
            indices.push_back(edge2);
            indices.push_back(edge1);
        }

        hemisphere_mesh.vertex_buffer = new veekay::graphics::Buffer(
            vertices.size() * sizeof(Vertex), vertices.data(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        hemisphere_mesh.index_buffer = new veekay::graphics::Buffer(
            indices.size() * sizeof(uint32_t), indices.data(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        hemisphere_mesh.indices = uint32_t(indices.size());
    }

    createMaterial(cmd, "default");
    createMaterial(cmd, "light_indicator");
    createMaterial(cmd, "epstein", "textures/Epstein_2013_mugshot.png");
    createMaterial(cmd, "epstein_house", "textures/house.png");
    createMaterial(cmd, "land", "textures/land.png");


    // NOTE: Add models to scene

    models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                    .position = {-2.0f, -1.35f, 10.0f},
                    .scale = {3.0f, 3.0f, 3.0f},
                    .rotation = {0.0f, 0.0f, 0.0f}
            },
            .albedo_color = {0.1f, 0.1f, 0.1f},
            .specular_color = {0.0f, 0.0f, 0.0f},
            .shininess = 64.0f,
            .material_name = "epstein"
    });

    models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                    .position = {1.3f, -0.8f, 9.3f},
                    .scale = {2.0f, 2.0f, 2.0f},
                    .rotation = {0.0f, 0.0f, 0.0f}
            },
            .albedo_color = {0.1f, 0.1f, 0.1f},
            .specular_color = {0.0f, 0.0f, 0.0f},
            .shininess = 64.0f,
            .material_name = "epstein_house"
    });


    models.emplace_back(Model{
        .mesh = plane_mesh,
        .transform = Transform{.scale = {5.f, 1.f, 5.f}},
        .albedo_color = {0.1f, 0.1f, 0.1f},
        .specular_color = {0.0f, 0.0f, 0.0f},
        .shininess = 64.0f,
        .material_name = "land",
        .tilling = {1.0f, 1.0f},
    });
    


    count_lamps = 0;

    point_lights.emplace_back(PointLight{.position = {20.0, -5.0, -3.0}});
    point_lights.emplace_back(PointLight{.position = {25.0, -5.0, -3.0}});
    point_lights.emplace_back(PointLight{.position = {30.0, -5.0, -3.0}});



    spot_lights.emplace_back(SpotLight{
        .position = {-0.4f, -0.8f, 14.0f},
        .radius = 10.0f,
        .direction = {0, -0.6, 0.4},
        .angle = 0.6f,
        .color = {1.0f * 30, 0.9f * 30, 0.3f * 30}
    });

        spot_lights.emplace_back(SpotLight{
        .position = {-0.4f, -0.8f, 7.0f},
        .radius = 10.0f,
        .direction = {0, -0.6, 0.4},
        .angle = 0.6f,
        .color = {1.0f * 30, 0.9f * 30, 0.3f * 30}
    });

    spot_lights.emplace_back(SpotLight{
        .position = {-20.0, -5.0, -3.0},
    });
}


// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
    VkDevice& device = veekay::app.vk_device;

    for (auto& [name, material] : materials) {
        if (material.albedo_texture != missing_texture &&
            material.albedo_texture != white_texture &&
            material.albedo_texture != black_texture) {
            delete material.albedo_texture;
            }
        if (material.specular_texture != missing_texture &&
            material.specular_texture != white_texture &&
            material.specular_texture != black_texture) {
            delete material.specular_texture;
            }
        if (material.emissive_texture != missing_texture &&
            material.emissive_texture != white_texture &&
            material.emissive_texture != black_texture) {
            delete material.emissive_texture;
            }
    }
    materials.clear();

    vkDestroySampler(device, texture_sampler, nullptr);
    vkDestroySampler(device, missing_texture_sampler, nullptr);

    delete missing_texture;
    delete white_texture;
    delete black_texture;

    delete cube_mesh.index_buffer;
    delete cube_mesh.vertex_buffer;

    delete hemisphere_mesh.index_buffer;
    delete hemisphere_mesh.vertex_buffer;

    delete plane_mesh.index_buffer;
    delete plane_mesh.vertex_buffer;

    delete model_uniforms_buffer;
    delete scene_uniforms_buffer;
    delete point_lights_buffer;
    delete spot_lights_buffer;

    vkDestroyDescriptorSetLayout(device, material_descriptor_set_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
       static auto sun_light_dir = veekay::vec3::normalized(veekay::vec3{0.3f, 1.0f, 0.2f});
    static veekay::vec3 sun_light_color = {1.0f, 1.0f, 1.0f};
    static float sun_intensity = 8.5f;
    static float sun_yaw = 187.0f;
    static float sun_pitch = 30.0f;

    static veekay::vec3 ambient_color = {1.0f, 1.0f, 1.0f};
    static float ambient_intensity = 0.01f;

    static veekay::vec3 point_color = {1.0, 1.0, 1.0};
    static float point_intensity = 5.0f;
    static float point_radius = 100.0f;

    static veekay::vec3 spot_color = {1.0, 1.0, 1.0};
    static float spot_intensity = 50.0f;
    static float spot_radius = 100.0f;
    static float spot_angle_deg = 75.0f;
    static float spot_yaw = 177.0f;
    static float spot_pitch = 35.0f;

	ImGui::Begin("Controls:");
    if (ImGui::Button(camera.use_lookat ? "LookAt Camera" : "Transform Camera")) {
        camera.use_lookat = !camera.use_lookat;
    }
    ImGui::Separator();
    ImGui::Text("Sun Controls:");
    ImGui::ColorEdit3("Sun color", sun_light_color.elements);
    ImGui::SliderFloat("Sun intensity", &sun_intensity, 0.0f, 10.0f);
    ImGui::SliderFloat("Sun yaw", &sun_yaw, 0.0f, 360.0f);
    ImGui::SliderFloat("Sun pitch", &sun_pitch, -90.0f, 90.0f);

    ImGui::Separator();
    ImGui::Text("Ambient:");
    ImGui::ColorEdit3("Ambient color", ambient_color.elements);
    ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 0.5f);


    ImGui::Separator();
    ImGui::Text("Spot:");
    ImGui::ColorEdit3("Spot color", spot_color.elements);
    ImGui::SliderFloat("Spot intensity", &spot_intensity, 0.0f, 50.0f);
    ImGui::SliderFloat("Spot radius", &spot_radius, 0.0f, 100.0f);
    ImGui::SliderFloat("Spot angle", &spot_angle_deg, 0.0f, 180.0f);
    ImGui::SliderFloat("Spot yaw", &spot_yaw, 0.0f, 360.0f);
    ImGui::SliderFloat("Spot pitch", &spot_pitch, -90.0f, 90.0f);
	ImGui::End();


    // Apply changes to all point lights (except first count_lamps)
    for (size_t i = count_lamps; i < point_lights.size(); ++i) {
        point_lights[i].color = point_color * point_intensity;
        point_lights[i].radius = point_radius;
    }
    
    // Apply changes to all spot lights
    for (size_t i = 0; i < spot_lights.size(); ++i) {
        spot_lights[i].angle = cos(toRadians(spot_angle_deg));
        spot_lights[i].color = spot_color * spot_intensity;
        spot_lights[i].radius = spot_radius;
        spot_lights[i].direction = angles_to_vec3(spot_pitch, spot_yaw);
    }

	if (!ImGui::IsWindowHovered()) {
	    using namespace veekay::input;

	    // Mouse rotation only when RMB is pressed
	    if (mouse::isButtonDown(mouse::Button::right)) {
	        auto move_delta = mouse::cursorDelta();
	        camera.rotation.y -= move_delta.x * 0.1f;
	        camera.rotation.x -= move_delta.y * 0.1f;

	        if (camera.rotation.x > 89.0f) camera.rotation.x = 89.0f;
	        if (camera.rotation.x < -89.0f) camera.rotation.x = -89.0f;
	    }

	    // Keyboard movement always active
	    auto view_matrix = camera.view();
	    veekay::vec3 right = {view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]};
	    veekay::vec3 up = {-view_matrix[0][1], -view_matrix[1][1], -view_matrix[2][1]};
	    veekay::vec3 front = {view_matrix[0][2], view_matrix[1][2], view_matrix[2][2]};
	    right = veekay::vec3::normalized(right);
	    up = veekay::vec3::normalized(up);
	    front = veekay::vec3::normalized(front);

	    if (keyboard::isKeyDown(keyboard::Key::w))
	        camera.position += front * 0.1f;

	    if (keyboard::isKeyDown(keyboard::Key::s))
	        camera.position -= front * 0.1f;

	    if (keyboard::isKeyDown(keyboard::Key::d))
	        camera.position += right * 0.1f;

	    if (keyboard::isKeyDown(keyboard::Key::a))
	        camera.position -= right * 0.1f;

	    if (keyboard::isKeyDown(keyboard::Key::q))
	        camera.position += up * 0.1f;

	    if (keyboard::isKeyDown(keyboard::Key::z))
	        camera.position -= up * 0.1f;
	}


	   float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

    sun_light_dir = angles_to_vec3(sun_pitch, sun_yaw);
    Camera light_camera;
    light_camera.position = -sun_light_dir * 50.0f;
    shadow.matrix = light_camera.lookat_view({0, 0, 0}) *
                    veekay::mat4::orthographic_projection(50.0f, 1.0, 0.0001f, 100.0f);

    *reinterpret_cast<veekay::mat4 *>(shadow.uniform_buffer->mapped_region) = shadow.matrix;

	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
	    .shadow_projection = shadow.matrix,
	    .view_position = camera.position,
	    .time = float(time),
        .ambient_light_intensity = ambient_color * ambient_intensity,
	    .sun_light_direction = sun_light_dir,
        .sun_light_color = sun_light_color * sun_intensity,
	    .point_light_count = static_cast<uint32_t>(point_lights.size()),
	    .spot_light_count = static_cast<uint32_t>(spot_lights.size()),
	};

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];

		uniforms.model = model.transform.matrix();
		uniforms.albedo_color = model.albedo_color;
	    uniforms.specular_color = model.specular_color;
	    uniforms.shininess = model.shininess;
	    uniforms.tilling = model.tilling;
	}
    std::copy(point_lights.begin(), point_lights.end(), reinterpret_cast<PointLight*> (point_lights_buffer->mapped_region));
    std::copy(spot_lights.begin(), spot_lights.end(), reinterpret_cast<SpotLight*> (spot_lights_buffer->mapped_region));

	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

	const size_t alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
		const ModelUniforms& uniforms = model_uniforms[i];

		char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
		*reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
	}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

    VkRenderingAttachmentInfoKHR depth_attachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
        .imageView = shadow.depth_image_view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.depthStencil = {1.0f, 0}},
    };

    VkRenderingInfoKHR rendering_info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
        .renderArea = {{0, 0}, {shadow_map_size, shadow_map_size}},
        .layerCount = 1,
        .viewMask = 0,
        .colorAttachmentCount = 0,
        .pColorAttachments = nullptr,
        .pDepthAttachment = &depth_attachment,
        .pStencilAttachment = nullptr,
    };

    auto vkCmdBeginRenderingKHR =
        (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(
            veekay::app.vk_device, "vkCmdBeginRenderingKHR");
    auto vkCmdEndRenderingKHR = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(
        veekay::app.vk_device, "vkCmdEndRenderingKHR");
    
if (!vkCmdBeginRenderingKHR || !vkCmdEndRenderingKHR) {
    std::cerr << "Dynamic rendering not supported!\n";
    veekay::app.running = false;
    return;
}
    if (vkCmdBeginRenderingKHR && vkCmdEndRenderingKHR) {
        vkCmdBeginRenderingKHR(cmd, &rendering_info);

        VkViewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = float(shadow_map_size),
            .height = float(shadow_map_size),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor = {{0, 0}, {shadow_map_size, shadow_map_size}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdSetDepthBias(cmd, 2.5f, 0.0f, 5.0f);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline);

        constexpr VkDeviceSize zero_offset = 0;
        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        const size_t model_uniforms_alignment =
          veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model &model = models[i];
            const auto &[vertex_buffer, index_buffer, indices] = model.mesh;

            if (current_vertex_buffer != vertex_buffer->buffer) {
              current_vertex_buffer = vertex_buffer->buffer;
              vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer,
                                     &zero_offset);
            }

            if (current_index_buffer != index_buffer->buffer) {
              current_index_buffer = index_buffer->buffer;
              vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset,
                                   VK_INDEX_TYPE_UINT32);
            }

            uint32_t offset = i * model_uniforms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    shadow.pipeline_layout, 0, 1,
                                    &shadow.descriptor_set, 1, &offset);

            vkCmdDrawIndexed(cmd, indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderingKHR(cmd);
    } else {
        std::cerr << "Dynamic rendering functions not available!\n";
        veekay::app.running = false;
        return;
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

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
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

		uint32_t offset = i * model_uniorms_alignment;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

	    auto material_it = materials.find(model.material_name);
	    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                1, 1, &material_it->second.descriptor_set, 0, nullptr);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

}

int main() {
    try {
        return veekay::run({
            .init = initialize,
            .shutdown = shutdown,
            .update = update,
            .render = render,
        });
    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return -1;
    }
}
