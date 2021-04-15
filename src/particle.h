#include <array>

struct Particle {
    alignas(8) glm::vec2 pos;
    alignas(8) glm::vec2 vel;
    alignas(16) glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Particle);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
	    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions {};

	    attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Particle, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Particle, vel);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Particle, color);

	    return attributeDescriptions;
	}
};

std::vector<Particle> particles = {
    // {{-0.5f, -0.5f}, { 1.0f,  0.0f}, {1.0f, 0.0f, 0.0f}},
    // {{ 0.5f, -0.5f}, { 1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}},
    // {{ 0.5f,  0.5f}, {-1.0f, -1.0f}, {1.0f, 0.0f, 1.0f}},
    // {{-0.5f,  0.5f}, {-1.0f,  0.0f}, {1.0f, 1.0f, 1.0f}},
};

size_t PARTICLES_CNT = particles.size();