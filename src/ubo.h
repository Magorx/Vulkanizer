struct UniformBufferObject {
    alignas(8) glm::vec2 mouse_pos;
    alignas(4) float dt;
    alignas(4) int mouse_hold;
    alignas(4) int PARTICLE_COUNT;
    alignas(4) float power_delta;
};

