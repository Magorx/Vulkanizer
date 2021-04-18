#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inAnchor;
layout(location = 1) in vec2 inPosition;
layout(location = 2) in vec3 inVelocity;
layout(location = 3) in vec3 inColor;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragPos;

void main() {
    // gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    gl_Position = vec4(inPosition, 0.0, 1.0);
    gl_PointSize = 1.0;
    fragColor = inColor * (1 + length(inVelocity) / 1.2) + vec3(abs(inVelocity.x) / 1.2 + abs(inVelocity.y) / 1.2, 0, 0);
    fragPos = inPosition;
}