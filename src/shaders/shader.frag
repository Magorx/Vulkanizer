#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
	int x = int(gl_FragCoord.x);
    if ((x % 3) == 0) {
    	outColor = vec4(fragColor, 1.0);
    } else {
    	discard;
    }
}
