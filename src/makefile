CUR_PROG = vulkanizer

ifndef VERBOSE
.SILENT:
endif

CC = gcc
CPP = g++

WARNINGS = -Wall -Wextra -Wno-multichar -Wno-unused-parameter
STANDARD = -std=c++17
OPTIMIZATION = -O2

LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

CFLAGS = $(STANDARD) $(WARNINGS) $(OPTIMIZATION) -lm -march=native

ifeq ($(BUILD),debug)   
# "Debug" build - no optimization, and debugging symbols
CFLAGS += -O0 -g
else
# "Release" build - optimization, and no debug symbols
CFLAGS += $(OPTIMIZATION) -s -DNDEBUG
endif

run: VulkanTest
	./$(CUR_PROG)

VulkanTest: main.cpp vertex.h
	make compile_shaders
	$(CPP) $(CFLAGS) -o $(CUR_PROG) main.cpp $(LDFLAGS)

all: VulkanTest

compile_shaders: shaders/shader.vert shaders/shader.frag
	glslc shaders/shader.vert -o vert.spv
	glslc shaders/shader.frag -o frag.spv
	glslc shaders/shader.comp -o comp.spv