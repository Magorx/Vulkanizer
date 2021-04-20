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

VulkanTest: main.cpp vertex.h color.o color_map.o
	make compile_shaders
	$(CPP) $(CFLAGS) -o $(CUR_PROG) main.cpp color.o color_map.o $(LDFLAGS) -lsfml-graphics -lsfml-window -lsfml-system

color.o: color_p4.cpp color_p4.h
	$(CPP) $(CFLAGS) color_p4.cpp -c -o color.o

color_map.o: color_map_p4.cpp color_map_p4.h
	$(CPP) $(CFLAGS) color_map_p4.cpp -c -o color_map.o

all: VulkanTest

compile_shaders: shaders/shader.vert shaders/shader.frag
	glslc shaders/shader.vert -o vert.spv
	glslc shaders/shader.frag -o frag.spv
	glslc shaders/shader.comp -o comp.spv