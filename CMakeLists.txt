cmake_minimum_required (VERSION 3.13)
project (octopus-engine LANGUAGES CXX CUDA)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 17)

#find cuda architecture
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
#add_compile_options(-Wno-narrowing)


# CMake setup
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(ConfigureOpenGL)

# Build static
set(BUILD_SHARED_LIBS OFF CACHE STRING "" FORCE)

# Include submodules
add_subdirectory(extern/glfw)
add_subdirectory(extern/glew)
add_subdirectory(extern/glm)
add_subdirectory(extern/imgui-cmake)
add_subdirectory(extern/spdlog)

add_subdirectory(octopus)

