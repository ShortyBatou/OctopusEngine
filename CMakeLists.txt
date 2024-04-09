cmake_minimum_required (VERSION 3.8)
project (octopus-engine LANGUAGES CXX)

#add_compile_options(-Wno-narrowing)

# Options
option(TOBANTEGAMING_TEMPLATE_OPENGL_EXAMPLES  "Build examples" ON)
option(TOBANTEGAMING_TEMPLATE_OPENGL_TESTS     "Build unit tests" ON)

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

