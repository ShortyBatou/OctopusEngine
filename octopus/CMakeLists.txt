project(octopus-engine)

# GETTING STARTED
set(OCTOPUS ${PROJECT_NAME})

file(GLOB_RECURSE cuda_files "*.cu")
file(GLOB_RECURSE cpp_files "*.cpp")
set_source_files_properties(${cpp_files} PROPERTIES LANGUAGE CUDA)
add_executable(${OCTOPUS} main.cu  ${cuda_files} ${cpp_files})

set_property(TARGET ${OCTOPUS} PROPERTY CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dlto")
set_target_properties(${OCTOPUS} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${OCTOPUS} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_compile_options(${OCTOPUS} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --default-stream per-thread>)

target_compile_features(${OCTOPUS} PUBLIC cxx_std_17)
target_compile_definitions(${OCTOPUS} PUBLIC GLEW_STATIC=1)

target_include_directories(${OCTOPUS} PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${OCTOPUS} 
                    glfw 
                    ${GLFW_LIBRARIES} 
                    libglew_static
                    glm
                    spdlog::spdlog
                    imgui-cmake
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../assets DESTINATION ${CMAKE_CURRENT_BINARY_DIR})