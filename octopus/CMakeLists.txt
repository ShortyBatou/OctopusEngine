project(octopus-engine)

# GETTING STARTED
set(OCTOPUS ${PROJECT_NAME})
add_executable(${OCTOPUS} main.cpp)
target_compile_features(${OCTOPUS} PUBLIC cxx_std_17)
target_compile_definitions(${OCTOPUS} PUBLIC GLEW_STATIC=1)
target_include_directories(${OCTOPUS} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${OCTOPUS} 
                    glfw 
                    ${GLFW_LIBRARIES} 
                    libglew_static
                    glm
                    spdlog::spdlog
                    imgui-cmake
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../assets DESTINATION ${CMAKE_CURRENT_BINARY_DIR})