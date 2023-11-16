project(imgui-cmake)
set(IMGUI_SRC_DIR ${PROJECT_SOURCE_DIR}/imgui)
# Add library
aux_source_directory(${IMGUI_SRC_DIR} IMGUI_SRC_FILES)
add_library(${PROJECT_NAME} STATIC
        ${IMGUI_SRC_FILES} 
        ${IMGUI_SRC_DIR}/backends/imgui_impl_glfw.h
        ${IMGUI_SRC_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_SRC_DIR}/backends/imgui_impl_opengl3.h
        ${IMGUI_SRC_DIR}/backends/imgui_impl_opengl3.cpp
)

# Add alias
add_library(imgui ALIAS ${PROJECT_NAME})

# Settings
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/imgui
                                           PUBLIC ${PROJECT_SOURCE_DIR}/imgui/backends
                                           PUBLIC ${PROJECT_SOURCE_DIR}/imgui/include)
target_compile_definitions(${PROJECT_NAME} PUBLIC
        GLEW_STATIC=1
        IMGUI_IMPL_OPENGL_LOADER_GLEW=1
)

# Link
target_link_libraries(${PROJECT_NAME} PRIVATE
        glfw
        ${GLFW_LIBRARIES}
        libglew_static
        tobanteGaming::CompilerWarnings
        tobanteGaming::CompilerOptions
)
