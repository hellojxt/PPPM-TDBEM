cmake_minimum_required(VERSION 3.0.0)
project(PPPM LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 75 86)
set(CMAKE_CXX_STANDARD 17)


add_subdirectory(libs)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${SRC_DIR}/core ${SRC_DIR}/ui)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
add_executable(PPPM main.cu)
target_link_libraries(PPPM PUBLIC imgui)
target_compile_definitions(PPPM PRIVATE IMGUI_CONFIG_FILE="${CMAKE_SOURCE_DIR}/src/ui/imgui.ini")


