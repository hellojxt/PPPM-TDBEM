cmake_minimum_required(VERSION 3.0.0)
project(PPPM LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 75 86)
set(CMAKE_CXX_STANDARD 17)
set(OpenGL_GL_PREFERENCE GLVND)
# For CUDA-GDB
# add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
add_definitions(-DASSET_DIR="${CMAKE_SOURCE_DIR}/assets")

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(
                    ${SRC_DIR}/core 
                    ${SRC_DIR}/ui
                    ${SRC_DIR}/algorithm
                    ${SRC_DIR}/utils
)
add_subdirectory(libs)
add_subdirectory(${SRC_DIR}/ui)
add_subdirectory(${SRC_DIR}/algorithm)
add_subdirectory(${SRC_DIR}/utils)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
add_executable(PPPM main.cu)
target_link_libraries(PPPM uilib alglib utillib)



