cmake_minimum_required(VERSION 3.12.0)
project(PPPM LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 75 86)
set(CMAKE_CXX_STANDARD 17)
set(OpenGL_GL_PREFERENCE GLVND)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# For CUDA-GDB
# add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
# For nsight compute
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-DMEMORY_CHECK>)
# add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-DNDEBUG>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
add_definitions(-DASSET_DIR="${CMAKE_SOURCE_DIR}/assets/")
add_definitions(-DEXP_DIR="${CMAKE_SOURCE_DIR}/experiments/")
add_definitions(-DDATASET_DIR="${CMAKE_SOURCE_DIR}/dataset/")
add_definitions(-DROOT_DIR="${CMAKE_SOURCE_DIR}/")

################################
# For Config
add_compile_definitions(LINE_GAUSS_NUM=2)
add_compile_definitions(TRI_GAUSS_NUM=3)
add_compile_definitions(STEP_NUM=16) # number of time steps for history of each particle

################################

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(
                    ${SRC_DIR}/core 
                    ${SRC_DIR}/ui
                    ${SRC_DIR}/algorithm
                    ${SRC_DIR}/utils
                    ${SRC_DIR}/helmholtz
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
add_subdirectory(libs)
add_subdirectory(${SRC_DIR}/ui)
add_subdirectory(${SRC_DIR}/algorithm)
add_subdirectory(${SRC_DIR}/utils)

find_package(Catch2 3 REQUIRED)
add_subdirectory(tests)
add_subdirectory(experiments)


