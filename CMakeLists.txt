cmake_minimum_required(VERSION 3.0.0)
project(PPPM LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 75 86)
set(CMAKE_CXX_STANDARD 17)
set(OpenGL_GL_PREFERENCE GLVND)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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
                    ${SRC_DIR}/helmholtz
)
add_subdirectory(libs)
add_subdirectory(${SRC_DIR}/ui)
add_subdirectory(${SRC_DIR}/algorithm)
add_subdirectory(${SRC_DIR}/utils)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
add_executable(PPPM main.cu)
target_link_libraries(PPPM uilib alglib utillib)

find_package(Catch2 3 REQUIRED)
add_executable(test_particle_grid tests/ParticleGridTest.cu)
target_link_libraries(test_particle_grid PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(test_tdbem tests/TDBEMTest.cu)
target_link_libraries(test_tdbem PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(test_local_correction tests/LocalCorrectionTest.cu)
target_link_libraries(test_local_correction PRIVATE Catch2::Catch2WithMain uilib alglib utillib)



