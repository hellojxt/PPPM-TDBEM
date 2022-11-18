cmake_minimum_required(VERSION 3.12.0)
project(PPPM LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 75 86)
set(CMAKE_CXX_STANDARD 17)
set(OpenGL_GL_PREFERENCE GLVND)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# For CUDA-GDB
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
# For nsight compute
# add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-DMEMORY_CHECK>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
add_definitions(-DASSET_DIR="${CMAKE_SOURCE_DIR}/assets")

################################
# For Config
add_compile_definitions(LINE_GAUSS_NUM=1)
add_compile_definitions(TRI_GAUSS_NUM=1)
add_compile_definitions(STEP_NUM=16) # number of time steps for history of each particle

################################

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(
                    ${SRC_DIR}/core 
                    ${SRC_DIR}/ui
                    ${SRC_DIR}/algorithm
                    ${SRC_DIR}/utils
                    ${SRC_DIR}/helmholtz
                    tests
)
add_subdirectory(libs)
add_subdirectory(${SRC_DIR}/ui)
add_subdirectory(${SRC_DIR}/algorithm)
add_subdirectory(${SRC_DIR}/utils)

find_package(Catch2 3 REQUIRED)
add_executable(ParticleGridTest tests/ParticleGridTest.cu)
target_link_libraries(ParticleGridTest PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(TDBEMTest tests/TDBEMTest.cu)
target_link_libraries(TDBEMTest PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(LocalCorrectionTest tests/LocalCorrectionTest.cu)
target_link_libraries(LocalCorrectionTest PRIVATE uilib alglib utillib)

add_executable(UITest tests/UITest.cu)
target_link_libraries(UITest PRIVATE uilib alglib utillib)

add_executable(FDTDTest tests/FDTDTest.cu)
target_link_libraries(FDTDTest PRIVATE uilib alglib utillib)

add_executable(FarFieldCacheTest tests/FarFieldCacheTest.cu)
target_link_libraries(FarFieldCacheTest PRIVATE uilib alglib utillib)

add_executable(GridCacheTest tests/GridCacheTest.cu)
target_link_libraries(GridCacheTest PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(ParticleCacheTest tests/ParticleCacheTest.cu)
target_link_libraries(ParticleCacheTest PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(FFTTest tests/FFTTest.cu)
target_link_libraries(FFTTest PRIVATE Catch2::Catch2WithMain alglib utillib)

add_executable(PPPMTest tests/PPPMTest.cu)
target_link_libraries(PPPMTest PRIVATE uilib alglib utillib)

add_executable(PPPMDynamicTest tests/PPPMDynamicTest.cu)
target_link_libraries(PPPMDynamicTest PRIVATE uilib alglib utillib)

add_executable(CellClassificationTest tests/CellClassificationTest.cu)
target_link_libraries(CellClassificationTest PRIVATE uilib alglib utillib)

add_executable(GhostCellTest tests/GhostCellTest.cu)
target_link_libraries(GhostCellTest PRIVATE uilib alglib utillib)

add_executable(CusparseTest tests/CusparseTest.cu)
target_link_libraries(CusparseTest PRIVATE Catch2::Catch2WithMain alglib utillib)

add_executable(SVDTest tests/SVDTest.cu)
target_link_libraries(SVDTest PRIVATE Catch2::Catch2WithMain alglib utillib)