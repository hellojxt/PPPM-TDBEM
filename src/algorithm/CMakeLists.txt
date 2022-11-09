add_library(alglib 
        fdtd.cu 
        pppm.cu
        pppm_direct.cu
        pppm_cache.cu
        particle_grid.cu
        sparse.cu
        ghost_cell_util.cu
        ghost_cell.cu
        svd.cu
)
find_package(CUDAToolkit REQUIRED)
target_include_directories(alglib
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(alglib PUBLIC cusolver cudart cusparse cublas)