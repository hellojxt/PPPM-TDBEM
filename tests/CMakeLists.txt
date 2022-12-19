add_executable(ParticleGridTest ParticleGridTest.cu)
target_link_libraries(ParticleGridTest PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(TDBEMTest TDBEMTest.cu)
target_link_libraries(TDBEMTest PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(LocalCorrectionTest LocalCorrectionTest.cu)
target_link_libraries(LocalCorrectionTest PRIVATE uilib alglib utillib)

add_executable(UITest UITest.cu)
target_link_libraries(UITest PRIVATE uilib alglib utillib)

add_executable(FDTDTest FDTDTest.cu)
target_link_libraries(FDTDTest PRIVATE uilib alglib utillib)

add_executable(FarFieldCacheTest FarFieldCacheTest.cu)
target_link_libraries(FarFieldCacheTest PRIVATE uilib alglib utillib)

add_executable(GridCacheTest GridCacheTest.cu)
target_link_libraries(GridCacheTest PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(FaceCacheTest FaceCacheTest.cu)
target_link_libraries(FaceCacheTest PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(FFTTest FFTTest.cu)
target_link_libraries(FFTTest PRIVATE Catch2::Catch2WithMain alglib utillib)

add_executable(PPPMTest PPPMTest.cu)
target_link_libraries(PPPMTest PRIVATE uilib alglib utillib)

add_executable(FastCacheTest FastCacheTest.cu)
target_link_libraries(FastCacheTest PRIVATE uilib alglib utillib)

add_executable(CellClassificationTest CellClassificationTest.cu)
target_link_libraries(CellClassificationTest PRIVATE uilib alglib utillib)

add_executable(GhostCellTest GhostCellTest.cu)
target_link_libraries(GhostCellTest PRIVATE uilib alglib utillib)

add_executable(CusparseTest CusparseTest.cu)
target_link_libraries(CusparseTest PRIVATE Catch2::Catch2WithMain alglib utillib)

add_executable(SVDTest SVDTest.cu)
target_link_libraries(SVDTest PRIVATE Catch2::Catch2WithMain alglib utillib)

add_executable(CatchTest ParticleGridTest.cu TDBEMTest.cu GridCacheTest.cu FaceCacheTest.cu FFTTest.cu CusparseTest.cu SVDTest.cu)
target_link_libraries(CatchTest PRIVATE Catch2::Catch2WithMain uilib alglib utillib)
