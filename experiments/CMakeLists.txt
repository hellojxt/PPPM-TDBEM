add_executable(static static/main.cu)
target_link_libraries(static PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(ghost_rigidbody ghost_rigidbody/main.cu)
target_link_libraries(ghost_rigidbody PRIVATE Catch2::Catch2WithMain uilib utillib alglib)

add_executable(pppm_rigidbody pppm_rigidbody/main.cu)
target_link_libraries(pppm_rigidbody PRIVATE Catch2::Catch2WithMain uilib utillib alglib)

add_executable(pppm_objectCollection pppm_rigidbody/main2.cu)
target_link_libraries(pppm_objectCollection PRIVATE Catch2::Catch2WithMain uilib utillib alglib)