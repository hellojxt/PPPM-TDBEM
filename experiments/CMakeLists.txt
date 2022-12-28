add_executable(static static/main.cu)
target_link_libraries(static PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(rigidbody rigidbody/main.cu)
target_link_libraries(rigidbody PRIVATE Catch2::Catch2WithMain uilib utillib alglib)