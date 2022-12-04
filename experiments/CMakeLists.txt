add_executable(sphere sphere/main.cu)
target_link_libraries(sphere PRIVATE Catch2::Catch2WithMain uilib alglib utillib)
add_executable(plane plane/main.cu)
target_link_libraries(plane PRIVATE Catch2::Catch2WithMain uilib alglib utillib)