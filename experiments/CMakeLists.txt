add_executable(static static/main.cu)
target_link_libraries(static PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(teaser teaser/main.cu)
target_link_libraries(teaser PRIVATE Catch2::Catch2WithMain uilib alglib utillib)

add_executable(demo demo/main.cu)
target_link_libraries(demo PRIVATE Catch2::Catch2WithMain uilib utillib alglib)

add_executable(acoustic acoustic/main.cu)
target_link_libraries(acoustic PRIVATE Catch2::Catch2WithMain uilib alglib utillib)