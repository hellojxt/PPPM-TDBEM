add_library(uilib window.cu gui.cu)
target_link_libraries(uilib imgui)
target_compile_definitions(uilib PRIVATE IMGUI_CONFIG_FILE="${CMAKE_SOURCE_DIR}/src/ui/imgui.ini")