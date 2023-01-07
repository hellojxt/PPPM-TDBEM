find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
add_library(utillib objIO.cu visualize.cu Water.cu Object.cu RigidBody.cu ObjectCollection.cu)
target_link_libraries(utillib uilib ${PNG_LIBRARY})