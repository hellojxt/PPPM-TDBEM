find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
add_library(utillib objIO.cu visualize.cu RigidBody.cu Water.cu ObjectCollection.cu)
target_link_libraries(utillib ${PNG_LIBRARY})