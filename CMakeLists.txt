cmake_minimum_required (VERSION 2.6)
project (patchTextons)
 
# add the binary tree to the search path for include files
# add_subdirectory(src)
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("/opencv/build/include/opencv")
include_directories("/opencv/build/include/opencv2")
include_directories("/opencv/build/include")
# include_directories("${PROJECT_BINARY_DIR}/src/lib/")
include_directories("/boost_1_59_0/")
link_directories("/boost_1_59_0/lib/")

# set (CMAKE_CXX_FLAGS "-g")

# add the executable
add_executable(patchTextons
main.cpp
src/ImgData.cpp
src/PTexton.cpp
src/patchVector.cpp
src/ClassificationError.cpp
)

target_link_libraries(patchTextons opencv_core2410)
target_link_libraries(patchTextons opencv_contrib2410)
target_link_libraries(patchTextons opencv_highgui2410)
target_link_libraries(patchTextons opencv_imgproc2410)
target_link_libraries(patchTextons opencv_legacy2410)
target_link_libraries(patchTextons opencv_ml2410)
# target_link_libraries(patchTextons boost_thread)
# target_link_libraries(patchTextons boost_system)






