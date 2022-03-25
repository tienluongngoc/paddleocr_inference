set(LIBRARY_NAME "paddle_ocr")

find_package(OpenCV REQUIRED core imgcodecs imgproc)
list(APPEND INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND DEPS opencv_core opencv_imgcodecs opencv_imgproc)

list(APPEND INCLUDE_DIRS "/paddle/cpp_infer/include/")
list(APPEND INCLUDE_DIRS "/paddle/cpp_infer/paddle/")
list(APPEND DEPS "/paddle/cpp_infer/libs/libpaddle_inference.so")

file(GLOB_RECURSE HEADER_LIST "/paddle/cpp_infer/include/*[.hpp, .h]")
file(GLOB_RECURSE SOURCE_LIST "/paddle/cpp_infer/src/*.cpp")

add_library(${LIBRARY_NAME} ${SOURCE_LIST} ${HEADER_LIST})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)
target_include_directories(${LIBRARY_NAME} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${DEPS})
