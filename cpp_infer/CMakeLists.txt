cmake_minimum_required(VERSION 3.10.2)
project(paddleocr)

add_subdirectory(src)
add_subdirectory(apps)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
