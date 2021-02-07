cmake_minimum_required (VERSION 3.3)
project(tim-vx)

set(CMAKE_C_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Werror -fPIC")
set(CMAKE_CXX_FLAGS "--std=c++14 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Werror -fPIC")

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(OVXLIB_API_ATTR "__attribute__\(\(visibility\(\"default\"\)\)\)")
message(${OVXLIB_API_ATTR})
add_definitions(-DOVXLIB_API=${OVXLIB_API_ATTR})

if("${HARDWARE}" STREQUAL "A311D")
    include(cmake/A311D.cmake)
else()
    include(cmake/X86_64.cmake)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${OVXDRV_INCLUDE_DIRS})

add_subdirectory("src/tim/vx")
add_subdirectory("samples/lenet")
