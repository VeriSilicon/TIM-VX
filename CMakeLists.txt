cmake_minimum_required (VERSION 3.3)
project(tim-vx)

set(CMAKE_C_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Werror -fPIC -Wno-enum-conversion")
set(CMAKE_CXX_FLAGS "--std=c++14 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Werror -fPIC")

OPTION(ENABLE_LAYOUT_INFER "add layout inference support" ON)

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(OVXLIB_API_ATTR "__attribute__\(\(visibility\(\"default\"\)\)\)")
message(${OVXLIB_API_ATTR})
add_definitions(-DOVXLIB_API=${OVXLIB_API_ATTR})

if(EXTERNAL_VIV_SDK AND EXISTS ${EXTERNAL_VIV_SDK})
    # this is for internal development purpose
    include(cmake/local_sdk.cmake)
else()
    if("${CONFIG}" STREQUAL "A311D")
        include(cmake/A311D.cmake)
    elseif("${CONFIG}" STREQUAL "S905D3")
        include(cmake/S905D3.cmake)
    elseif("${CONFIG}" STREQUAL "vim3_android")
        include(cmake/vim3_android.cmake)
    else()
        include(cmake/X86_64_linux.cmake)
    endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${OVXDRV_INCLUDE_DIRS})

add_subdirectory("src/tim/")
add_subdirectory("samples/lenet")
