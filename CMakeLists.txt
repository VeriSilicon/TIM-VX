cmake_minimum_required (VERSION 3.14)
project(tim-vx LANGUAGES C CXX)

OPTION(TIM_VX_ENABLE_TEST           "Build the unit test"                   ON)
OPTION(TIM_VX_ENABLE_LAYOUT_INFER   "Enable layout inference support"       ON)
OPTION(TIM_VX_CODE_COVERAGE         "Run code coverage with gconv(gcc only" OFF)
option(TIM_VX_USE_EXTERNAL_OVXLIB   "Use external OVXLIB"                   OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror")
set(CONFIG "" CACHE STRING "Target Platform")

if(${TIM_VX_CODE_COVERAGE})
    set(CMAKE_CXX_FLAGS "-g -O0 --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "-g -O0 --coverage -fprofile-arcs -ftest-coverage")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "" FORCE)
endif()

set(CMAKE_C_VISIBILITY_PRESET hidden)

if(EXTERNAL_VIV_SDK AND EXISTS ${EXTERNAL_VIV_SDK})
    # this is for internal development purpose
    include(cmake/local_sdk.cmake)
else()
    set(TIM_VX_ENABLE_TEST OFF)
    if("${CONFIG}" STREQUAL "A311D")
        include(cmake/A311D.cmake)
    elseif("${CONFIG}" STREQUAL "S905D3")
        include(cmake/S905D3.cmake)
    elseif("${CONFIG}" STREQUAL "vim3_android")
        include(cmake/vim3_android.cmake)
    elseif("${CONFIG}" STREQUAL "YOCTO")
        include(cmake/YOCTO.cmake)
    else()
        set(TIM_VX_ENABLE_TEST ON)
        include(cmake/X86_64_linux.cmake)
    endif()
endif()

if(TIM_VX_ENABLE_TEST)
    include(FetchContent)
    FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.10.0
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
    endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${OVXDRV_INCLUDE_DIRS})

add_subdirectory("src/tim/")
add_subdirectory("samples/benchmark_test")
add_subdirectory("samples/lenet")
