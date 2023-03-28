cmake_minimum_required (VERSION 3.14)
project(tim-vx LANGUAGES C CXX)

option(BUILD_SHARED_LIBS                "Build using shared libraries"          ON)
option(TIM_VX_ENABLE_CUSTOM_OP          "Enable custom op support"              OFF)
option(TIM_VX_ENABLE_TEST               "Build the unit test"                   OFF)
option(TIM_VX_ENABLE_LAYOUT_INFER       "Enable layout inference support"       ON)
option(TIM_VX_ENABLE_NBG_PARSER         "Enable NBG parser"                     OFF)
option(TIM_VX_CODE_COVERAGE             "Run code coverage with gconv(gcc only" OFF)
option(TIM_VX_USE_EXTERNAL_OVXLIB       "Use external OVXLIB"                   OFF)
option(TIM_VX_BUILD_EXAMPLES            "Build demos show general usage"        OFF)
option(TIM_VX_ENABLE_VIPLITE            "Enable lite driver api support"        OFF)
option(TIM_VX_ENABLE_40BIT              "Enable large memory support"           OFF)
option(TIM_VX_ENABLE_PLATFORM           "Enable multi devices support"          OFF)
option(TIM_VX_ENABLE_PLATFORM_LITE      "Enable lite multi-device support"      OFF)
option(TIM_VX_ENABLE_GRPC               "Enable gPRC support"                   OFF)
option(TIM_VX_DBG_ENABLE_TENSOR_HNDL    "Enable built-in tensor from handle: use malloced memory instead of VideoMemory by kernel driver" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CONFIG "" CACHE STRING "Target Platform")

if(${TIM_VX_CODE_COVERAGE})
    set(CMAKE_CXX_FLAGS "-g -O0 --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "-g -O0 --coverage -fprofile-arcs -ftest-coverage")
endif()

if(${TIM_VX_ENABLE_PLATFORM_LITE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_PLATFORM_LITE")
endif()

if(${TIM_VX_DBG_ENABLE_TENSOR_HNDL})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_TENSOR_HNDL=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_TENSOR_HNDL=0")
endif()

include(GNUInstallDirs)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "" FORCE)
endif()

if(${TIM_VX_ENABLE_40BIT})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVSI_40BIT_VA_SUPPORT")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVSI_40BIT_VA_SUPPORT")
endif()

if(${TIM_VX_ENABLE_CUSTOM_OP})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTIM_VX_ENABLE_CUSTOM_OP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTIM_VX_ENABLE_CUSTOM_OP")
endif()

set(CMAKE_C_VISIBILITY_PRESET hidden)

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
    elseif("${CONFIG}" STREQUAL "YOCTO")
        include(cmake/YOCTO.cmake)
    elseif("${CONFIG}" STREQUAL "BUILDROOT")
        include(cmake/YOCTO.cmake)
    else()
        include(cmake/X86_64_linux.cmake)
    endif()
endif()

if(TIM_VX_ENABLE_TEST)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.0)

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
    endif()
endif()

if(TIM_VX_ENABLE_GRPC)
    include(cmake/gRPC.cmake)
endif()

add_subdirectory("src/tim")

if(TIM_VX_BUILD_EXAMPLES)
    add_subdirectory("samples")
endif()
