if(${TIM_VX_ENABLE_TENSOR_CACHE})

# set(OPENSSL_CMAKE_URL ON CACHE STRING "https://github.com/viaduck/openssl-cmake")
# git@gitlab-cn.verisilicon.com:npu_sw/openssl/openssl-cmake.git
message("If use -DFEATCHCONTENT_SOURCE_DIR_OPENSSL-CMAKE, apply patch ${CMAKE_SOURCE_DIR}/cmake/openssl.patch required")

if(NOT OPENSSL_CMAKE_URL)
    set(OPENSSL_CMAKE_URL "https://github.com/viaduck/openssl-cmake")
endif()

message("Using openssl cmake project from ${OPENSSL_CMAKE_URL}")
include(FetchContent)
FetchContent_Declare(
    openssl-cmake
    GIT_REPOSITORY ${OPENSSL_CMAKE_URL}
    GIT_TAG 79c122d1606556610477cfae07ff27d8c6e5f260
    PATCH_COMMAND echo && git reset --hard 79c122d1606556610477cfae07ff27d8c6e5f260 && git apply ${CMAKE_SOURCE_DIR}/cmake/openssl.patch
    )

set(openssl_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_openssl OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(openssl-cmake)
endif()
