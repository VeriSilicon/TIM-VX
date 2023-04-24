include(FetchContent)
message("OpenSSL version 3.0.0")
if(TIM_VX_LOCAL_BUILD)
    FetchContent_Declare(
        openssl
        GIT_REPOSITORY ${LOCAL_BUILD_URL}/openssl/openssl-cmake.git
        GIT_TAG 79c122d1606556610477cfae07ff27d8c6e5f260)
else()
    FetchContent_Declare(
        openssl
        GIT_REPOSITORY https://github.com/viaduck/openssl-cmake
        GIT_TAG 79c122d1606556610477cfae07ff27d8c6e5f260)
endif()
set(openssl_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_openssl OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(openssl)
message("Apply dynamic_openssl patch on openssl-cmake")
execute_process(COMMAND git apply ${CMAKE_SOURCE_DIR}/cmake/dynamic_openssl.patch
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/_deps/openssl-src)