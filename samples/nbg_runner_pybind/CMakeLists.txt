cmake_minimum_required(VERSION 3.18)
project(nbg_runner LANGUAGES C CXX)

# Set C/C++ standard.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable warnings as errors.
add_compile_options(-Wall -Werror)

# Add custom CMake modules.
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Find dependencies.
include(FindPyBind11)
include(FindVivanteSDK)

add_subdirectory(src)