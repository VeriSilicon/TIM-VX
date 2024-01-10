# Try to use installed pybind11 CMake module.
find_package(pybind11)

if(NOT ${pybind11_FOUND})
  include(FetchContent)

  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
    GIT_TAG "v2.11.1"
  )
  FetchContent_MakeAvailable(pybind11)
endif()