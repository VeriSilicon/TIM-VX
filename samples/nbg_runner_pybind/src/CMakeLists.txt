include_directories(${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(_nbg_runner ${CMAKE_CURRENT_SOURCE_DIR}/nbg_runner_pybind.cpp)

add_subdirectory(vx)
target_link_libraries(_nbg_runner
  PRIVATE
  nbg_runner::vx
)
