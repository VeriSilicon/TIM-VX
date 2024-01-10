add_library(nbg_runner_vx INTERFACE)
add_library(nbg_runner::vx ALIAS nbg_runner_vx)

target_sources(nbg_runner_vx INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/ovx_executor.cpp
)

target_link_libraries(nbg_runner_vx INTERFACE viv_sdk)