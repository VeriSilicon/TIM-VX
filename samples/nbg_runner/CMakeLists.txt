if(TIM_VX_ENABLE_NB_PARSER_EXAMPLE)

include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(. nbg_runner_src)
add_executable(nbg_runner ${nbg_runner_src})
target_link_libraries(nbg_runner tim-vx-static nbg_parser)

endif()
