if(TIM_VX_ENABLE_NB_PARSER_EXAMPLE)

aux_source_directory(./nbg_parser nbg_parser_src)
add_library(nbg_parser STATIC ${nbg_parser_src})

endif()