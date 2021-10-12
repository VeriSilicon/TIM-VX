message("samples/nbg_runner")

set(TARGET_NAME "nbg_runner")

aux_source_directory(. ${TARGET_NAME}_SRCS)
add_executable(${TARGET_NAME} ${${TARGET_NAME}_SRCS})

target_link_libraries(${TARGET_NAME} PRIVATE tim-vx nbg_parser)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)