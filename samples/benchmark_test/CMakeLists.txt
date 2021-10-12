message("samples/benchmark_test")

set(TARGET_NAME "benchmark_test")

aux_source_directory(. ${TARGET_NAME}_SRCS)
add_executable(${TARGET_NAME} ${${TARGET_NAME}_SRCS})

target_link_libraries(${TARGET_NAME} PRIVATE tim-vx)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)