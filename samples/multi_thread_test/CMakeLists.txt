message("samples/multi_thread_test")

set(TARGET_NAME "multi_thread_test")

aux_source_directory(. SRC)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(./)

add_executable(${TARGET_NAME} ${SRC})
target_link_libraries(${TARGET_NAME} ${OVXDRV_LIBRARIES}  tim-vx-static pthread)