message("samples/lenet")

set(target_name "lenet")

aux_source_directory(. SRC)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(./)

add_executable(${target_name} ${SRC})
target_link_libraries(${target_name} ${OVXDRV_LIBRARIES}  tim-vx)