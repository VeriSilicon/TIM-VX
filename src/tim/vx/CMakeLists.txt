message("src/tim/vx")

set(TARGET_NAME "tim-vx")

add_subdirectory("internal")

aux_source_directory(. VX_SRC)
aux_source_directory(ops OPS_SRC)

set(SRC)
list(APPEND SRC
    ${VX_SRC}
    ${OPS_SRC}
    )
list(REMOVE_ITEM SRC ./context_test.cc)
list(REMOVE_ITEM SRC ./graph_test.cc)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx/internal/include)

add_library(${TARGET_NAME} SHARED ${SRC})
target_link_libraries(${TARGET_NAME} PRIVATE
    -Wl,--whole-archive tim_internal -Wl,--no-whole-archive)

add_library(${TARGET_NAME}-static STATIC ${SRC})
target_link_libraries(${TARGET_NAME}-static PRIVATE tim_internal)
