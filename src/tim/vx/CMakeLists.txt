message("src/tim/vx")

set(target_name "tim-vx")

add_subdirectory("internal")

aux_source_directory(. VX_SRC)
aux_source_directory(ops OPS_SRC)

set(SRC)
list(APPEND SRC
    ${VX_SRC}
    ${OPS_SRC}
    )

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx/internal/include)

add_library(${target_name} SHARED ${SRC})
target_link_libraries(${target_name} PRIVATE tim_internal)