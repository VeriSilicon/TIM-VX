message("src/tim/vx")

set(TARGET_NAME "tim-vx")

add_subdirectory("vx/internal")

aux_source_directory(./vx VX_SRC)
aux_source_directory(./vx/ops OPS_SRC)

set(SRC)
list(APPEND SRC
    ${VX_SRC}
    ${OPS_SRC}
    )
list(REMOVE_ITEM SRC ./vx/context_test.cc)
list(REMOVE_ITEM SRC ./vx/graph_test.cc)
list(REMOVE_ITEM SRC ./vx/ops/reorg_test.cc)
list(REMOVE_ITEM SRC ./vx/ops/simple_operations_test.cc)
list(REMOVE_ITEM SRC ./vx/ops/relational_operations_test.cc)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx/internal/include)

if(ENABLE_LAYOUT_INFER)
    include_directories(${PROJECT_SOURCE_DIR}/)

    aux_source_directory(./transform LAYOUT_INFER_FRAMEWORK_SRCS)
    aux_source_directory(./transform/ops LAYOUT_INFER_OP_SRCS)

    list(APPEND SRC
        ${LAYOUT_INFER_FRAMEWORK_SRCS}
        ${LAYOUT_INFER_OP_SRCS}
    )
    list(REMOVE_ITEM SRC ./transform/layout_inference_test.cc)
endif()

add_library(${TARGET_NAME} SHARED ${SRC})
target_link_libraries(${TARGET_NAME} PRIVATE
    -Wl,--whole-archive tim_internal -Wl,--no-whole-archive)

add_library(${TARGET_NAME}-static STATIC ${SRC})
target_link_libraries(${TARGET_NAME}-static PRIVATE
	-Wl,--whole-archive tim_internal -Wl,--no-whole-archive)

install(TARGETS ${TARGET_NAME} ${TARGET_NAME}-static
        DESTINATION ${CMAKE_BINARY_DIR}/install/lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/vx DESTINATION ${CMAKE_BINARY_DIR}/install/include/tim/)

if(ENABLE_LAYOUT_INFER)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/transform DESTINATION ${CMAKE_BINARY_DIR}/install/include/tim/)
endif()
