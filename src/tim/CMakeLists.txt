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

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx/internal/include)

if(TIM_VX_ENABLE_LAYOUT_INFER)
    include_directories(${PROJECT_SOURCE_DIR}/)

    aux_source_directory(./transform LAYOUT_INFER_FRAMEWORK_SRCS)
    aux_source_directory(./transform/ops LAYOUT_INFER_OP_SRCS)

    list(APPEND SRC
        ${LAYOUT_INFER_FRAMEWORK_SRCS}
        ${LAYOUT_INFER_OP_SRCS}
    )
endif()

set(UT_SRC)
aux_source_directory(./vx/ut VX_UT_SRC)
list(APPEND UT_SRC ${VX_UT_SRC})
foreach(src_file ${SRC})
    if(${src_file} MATCHES ".*_test\.cc")
        list(REMOVE_ITEM SRC ${src_file})
        list(APPEND UT_SRC ${src_file})
    endif()
endforeach()

add_library(${TARGET_NAME} SHARED ${SRC})
target_link_libraries(${TARGET_NAME} PRIVATE
    -Wl,--whole-archive tim_internal -Wl,--no-whole-archive)

add_library(${TARGET_NAME}-static STATIC ${SRC})
target_link_libraries(${TARGET_NAME}-static PRIVATE
	-Wl,--whole-archive tim_internal -Wl,--no-whole-archive)

install(TARGETS ${TARGET_NAME} ${TARGET_NAME}-static
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/vx DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tim/)

if(TIM_VX_ENABLE_LAYOUT_INFER)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/transform DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tim/)
endif()

if (TIM_VX_ENABLE_TEST)
    include(GoogleTest)

    add_executable(unit_test ${UT_SRC})
    target_link_libraries(unit_test gtest gtest_main ${TARGET_NAME}-static)

    install(TARGETS unit_test DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
endif()
