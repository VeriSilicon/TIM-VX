message("src/tim/vx")

set(TARGET_NAME "tim-vx")

if(${TIM_VX_USE_EXTERNAL_OVXLIB})
    find_library(OVXLIB_LIB NAMES "ovxlib")
    message(STATUS "Using external OVXLIB from ${OVXLIB_LIB}")
    add_library(tim_internal SHARED IMPORTED)
    set_target_properties(tim_internal PROPERTIES IMPORTED_LOCATION ${OVXLIB_LIB})
    target_link_libraries(tim_internal INTERFACE ${OVXDRV_LIBRARIES})
else()
    add_subdirectory("vx/internal")
endif()

aux_source_directory(./vx VX_SRC)
aux_source_directory(./vx/ops OPS_SRC)

set(${TARGET_NAME}_SRCS)
list(APPEND ${TARGET_NAME}_SRCS
    ${VX_SRC}
    ${OPS_SRC}
)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx)

if(${TIM_VX_USE_EXTERNAL_OVXLIB})
    if(NOT OVXLIB_INC)
        message(FATAL_ERROR "Set OVXLIB_INC if using external OVXLIB (TIM_VX_USE_EXTERNAL_OVXLIB)")
    endif()
    set(OVXLIB_INCLUDE_DIR ${OVXLIB_INC})
else()
    set(OVXLIB_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src/tim/vx/internal/include")
endif()
message(STATUS "OVXLIB include directory: ${OVXLIB_INCLUDE_DIR}")
include_directories(${OVXLIB_INCLUDE_DIR})

if(TIM_VX_ENABLE_LAYOUT_INFER)
    include_directories(${PROJECT_SOURCE_DIR}/)

    aux_source_directory(./transform LAYOUT_INFER_FRAMEWORK_SRCS)
    aux_source_directory(./transform/ops LAYOUT_INFER_OP_SRCS)

    list(APPEND ${TARGET_NAME}_SRCS
        ${LAYOUT_INFER_FRAMEWORK_SRCS}
        ${LAYOUT_INFER_OP_SRCS}
    )
endif()

foreach(src_file ${${TARGET_NAME}_SRCS})
    if(${src_file} MATCHES ".*_test\.cc")
        list(REMOVE_ITEM ${TARGET_NAME}_SRCS ${src_file})
        list(APPEND ${TARGET_NAME}_TEST_SRCS ${src_file})
    endif()
endforeach()

add_library(${TARGET_NAME} ${${TARGET_NAME}_SRCS})
target_link_libraries(${TARGET_NAME} PRIVATE
    -Wl,--whole-archive tim_internal -Wl,--no-whole-archive)

install(TARGETS ${TARGET_NAME} ${TARGET_NAME}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/vx 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tim)

if(TIM_VX_ENABLE_LAYOUT_INFER)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/transform 
            DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tim)
endif()

if(TIM_VX_ENABLE_TEST)
    include(GoogleTest)

    add_executable(unit_test ${${TARGET_NAME}_TEST_SRCS})
    target_link_libraries(unit_test gtest gtest_main gmock gmock_main ${TARGET_NAME})

    install(TARGETS unit_test DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
endif()

add_subdirectory("utils")
