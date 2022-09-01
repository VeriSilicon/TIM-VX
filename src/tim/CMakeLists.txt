message("src/tim/vx")

set(TARGET_NAME "tim-vx")

aux_source_directory(./vx VX_SRC)
aux_source_directory(./vx/ops OPS_SRC)

if(NOT ${TIM_VX_ENABLE_CUSTOM_OP})
    list(REMOVE_ITEM OPS_SRC "./vx/ops/custom_base.cc")
endif()

set(${TARGET_NAME}_SRCS)
list(APPEND ${TARGET_NAME}_SRCS
    ${VX_SRC}
    ${OPS_SRC}
)

if(${TIM_VX_USE_EXTERNAL_OVXLIB})
    find_library(OVXLIB_LIB NAMES "ovxlib")
    message(STATUS "Using external OVXLIB from ${OVXLIB_LIB}")
    add_library(tim_internal SHARED IMPORTED)
    set_target_properties(tim_internal PROPERTIES IMPORTED_LOCATION ${OVXLIB_LIB})
    target_link_libraries(tim_internal INTERFACE ${OVXDRV_LIBRARIES})
else()
    include(vx/internal/tim_internal.cmake)
endif()

if(${TIM_VX_USE_EXTERNAL_OVXLIB})
    if(NOT OVXLIB_INC)
        message(FATAL_ERROR "Set OVXLIB_INC if using external OVXLIB (TIM_VX_USE_EXTERNAL_OVXLIB)")
    endif()
    set(OVXLIB_INCLUDE_DIR ${OVXLIB_INC})
else()
    set(OVXLIB_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src/tim/vx/internal/include")
endif()
message(STATUS "OVXLIB include directory: ${OVXLIB_INCLUDE_DIR}")

if(TIM_VX_ENABLE_LAYOUT_INFER)
    aux_source_directory(./transform LAYOUT_INFER_FRAMEWORK_SRCS)
    aux_source_directory(./transform/ops LAYOUT_INFER_OP_SRCS)

    list(APPEND ${TARGET_NAME}_SRCS
        ${LAYOUT_INFER_FRAMEWORK_SRCS}
        ${LAYOUT_INFER_OP_SRCS}
    )
endif()

if(TIM_VX_ENABLE_PLATFORM)
    message(STATUS "Using paltform")
    aux_source_directory(./vx/platform PLATFORM_SRC)
    list(APPEND ${TARGET_NAME}_SRCS
        ${PLATFORM_SRC}
    )
endif()

foreach(src_file ${${TARGET_NAME}_SRCS})
    if(${src_file} MATCHES ".*_test\.cc")
        list(REMOVE_ITEM ${TARGET_NAME}_SRCS ${src_file})
        list(APPEND ${TARGET_NAME}_TEST_SRCS ${src_file})
    endif()
endforeach()

set(EXTERNAL_LIBS)
set(INC_DIRS)
list(APPEND INC_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vx
    ${CMAKE_CURRENT_SOURCE_DIR}/transform
    ${OVXLIB_INCLUDE_DIR}
    ${OVXDRV_INCLUDE_DIRS}
)

if(${TIM_VX_ENABLE_VIPLITE})
    aux_source_directory(./lite LITE_SRC)
    list(APPEND ${TARGET_NAME}_SRCS
        ${LITE_SRC})
    list(APPEND EXTERNAL_LIBS ${VIPLITE_DRV_LIBRARIES})
    list(APPEND INC_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/lite
        ${VIPLITE_DRV_INCLUDE_DIR}
    )
endif()

# convert op list as compile flags so that we can implement compile compatable easier
if(${TIM_VX_USE_EXTERNAL_OVXLIB})
    file(STRINGS "${OVXLIB_INC}/interface/ops.def" ops_file_content)
else()
    file(STRINGS "./vx/internal/include/interface/ops.def" ops_file_content)
endif()
string(LENGTH "/*;   Add new ops to the end.;*_/" comment_len)
string(SUBSTRING "${ops_file_content}" ${comment_len} -1 op_list_only)
string(REGEX REPLACE "DEF_OP\\(" "-DVSI_FEAT_OP_" op_list_tmp ${op_list_only})
string(REGEX REPLACE "\\)" " " op_as_flags ${op_list_tmp})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${op_as_flags}")

add_library(${TARGET_NAME} ${${TARGET_NAME}_SRCS})
target_include_directories(${TARGET_NAME} PRIVATE ${INC_DIRS})
target_link_libraries(${TARGET_NAME} PUBLIC
    -Wl,--no-whole-archive ${EXTERNAL_LIBS} ${OVXDRV_LIBRARIES})

if(${TIM_VX_USE_EXTERNAL_OVXLIB})
  #-Wl,--whole-archive should not applied to external library, but only for shared library
  target_link_libraries(${TARGET_NAME} PUBLIC tim_internal)
endif()

if (NOT CMAKE_INSTALL_LIBDIR)
	set(CMAKE_INSTALL_LIBDIR "lib")
endif()

install(TARGETS ${TARGET_NAME} ${TARGET_NAME}
	DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/vx 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/lite
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim)

if(TIM_VX_ENABLE_LAYOUT_INFER)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/transform 
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim)
endif()

if(TIM_VX_ENABLE_TEST)
    include(GoogleTest)

    add_executable(unit_test ${${TARGET_NAME}_TEST_SRCS})
    target_link_libraries(unit_test PRIVATE gtest gtest_main gmock gmock_main ${TARGET_NAME} ${OVXDRV_LIBRARIES})
    target_include_directories(unit_test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/vx
        ${OVXLIB_INCLUDE_DIR}
        ${INC_DIRS}
    )

    install(TARGETS unit_test DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
endif()

if(TIM_VX_ENABLE_PLATFORM)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/vx/platform
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim/vx)
endif()

add_subdirectory("utils")
