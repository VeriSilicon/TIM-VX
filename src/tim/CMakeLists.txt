message("src/tim/vx")

set(TARGET_NAME "tim-vx")

aux_source_directory(./vx VX_SRC)
aux_source_directory(./vx/ops OPS_SRC)

if(NOT ${TIM_VX_ENABLE_CUSTOM_OP})
    list(REMOVE_ITEM OPS_SRC "./vx/ops/custom_base.cc")
endif()

set(${TARGET_NAME}_SRCS)
set(LITE_INC_DIRS)
set(LITE_EXTERNAL_LIBS)
list(APPEND ${TARGET_NAME}_SRCS
    ${VX_SRC}
    ${OPS_SRC})

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
    list(APPEND OVXLIB_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src/tim/vx/internal/src")
endif()
message(STATUS "OVXLIB include directory: ${OVXLIB_INCLUDE_DIR}")

if(TIM_VX_ENABLE_LAYOUT_INFER)
    aux_source_directory(./transform LAYOUT_INFER_FRAMEWORK_SRCS)
    aux_source_directory(./transform/ops LAYOUT_INFER_OP_SRCS)

    list(APPEND ${TARGET_NAME}_SRCS
        ${LAYOUT_INFER_FRAMEWORK_SRCS}
        ${LAYOUT_INFER_OP_SRCS})
endif()

if(TIM_VX_ENABLE_PLATFORM)
    message(STATUS "Using platform")
    aux_source_directory(./vx/platform PLATFORM_SRC)
    list(APPEND ${TARGET_NAME}_SRCS
        ${PLATFORM_SRC})

    if(${TIM_VX_ENABLE_PLATFORM_LITE})
        message(STATUS "Using lite paltform")
        list(APPEND ${TARGET_NAME}_SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/vx/platform/lite/lite_native.cc)
        if(NOT VIP_LITE_SDK)
            message(FATAL_ERROR "Please set VIP_LITE_SDK when using lite platform(TIM_VX_ENABLE_PLATFORM_LITE)")
        endif()
        list(APPEND LITE_EXTERNAL_LIBS
            ${VIP_LITE_SDK}/drivers/libNBGlinker.so
            ${VIP_LITE_SDK}/drivers/libVIPlite.so)
        list(APPEND LITE_INC_DIRS ${VIP_LITE_SDK}/include)
    endif()

    if(TIM_VX_ENABLE_GRPC)
        list(APPEND ${TARGET_NAME}_SRCS
            "${CMAKE_CURRENT_SOURCE_DIR}/vx/platform/grpc/grpc_platform_client.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/vx/platform/grpc/grpc_remote.cc")

        # Proto file
        get_filename_component(gp_proto "${CMAKE_CURRENT_SOURCE_DIR}/vx/platform/grpc/grpc_platform.proto" ABSOLUTE)
        get_filename_component(gp_proto_path "${gp_proto}" PATH)

        # Generate sources
        set(gp_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/grpc_platform.pb.cc")
        set(gp_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/grpc_platform.pb.h")
        set(gp_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/grpc_platform.grpc.pb.cc")
        set(gp_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/grpc_platform.grpc.pb.h")
        add_custom_command(
            OUTPUT "${gp_proto_srcs}" "${gp_proto_hdrs}" "${gp_grpc_srcs}" "${gp_grpc_hdrs}"
            COMMAND ${PROTOBUF_PROTOC}
            ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
                --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
                -I "${gp_proto_path}"
                --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN_EXECUTABLE}"
                    "${gp_proto}"
            DEPENDS "${gp_proto}")

        include_directories(${CMAKE_CURRENT_BINARY_DIR})

        list(APPEND ${TARGET_NAME}_SRCS
            ${gp_grpc_srcs}
            ${gp_grpc_hdrs}
            ${gp_proto_srcs}
            ${gp_proto_hdrs})
    endif()
endif()

foreach(src_file ${${TARGET_NAME}_SRCS})
    if(${src_file} MATCHES ".*_test\.cc")
        list(REMOVE_ITEM ${TARGET_NAME}_SRCS ${src_file})
        list(APPEND ${TARGET_NAME}_TEST_SRCS ${src_file})
    endif()
endforeach()

list(APPEND LITE_INC_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vx
    ${CMAKE_CURRENT_SOURCE_DIR}/transform
    ${OVXLIB_INCLUDE_DIR}
    ${OVXDRV_INCLUDE_DIRS})

if(${TIM_VX_ENABLE_VIPLITE})
    aux_source_directory(./lite LITE_SRC)
    list(APPEND ${TARGET_NAME}_SRCS
        ${LITE_SRC})
    list(APPEND LITE_EXTERNAL_LIBS ${VIPLITE_DRV_LIBRARIES})
    list(APPEND LITE_INC_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/lite
        ${VIPLITE_DRV_INCLUDE_DIR})
endif()
include_directories(${LITE_INC_DIRS})
# convert op list as compile flags so that we can implement compile compatable easier
if(${TIM_VX_USE_EXTERNAL_OVXLIB})
    file(STRINGS "${OVXLIB_INC}/interface/ops.def" ops_file_content)
    file(STRINGS "${OVXLIB_INC}/custom/custom_ops.def" custom_ops_file_content)
else()
    file(STRINGS "./vx/internal/include/interface/ops.def" ops_file_content)
    file(STRINGS "./vx/internal/include/custom/custom_ops.def" custom_ops_file_content)
endif()
string(LENGTH "/*;   Add new ops to the end.;*_/" comment_len)
string(SUBSTRING "${ops_file_content}" ${comment_len} -1 op_list_only)
string(REGEX REPLACE "DEF_OP\\(" "-DVSI_FEAT_OP_" op_list_tmp ${op_list_only})
string(REGEX REPLACE "\\)" " " op_as_flags ${op_list_tmp})

string(LENGTH "/*;   Add custom ops to the end.;*_/" custom_comment_len)
string(SUBSTRING "${custom_ops_file_content}" ${custom_comment_len} -1 custom_op_list_only)
string(REGEX REPLACE "DEF_OP\\(" "-DVSI_FEAT_OP_" custom_op_list_tmp ${custom_op_list_only})
string(REGEX REPLACE "\\)" " " custom_op_as_flags ${custom_op_list_tmp})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${op_as_flags} ${custom_op_as_flags}")

add_library(${TARGET_NAME} ${${TARGET_NAME}_SRCS})
target_include_directories(${TARGET_NAME} PRIVATE ${LITE_INC_DIRS})
target_link_libraries(${TARGET_NAME} PUBLIC
    -Wl,--no-whole-archive  ${OVXDRV_LIBRARIES} ${LITE_EXTERNAL_LIBS})

if(${TIM_VX_ENABLE_TENSOR_CACHE})
    add_dependencies(${TARGET_NAME} crypto)
    target_link_libraries(${TARGET_NAME} PUBLIC crypto)
endif()

if(${TIM_VX_USE_EXTERNAL_OVXLIB})
  #-Wl,--whole-archive should not applied to external library, but only for shared library
    target_link_libraries(${TARGET_NAME} PUBLIC tim_internal)
endif()

if(TIM_VX_ENABLE_PLATFORM AND TIM_VX_ENABLE_GRPC)
    target_link_libraries(${TARGET_NAME} PUBLIC
        ${GRPCPP_REFLECTION}
        ${GRPC_GRPCPP}
        ${PROTOBUF_LIBPROTOBUF})

    add_executable(grpc_platform_server
        ${CMAKE_CURRENT_SOURCE_DIR}/vx/platform/grpc/grpc_platform_server.cc)
    target_link_libraries(grpc_platform_server -Wl,--whole-archive ${TARGET_NAME})
    install(TARGETS grpc_platform_server grpc_platform_server
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
endif()
if(NOT CMAKE_INSTALL_LIBDIR)
	set(CMAKE_INSTALL_LIBDIR "lib")
endif()

# Install
install(TARGETS ${TARGET_NAME} ${TARGET_NAME}
	DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

if(TIM_VX_ENABLE_TENSOR_CACHE)
file(GLOB OPENSSL_H ${openssl-cmake_BINARY_DIR}/usr/local/include/openssl/*.h)
install(
    FILES ${OPENSSL_H}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/openssl)
endif()
install(
    FILES
        ${CMAKE_SOURCE_DIR}/include/tim/vx/builtin_op.h
        ${CMAKE_SOURCE_DIR}/include/tim/vx/compile_option.h
        ${CMAKE_SOURCE_DIR}/include/tim/vx/context.h
        ${CMAKE_SOURCE_DIR}/include/tim/vx/graph.h
        ${CMAKE_SOURCE_DIR}/include/tim/vx/operation.h
        ${CMAKE_SOURCE_DIR}/include/tim/vx/ops.h
        ${CMAKE_SOURCE_DIR}/include/tim/vx/tensor.h
        ${CMAKE_SOURCE_DIR}/include/tim/vx/types.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim/vx)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/vx/ops
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim/vx)

if(TIM_VX_ENABLE_VIPLITE)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/lite
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim)
endif()

if(TIM_VX_ENABLE_LAYOUT_INFER)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/transform
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim)
endif()

if(TIM_VX_ENABLE_PLATFORM)
    install(
        FILES
            ${CMAKE_SOURCE_DIR}/include/tim/vx/platform/platform.h
            ${CMAKE_SOURCE_DIR}/include/tim/vx/platform/native.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim/vx/platform)
    if(TIM_VX_ENABLE_PLATFORM_LITE)
        install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/vx/platform/lite
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim/vx/platform)
    endif()
    if(TIM_VX_ENABLE_GRPC)
        install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tim/vx/platform/grpc
            DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/tim/vx/platform)
    endif()
endif()

if(TIM_VX_ENABLE_TEST)
    include(GoogleTest)

    if(TIM_VX_ENABLE_API_TRACE)
        add_definitions("-DENABLE_API_TRACE")
    endif()

    add_executable(unit_test ${${TARGET_NAME}_TEST_SRCS})
    target_link_libraries(unit_test PRIVATE
        -Wl,--whole-archive ${TARGET_NAME}
        -Wl,--no-whole-archive gtest gtest_main gmock gmock_main ${OVXDRV_LIBRARIES})
    target_include_directories(unit_test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/vx
        ${OVXLIB_INCLUDE_DIR}
        ${LITE_INC_DIRS}
        ${PROJECT_SOURCE_DIR}
    )

    install(TARGETS unit_test DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
endif()

add_subdirectory("utils")
