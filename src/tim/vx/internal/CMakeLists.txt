message("src/tim/vx/internal")

set(TARGET_NAME "tim_internal")
if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    add_definitions(-DOVXLIB_API=)
else()
    set(OVXLIB_API_ATTR "__attribute__\(\(visibility\(\"default\"\)\)\)")
    add_definitions(-DOVXLIB_API=${OVXLIB_API_ATTR})
endif()


aux_source_directory(src INTERNAL_SRC)
aux_source_directory(src/kernel INTERNAL_KERNEL)
aux_source_directory(src/kernel/cl INTERNAL_KERNEL_CL)
aux_source_directory(src/kernel/cpu INTERNAL_KERNEL_CPU)
aux_source_directory(src/kernel/evis INTERNAL_KERNEL_EVIS)
aux_source_directory(src/kernel/vx INTERNAL_KERNEL_VX)
aux_source_directory(src/ops INTERNAL_OPS)
aux_source_directory(src/libnnext INTERNAL_LIBNNEXT)
aux_source_directory(src/quantization INTERNAL_QUANTIZATION)
aux_source_directory(src/custom/ops INTERNAL_CUSTOM_OPS)
aux_source_directory(src/custom/ops/kernel INTERNAL_CUSTOM_OPS_KERNEL)
aux_source_directory(src/utils INTERNAL_UTILS)

set(${TARGET_NAME}_SRCS)
list(APPEND ${TARGET_NAME}_SRCS
    ${INTERNAL_SRC}
    ${INTERNAL_KERNEL}
    ${INTERNAL_KERNEL_CL}
    ${INTERNAL_KERNEL_CPU}
    ${INTERNAL_KERNEL_EVIS}
    ${INTERNAL_KERNEL_VX}
    ${INTERNAL_OPS}
    ${INTERNAL_LIBNNEXT}
    ${INTERNAL_QUANTIZATION}
    ${INTERNAL_CUSTOM_OPS}
    ${INTERNAL_CUSTOM_OPS_KERNEL}
    ${INTERNAL_UTILS}
)

add_library(${TARGET_NAME} STATIC ${${TARGET_NAME}_SRCS})
target_link_libraries(${TARGET_NAME} PRIVATE ${OVXDRV_LIBRARIES})
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OVXDRV_INCLUDE_DIRS}
)
