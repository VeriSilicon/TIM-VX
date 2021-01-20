message("src/tim/vx")

set(target_name "tim-vx")

aux_source_directory(. VX_SRC)
aux_source_directory(ops OPS_SRC)
aux_source_directory(internal/src INTERNAL_SRC)
aux_source_directory(internal/src/kernel INTERNAL_KERNEL)
aux_source_directory(internal/src/kernel/cl INTERNAL_KERNEL_CL)
aux_source_directory(internal/src/kernel/cpu INTERNAL_KERNEL_CPU)
aux_source_directory(internal/src/kernel/evis INTERNAL_KERNEL_EVIS)
aux_source_directory(internal/src/kernel/vx INTERNAL_KERNEL_VX)
aux_source_directory(internal/src/ops INTERNAL_OPS)
aux_source_directory(internal/src/client INTERNAL_CLIENT)
aux_source_directory(internal/src/libnnext INTERNAL_LIBNNEXT)
aux_source_directory(internal/src/libnnext/ops/kernel INTERNAL_LIBNNEXT_OPS_KERNEL)
aux_source_directory(internal/src/quantization INTERNAL_QUANTIZATION)
aux_source_directory(internal/src/custom/ops INTERNAL_CUSTOM_OPS)
aux_source_directory(internal/src/custom/ops/kernel INTERNAL_CUSTOM_OPS_KERNEL)
aux_source_directory(internal/src/utils INTERNAL_UTILS)

set(SRC)
list(APPEND SRC
    ${VX_SRC}
    ${OPS_SRC}
    ${INTERNAL_SRC}
    ${INTERNAL_KERNEL}
    ${INTERNAL_KERNEL_CL}
    ${INTERNAL_KERNEL_CPU}
    ${INTERNAL_KERNEL_EVIS}
    ${INTERNAL_KERNEL_VX}
    ${INTERNAL_OPS}
    ${INTERNAL_CLIENT}
    ${INTERNAL_LIBNNEXT}
    ${INTERNAL_LIBNNEXT_OPS_KERNEL}
    ${INTERNAL_QUANTIZATION}
    ${INTERNAL_CUSTOM_OPS}
    ${INTERNAL_CUSTOM_OPS_KERNEL}
    ${INTERNAL_UTILS}
    )

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx)
include_directories(${PROJECT_SOURCE_DIR}/src/tim/vx/internal/include)
include_directories(${OVXDRV_INCLUDE_DIRS})

add_library(${target_name} SHARED ${SRC})
target_link_libraries(${target_name} PRIVATE ${OVXDRV_LIBRARIES})