message("src/tim/vx/internal")

set(OVXLIB_API_ATTR "__attribute__\(\(visibility\(\"default\"\)\)\)")
add_definitions(-DOVXLIB_API=${OVXLIB_API_ATTR})

aux_source_directory(./vx/internal/src INTERNAL_SRC)
aux_source_directory(./vx/internal/src/kernel INTERNAL_KERNEL)
aux_source_directory(./vx/internal/src/kernel/cl INTERNAL_KERNEL_CL)
aux_source_directory(./vx/internal/src/kernel/cpu INTERNAL_KERNEL_CPU)
aux_source_directory(./vx/internal/src/kernel/evis INTERNAL_KERNEL_EVIS)
aux_source_directory(./vx/internal/src/kernel/vx INTERNAL_KERNEL_VX)
aux_source_directory(./vx/internal/src/ops INTERNAL_OPS)
aux_source_directory(./vx/internal/src/libnnext INTERNAL_LIBNNEXT)
aux_source_directory(./vx/internal/src/quantization INTERNAL_QUANTIZATION)
aux_source_directory(./vx/internal/src/custom/ops INTERNAL_CUSTOM_OPS)
aux_source_directory(./vx/internal/src/custom/ops/kernel INTERNAL_CUSTOM_OPS_KERNEL)
aux_source_directory(./vx/internal/src/utils INTERNAL_UTILS)
aux_source_directory(./vx/internal/src/POST POST)

list(APPEND ${TARGET_NAME}_SRCS
    ${INTERNAL_SRC}
    ${INTERNAL_KERNEL}
    ${INTERNAL_KERNEL_CL}
    ${INTERNAL_KERNEL_EVIS}
    ${INTERNAL_KERNEL_VX}
    ${INTERNAL_OPS}
    ${INTERNAL_LIBNNEXT}
    ${INTERNAL_QUANTIZATION}
    ${INTERNAL_CUSTOM_OPS}
    ${INTERNAL_CUSTOM_OPS_KERNEL}
    ${INTERNAL_UTILS}
    ${POST}
)
