message("src/tim/vx/internal")

set(lib_name "tim_internal")
set(OVXLIB_API_ATTR "__attribute__\(\(visibility\(\"default\"\)\)\)")
add_definitions(-DOVXLIB_API=${OVXLIB_API_ATTR})
add_compile_options(-Wno-strict-aliasing -Wno-unused-but-set-variable -Wno-maybe-uninitialized)

aux_source_directory(src INTERNAL_SRC)
aux_source_directory(src/kernel INTERNAL_KERNEL)
aux_source_directory(src/kernel/cl INTERNAL_KERNEL_CL)
aux_source_directory(src/kernel/cpu INTERNAL_KERNEL_CPU)
aux_source_directory(src/kernel/evis INTERNAL_KERNEL_EVIS)
aux_source_directory(src/kernel/vx INTERNAL_KERNEL_VX)
aux_source_directory(src/ops INTERNAL_OPS)
aux_source_directory(src/libnnext INTERNAL_LIBNNEXT)
aux_source_directory(src/libnnext/ops/kernel INTERNAL_LIBNNEXT_OPS_KERNEL)
aux_source_directory(src/quantization INTERNAL_QUANTIZATION)
aux_source_directory(src/custom/ops INTERNAL_CUSTOM_OPS)
aux_source_directory(src/custom/ops/kernel INTERNAL_CUSTOM_OPS_KERNEL)
aux_source_directory(src/utils INTERNAL_UTILS)

set(SRC)
list(APPEND SRC
    ${INTERNAL_SRC}
    ${INTERNAL_KERNEL}
    ${INTERNAL_KERNEL_CL}
    ${INTERNAL_KERNEL_CPU}
    ${INTERNAL_KERNEL_EVIS}
    ${INTERNAL_KERNEL_VX}
    ${INTERNAL_OPS}
    ${INTERNAL_LIBNNEXT}
    ${INTERNAL_LIBNNEXT_OPS_KERNEL}
    ${INTERNAL_QUANTIZATION}
    ${INTERNAL_CUSTOM_OPS}
    ${INTERNAL_CUSTOM_OPS_KERNEL}
    ${INTERNAL_UTILS}
)

include_directories(include)
include_directories(${OVXDRV_INCLUDE_DIRS})

add_library(${lib_name} ${SRC})
target_link_libraries(${lib_name} PRIVATE ${OVXDRV_LIBRARIES})
