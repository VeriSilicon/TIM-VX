message("src/tim/vx/internal")

option(USE_VXC_BINARY                      "Use VXC binary file"             OFF)

set(OVXLIB_API_ATTR "__attribute__\(\(visibility\(\"default\"\)\)\)")
add_definitions(-DOVXLIB_API=${OVXLIB_API_ATTR})

if(USE_VXC_BINARY)
    if(EXTERNAL_VIV_SDK)
        set(VIV_SDK_PATH ${EXTERNAL_VIV_SDK})
    else()
        set(VIV_SDK_PATH ${PROJECT_SOURCE_DIR}/prebuilt-sdk/x86_64_linux)
    endif()
    if(NOT VCCOMPILER_PATH)
        set(VCCOMPILER_PATH ${PROJECT_SOURCE_DIR}/prebuilt-sdk/x86_64_linux/bin/vcCompiler)
    endif()
    if(NOT GPU_CONFIG_FILE)
        message(FATAL_ERROR "Need set GPU_CONFIG_FILE for vxc binary")
    endif()

    execute_process(COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/vx/internal/ovxlib_bin_build.sh
        ${VIV_SDK_PATH} ${VCCOMPILER_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/vx/internal/ ${GPU_CONFIG_FILE})
    add_definitions(-DVSI_USE_VXC_BINARY=1)
endif()


aux_source_directory(./vx/internal/src INTERNAL_SRC)
aux_source_directory(./vx/internal/src/kernel INTERNAL_KERNEL)
aux_source_directory(./vx/internal/src/kernel/cl INTERNAL_KERNEL_CL)
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

if(TIM_VX_ENABLE_PLATFORM)
    message(STATUS "Using ovxlib vip")
    aux_source_directory(./vx/internal/src/vip INTERNAL_VIPS)
    list(APPEND ${TARGET_NAME}_SRCS
        ${INTERNAL_VIPS}
    )
endif()
