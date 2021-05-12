set(PKG_NAME "OVXDRV")

message("include driver sdk from ${VIP_SDK_DIR}")
set(OVXDRV_INCLUDE_DIRS)
list(APPEND OVXDRV_INCLUDE_DIRS
        ${VIP_SDK_DIR}include
        ${VIP_SDK_DIR}include/CL)

set(OVXDRV_LIBRARIES)
list(APPEND OVXDRV_LIBRARIES
    ${VIP_SDK_DIR}/drivers/libCLC.so
    ${VIP_SDK_DIR}/drivers/libGAL.so
    ${VIP_SDK_DIR}/drivers/libOpenVX.so
    ${VIP_SDK_DIR}/drivers/libOpenVXU.so
    ${VIP_SDK_DIR}/drivers/libVSC.so
    ${VIP_SDK_DIR}/drivers/libArchModelSw.so
    ${VIP_SDK_DIR}/drivers/libNNArchPerf.so)

mark_as_advanced(${OVXDRV_INCLUDE_DIRS} ${OVXDRV_LIBRARIES})