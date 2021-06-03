set(PKG_NAME "OVXDRV")

message("include driver sdk from ${EXTERNAL_VIV_SDK}")
set(OVXDRV_INCLUDE_DIRS)
list(APPEND OVXDRV_INCLUDE_DIRS
    ${EXTERNAL_VIV_SDK}/include
    ${EXTERNAL_VIV_SDK}/include/CL)

set(OVXDRV_LIBRARIES)
list(APPEND OVXDRV_LIBRARIES
    ${EXTERNAL_VIV_SDK}/drivers/libCLC.so
    ${EXTERNAL_VIV_SDK}/drivers/libGAL.so
    ${EXTERNAL_VIV_SDK}/drivers/libOpenVX.so
    ${EXTERNAL_VIV_SDK}/drivers/libOpenVXU.so
    ${EXTERNAL_VIV_SDK}/drivers/libVSC.so
    ${EXTERNAL_VIV_SDK}/drivers/libArchModelSw.so
    ${EXTERNAL_VIV_SDK}/drivers/libNNArchPerf.so)

mark_as_advanced(${OVXDRV_INCLUDE_DIRS} ${OVXDRV_LIBRARIES})
