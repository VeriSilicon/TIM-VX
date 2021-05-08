set(PKG_NAME "OVXDRV")

set(SDK_URL "https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/arm_android9_A311D_6.4.3.tgz")
# set(SDK_URL "https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/arm_android10_A311D_6.4.3.tgz")
set(TARGET "arm_android9_A311D_6.4.3")

message("Downloading android vim3 SDK ...")
file(DOWNLOAD ${SDK_URL}
    ${PROJECT_BINARY_DIR}/${TARGET}.tgz
    EXPECTED_MD5 "92186fa61db7919aeb166d8823c1c214"
    SHOW_PROGRESS)
execute_process(COMMAND
    tar xf ${PROJECT_BINARY_DIR}/${TARGET}.tgz)

set(OVXDRV_INCLUDE_DIRS)
list(APPEND OVXDRV_INCLUDE_DIRS
    ${PROJECT_BINARY_DIR}/${TARGET}/include
    ${PROJECT_BINARY_DIR}/${TARGET}/include/CL)

set(OVXDRV_LIBRARIES)
list(APPEND OVXDRV_LIBRARIES
    ${PROJECT_BINARY_DIR}/${TARGET}/lib/libCLC.so
    ${PROJECT_BINARY_DIR}/${TARGET}/lib/libGAL.so
    ${PROJECT_BINARY_DIR}/${TARGET}/lib/libOpenVX.so
    ${PROJECT_BINARY_DIR}/${TARGET}/lib/libOpenVXU.so
    ${PROJECT_BINARY_DIR}/${TARGET}/lib/libVSC.so
    ${PROJECT_BINARY_DIR}/${TARGET}/lib/libarchmodelSw.so
    ${PROJECT_BINARY_DIR}/${TARGET}/lib/libNNArchPerf.so)

mark_as_advanced(${OVXDRV_INCLUDE_DIRS} ${OVXDRV_LIBRARIES})