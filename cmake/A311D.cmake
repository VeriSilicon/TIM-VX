set(PKG_NAME "OVXDRV")
message("Downloading A311D toolchain ...")
file(DOWNLOAD  "https://cnbj1.fds.api.xiaomi.com/mace/third-party/gcc-linaro/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz"
    ${PROJECT_BINARY_DIR}/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz
    EXPECTED_MD5 "6ec789d642584a01e240ab3366599dbb"
    SHOW_PROGRESS)
execute_process(COMMAND
    tar xf ${PROJECT_BINARY_DIR}/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz)

set(TOOLCHAIN_DIR  ${PROJECT_BINARY_DIR}/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-linux-gnu-g++)
set(CMAKE_AR ${TOOLCHAIN_DIR}/bin/aarch64-linux-gnu-gcc-ar)
set(CMAKE_AS ${TOOLCHAIN_DIR}/bin/aarch64-linux-gnu-gcc-as)
set(CMAKE_LD ${TOOLCHAIN_DIR}/bin/aarch64-linux-gnu-gcc-ld)

message("Downloading A311D SDK ...")
file(DOWNLOAD "https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.34/aarch64_A311D_6.4.8.tgz"
    ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8.tgz
    EXPECTED_MD5 "da530e28f73fd8b143330b6d1b97a1d8"
    SHOW_PROGRESS)
execute_process(COMMAND
    tar xf ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8.tgz)

set(OVXDRV_INCLUDE_DIRS)
list(APPEND OVXDRV_INCLUDE_DIRS
    ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8/include
    ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8/include/CL)

set(OVXDRV_LIBRARIES)
list(APPEND OVXDRV_LIBRARIES
    ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8/lib/libCLC.so
    ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8/lib/libGAL.so
    ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8/lib/libOpenVX.so
    ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8/lib/libOpenVXU.so
    ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8/lib/libVSC.so
    ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8/lib/libArchModelSw.so
    ${PROJECT_BINARY_DIR}/aarch64_A311D_6.4.8/lib/libNNArchPerf.so)

mark_as_advanced(${OVXDRV_INCLUDE_DIRS} ${OVXDRV_LIBRARIES})
