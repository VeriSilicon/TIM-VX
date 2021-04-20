set(PKG_NAME "OVXDRV")
message(STATUS "Using YOCTO Project configuration.")

# The include directories are available in SDK
set(OVXDRV_INCLUDE_DIRS)
list(APPEND OVXDRV_INCLUDE_DIRS
    ${CMAKE_SYSROOT}/usr/include/
    ${CMAKE_SYSROOT}/usr/include/CL/
)

set(OVXDRV_LIBRARIES)
list(APPEND OVXDRV_LIBRARIES
    libCLC.so
    libGAL.so
    libOpenVX.so
    libOpenVXU.so
    libVSC.so
    libArchModelSw.so
    libNNArchPerf.so)

mark_as_advanced(${OVXDRV_INCLUDE_DIRS} ${OVXDRV_LIBRARIES})