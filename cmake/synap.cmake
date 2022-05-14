if(NOT EXISTS ${SYNAP_DIR})
    message(FATAL_ERROR "not existing SYNAP_DIR: ${SYNAP_DIR}")
endif()

message("include SyNAP driver sdk from ${SYNAP_DIR}")

if(CMAKE_CROSSCOMPILING)
    if(ANDROID)
        set(PLATFORM armv7a-android-ndk-api30)
    elseif(NOT PLATFORM)
        message(FATAL_ERROR "Unsupported platform")
    endif()
else()
    set(PLATFORM x86_64-linux-gcc)
endif()

set(TIM_VX_USE_EXTERNAL_OVXLIB ON)
set(TIM_VX_ENABLE_VIPLITE OFF)
set(OVXLIB_LIB ${SYNAP_DIR}/lib/${PLATFORM}/libovxlib.so)
set(OVXLIB_INC ${SYNAP_DIR}/include/ovxlib)
set(OVXDRV_INCLUDE_DIRS
    ${SYNAP_DIR}/include
    ${SYNAP_DIR}/include/CL)
