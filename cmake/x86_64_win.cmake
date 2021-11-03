message("Processing x86_64_win32.cmake ...")

set(PKG_NAME "OVXDRV")

set(OVXDRV_INCLUDE_DIRS)
set(TARGET_SDK_NAME x86_64_win)

list(APPEND OVXDRV_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/prebuilt-sdk/${TARGET_SDK_NAME}/include
    ${PROJECT_SOURCE_DIR}/prebuilt-sdk/${TARGET_SDK_NAME}/include/CL)


set(LIBS libCLC libOpenVX libOpenVXU libVSC libArchModelSw libNNArchPerf)
list(LENGTH LIBS count)
math(EXPR count "${count} - 1")
foreach(i RANGE ${count})
    list(GET LIBS ${i} lib)
    find_library(${PKG_NAME}_FOUND_LIB_${i}
        NAMES ${lib}
        PATHS ${PROJECT_SOURCE_DIR}/prebuilt-sdk/${TARGET_SDK_NAME}/lib
    )
    message("found lib ${lib} at ${${PKG_NAME}_FOUND_LIB_${i}}")
    list(APPEND ${PKG_NAME}_LIBRARIES ${${PKG_NAME}_FOUND_LIB_${i}})
    message("--->" ${PKG_NAME}_LIBRARIES ${${PKG_NAME}_FOUND_LIB_${i}})
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PKG_NAME} DEFAULT_MSG
  ${PKG_NAME}_LIBRARIES)

mark_as_advanced(${PKG_NAME}_INCLUDE_DIRS ${PKG_NAME}_LIBRARIES)