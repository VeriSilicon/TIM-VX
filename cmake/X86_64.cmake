message("Processing X86_64.cmake ...")
set(PKG_NAME "OVXDRV")

set(OVXDRV_INCLUDE_DIRS)

list(APPEND OVXDRV_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/prebuilt-sdk/x86_64_linux/include
    ${PROJECT_SOURCE_DIR}/prebuilt-sdk/x86_64_linux/include/CL)


set(LIBS CLC GAL OpenVX OpenVXU VSC ArchModelSw NNArchPerf)
list(LENGTH LIBS count)
math(EXPR count "${count} - 1")
foreach(i RANGE ${count})
    list(GET LIBS ${i} lib)
    find_library(${PKG_NAME}_FOUND_LIB_${i}
        NAMES ${lib}
        PATHS ${PROJECT_SOURCE_DIR}/prebuilt-sdk/x86_64_linux/lib
    )
    message("found lib ${lib} at ${${PKG_NAME}_FOUND_LIB_${i}}")
    list(APPEND ${PKG_NAME}_LIBRARIES ${${PKG_NAME}_FOUND_LIB_${i}})
    message("--->" ${PKG_NAME}_LIBRARIES ${${PKG_NAME}_FOUND_LIB_${i}})
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${PKG_NAME} DEFAULT_MSG
  ${PKG_NAME}_LIBRARIES)

mark_as_advanced(${PKG_NAME}_INCLUDE_DIRS ${PKG_NAME}_LIBRARIES)