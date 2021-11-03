/*
 * Copyright (c) 2012-2018 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

#ifndef _OPENVX_IMPORT_KERNEL_H_
#define _OPENVX_IMPORT_KERNEL_H_

#include <VX/vx.h>

/*!
 * \file
 * \brief The OpenVX import kernel extension API.
 */
#define OPENVX_KHR_IMPORT_KERNEL  "vx_khr_import_kernel"

/*! \brief The import kernel extension library set
 *  \ingroup group_import_kernel
 */
#define VX_LIBRARY_KHR_IMPORT_KERNEL_EXTENSION (0x5)

/*
define type for vxImportKernelFromURL() function
*/
#define VX_VIVANTE_IMPORT_KERNEL_FROM_FILE          "vx_vivante_file"
#define VX_VIVANTE_IMPORT_KERNEL_FROM_FOLDER        "vx_vivante_folder"
#define VX_VIVANTE_IMPORT_KERNEL_FROM_LABEL         "vx_vivante_label"
#define VX_VIVANTE_IMPORT_KERNEL_FROM_POINTER       "vx_vivante_pointer"

#ifdef  __cplusplus
extern "C" {
#endif

/*! \brief Import a kernel from binary specified by URL.
 *
 * The name of kernel parameters can be queried using the vxQueryReference API
 * with vx_parameter as ref and VX_REFERENCE_NAME as attribute.
 *
 * \param context [in] The OpenVX context
 * \param type [in] Vendor-specific identifier that indicates to the implementation
 *   how to interpret the url. For example, if an implementation can interpret the url
 *   as a file, a folder a symbolic label, or a pointer, then a vendor may choose
 *   to use "vx_<vendor>_file", "vx_<vendor>_folder", "vx_<vendor>_label", and
 *   "vx_<vendor>_pointer", respectively for this field. Container types starting
 *   with "vx_khr_" are reserved. Refer to vendor documentation for list of
 *   container types supported
 * \param url [in] URL to binary container.
 *
 * \retval On success, a valid vx_kernel object. Calling vxGetStatus with the return value
 *   as a parameter will return VX_SUCCESS if the function was successful.
 *
 * \ingroup group_import_kernel
 */
VX_API_ENTRY vx_kernel VX_API_CALL vxImportKernelFromURL(
        vx_context context,
        const vx_char * type,
        const vx_char * url
    );

#ifdef  __cplusplus
}
#endif

#endif

