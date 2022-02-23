/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
/** @file */
#ifndef _VSI_NN_VERSION_H_
#define _VSI_NN_VERSION_H_

#include "vsi_nn_types.h"

#if defined(__cplusplus)
extern "C"{
#endif

#define VSI_NN_VERSION_MAJOR 1
#define VSI_NN_VERSION_MINOR 1
#define VSI_NN_VERSION_PATCH 39
#define VSI_NN_VERSION \
    (VSI_NN_VERSION_MAJOR * 10000 + VSI_NN_VERSION_MINOR * 100 + VSI_NN_VERSION_PATCH)

/**
 * Ovxlib version check
 * Ovxlib will check the suitable version at compile time.
 * @note Ovxlib version should be always greater or equal to case version.
 */
#define _version_assert _compiler_assert

/**
 * Get ovxlib version
 * Get ovxlib version string.
 */
OVXLIB_API const char *vsi_nn_GetVersion(void);

/**
 * Get ovxlib version major
 * Get ovxlib version major, return integer value.
 */
OVXLIB_API uint32_t vsi_nn_GetVersionMajor(void);

/**
 * Get ovxlib version minor
 * Get ovxlib version minor, return integer value.
 */
OVXLIB_API uint32_t vsi_nn_GetVersionMinor(void);

/**
 * Get ovxlib version patch
 * Get ovxlib version patch, return integer value.
 */
OVXLIB_API uint32_t vsi_nn_GetVersionPatch(void);

#if defined(__cplusplus)
}
#endif
#endif
