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

/* WARNING! AUTO-GENERATED, DO NOT MODIFY MANUALLY */

#ifndef _VSI_NN_LIBNNEXT_RESOURCE_H
#define _VSI_NN_LIBNNEXT_RESOURCE_H

#include "kernel/vsi_nn_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load gpu source code
 */
const char* vsi_nn_resource_load_source_code
    (
    const char* source_name,
    size_t* size,
    vsi_nn_kernel_type_e type
    );

#ifdef __cplusplus
}
#endif

#endif /* _VSI_NN_LIBNNEXT_RESOURCE_H */
