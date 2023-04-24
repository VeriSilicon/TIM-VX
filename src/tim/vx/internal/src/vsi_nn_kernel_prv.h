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
#ifndef _VSI_NN_KERNEL_PRV_H
#define _VSI_NN_KERNEL_PRV_H

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "kernel/vsi_nn_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set sp node name
 *
 * @param[in] node Node handle
 * @param[in] kernel_name Kernel name.
 * @return VSI_SUCCESS on success, or appropriate error code otherwise.
 */
vsi_status vsi_nn_set_sp_kernel_name
    (
        vsi_nn_kernel_node_t node,
        char* kernel_name
    );

vsi_bool vsi_nn_is_sp_supported_broadcast
    (
        vsi_nn_graph_t*   graph,
        vsi_nn_tensor_t** inputs,
        uint32_t          input_num,
        vsi_nn_tensor_t*  output
    );

#ifdef __cplusplus
}
#endif

#endif
