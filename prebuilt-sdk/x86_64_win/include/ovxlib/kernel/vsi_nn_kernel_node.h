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

#ifndef _VSI_NN_KERNEL_NODE_H
#define _VSI_NN_KERNEL_NODE_H

#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include "vsi_nn_prv.h"
#include "vsi_nn_types.h"
#include "kernel/vsi_nn_kernel.h"

vsi_nn_kernel_tensor_t kernel_pad_node
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_tensor_t tensor,
    int32_t * pad_front,
    int32_t * pad_end,
    size_t pad_size,
    vsi_nn_pad_mode_e mode,
    int32_t pad_value,
    vsi_nn_kernel_node_t * out_node
    );

#endif

