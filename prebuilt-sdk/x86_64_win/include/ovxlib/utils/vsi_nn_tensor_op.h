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
#ifndef __VSI_NN_TENSOR_OP_H__
#define __VSI_NN_TENSOR_OP_H__

#include "vsi_nn_graph.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_types.h"

vsi_nn_tensor_t* vsi_nn_Concat
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t** tensors,
    uint32_t tensor_num,
    uint32_t axis
    );

vsi_nn_tensor_t* vsi_nn_ConvertTensorDtype
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t* tensor,
    const vsi_nn_dtype_t* dst_dtype
    );

vsi_nn_tensor_t* vsi_nn_TensorAdd
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t** tensors,
    uint32_t tensor_num,
    vsi_nn_tensor_attr_t output_attr
    );

#endif
