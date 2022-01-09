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
#ifndef _VSI_NN_OP_RELU_KERAS_INTERNAL_H
#define _VSI_NN_OP_RELU_KERAS_INTERNAL_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define _VSI_NN_RELU_KERAS_INTERNAL_LOCAL_TENSOR_NUM 2

typedef struct _vsi_nn_relu_keras_internal_lcl_data
{
    vx_tensor   local_tensor[_VSI_NN_RELU_KERAS_INTERNAL_LOCAL_TENSOR_NUM];
    uint32_t hash_idx;
    vsi_bool execute_on_sw;
} vsi_nn_relu_keras_internal_lcl_data;

typedef struct _vsi_nn_relu_keras_internal_param
{
    vsi_nn_relu_keras_internal_lcl_data local;

    float     alpha;
    float     max_value;
    float     threshold;
} vsi_nn_relu_keras_internal_param;

#ifdef __cplusplus
}
#endif

#endif
