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
#ifndef _VSI_NN_OP_POOL_H
#define _VSI_NN_OP_POOL_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define _VSI_NN_POOLWITHARGMAX_LOCAL_TENSOR_NUM 3

typedef struct _vsi_nn_poolwithargmax_lcl_data
{
    vx_tensor   local_tensor[_VSI_NN_POOLWITHARGMAX_LOCAL_TENSOR_NUM];
} vsi_nn_poolwithargmax_lcl_data;

typedef struct _vsi_nn_pool_param
{
    vsi_enum     type;
    /* round_type is used to calculate the output shape */
    vsi_nn_round_type_e round_type;
    uint32_t     ksize[2];
    uint32_t     stride[2];
    /* Pad left, right, top, bottom */
    uint32_t     pad[4];
    /* Pad type default value shall be AUTO */
    vsi_nn_pad_e pad_type;
    /* poolwithargmax layer local data structure */
    vsi_nn_poolwithargmax_lcl_data local;
} vsi_nn_pool_param;

#ifdef __cplusplus
}
#endif

#endif

