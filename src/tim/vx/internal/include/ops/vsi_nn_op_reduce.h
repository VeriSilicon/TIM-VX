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

#ifndef _VSI_NN_OP_REDUCE_H
#define _VSI_NN_OP_REDUCE_H

#include "vsi_nn_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef vx_uint32 vsi_nn_reduce_t; enum
{
    VSI_NN_REDUCE_MEAN = 1,
    VSI_NN_REDUCE_MAX,
    VSI_NN_REDUCE_MIN,
    VSI_NN_REDUCE_SUM,
    VSI_NN_REDUCE_ALL,
    VSI_NN_REDUCE_ANY,
    VSI_NN_REDUCE_PROD,
};

typedef struct _vsi_nn_reduce_lcl_data_t
{
    vsi_nn_tensor_t *axis_tensor;
} vsi_nn_reduce_lcl_data_t;

typedef struct _vsi_nn_reduce_param
{
    /* local data must be the first. */
    vsi_nn_reduce_lcl_data_t local;
    vx_enum     type;
    const int32_t *axis;
    vx_uint32   axis_num;
    vx_bool     keep_dim;
    struct _vsi_nn_reduce_lcl2_data_t* local2;
} vsi_nn_reduce_param;

#ifdef __cplusplus
}
#endif

#endif
