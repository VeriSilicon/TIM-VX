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
#ifndef _VSI_NN_OP_CLIENT_INSTANCENORMALIZE_H
#define _VSI_NN_OP_CLIENT_INSTANCENORMALIZE_H

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VSI_NN_SUMSQR_SH_KERNEL_IDX(_INPUT0_TYPE, _RESHAPE_FLAG) \
    VSI_NN_SUMSQR_##_INPUT0_TYPE##_RESHAPE_FLAG##_KERNEL,

#define VSI_NN_INSTANCENORM_SH_KERNEL_IDX(_INPUT0_TYPE, _OUTPUT_TYPE, _RESHAPE_FLAG) \
    VSI_NN_INSTANCENORM_##_INPUT0_TYPE##TO##_OUTPUT_TYPE##_RESHAPE_FLAG##_KERNEL,

enum {
    INSTANCENORM_CPU_KERNEL,
    VSI_NN_SUMSQR_SH_KERNEL_IDX(U8, 1)
    VSI_NN_SUMSQR_SH_KERNEL_IDX(I8, 1)
    VSI_NN_SUMSQR_SH_KERNEL_IDX(I16, 1)
    VSI_NN_SUMSQR_SH_KERNEL_IDX(F16, 1)
    VSI_NN_SUMSQR_SH_KERNEL_IDX(U8, 0)
    VSI_NN_SUMSQR_SH_KERNEL_IDX(I8, 0)
    VSI_NN_SUMSQR_SH_KERNEL_IDX(I16, 0)
    VSI_NN_SUMSQR_SH_KERNEL_IDX(F16, 0)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(U8, U8, 1)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(U8, F16, 1)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(I8, I8, 1)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(I8, F16, 1)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(I16, I16, 1)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(I16, F16, 1)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(F16, F16, 1)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(U8, U8, 0)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(U8, F16, 0)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(I8, I8, 0)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(I8, F16, 0)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(I16, I16, 0)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(I16, F16, 0)
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(F16, F16, 0)
};

#define _VSI_NN_INSTANCENORM_LOCAL_TENSOR_NUM 5

typedef struct _vsi_nn_instancenorm_lcl_data2
{
    uint32_t reshapeFlg;
    uint32_t hash_idx;
    vsi_bool execute_on_sw;

    /* handle 3D instance norm */
    vsi_nn_tensor_t *reshaped_input;
    vsi_nn_tensor_t *reshaped_output;
} vsi_nn_instancenorm_lcl_data2;

typedef struct _vsi_nn_instancenorm_lcl_data
{
    vx_tensor   local_tensor[_VSI_NN_INSTANCENORM_LOCAL_TENSOR_NUM];
} vsi_nn_instancenorm_lcl_data;

typedef struct _vsi_nn_instancenormalize_param
{
    /* local data must be the first. */
    vsi_nn_instancenorm_lcl_data local;
    float eps;
    int axis_num;
    int* axis;
    vsi_nn_instancenorm_lcl_data2* lcl2_data;
} vsi_nn_instancenormalize_param;

#ifdef __cplusplus
}
#endif

#endif
