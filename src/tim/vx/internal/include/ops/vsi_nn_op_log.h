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
#ifndef _VSI_NN_OP_LOG_H
#define _VSI_NN_OP_LOG_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VSI_NN_LOG_SH_KERNEL_IDX(_INPUT_TYPE, _OUTPUT_TYPE, _IMAGE_DIMS) \
    VSI_NN_LOG_##_INPUT_TYPE##TO##_OUTPUT_TYPE##_##_IMAGE_DIMS##_KERNEL,

enum {
    LOG_CPU_KERNEL,

    VSI_NN_LOG_SH_KERNEL_IDX(F16,  F16,  IMAGE_3D)
    VSI_NN_LOG_SH_KERNEL_IDX(F16,  I16,  IMAGE_3D)
    VSI_NN_LOG_SH_KERNEL_IDX(F16,  I8,   IMAGE_3D)
    VSI_NN_LOG_SH_KERNEL_IDX(F16,  U8,   IMAGE_3D)
    VSI_NN_LOG_SH_KERNEL_IDX(I16,  I16,  IMAGE_3D)
    VSI_NN_LOG_SH_KERNEL_IDX(I16,  F16,  IMAGE_3D)
    VSI_NN_LOG_SH_KERNEL_IDX(I8,   I8,   IMAGE_3D)
    VSI_NN_LOG_SH_KERNEL_IDX(I8,   F16,  IMAGE_3D)
    VSI_NN_LOG_SH_KERNEL_IDX(U8,   U8,   IMAGE_3D)
    VSI_NN_LOG_SH_KERNEL_IDX(U8,   F16,  IMAGE_3D)
    VSI_NN_LOG_SH_KERNEL_IDX(BF16, BF16, IMAGE_3D)

    VSI_NN_LOG_SH_KERNEL_IDX(F16,  F16,  IMAGE_2D)
    VSI_NN_LOG_SH_KERNEL_IDX(F16,  I16,  IMAGE_2D)
    VSI_NN_LOG_SH_KERNEL_IDX(F16,  I8,   IMAGE_2D)
    VSI_NN_LOG_SH_KERNEL_IDX(F16,  U8,   IMAGE_2D)
    VSI_NN_LOG_SH_KERNEL_IDX(I16,  I16,  IMAGE_2D)
    VSI_NN_LOG_SH_KERNEL_IDX(I16,  F16,  IMAGE_2D)
    VSI_NN_LOG_SH_KERNEL_IDX(I8,   I8,   IMAGE_2D)
    VSI_NN_LOG_SH_KERNEL_IDX(I8,   F16,  IMAGE_2D)
    VSI_NN_LOG_SH_KERNEL_IDX(U8,   U8,   IMAGE_2D)
    VSI_NN_LOG_SH_KERNEL_IDX(U8,   F16,  IMAGE_2D)
    VSI_NN_LOG_SH_KERNEL_IDX(BF16, BF16, IMAGE_2D)
};

enum {
    TENSOR_LOG_INPUT,

    TENSOR_LOG_INPUTS_COUNT,

    TENSOR_LOG_OUTPUT = 0,

    TENSOR_LOG_OUTUTS_COUNT,

    TENSOR_LOG_PARAM_COUT = TENSOR_LOG_INPUTS_COUNT + TENSOR_LOG_OUTUTS_COUNT,
};

enum {
    TENSOR_LOG_CPU_KERNEL,

    TENSOR_LOG_F16TOF16_KERNEL,
    TENSOR_LOG_F16TOI16_KERNEL,
    TENSOR_LOG_F16TOI8_KERNEL,
    TENSOR_LOG_F16TOU8_KERNEL,
    TENSOR_LOG_I16TOI16_KERNEL,
    TENSOR_LOG_I16TOF16_KERNEL,
    TENSOR_LOG_I8TOI8_KERNEL,
    TENSOR_LOG_I8TOF16_KERNEL,
    TENSOR_LOG_U8TOU8_KERNEL,
    TENSOR_LOG_U8TOF16_KERNEL,

    TENSOR_LOG_F16TOF16_2D_KERNEL,
    TENSOR_LOG_F16TOI16_2D_KERNEL,
    TENSOR_LOG_F16TOI8_2D_KERNEL,
    TENSOR_LOG_F16TOU8_2D_KERNEL,
    TENSOR_LOG_I16TOI16_2D_KERNEL,
    TENSOR_LOG_I16TOF16_2D_KERNEL,
    TENSOR_LOG_I8TOI8_2D_KERNEL,
    TENSOR_LOG_I8TOF16_2D_KERNEL,
    TENSOR_LOG_U8TOU8_2D_KERNEL,
    TENSOR_LOG_U8TOF16_2D_KERNEL,

    TENSOR_LOG_KERNEL_COUNTS,
};

#define _VSI_NN_LOG_LOCAL_TENSOR_NUM 2

typedef struct _vsi_nn_log_lcl_data
{
    vx_tensor   local_tensor[_VSI_NN_EXP_LOCAL_TENSOR_NUM];
    uint32_t    hash_idx;
    vsi_bool    execute_on_sw;
} vsi_nn_log_lcl_data;

typedef struct _vsi_nn_log_param
{
    /* log layer local data structure */
    vsi_nn_log_lcl_data local;
} vsi_nn_log_param;

#ifdef __cplusplus
}
#endif

#endif
