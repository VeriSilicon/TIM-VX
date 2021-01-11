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
#ifndef _VSI_NN_OP_PRELU_H
#define _VSI_NN_OP_PRELU_H

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VSI_NN_PRELU_SH_KERNEL_IDX(_AXIS, _INPUT_TYPE, _ALPHA_TYPE, _OUTPUT_TYPE, _IMAGE_DIMS) \
    VSI_NN_PRELU_AXIS##_AXIS##_##_INPUT_TYPE##_ALPHA_TYPE##TO##_OUTPUT_TYPE##_##_IMAGE_DIMS##_KERNEL,

enum {
    PRELU_CPU_KERNEL,
    VSI_NN_PRELU_SH_KERNEL_IDX(0, BF16, F16, BF16, IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, BF16, BF16, BF16, IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, F16,  F16, F16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, F16,  F16, I16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, F16,  F16, I8,   IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, F16,  F16, U8,   IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, I16,  F16, I16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, I8,   F16, I8,   IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, U8,   F16, U8,   IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, I16,  F16, F16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, I8,   F16, F16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, U8,   F16, F16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, BF16, F16, BF16, IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, BF16, BF16, BF16, IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, F16,  F16, F16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, F16,  F16, I16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, F16,  F16, I8,   IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, F16,  F16, U8,   IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, I16,  F16, I16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, I8,   F16, I8,   IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, U8,   F16, U8,   IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, I16,  F16, F16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, I8,   F16, F16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(0, U8,   F16, F16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, BF16, F16, BF16, IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, BF16, BF16, BF16, IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, F16,  F16, F16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, F16,  F16, I16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, F16,  F16, I8,   IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, F16,  F16, U8,   IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, I16,  F16, I16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, I8,   F16, I8,   IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, U8,   F16, U8,   IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, I16,  F16, F16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, I8,   F16, F16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, U8,   F16, F16,  IMAGE)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, BF16, F16, BF16, IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, BF16, BF16, BF16, IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, F16,  F16, F16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, F16,  F16, I16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, F16,  F16, I8,   IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, F16,  F16, U8,   IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, I16,  F16, I16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, I8,   F16, I8,   IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, U8,   F16, U8,   IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, I16,  F16, F16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, I8,   F16, F16,  IMAGE_2D)
    VSI_NN_PRELU_SH_KERNEL_IDX(1, U8,   F16, F16,  IMAGE_2D)
    PRELU_KERNEL_COUNTS,
};

enum {
    PRELLU_INPUT = 0,

    PRELLU_INPUT1,

    PRELLU_INPUTS_COUNT,

    PRELLU_OUTPUT = 0,

    PRELLU_OUTPUTS_COUNT,

    PRELLU_PARAM_COUT = PRELLU_INPUTS_COUNT + PRELLU_OUTPUTS_COUNT,
};

#define _VSI_NN_PRELU_LOCAL_TENSOR_NUM 3

enum {
    PRELLU_STYLE_ORIGINAL = 0,
    PRELLU_STYLE_ANDROID_NN = 1,
    PRELLU_STYLE_CAN_TRANS_ORIGINAL = 2,
};

typedef struct _vsi_nn_prelu_lcl_data
{
    vx_tensor   local_tensor[_VSI_NN_PRELU_LOCAL_TENSOR_NUM];
    uint32_t    hash_idx;
    vsi_bool    execute_on_sw;
    uint32_t    style;
} vsi_nn_prelu_lcl_data;

typedef struct _vsi_nn_prelu_param
{
    /* prelu layer local data structure */
    vsi_nn_prelu_lcl_data *local;
    int32_t    axis;
} vsi_nn_prelu_param;

#ifdef __cplusplus
}
#endif

#endif

