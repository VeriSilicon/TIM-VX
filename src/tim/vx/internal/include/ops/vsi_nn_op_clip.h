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
#ifndef _VSI_NN_OP_CLIP_H
#define _VSI_NN_OP_CLIP_H

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VSI_NN_CLIP_SH_KERNEL_IDX(_INPUT_TYPE, _OUTPUT_TYPE, _IMAGE_DIMS) \
    VSI_NN_CLIP_##_INPUT_TYPE##TO##_OUTPUT_TYPE##_##_IMAGE_DIMS##_KERNEL,


enum {
    CLIP_CPU_KERNEL,
    VSI_NN_CLIP_SH_KERNEL_IDX(F16,  F16,  IMAGE)
    VSI_NN_CLIP_SH_KERNEL_IDX(F16,  I16,  IMAGE)
    VSI_NN_CLIP_SH_KERNEL_IDX(F16,  I8,   IMAGE)
    VSI_NN_CLIP_SH_KERNEL_IDX(F16,  U8,   IMAGE)
    VSI_NN_CLIP_SH_KERNEL_IDX(I16,  F16,  IMAGE)
    VSI_NN_CLIP_SH_KERNEL_IDX(I8,   F16,  IMAGE)
    VSI_NN_CLIP_SH_KERNEL_IDX(U8,   F16,  IMAGE)
    VSI_NN_CLIP_SH_KERNEL_IDX(I16,  I16,  IMAGE)
    VSI_NN_CLIP_SH_KERNEL_IDX(I8,   I8,   IMAGE)
    VSI_NN_CLIP_SH_KERNEL_IDX(U8,   U8,   IMAGE)
    VSI_NN_CLIP_SH_KERNEL_IDX(F16,  F16,  IMAGE_2D)
    VSI_NN_CLIP_SH_KERNEL_IDX(F16,  I16,  IMAGE_2D)
    VSI_NN_CLIP_SH_KERNEL_IDX(F16,  I8,   IMAGE_2D)
    VSI_NN_CLIP_SH_KERNEL_IDX(F16,  U8,   IMAGE_2D)
    VSI_NN_CLIP_SH_KERNEL_IDX(I16,  F16,  IMAGE_2D)
    VSI_NN_CLIP_SH_KERNEL_IDX(I8,   F16,  IMAGE_2D)
    VSI_NN_CLIP_SH_KERNEL_IDX(U8,   F16,  IMAGE_2D)
    VSI_NN_CLIP_SH_KERNEL_IDX(I16,  I16,  IMAGE_2D)
    VSI_NN_CLIP_SH_KERNEL_IDX(I8,   I8,   IMAGE_2D)
    VSI_NN_CLIP_SH_KERNEL_IDX(U8,   U8,   IMAGE_2D)
    CLIP_KERNEL_COUNTS,
};

enum {
    CLIP_INPUT = 0,

    CLIP_INPUTS_COUNT,

    CLIP_OUTPUT = 0,

    CLIP_OUTPUTS_COUNT,

    CLIP_PARAM_COUT = CLIP_INPUTS_COUNT + CLIP_OUTPUTS_COUNT,
};

#define _VSI_NN_CLIP_LOCAL_TENSOR_NUM 2

typedef struct _vsi_nn_clip_lcl_data
{
    vx_tensor   local_tensor[_VSI_NN_CLIP_LOCAL_TENSOR_NUM];
} vsi_nn_clip_lcl_data;

typedef struct _vsi_nn_clip_lcl2_data
{
    vsi_bool is_internal_node;
} vsi_nn_clip_lcl2_data;

typedef struct _vsi_nn_clip_param
{
    /* local data must be the first. */
    vsi_nn_clip_lcl_data local;
    float min;
    float max;
    vsi_nn_clip_lcl2_data *local2;
} vsi_nn_clip_param;

#ifdef __cplusplus
}
#endif

#endif
