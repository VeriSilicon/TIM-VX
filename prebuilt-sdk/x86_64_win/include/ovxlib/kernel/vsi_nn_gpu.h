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

#ifndef _VSI_NN_GPU_H
#define _VSI_NN_GPU_H

#include "vsi_nn_gpu_config.h"

#define gpu_min(x, y)       (((x) <= (y)) ?  (x) :  (y))

#define gpu_max(x, y)       (((x) >= (y)) ?  (x) :  (y))

#define gpu_postshift( x )  (gpu_min( x, GPU_MAX_POST_SHIFT_BITS))
#define gpu_multiplier( x )  (gpu_min( x, GPU_MAX_MULTIPLIER_NUM))

// Alignment with a power of two value.
#define gpu_align_np2(n, align) (((n) + (align) - 1) - (((n) + (align) - 1) % (align)))

#define gpu_align_np2_safe(n, align)                                        \
(                                                                          \
    (gpu_align_np2((n) & ~0ULL, (align) & ~0ULL) ^ gpu_align_np2(n, align)) ?   \
        (n) : gpu_align_np2(n, align)                                       \
)
#define gpu_align_p2(n, align) ((n) + ((align) - 1)) & ~((align) - 1)

#define GPU_MAX_DIMENSION_SIZE      (3)

typedef enum
{
    GPU_DP_TYPE_16,
    GPU_DP_TYPE_32,
} gpu_dp_type_e;

typedef struct
{
    // 512 byte data
    uint32_t data[16];
    gpu_dp_type_e type;
} gpu_dp_inst_t;

typedef struct
{
    uint32_t dim;
    size_t   global_offset[GPU_MAX_DIMENSION_SIZE];
    size_t   global_scale[GPU_MAX_DIMENSION_SIZE];
    size_t   local_size[GPU_MAX_DIMENSION_SIZE];
    size_t   global_size[GPU_MAX_DIMENSION_SIZE];
} gpu_param_t;

void gpu_dp_inst_update_postshfit
    (
    gpu_dp_inst_t * dp_inst,
    int32_t shift
    );

void gpu_dp_inst_update_multiplier
    (
    gpu_dp_inst_t * dp_inst,
    int32_t start,
    int32_t end,
    int32_t multiplier
    );

void gpu_quantize_multiplier_16bit
    (
    double double_multipier,
    uint16_t * quantize_multiplier,
    int32_t * shift
    );

void gpu_quantize_multiplier_32bit
    (
    double double_multipier,
    uint32_t * quantize_multiplier,
    int32_t * shift
    );

#endif

