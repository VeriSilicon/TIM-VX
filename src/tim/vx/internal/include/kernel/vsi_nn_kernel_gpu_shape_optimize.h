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

#ifndef _VSI_NN_KERNEL_GPU_SHAPE_OPTIMIZE_H
#define _VSI_NN_KERNEL_GPU_SHAPE_OPTIMIZE_H

#include <stdint.h>
#include "kernel/vsi_nn_kernel.h"

vsi_bool vsi_nn_kernel_optimize_reduce_shape
    (
    const int32_t* shape_x, const size_t rank_x,
    const int32_t *axis, const size_t axis_size,
    const int32_t* shape_output, const size_t rank_output,
    int32_t* out_shape_x, uint32_t* out_rank_x,
    int32_t* out_shape_output, uint32_t* out_rank_output,
    int32_t* out_axis, uint32_t* out_axis_size
    );

vsi_bool vsi_nn_kernel_optimize_element_shape
    (
    const int32_t* shape_x, const size_t rank_x,
    int32_t* out_shape_x, int32_t* out_rank_x
    );

vsi_bool vsi_nn_kernel_optimize_softmax_shape
    (
    const int32_t* shape_x, const size_t rank_x, const int32_t axis,
    int32_t* out_shape_x, uint32_t* out_rank_x,int32_t* out_axis
    );

vsi_bool vsi_nn_kernel_optimize_tile_shape
    (
    const int32_t* shape_x,   const size_t rank_x,
    const int32_t* multiples, const size_t rank,
    const int32_t* shape_output, const size_t rank_output,
    int32_t* out_shape_x, int32_t* out_shape_y,
    int32_t* out_shape_output, uint32_t* out_rank_output
    );

#endif
