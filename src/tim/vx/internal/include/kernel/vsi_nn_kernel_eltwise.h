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

#ifndef _VSI_NN_KERNEL_ELTWISE_H
#define _VSI_NN_KERNEL_ELTWISE_H

#include <stdint.h>
#include "kernel/vsi_nn_kernel.h"

vsi_bool vsi_nn_kernel_optimize_eltwise_shape
    (
    const vsi_size_t* shape_x, const vsi_size_t rank_x,
    const vsi_size_t* shape_y, const vsi_size_t rank_y,
    const vsi_size_t* shape_output, const vsi_size_t rank_output,
    vsi_size_t* out_shape_x, vsi_size_t* out_shape_y,
    vsi_size_t* out_shape_output, vsi_size_t* out_rank_output
    );

vsi_bool vsi_nn_kernel_optimize_broadcast_shape
    (
    const vsi_size_t** shape_in, const vsi_size_t* rank_in,
    const int32_t   input_num,
    const vsi_size_t*  shape_output, const vsi_size_t rank_output,
    vsi_size_t** out_shape_in,
    vsi_size_t* out_shape_output, uint32_t* out_rank_output
    );

#endif
