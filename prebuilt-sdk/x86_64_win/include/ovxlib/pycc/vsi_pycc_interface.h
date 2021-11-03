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
#ifndef _VSI_PYCC_INTERFACE_H
#define _VSI_PYCC_INTERFACE_H

#include <stdint.h>

#include "vsi_nn_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _vsi_pycc_params_t
{
    uint32_t   op;
    uint32_t   weights;

    uint32_t   ksize_h;
    uint32_t   ksize_w;
    uint32_t   stride_h;
    uint32_t   stride_w;
    uint32_t   pad_left;
    uint32_t   pad_right;
    uint32_t   pad_top;
    uint32_t   pad_bottom;
    uint32_t   dilation_h;
    uint32_t   dilation_w;

    uint32_t   pool_type;
    vsi_nn_round_type_e   pool_round_type;
    uint32_t   pool_ksize_h;
    uint32_t   pool_ksize_w;
    uint32_t   pool_stride_h;
    uint32_t   pool_stride_w;
    uint32_t   pool_pad_left;
    uint32_t   pool_pad_right;
    uint32_t   pool_pad_top;
    uint32_t   pool_pad_bottom;

    int32_t    enable_relu;

    uint8_t  * weight_stream;
    uint32_t   weight_stream_size;
    uint8_t  * bias_stream;
    uint32_t   bias_stream_size;

    vsi_nn_tensor_attr_t input_attr;
    vsi_nn_tensor_attr_t output_attr;
    vsi_nn_tensor_attr_t weight_attr;
    vsi_nn_tensor_attr_t bias_attr;

    int32_t    onlyHeaderStream;
} vsi_pycc_params;

void vsi_pycc_VdataGeneratorInit();

void vsi_pycc_VdataGeneratorDeinit();

uint32_t vsi_pycc_VdataCreate
    (
    vsi_pycc_params * pycc_params,
    uint8_t * buffer
    );

#ifdef __cplusplus
}
#endif

#endif
