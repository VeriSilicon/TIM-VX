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
#ifndef _VSI_NN_OP_CONV1D_H
#define _VSI_NN_OP_CONV1D_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _vsi_nn_conv1d_param
{
    /* local data must be the first. */
    union
    {
        struct _conv1d_local_data_t *local;

        struct {
            vx_tensor input_tensor;
            vx_tensor weight_tensor;
            vx_tensor output_tensor;
        } reserved;
    };

    uint32_t     ksize;
    uint32_t     stride;
    /* Pad left, right */
    uint32_t     pad[2];
    /* Pad type default value shall be AUTO */
    vsi_nn_pad_e pad_type;
    uint32_t     weights;
    uint32_t     group;
    uint32_t     dilation;
    int32_t      multiplier;
} vsi_nn_conv1d_param;
_compiler_assert(offsetof(vsi_nn_conv1d_param, local) == 0, \
    vsi_nn_vsi_nn_conv1d_h );

#ifdef __cplusplus
}
#endif

#endif

