/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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

#ifndef _VSI_NN_KERNEL_LUT_H
#define _VSI_NN_KERNEL_LUT_H

#include <stdint.h>

__BEGIN_DECLS

typedef int32_t vsi_nn_kernel_lut_act_e; enum
{
    VSI_NN_KERNEL_LUT_NONE             = 0,
    VSI_NN_KERNEL_LUT_MISH             = 1,
    VSI_NN_KERNEL_LUT_LOG              = 2,
    VSI_NN_KERNEL_LUT_EXP              = 3,
    VSI_NN_KERNEL_LUT_SELU             = 4,
    VSI_NN_KERNEL_LUT_NEG              = 5,
    VSI_NN_KERNEL_LUT_HSIGMOID         = 6,
    VSI_NN_KERNEL_LUT_SOFT_PLUS        = 7,
    VSI_NN_KERNEL_LUT_ERF              = 8,
    VSI_NN_KERNEL_LUT_GELU             = 9,
    VSI_NN_KERNEL_LUT_HGELU            = 10,
    VSI_NN_KERNEL_LUT_RELU_KERAS       = 11,
    VSI_NN_KERNEL_LUT_CLIP             = 12,
    VSI_NN_KERNEL_LUT_SQUARE           = 13,
    VSI_NN_KERNEL_LUT_CELU             = 14,
    VSI_NN_KERNEL_LUT_RCP              = 15,
    VSI_NN_KERNEL_LUT_SOFTSIGN         = 16,
};

#define VSI_NN_KERNEL_LUT_MAX_SIZE  (1024)
#define VSI_NN_KERNEL_LUT_FP16_MAX  (57344)
#define VSI_NN_KERNEL_LUT_FP16_MIN  (-57344)

typedef struct _vsi_nn_kernel_lut_
{
    float index;
    float val;
} vsi_nn_kernel_lut_t;

typedef struct  _vsi_nn_kernel_lut_params
{
    vsi_enum act_type;
    float params[16];
} vsi_nn_kernel_lut_params;

vsi_status vsi_nn_kernel_lut
    (
    vx_lut index_lut,
    vx_lut output_lut,
    vsi_nn_kernel_lut_params *param
    );

__END_DECLS

#endif
