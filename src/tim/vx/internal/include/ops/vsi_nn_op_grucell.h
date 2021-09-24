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
#ifndef _VSI_NN_OP_GRUCELL_H
#define _VSI_NN_OP_GRUCELL_H

#include "vsi_nn_types.h"

enum
{
    GRUCELL_GATES_Z = 0,
    GRUCELL_GATES_R = 1,
    GRUCELL_GATES_H = 2,

    GRUCELL_GATE_CNT
};

/* Define the inputs and outputs for GRUCell */
enum
{
    GRUCELL_IN_INPUT        = 0,
    GRUCELL_IN_H_STATE      = 1,

    /* input kernel */
    GRUCELL_IN_KERNEL_I2Z   = 2,
    GRUCELL_IN_KERNEL_I2R   = 3,
    GRUCELL_IN_KERNEL_I2H   = 4,

    /* recurrent kernel */
    GRUCELL_IN_KERNEL_R2Z   = 5,
    GRUCELL_IN_KERNEL_R2R   = 6,
    GRUCELL_IN_KERNEL_R2H   = 7,

    /* input bias */
    GRUCELL_IN_BIAS_I2Z     = 8,
    GRUCELL_IN_BIAS_I2R     = 9,
    GRUCELL_IN_BIAS_I2H     = 10,

    /* recurrent bias */
    GRUCELL_IN_BIAS_R2Z     = 11,
    GRUCELL_IN_BIAS_R2R     = 12,
    GRUCELL_IN_BIAS_R2H     = 13,

    GRUCELL_IN_CNT,

    GRUCELL_OUT_OUTPUT      = 0,
    GRUCELL_OUT_H_STATE     = 1,

    GRUCELL_OUT_CNT
};

typedef struct _vsi_nn_grucell_param
{
    struct _vsi_nn_grucell_local * local;

    uint32_t num_units;
    vsi_nn_activation_e activation;
    vsi_nn_activation_e recurrent_activation;
    vsi_bool reset_after;
} vsi_nn_grucell_param;
_compiler_assert(offsetof(vsi_nn_grucell_param, local) == 0, \
                 vsi_nn_conv1d_h );

#endif