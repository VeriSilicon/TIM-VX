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
#ifndef _VSI_NN_OP_GRUCELL_OVXLIB_H
#define _VSI_NN_OP_GRUCELL_OVXLIB_H

#include "vsi_nn_types.h"
#include "vsi_nn_op_grucell_ovxlib.h"

#define GRUCELL_RZ_GATE_COUNT 2

/* enum for inputs/outputs */
enum
{
    GRUCELL_INPUT_INPUT        = 0,
    GRUCELL_INPUT_H_STATE      = 1,

    GRUCELL_INPUT_WEIGHT_I2R   = 2,
    GRUCELL_INPUT_WEIGHT_I2Z   = 3,

    GRUCELL_INPUT_WEIGHT_H2R   = 4,
    GRUCELL_INPUT_WEIGHT_H2Z   = 5,

    GRUCELL_INPUT_BIAS_I2R     = 6,
    GRUCELL_INPUT_BIAS_I2Z     = 7,

    GRUCELL_INPUT_BIAS_H2R     = 8,
    GRUCELL_INPUT_BIAS_H2Z     = 9,

    GRUCELL_INPUT_WEIGHT_I2C   = 10,
    GRUCELL_INPUT_WEIGHT_H2C   = 11,

    GRUCELL_INPUT_BIAS_I2C     = 12,
    GRUCELL_INPUT_BIAS_H2C     = 13,

    GRUCELL_INPUT_COND_RESET   = 14,
    GRUCELL_INPUT_COND_UPDATE  = 15,
    GRUCELL_INPUT_COND_CANDIDATE    = 16,

    GRUCELL_INPUT_CNT,

    GRUCELL_OUTPUT_OUTPUT      = 0,
    GRUCELL_OUTPUT_H_STATE     = 1,

    GRUCELL_OUTPUT_CNT
};

enum
{
    GRUCELL_QUANTIZE_PARAM_I2R,
    GRUCELL_QUANTIZE_PARAM_I2Z,
    GRUCELL_QUANTIZE_PARAM_H2R,
    GRUCELL_QUANTIZE_PARAM_H2Z,
    GRUCELL_QUANTIZE_PARAM_I2C,
    GRUCELL_QUANTIZE_PARAM_H2C,

    GRUCELL_QUANTIZE_PARAM_COUNT,

    GRUCELL_CUDNN_QUANTIZE_PARAM_INPUT = 0,
    GRUCELL_CUDNN_QUANTIZE_PARAM_HIDDEN,
    GRUCELL_CUDNN_QUANTIZE_PARAM_COUNT
};

enum
{
    GRUCELL_GATE_R = 0,
    GRUCELL_GATE_Z = 1,

    GRUCELL_GATE_COUNT
};

typedef struct _vsi_nn_grucell_ovxlib_param
{
    struct _grucell_ovxlib_local_data_t* local;
    uint32_t num_units;
    vsi_nn_activation_e activation;
    vsi_nn_activation_e recurrent_activation;
    uint32_t linear_before_reset;
    vsi_bool use_cudnn_implementation;
    uint32_t cudnn_implementation_version;
    vsi_nn_dtype_t internal_dtype[GRUCELL_QUANTIZE_PARAM_COUNT];
} vsi_nn_grucell_ovxlib_param;
_compiler_assert(offsetof(vsi_nn_grucell_ovxlib_param, local) == 0, \
    vsi_nn_vsi_nn_grucell_ovxlib_h );

#endif
