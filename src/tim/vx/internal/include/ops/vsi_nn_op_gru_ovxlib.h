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
#ifndef _VSI_NN_OP_GRU_OVXLIB_H
#define _VSI_NN_OP_GRU_OVXLIB_H

#include "vsi_nn_types.h"
#include "vsi_nn_op_grucell_ovxlib.h"

/* enum for inputs/outputs */
enum
{
    GRU_INPUT_INPUT        = 0,
    GRU_INPUT_H_STATE      = 1,

    GRU_INPUT_WEIGHT_I2R   = 2,
    GRU_INPUT_WEIGHT_I2Z   = 3,

    GRU_INPUT_WEIGHT_H2R   = 4,
    GRU_INPUT_WEIGHT_H2Z   = 5,

    GRU_INPUT_BIAS_I2R     = 6,
    GRU_INPUT_BIAS_I2Z     = 7,

    GRU_INPUT_BIAS_H2R     = 8,
    GRU_INPUT_BIAS_H2Z     = 9,

    GRU_INPUT_WEIGHT_I2C   = 10,
    GRU_INPUT_WEIGHT_H2C   = 11,

    GRU_INPUT_BIAS_I2C     = 12,
    GRU_INPUT_BIAS_H2C     = 13,

    GRU_INPUT_CNT,

    GRU_OUTPUT_OUTPUT      = 0,
    GRU_OUTPUT_H_STATE     = 1,

    GRU_OUTPUT_CNT
};

typedef struct _vsi_nn_gru_ovxlib_param
{
    uint32_t num_units;
    vsi_bool time_major;
    vsi_nn_activation_e activation;
    vsi_nn_activation_e recurrent_activation;
    vsi_bool return_sequences;
    uint32_t linear_before_reset;
    vsi_nn_dtype_t internal_dtype[GRUCELL_QUANTIZE_PARAM_COUNT];

    struct _gru_ovxlib_local_data_t *local;
    vsi_bool use_cudnn_implementation;
    uint32_t cudnn_implementation_version;
} vsi_nn_gru_ovxlib_param;

#endif

