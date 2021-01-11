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
#ifndef _VSI_NN_OP_LSTM_OVXLIB_H
#define _VSI_NN_OP_LSTM_OVXLIB_H

#include "vsi_nn_types.h"
#include "vsi_nn_op_lstmunit.h"

enum
{
    LSTM_INPUT_INPUT        = 0,
    LSTM_INPUT_H_STATE      = 1,
    LSTM_INPUT_C_STATE      = 2,

    LSTM_INPUT_WEIGHT_I2I   = 3,
    LSTM_INPUT_WEIGHT_I2F   = 4,
    LSTM_INPUT_WEIGHT_I2C   = 5,
    LSTM_INPUT_WEIGHT_I2O   = 6,

    LSTM_INPUT_WEIGHT_R2I   = 7,
    LSTM_INPUT_WEIGHT_R2F   = 8,
    LSTM_INPUT_WEIGHT_R2C   = 9,
    LSTM_INPUT_WEIGHT_R2O   = 10,

    LSTM_INPUT_WEIGHT_C2I   = 11,
    LSTM_INPUT_WEIGHT_C2F   = 12,
    LSTM_INPUT_WEIGHT_C2O   = 13,

    LSTM_INPUT_BIAS_I       = 14,
    LSTM_INPUT_BIAS_F       = 15,
    LSTM_INPUT_BIAS_C       = 16,
    LSTM_INPUT_BIAS_O       = 17,

    LSTM_INPUT_WEIGHT_PROJ  = 18,
    LSTM_INPUT_BIAS_PROJ    = 19,

    LSTM_INPUT_LAYERNORM_I  = 20,
    LSTM_INPUT_LAYERNORM_F  = 21,
    LSTM_INPUT_LAYERNORM_C  = 22,
    LSTM_INPUT_LAYERNORM_O  = 23,

    LSTM_INPUT_AUX_INPUT    = 24,
    LSTM_INPUT_AUX_WEIGHT_I2I = 25,
    LSTM_INPUT_AUX_WEIGHT_I2F = 26,
    LSTM_INPUT_AUX_WEIGHT_I2C = 27,
    LSTM_INPUT_AUX_WEIGHT_I2O = 28,

    LSTM_INPUT_CNT,

    LSTM_OUTPUT_OUTPUT      = 0,
    LSTM_OUTPUT_H_STATE     = 1,
    LSTM_OUTPUT_C_STATE     = 2,

    LSTM_OUTPUT_CNT
};

typedef struct _vsi_nn_lstm_ovxlib_lcl_data_t
{
    vsi_bool use_cifg;
    vsi_bool use_layer_norm;
    vsi_bool use_projection;
    vsi_bool use_projection_bias;
    vsi_bool use_hybrid;
    vsi_bool multi_batch;
} vsi_nn_lstm_ovxlib_lcl_data_t;

typedef struct _vsi_nn_lstm_ovxlib_param
{
    vsi_nn_lstm_ovxlib_lcl_data_t local;

    float cell_clip;
    float proj_clip;
    vsi_nn_activation_e activation;
    float forget_bias;
    vsi_bool time_major;
    vsi_nn_dtype_t internal_dtype[LSTMUNIT_QUANTIZE_PARAM_COUNT];
    vsi_nn_activation_e recurrent_activation;
    vsi_bool return_sequences;
    uint32_t weights; /* compatible with LSTM, NOT used */
} vsi_nn_lstm_ovxlib_param;

#endif

