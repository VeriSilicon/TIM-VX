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
#ifndef _VSI_NN_OP_BIDIRECTIONAL_SEQUENCE_LSTM_H
#define _VSI_NN_OP_BIDIRECTIONAL_SEQUENCE_LSTM_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    BI_LSTM_INPUT_INPUT             = 0,

    BI_LSTM_FW_INPUT_WEIGHT_I2I     = 1,
    BI_LSTM_FW_INPUT_WEIGHT_I2F     = 2,
    BI_LSTM_FW_INPUT_WEIGHT_I2C     = 3,
    BI_LSTM_FW_INPUT_WEIGHT_I2O     = 4,

    BI_LSTM_FW_INPUT_WEIGHT_R2I     = 5,
    BI_LSTM_FW_INPUT_WEIGHT_R2F     = 6,
    BI_LSTM_FW_INPUT_WEIGHT_R2C     = 7,
    BI_LSTM_FW_INPUT_WEIGHT_R2O     = 8,

    BI_LSTM_FW_INPUT_WEIGHT_C2I     = 9,
    BI_LSTM_FW_INPUT_WEIGHT_C2F     = 10,
    BI_LSTM_FW_INPUT_WEIGHT_C2O     = 11,

    BI_LSTM_FW_INPUT_BIAS_I2I         = 12,
    BI_LSTM_FW_INPUT_BIAS_I2F         = 13,
    BI_LSTM_FW_INPUT_BIAS_I2C         = 14,
    BI_LSTM_FW_INPUT_BIAS_I2O         = 15,

    BI_LSTM_FW_INPUT_BIAS_R2I        = 16,
    BI_LSTM_FW_INPUT_BIAS_R2F         = 17,
    BI_LSTM_FW_INPUT_BIAS_R2C         = 18,
    BI_LSTM_FW_INPUT_BIAS_R2O         = 19,

    BI_LSTM_FW_INPUT_WEIGHT_PROJ    = 20,
    BI_LSTM_FW_INPUT_BIAS_PROJ      = 21,

    BI_LSTM_BW_INPUT_WEIGHT_I2I     = 22,
    BI_LSTM_BW_INPUT_WEIGHT_I2F     = 23,
    BI_LSTM_BW_INPUT_WEIGHT_I2C     = 24,
    BI_LSTM_BW_INPUT_WEIGHT_I2O     = 25,

    BI_LSTM_BW_INPUT_WEIGHT_R2I     = 26,
    BI_LSTM_BW_INPUT_WEIGHT_R2F     = 27,
    BI_LSTM_BW_INPUT_WEIGHT_R2C     = 28,
    BI_LSTM_BW_INPUT_WEIGHT_R2O     = 29,

    BI_LSTM_BW_INPUT_WEIGHT_C2I     = 30,
    BI_LSTM_BW_INPUT_WEIGHT_C2F     = 31,
    BI_LSTM_BW_INPUT_WEIGHT_C2O     = 32,

    BI_LSTM_BW_INPUT_BIAS_I2I         = 33,
    BI_LSTM_BW_INPUT_BIAS_I2F         = 34,
    BI_LSTM_BW_INPUT_BIAS_I2C         = 35,
    BI_LSTM_BW_INPUT_BIAS_I2O         = 36,

    BI_LSTM_BW_INPUT_BIAS_R2I         = 37,
    BI_LSTM_BW_INPUT_BIAS_R2F         = 38,
    BI_LSTM_BW_INPUT_BIAS_R2C         = 39,
    BI_LSTM_BW_INPUT_BIAS_R2O         = 40,

    BI_LSTM_BW_INPUT_WEIGHT_PROJ    = 41,
    BI_LSTM_BW_INPUT_BIAS_PROJ      = 42,

    BI_LSTM_FW_INPUT_H_STATE        = 43,
    BI_LSTM_FW_INPUT_C_STATE        = 44,

    BI_LSTM_BW_INPUT_H_STATE        = 45,
    BI_LSTM_BW_INPUT_C_STATE        = 46,

    BI_LSTM_AUX_INPUT               = 47,

    BI_LSTM_FW_AUX_INPUT_WEIGHT_I2I = 48,
    BI_LSTM_FW_AUX_INPUT_WEIGHT_I2F = 49,
    BI_LSTM_FW_AUX_INPUT_WEIGHT_I2C = 50,
    BI_LSTM_FW_AUX_INPUT_WEIGHT_I2O = 51,

    BI_LSTM_BW_AUX_INPUT_WEIGHT_I2I = 52,
    BI_LSTM_BW_AUX_INPUT_WEIGHT_I2F = 53,
    BI_LSTM_BW_AUX_INPUT_WEIGHT_I2C = 54,
    BI_LSTM_BW_AUX_INPUT_WEIGHT_I2O = 55,

    BI_LSTM_FW_INPUT_LAYERNORM_I    = 56,
    BI_LSTM_FW_INPUT_LAYERNORM_F    = 57,
    BI_LSTM_FW_INPUT_LAYERNORM_C    = 58,
    BI_LSTM_FW_INPUT_LAYERNORM_O    = 59,

    BI_LSTM_BW_INPUT_LAYERNORM_I    = 60,
    BI_LSTM_BW_INPUT_LAYERNORM_F    = 61,
    BI_LSTM_BW_INPUT_LAYERNORM_C    = 62,
    BI_LSTM_BW_INPUT_LAYERNORM_O    = 63,

    BI_LSTM_INPUT_CNT,

    BI_LSTM_FW_OUTPUT_OUTPUT      = 0,
    BI_LSTM_BW_OUTPUT_OUTPUT      = 1,
    BI_LSTM_FW_OUTPUT_H_STATE     = 2,
    BI_LSTM_BW_OUTPUT_H_STATE     = 3,
    BI_LSTM_FW_OUTPUT_C_STATE     = 4,
    BI_LSTM_BW_OUTPUT_C_STATE     = 5,
    
    BI_LSTM_OUTPUT_CNT
};

typedef struct _vsi_nn_bidirectional_sequence_lstm_lcl_data_t
{
    vsi_bool use_cifg;
    vsi_bool use_layer_norm;
    vsi_bool use_projection;
    vsi_bool use_projection_bias;
    vsi_bool use_hybrid;
    vsi_bool multi_batch;
} vsi_nn_bidirectional_sequence_lstm_lcl_data_t;

typedef struct _vsi_nn_bidirectional_sequence_lstm_param
{
    vsi_nn_bidirectional_sequence_lstm_lcl_data_t *local;
    vsi_bool time_major;
    vsi_bool merge_outputs;
    vsi_nn_activation_e activation;
    float cell_clip;
    float proj_clip;
    float forget_bias;
    vsi_nn_activation_e recurrent_activation;
    vsi_nn_dtype_t *internal_dtype;
} vsi_nn_bidirectional_sequence_lstm_param;

#ifdef __cplusplus
}
#endif

#endif
