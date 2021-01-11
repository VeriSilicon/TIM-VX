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
#ifndef _VSI_NN_OP_QUANTIZED_16BIT_LSTM_H
#define _VSI_NN_OP_QUANTIZED_16BIT_LSTM_H

#include "vsi_nn_types.h"

enum
{
    Q16_LSTM_INPUT_INPUT        = 0,

    Q16_LSTM_INPUT_WEIGHT_I2I   = 1,
    Q16_LSTM_INPUT_WEIGHT_I2F   = 2,
    Q16_LSTM_INPUT_WEIGHT_I2C   = 3,
    Q16_LSTM_INPUT_WEIGHT_I2O   = 4,

    Q16_LSTM_INPUT_WEIGHT_R2I   = 5,
    Q16_LSTM_INPUT_WEIGHT_R2F   = 6,
    Q16_LSTM_INPUT_WEIGHT_R2C   = 7,
    Q16_LSTM_INPUT_WEIGHT_R2O   = 8,

    Q16_LSTM_INPUT_BIAS_I       = 9,
    Q16_LSTM_INPUT_BIAS_F       = 10,
    Q16_LSTM_INPUT_BIAS_C       = 11,
    Q16_LSTM_INPUT_BIAS_O       = 12,

    Q16_LSTM_INPUT_C_STATE      = 13,
    Q16_LSTM_INPUT_H_STATE      = 14,

    Q16_LSTM_INPUT_CNT,

    Q16_LSTM_OUTPUT_C_STATE     = 0,
    Q16_LSTM_OUTPUT_OUTPUT      = 1,
    Q16_LSTM_OUTPUT_CNT
};

typedef struct _vsi_nn_quantized_16bit_lstm_param
{
    void* local;
} vsi_nn_quantized_16bit_lstm_param;

#endif

