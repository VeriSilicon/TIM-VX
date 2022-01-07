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
#ifndef _VSI_NN_OP_CONV2D_LSTM_H
#define _VSI_NN_OP_CONV2D_LSTM_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    CONV2D_LSTM_IN_INPUT        = 0,
    CONV2D_LSTM_IN_H_STATE      = 1,
    CONV2D_LSTM_IN_C_STATE      = 2,

    CONV2D_LSTM_IN_KERNEL_I2I   = 3,
    CONV2D_LSTM_IN_KERNEL_I2F   = 4,
    CONV2D_LSTM_IN_KERNEL_I2C   = 5,
    CONV2D_LSTM_IN_KERNEL_I2O   = 6,

    CONV2D_LSTM_IN_KERNEL_R2I   = 7,
    CONV2D_LSTM_IN_KERNEL_R2F   = 8,
    CONV2D_LSTM_IN_KERNEL_R2C   = 9,
    CONV2D_LSTM_IN_KERNEL_R2O   = 10,

    CONV2D_LSTM_IN_BIAS_I       = 11,
    CONV2D_LSTM_IN_BIAS_F       = 12,
    CONV2D_LSTM_IN_BIAS_C       = 13,
    CONV2D_LSTM_IN_BIAS_O       = 14,

    CONV2D_LSTM_IN_CNT,

    CONV2D_LSTM_OUT_OUTPUT      = 0,
    CONV2D_LSTM_OUT_H_STATE     = 1,
    CONV2D_LSTM_OUT_C_STATE     = 2,

    CONV2D_LSTM_OUT_CNT
};

typedef struct _vsi_nn_conv2d_lstm_local
{
    void * ptr;
} vsi_nn_conv2d_lstm_local;

typedef struct _vsi_nn_conv2d_lstm_param
{
    vsi_nn_conv2d_lstm_local * local;

    vsi_nn_activation_e activation;
    vsi_nn_activation_e recurrent_activation;
    vsi_nn_con2d_lstm_dataformat data_format;
    vsi_bool return_sequences;
    uint32_t filters;
    vsi_nn_conv2d_param conv2d;
} vsi_nn_conv2d_lstm_param;

#ifdef __cplusplus
}
#endif

#endif
