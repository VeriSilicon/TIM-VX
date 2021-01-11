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
#ifndef _VSI_NN_OP_BIDIRECTIONAL_SEQUENCE_RNN_H
#define _VSI_NN_OP_BIDIRECTIONAL_SEQUENCE_RNN_H

#include "vsi_nn_types.h"
#include "vsi_nn_op_rnn.h"

/* enum for inputs/outputs */
enum
{
    BI_RNN_INPUT_INPUT           = 0,

    BI_RNN_FW_INPUT_WEIGHT_I     = 1,
    BI_RNN_FW_INPUT_WEIGHT_H     = 2,
    BI_RNN_FW_INPUT_BIAS         = 3,
    BI_RNN_FW_INPUT_H_STATE      = 4,

    BI_RNN_BW_INPUT_WEIGHT_I     = 5,
    BI_RNN_BW_INPUT_WEIGHT_H     = 6,
    BI_RNN_BW_INPUT_BIAS         = 7,
    BI_RNN_BW_INPUT_H_STATE      = 8,

    BI_RNN_AUX_INPUT             = 9,
    BI_RNN_FW_AUX_INPUT_WEIGHT   = 10,
    BI_RNN_BW_AUX_INPUT_WEIGHT   = 11,

    BI_RNN_INPUT_CNT,

    BI_RNN_FW_OUTPUT_OUTPUT      = 0,
    BI_RNN_BW_OUTPUT_OUTPUT      = 1,
    BI_RNN_OUTPUT_CNT
};


typedef struct _vsi_nn_bidirectional_sequence_rnn_param
{
    vsi_bool time_major;
    vsi_bool merge_outputs;
    vsi_nn_activation_e activation;
    vsi_nn_dtype_t* internal_dtype;
} vsi_nn_bidirectional_sequence_rnn_param;

#endif

