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
#ifndef _VSI_NN_OP_RNN_H
#define _VSI_NN_OP_RNN_H

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* enum for inputs/outputs */
enum
{
    RNNCELL_INPUT_INPUT        = 0,
    RNNCELL_INPUT_WEIGHT_I     = 1,
    RNNCELL_INPUT_WEIGHT_H     = 2,
    RNNCELL_INPUT_BIAS_I       = 3,
    RNNCELL_INPUT_BIAS_H       = 4,
    RNNCELL_INPUT_H_STATE      = 5,

    RNNCELL_INPUT_AUX_INPUT    = 6,
    RNNCELL_INPUT_AUX_WEIGHT   = 7,
    RNNCELL_INPUT_CNT,

    RNNCELL_OUTPUT_H_STATE     = 0,
    RNNCELL_OUTPUT_OUTPUT      = 1,
    RNNCELL_OUTPUT_CNT
};

enum
{
    RNNCELL_QUANTIZE_PARAM_I,
    RNNCELL_QUANTIZE_PARAM_H,
    RNNCELL_QUANTIZE_PARAM_AUX,

    RNNCELL_QUANTIZE_PARAM_COUNT
};

typedef struct _vsi_nn_rnn_param
{
    vsi_nn_activation_e activation;
} vsi_nn_rnn_param;

#ifdef __cplusplus
}
#endif

#endif

