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

#ifndef _VSI_NN_OP_GRUCELL_ACTIVATION_INTERNAL_H
#define _VSI_NN_OP_GRUCELL_ACTIVATION_INTERNAL_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GRUCELL_ACTIVATION_INPUT_ZT_    = 0,
    GRUCELL_ACTIVATION_INPUT_HT__   = 1,
    GRUCELL_ACTIVATION_INPUT_HT_1   = 2,

    GRUCELL_ACTIVATION_INPUT_H_STATE        = 0,
    GRUCELL_ACTIVATION_INPUT_INPUT_FC_R     = 1,
    GRUCELL_ACTIVATION_INPUT_INPUT_FC_Z     = 2,
    GRUCELL_ACTIVATION_INPUT_INPUT_FC_C     = 3,
    GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_R = 4,
    GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_Z = 5,
    GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_C = 6,
    GRUCELL_ACTIVATION_INPUT_BIAS_R         = 7,
    GRUCELL_ACTIVATION_INPUT_BIAS_Z         = 8,
    GRUCELL_ACTIVATION_INPUT_BIAS_C         = 9,
    GRUCELL_ACTIVATION_INPUT_COND_R         = 10,
    GRUCELL_ACTIVATION_INPUT_COND_Z         = 11,
    GRUCELL_ACTIVATION_INPUT_COND_C         = 12,

    GRUCELL_ACTIVATION_INPUT_COUNT,

    GRUCELL_ACTIVATION_OUTPUT_OUTPUT    = 0,
    GRUCELL_ACTIVATION_OUTPUT_H_STATE   = 1,
    GRUCELL_ACTIVATION_OUTPUT_COUNT
};

enum {
    GRUCELL_INPUT_CATEGORY_DEFAULT,
    GRUCELL_INPUT_CATEGORY_CUDNN,

    GRUCELL_INPUT_CATEGORY_COUNT
};

typedef int32_t grucell_activation_input_layout_e; enum
{
    GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_NC,
    GRUCELL_ACTIVATION_INPUT_LAYOUT_INPUT_NC_FC_CN,
    GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_CN,

    GRUCELL_ACTIVATION_INPUT_LAYOUT_COUNT
};

typedef struct _vsi_nn_grucell_activation_internal_local {
    uint32_t placeholder;
} vsi_nn_grucell_activation_internal_local;

typedef struct _vsi_nn_grucell_activation_internal_param
{
    vsi_nn_grucell_activation_internal_local* local;
    vsi_nn_activation_e gate_activation;
    vsi_nn_activation_e candidate_activation;
    int32_t input_category;
    vsi_bool use_cudnn_implementation;
    grucell_activation_input_layout_e input_layout;
} vsi_nn_grucell_activation_internal_param;

#ifdef __cplusplus
}
#endif

#endif
