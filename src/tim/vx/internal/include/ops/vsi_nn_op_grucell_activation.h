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
#ifndef _VSI_NN_OP_GRUCELL_ACTIVATION_H
#define _VSI_NN_OP_GRUCELL_ACTIVATION_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GRUCELL_ACT_H_STATE = 0,
    GRUCELL_ACT_I_FC_Z  = 1,
    GRUCELL_ACT_I_FC_R  = 2,
    GRUCELL_ACT_I_FC_H  = 3,
    GRUCELL_ACT_H_FC_Z  = 4,
    GRUCELL_ACT_H_FC_R  = 5,
    GRUCELL_ACT_H_FC_H  = 6,

    GRUCELL_ACT_IN_CNT,

    GRUCELL_ACT_OUT_OUTPUT    = 0,
    GRUCELL_ACT_OUT_H_STATE   = 1,

    GRUCELL_ACT_OUT_CNT
};

typedef struct _vsi_nn_grucell_activation_param
{
    struct _vsi_nn_grucell_activation_local * local;

    vsi_nn_activation_e activation;
    vsi_nn_activation_e recurrent_activation;
} vsi_nn_grucell_activation_param;
_compiler_assert(offsetof(vsi_nn_grucell_activation_param, local) == 0, \
                 vsi_nn_grucell_activation_h );

#ifdef __cplusplus
}
#endif

#endif