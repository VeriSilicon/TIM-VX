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
#ifndef _VSI_NN_OP_LSTMUNIT_ACTIVATION_H
#define _VSI_NN_OP_LSTMUNIT_ACTIVATION_H

#include "vsi_nn_types.h"
#include "vsi_nn_op_lstmunit.h"

/* c -> cifg, l -> layer norm, p -> projection, h -> peephole, b -> hybrid bias fp32, s -> standard*/

enum {
    LSTMUNIT_ACT_INPUT_FC_I, //optional
    LSTMUNIT_ACT_INPUT_FC_F,
    LSTMUNIT_ACT_INPUT_FC_C,
    LSTMUNIT_ACT_INPUT_FC_O,
    LSTMUNIT_ACT_CSTATE_IN,
    LSTMUNIT_ACT_HSTATE_FC_I, //optional
    LSTMUNIT_ACT_HSTATE_FC_F, //optional
    LSTMUNIT_ACT_HSTATE_FC_C, //optional
    LSTMUNIT_ACT_HSTATE_FC_O, //optional
    LSTMUNIT_ACT_DATA_BI,     //optional
    LSTMUNIT_ACT_DATA_BF,     //optional
    LSTMUNIT_ACT_DATA_BC,     //optional
    LSTMUNIT_ACT_DATA_BO,     //optional
    LSTMUNIT_ACT_LN_WI,       //optional
    LSTMUNIT_ACT_LN_WF,       //optional
    LSTMUNIT_ACT_LN_WC,       //optional
    LSTMUNIT_ACT_LN_WO,       //optional

    LSTMUNIT_ACT_INPUTS_COUNT,

    LSTMUNIT_ACT_OUTPUT = 0,
    LSTMUNIT_ACT_CSTATE_OUT,
    LSTMUNIT_ACT_HSTATE_OUT, //optional

    LSTMUNIT_ACT_OUTUTS_COUNT,

    LSTMUNIT_ACT_PARAM_COUT = LSTMUNIT_ACT_INPUTS_COUNT + LSTMUNIT_ACT_OUTUTS_COUNT,
};


enum
{
    LSTMUNIT_ACT_TENSOR_BI,
    LSTMUNIT_ACT_TENSOR_BF,
    LSTMUNIT_ACT_TENSOR_BC,
    LSTMUNIT_ACT_TENSOR_BO,
    LSTMUNIT_ACT_TENSOR_LN_WI,
    LSTMUNIT_ACT_TENSOR_LN_WF,
    LSTMUNIT_ACT_TENSOR_LN_WC,
    LSTMUNIT_ACT_TENSOR_LN_WO,

    LSTMUNIT_ACT_TENSOR_CNT
};

typedef struct _vsi_nn_lstmunit_activation_lcl_data_t
{
    vsi_nn_tensor_t* tensors[LSTMUNIT_ACT_TENSOR_CNT];
    uint32_t hash_idx;
    vsi_bool execute_on_sw;
    vsi_nn_tensor_t *lstmunit_param;
} vsi_nn_lstmunit_activation_lcl_data_t;

typedef struct _vsi_nn_lstmunit_activation_param
{
    vsi_nn_lstmunit_activation_lcl_data_t local;
    float cell_clip;
    float proj_clip;
    float forget_bias;
    uint8_t is_cifg;
    uint8_t is_projection;
    uint8_t is_layer_norm;
    uint8_t is_peephole; /* not supported now*/
    uint8_t is_hybrid; /*hybrid mode and bias format is fp32 */
    vsi_nn_activation_e recurrent_activation;
} vsi_nn_lstmunit_activation_param;

#endif

