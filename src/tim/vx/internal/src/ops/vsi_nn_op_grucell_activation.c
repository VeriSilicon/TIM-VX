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
#include <string.h>
#include <stdlib.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_internal_node.h"
#include "vsi_nn_rnn_helper.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_tensor_op.h"
#include "utils/vsi_nn_util.h"
#include "ops/vsi_nn_op_grucell_activation.h"

typedef struct _vsi_nn_grucell_activation_local {
    void * placeholder;
} vsi_nn_grucell_activation_local;

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_internal_compute_node( self );
}

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return TRUE;
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool use_virtual_tensor= TRUE;
    vsi_nn_grucell_activation_param * p = &self->nn_param.grucell_activation;
    vsi_nn_internal_tensor_t * tmp_sub = NULL, * tmp_add = NULL, * tmp_mul = NULL;
    vsi_nn_internal_tensor_t * tmp_act = NULL;
    vsi_nn_internal_node_t * curr = NULL;

    vsi_nn_internal_init_node_wksp( self );

    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_ACT_OUT_OUTPUT]->attr.dim_num)
    {
        outputs[GRUCELL_ACT_OUT_OUTPUT]->attr.dim_num = 2;
        outputs[GRUCELL_ACT_OUT_OUTPUT]->attr.size[0] = inputs[GRUCELL_ACT_IN_H_STATE]->attr.size[0];
        outputs[GRUCELL_ACT_OUT_OUTPUT]->attr.size[1] = inputs[GRUCELL_ACT_IN_H_STATE]->attr.size[1];
    }

    /*
        hht = activation(fc_h + ht)
    */
    tmp_add = vsi_nn_rnn_create_binary_operator(
        self,
        VSI_NN_OP_ADD,
        inputs[GRUCELL_ACT_IN_INPUT_FC_H],
        inputs[GRUCELL_ACT_IN_H_T],
        &inputs[GRUCELL_ACT_IN_INPUT_FC_H]->attr.dtype,
        use_virtual_tensor
    );
    tmp_act = vsi_nn_rnn_create_activation(
        self,
        tmp_add->t,
        p->activation,
        &tmp_add->t->attr.dtype,
        use_virtual_tensor
    );

    /*
        new_h = zt * (hstate - hht) + hht
    */
    tmp_sub = vsi_nn_rnn_create_binary_operator(
        self,
        VSI_NN_OP_SUBTRACT,
        inputs[GRUCELL_ACT_IN_H_STATE],
        tmp_act->t,
        &tmp_act->t->attr.dtype,
        use_virtual_tensor
    );
    tmp_mul = vsi_nn_rnn_create_binary_operator(
        self,
        VSI_NN_OP_MULTIPLY,
        inputs[GRUCELL_ACT_IN_Z_T],
        tmp_sub->t,
        &tmp_sub->t->attr.dtype,
        use_virtual_tensor
    );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_ADD, 0, 0 );
    curr->inputs[0] = tmp_mul->t;
    curr->inputs[1] = tmp_act->t;
    curr->outputs[0] = outputs[GRUCELL_ACT_OUT_OUTPUT];
    vsi_nn_internal_setup_node(self, curr);

    /* copy outputs to h_state */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
    curr->inputs[0] = outputs[GRUCELL_ACT_OUT_OUTPUT];
    curr->outputs[0] = outputs[GRUCELL_ACT_OUT_H_STATE];
    vsi_nn_internal_setup_node(self, curr);

    return TRUE;
}

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    return VSI_SUCCESS;
}

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_internal_optimize_node( self, direction );
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GRUCELL_ACTIVATION,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ GRUCELL_ACT_IN_CNT,
    /* output_num */ GRUCELL_ACT_OUT_CNT
    );
#ifdef __cplusplus
}
#endif
