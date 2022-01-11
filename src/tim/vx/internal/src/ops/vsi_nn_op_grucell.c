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
#include "ops/vsi_nn_op_grucell.h"

typedef struct _vsi_nn_grucell_local
{
    void * placeholder;
} vsi_nn_grucell_local;

static vsi_nn_internal_tensor_t * _create_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t * fc_out = NULL;
    vsi_nn_internal_tensor_t * tmp_tensor = NULL;
    vsi_nn_tensor_t * bias_tensor = NULL;
     vsi_nn_internal_node_t* fc_node = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    if(NULL == bias)
    {
        /* create zero bias for NN/TP */
        tmp_tensor = vsi_nn_internal_create_zero_bias_tensor(self, &input->attr, &weight->attr, VSI_NN_OP_FCL, FALSE);
        bias_tensor = tmp_tensor->t;
    }
    else
    {
        bias_tensor = bias;
    }

    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if (input->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 ||
        input->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16)
    {
        attr.dtype.vx_type = input->attr.dtype.vx_type;
    }
    else
    {
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    fc_out = vsi_nn_internal_new_tensor(self, &attr, 0.0f);

    fc_node = vsi_nn_internal_new_node(self, VSI_NN_OP_FCL, 0, 0 );
    fc_node->node->nn_param.fcl.axis = 0;
    fc_node->node->nn_param.fcl.weights = (uint32_t)weight->attr.size[1];
    fc_node->inputs[0] = input;
    fc_node->inputs[1] = weight;
    fc_node->inputs[2] = bias_tensor;
    fc_node->outputs[0] = fc_out->t;
    vsi_nn_internal_setup_node(self, fc_node);

    return fc_out;
} /* () */

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_grucell_param * p = &self->nn_param.grucell;
    vsi_size_t hidden_size = p->num_units;
    vsi_size_t batch_size = inputs[GRUCELL_IN_INPUT]->attr.size[1];

    /* setup grucell output tensors' shape */
    /* output */
    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_OUT_OUTPUT]->attr.dim_num)
    {
        outputs[GRUCELL_OUT_OUTPUT]->attr.dim_num = 2;
        outputs[GRUCELL_OUT_OUTPUT]->attr.size[0] = hidden_size;
        outputs[GRUCELL_OUT_OUTPUT]->attr.size[1] = batch_size;
    }

    /* hstate output */
    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_OUT_H_STATE]->attr.dim_num)
    {
        outputs[GRUCELL_OUT_H_STATE]->attr.dim_num = 2;
        outputs[GRUCELL_OUT_H_STATE]->attr.size[0] = hidden_size;
        outputs[GRUCELL_OUT_H_STATE]->attr.size[1] = batch_size;
    }

    return TRUE;
} /* setup_op_shapes() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_internal_compute_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return TRUE;
}

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_internal_deinit_node_wksp( self );

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

#if 1
static vsi_bool op_setup_default
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    vsi_nn_internal_node_t * curr = NULL;
    vsi_nn_grucell_param * p = &self->nn_param.grucell;
    vsi_nn_internal_tensor_t * input_fc_outputs[GRUCELL_GATE_CNT] = { NULL };
    vsi_nn_internal_tensor_t * hstate_fc_outputs[GRUCELL_GATE_CNT] = { NULL };
    vsi_nn_internal_tensor_t * h_times_r = NULL;
    vsi_nn_tensor_attr_t attr;

    vsi_nn_internal_init_node_wksp( self );

    /* compute output tensor's shapes */
    setup_op_shapes(self, inputs, outputs);

    /* create input fc */
    for(i = 0; i < GRUCELL_GATE_CNT; i++)
    {
        input_fc_outputs[i] = _create_fc(
            self,
            inputs[GRUCELL_IN_INPUT],
            inputs[GRUCELL_IN_KERNEL_I2Z + i],
            inputs[GRUCELL_IN_BIAS_I2Z + i]
        );
    }

    /* create hstate fc */
    for(i = 0; i < GRUCELL_GATE_CNT - 1; i++)
    {
        hstate_fc_outputs[i] = _create_fc(
            self,
            inputs[GRUCELL_IN_H_STATE],
            inputs[GRUCELL_IN_KERNEL_R2Z + i],
            inputs[GRUCELL_IN_BIAS_R2Z + i]
        );
    }

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if (inputs[GRUCELL_IN_H_STATE]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 ||
        self->graph->ctx->config.support_stream_processor)
    {
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    }
    else
    {
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    h_times_r = vsi_nn_internal_new_tensor(self, &attr, 0.0f);

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_GRUCELL_H_TIMES_ACTIVATION_R, 3, 1 );
    curr->node->nn_param.grucell_h_times_activation_r.recurrent_activation = p->recurrent_activation;
    curr->inputs[0] = inputs[GRUCELL_IN_H_STATE];
    curr->inputs[1] = input_fc_outputs[GRUCELL_GATES_R]->t;
    curr->inputs[2] = hstate_fc_outputs[GRUCELL_GATES_R]->t;
    curr->outputs[0] = h_times_r->t;
    vsi_nn_internal_setup_node(self, curr);

    hstate_fc_outputs[GRUCELL_GATES_H] = _create_fc(
        self,
        h_times_r->t,
        inputs[GRUCELL_IN_KERNEL_R2H],
        inputs[GRUCELL_IN_BIAS_R2H]
    );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_GRUCELL_ACTIVATION_Z_H, 0, 0 );
    curr->node->nn_param.grucell_activation_z_h.activation = p->activation;
    curr->node->nn_param.grucell_activation_z_h.recurrent_activation = p->recurrent_activation;
    curr->inputs[GRUCELL_ACT_Z_H_HSTATE] = inputs[GRUCELL_IN_H_STATE];
    curr->inputs[GRUCELL_ACT_Z_H_I_FC_Z] = input_fc_outputs[GRUCELL_GATES_Z]->t;
    curr->inputs[GRUCELL_ACT_Z_H_I_FC_H] = input_fc_outputs[GRUCELL_GATES_H]->t;
    curr->inputs[GRUCELL_ACT_Z_H_H_FC_Z] = hstate_fc_outputs[GRUCELL_GATES_Z]->t;
    curr->inputs[GRUCELL_ACT_Z_H_H_FC_H] = hstate_fc_outputs[GRUCELL_GATES_H]->t;
    curr->outputs[GRUCELL_ACT_Z_H_OUT_OUTPUT] = outputs[GRUCELL_OUT_OUTPUT];
    curr->outputs[GRUCELL_ACT_Z_H_OUT_HSTATE] = outputs[GRUCELL_OUT_H_STATE];
    vsi_nn_internal_setup_node(self, curr);

    return TRUE;
}
#endif

static vsi_bool op_setup_reset_after
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    vsi_nn_internal_node_t * curr = NULL;
    vsi_nn_grucell_param * p = &self->nn_param.grucell;
    vsi_nn_internal_tensor_t * input_fc_outputs[GRUCELL_GATE_CNT] = { NULL };
    vsi_nn_internal_tensor_t * hstate_fc_outputs[GRUCELL_GATE_CNT] = { NULL };

    vsi_nn_internal_init_node_wksp( self );

    /* compute output tensor's shapes */
    setup_op_shapes(self, inputs, outputs);

    /* create input fc */
    for(i = 0; i < GRUCELL_GATE_CNT; i++)
    {
        input_fc_outputs[i] = _create_fc(
            self,
            inputs[GRUCELL_IN_INPUT],
            inputs[GRUCELL_IN_KERNEL_I2Z + i],
            inputs[GRUCELL_IN_BIAS_I2Z + i]
        );
    }

    /* create hstate fc */
    for(i = 0; i < GRUCELL_GATE_CNT; i++)
    {
        hstate_fc_outputs[i] = _create_fc(
            self,
            inputs[GRUCELL_IN_H_STATE],
            inputs[GRUCELL_IN_KERNEL_R2Z + i],
            inputs[GRUCELL_IN_BIAS_R2Z + i]
        );
    }

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_GRUCELL_ACTIVATION, 0, 0 );
    curr->node->nn_param.grucell_activation.activation = p->activation;
    curr->node->nn_param.grucell_activation.recurrent_activation = p->recurrent_activation;
    curr->inputs[GRUCELL_ACT_H_STATE] = inputs[GRUCELL_IN_H_STATE];
    curr->inputs[GRUCELL_ACT_I_FC_Z] = input_fc_outputs[GRUCELL_GATES_Z]->t;
    curr->inputs[GRUCELL_ACT_I_FC_R] = input_fc_outputs[GRUCELL_GATES_R]->t;
    curr->inputs[GRUCELL_ACT_I_FC_H] = input_fc_outputs[GRUCELL_GATES_H]->t;
    curr->inputs[GRUCELL_ACT_H_FC_Z] = hstate_fc_outputs[GRUCELL_GATES_Z]->t;
    curr->inputs[GRUCELL_ACT_H_FC_R] = hstate_fc_outputs[GRUCELL_GATES_R]->t;
    curr->inputs[GRUCELL_ACT_H_FC_H] = hstate_fc_outputs[GRUCELL_GATES_H]->t;
    curr->outputs[GRUCELL_ACT_OUT_OUTPUT] = outputs[GRUCELL_OUT_OUTPUT];
    curr->outputs[GRUCELL_ACT_OUT_H_STATE] = outputs[GRUCELL_OUT_H_STATE];
    vsi_nn_internal_setup_node(self, curr);

    return TRUE;
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if (self->nn_param.grucell.reset_after == TRUE)
    {
        return op_setup_reset_after(self, inputs, outputs);
    }
    else
    {
        return op_setup_default(self, inputs, outputs);
    }
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GRUCELL,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ GRUCELL_IN_CNT,
    /* output_num */ GRUCELL_OUT_CNT
    );
#ifdef __cplusplus
}
#endif
