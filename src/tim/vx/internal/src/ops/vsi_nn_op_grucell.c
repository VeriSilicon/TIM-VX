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
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
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

/*
    copmute the recurrent hstate gates
    equations:
      reset_after == True:
        ht = FC(hstate, kernel_rh, bias_rh)
        ht = rt * ht
      reset_after == False:
        ht = rt * hstate
        ht = FC(ht, kernel_rh, bias_rh)
*/
static vsi_nn_internal_tensor_t * _compute_ht
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input_rt,
    vsi_nn_tensor_t * hstate,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias
    )
{
    vsi_bool use_virtual_tensor = TRUE;
    vsi_nn_grucell_param * p = &self->nn_param.grucell;
    vsi_nn_internal_tensor_t * tensor1 = NULL, * tensor2 = NULL;

    if(p->reset_after == TRUE)
    {
        tensor1 = _create_fc(
            self,
            hstate,
            weight,
            bias
        );
        tensor2 = vsi_nn_rnn_create_binary_operator(
            self,
            VSI_NN_OP_MULTIPLY,
            input_rt,
            tensor1->t,
            &input_rt->attr.dtype,
            use_virtual_tensor
        );
    }
    else
    {
        tensor1 = vsi_nn_rnn_create_binary_operator(
            self,
            VSI_NN_OP_MULTIPLY,
            input_rt,
            hstate,
            &input_rt->attr.dtype,
            use_virtual_tensor
        );
        tensor2 = _create_fc(
            self,
            tensor1->t,
            weight,
            bias
        );
    }

    return tensor2;
} /* _compute_ht() */

/*
    compute the recurrent update gates or reset gates
    equations:
      xt = FC(hstate, kernel_xt, bias_xt)
      xt = input_xt + xt
      xt = recurrent_activation(xt)
*/
static vsi_nn_internal_tensor_t * _compute_recurrent_gate
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input_xt,
    vsi_nn_tensor_t * hstate,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias
    )
{
    vsi_bool use_virtual_tensor = TRUE;
    vsi_nn_grucell_param * p = &self->nn_param.grucell;
    vsi_nn_internal_tensor_t * tensor_add = NULL, * tensor_act;
    vsi_nn_internal_tensor_t * recurrent_fc_out = NULL;

    recurrent_fc_out = _create_fc(self, hstate, weight, bias);

    tensor_add = vsi_nn_rnn_create_binary_operator(
        self,
        VSI_NN_OP_ADD,
        recurrent_fc_out->t,
        input_xt,
        &recurrent_fc_out->t->attr.dtype,
        use_virtual_tensor
    );

    tensor_act = vsi_nn_rnn_create_activation(
        self,
        tensor_add->t,
        p->recurrent_activation,
        &tensor_add->t->attr.dtype,
        use_virtual_tensor
    );

    return tensor_act;
} /* _compute_recurrent_gate */

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

static vsi_bool op_setup
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
    vsi_nn_internal_tensor_t * zt = NULL, * rt = NULL, * ht = NULL;

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

    /* compute update gate and reset gate */
    zt = _compute_recurrent_gate(
        self,
        input_fc_outputs[GRUCELL_GATES_Z]->t,
        inputs[GRUCELL_IN_H_STATE],
        inputs[GRUCELL_IN_KERNEL_R2Z],
        inputs[GRUCELL_IN_BIAS_R2Z]
    );
    rt = _compute_recurrent_gate(
        self,
        input_fc_outputs[GRUCELL_GATES_R]->t,
        inputs[GRUCELL_IN_H_STATE],
        inputs[GRUCELL_IN_KERNEL_R2R],
        inputs[GRUCELL_IN_BIAS_R2R]
    );

    /* compute recurrent h with parameter 'reset_after' */
    ht = _compute_ht(
        self,
        rt->t,
        inputs[GRUCELL_IN_H_STATE],
        inputs[GRUCELL_IN_KERNEL_R2H],
        inputs[GRUCELL_IN_BIAS_R2H]
    );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_GRUCELL_ACTIVATION, 0, 0 );
    curr->node->nn_param.grucell_activation.activation = p->activation;
    curr->inputs[GRUCELL_ACT_IN_H_STATE] = inputs[GRUCELL_IN_H_STATE];
    curr->inputs[GRUCELL_ACT_IN_INPUT_FC_H] = input_fc_outputs[GRUCELL_GATES_H]->t;
    curr->inputs[GRUCELL_ACT_IN_H_T] = ht->t;
    curr->inputs[GRUCELL_ACT_IN_Z_T] = zt->t;
    curr->outputs[GRUCELL_ACT_OUT_OUTPUT] = outputs[GRUCELL_OUT_OUTPUT];
    curr->outputs[GRUCELL_ACT_OUT_H_STATE] = outputs[GRUCELL_OUT_H_STATE];
    vsi_nn_internal_setup_node(self, curr);

    return TRUE;
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
