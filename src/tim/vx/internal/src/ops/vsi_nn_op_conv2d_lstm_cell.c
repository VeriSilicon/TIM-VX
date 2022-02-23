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
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "vsi_nn_internal_node.h"
#include "vsi_nn_rnn_helper.h"

static vsi_nn_internal_tensor_t * reshape_tensor_to_act
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t * tensor
    )
{
    vsi_nn_internal_tensor_t * reshape_out = NULL;
    vsi_size_t i,dim,reshaped_size[2];
    vsi_size_t sz;

    memset(reshaped_size, 0, sizeof(vsi_size_t) * 2);
    dim = 2;
    sz = 1;
    for(i = 0; i < tensor->attr.dim_num - 1; i++)
    {
        sz *= tensor->attr.size[i];
    }

    /* reshape 4d tensor to [-1, 0] */
    reshaped_size[0] = sz;
    reshaped_size[1] = tensor->attr.size[tensor->attr.dim_num - 1];
    reshape_out = vsi_nn_rnn_create_reshape(self, tensor, NULL, reshaped_size, dim, TRUE);

    return reshape_out;
} /* reshape_tensor_to_act() */

static vsi_nn_internal_tensor_t * create_input_conv
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_conv2d_lstm_cell_param * p;
    vsi_nn_internal_tensor_t * input_conv_out = NULL, * reshape_out = NULL;
    vsi_nn_internal_node_t * input_conv= NULL;

    p = &self->nn_param.conv2d_lstm_cell;
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    input_conv_out = vsi_nn_internal_new_tensor(self, &attr, 0.0f);

    input_conv = vsi_nn_internal_new_node(self, VSI_NN_OP_CONV2D, 0, 0 );
    input_conv->node->nn_param.conv2d.group = 1;
    input_conv->node->nn_param.conv2d.ksize[0] = p->conv2d.ksize[0];
    input_conv->node->nn_param.conv2d.ksize[1] = p->conv2d.ksize[1];
    input_conv->node->nn_param.conv2d.weights = p->filters;
    input_conv->node->nn_param.conv2d.pad[0] = p->conv2d.pad[0];
    input_conv->node->nn_param.conv2d.pad[1] = p->conv2d.pad[1];
    input_conv->node->nn_param.conv2d.pad[2] = p->conv2d.pad[2];
    input_conv->node->nn_param.conv2d.pad[3] = p->conv2d.pad[3];
    input_conv->node->nn_param.conv2d.pad_type = VSI_NN_PAD_AUTO;
    input_conv->node->nn_param.conv2d.stride[0] = p->conv2d.stride[0];
    input_conv->node->nn_param.conv2d.stride[1] = p->conv2d.stride[1];
    input_conv->node->nn_param.conv2d.dilation[0] = p->conv2d.dilation[0];
    input_conv->node->nn_param.conv2d.dilation[1] = p->conv2d.dilation[1];
    input_conv->node->nn_param.conv2d.multiplier = 0;
    input_conv->node->vx_param.overflow_policy = self->vx_param.overflow_policy;
    input_conv->node->vx_param.rounding_policy = self->vx_param.rounding_policy;
    input_conv->node->vx_param.down_scale_size_rounding = self->vx_param.down_scale_size_rounding;

    input_conv->inputs[0] = input;
    input_conv->inputs[1] = weight;
    input_conv->inputs[2] = bias;
    input_conv->outputs[0] = input_conv_out->t;
    vsi_nn_internal_setup_node(self, input_conv);

    // reshape whcn --> xn
    reshape_out = reshape_tensor_to_act(self, input_conv_out->t);

    return reshape_out;
} /* create_input_conv() */

static vsi_nn_internal_tensor_t * create_recurrent_conv
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_conv2d_lstm_cell_param * p = NULL;
    vsi_nn_tensor_t * bias = NULL;
    vsi_nn_internal_tensor_t * recurrent_conv_out = NULL;
    vsi_nn_internal_tensor_t * internal_bias = NULL;
    vsi_nn_internal_tensor_t * reshape_out = NULL;
    vsi_nn_internal_node_t * recurrent_conv = NULL;

    p = &self->nn_param.conv2d_lstm_cell;
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    internal_bias = vsi_nn_internal_create_zero_bias_tensor(
        self, &input->attr, &weight->attr, VSI_NN_OP_CONV2D, FALSE);
    bias = internal_bias->t;

    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    recurrent_conv_out = vsi_nn_internal_new_tensor(self, &attr, 0.0f);

    recurrent_conv = vsi_nn_internal_new_node(self, VSI_NN_OP_CONV2D, 0, 0 );
    recurrent_conv->node->nn_param.conv2d.pad_type = VSI_NN_PAD_SAME;
    recurrent_conv->node->nn_param.conv2d.group = 1;
    recurrent_conv->node->nn_param.conv2d.ksize[0] = p->conv2d.ksize[0];
    recurrent_conv->node->nn_param.conv2d.ksize[1] = p->conv2d.ksize[1];
    recurrent_conv->node->nn_param.conv2d.stride[0] = 1;
    recurrent_conv->node->nn_param.conv2d.stride[1] = 1;
    recurrent_conv->node->nn_param.conv2d.dilation[0] = 1;
    recurrent_conv->node->nn_param.conv2d.dilation[1] = 1;
    recurrent_conv->node->nn_param.conv2d.weights = p->filters;
    recurrent_conv->node->nn_param.conv2d.multiplier = 0;

    recurrent_conv->node->vx_param.overflow_policy = self->vx_param.overflow_policy;
    recurrent_conv->node->vx_param.rounding_policy = self->vx_param.rounding_policy;
    recurrent_conv->node->vx_param.down_scale_size_rounding = self->vx_param.down_scale_size_rounding;

    recurrent_conv->inputs[0] = input;
    recurrent_conv->inputs[1] = weight;
    recurrent_conv->inputs[2] = bias;

    recurrent_conv->outputs[0] = recurrent_conv_out->t;
    vsi_nn_internal_setup_node(self, recurrent_conv);

    // reshape whcn --> xn
    reshape_out = reshape_tensor_to_act(self, recurrent_conv_out->t);
    return reshape_out;
} /* create_recurrent_conv() */

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    vsi_nn_conv2d_lstm_cell_param *p = &self->nn_param.conv2d_lstm_cell;
    vsi_size_t w_out, h_out, samples, out_channel;
    vsi_size_t ksize[_cnt_of_array(p->conv2d.ksize)];
    vsi_size_t i, pad[_cnt_of_array(p->conv2d.pad)] = {0};

    w_out = 0;
    h_out = 0;
    for( i = 0; i < _cnt_of_array(p->conv2d.ksize); i++ )
    {
        ksize[i] = (vsi_size_t)p->conv2d.ksize[i];
    }
    samples = inputs[CONV2D_LSTM_CELL_IN_INPUT]->attr.size[3];
    out_channel = p->filters;
    for(i = 0; i < _cnt_of_array(p->conv2d.pad); i++)
    {
        pad[i] = self->nn_param.conv2d.pad[i];
    }

    vsi_nn_compute_padding(
        inputs[CONV2D_LSTM_CELL_IN_INPUT]->attr.size,
        ksize,
        p->conv2d.stride,
        p->conv2d.dilation,
        p->conv2d.pad_type,
        pad
    );
    for(i = 0; i < _cnt_of_array(p->conv2d.ksize); i++)
    {
        self->nn_param.conv2d.ksize[i] = (uint32_t)ksize[i];
    }
    for(i = 0; i < _cnt_of_array(p->conv2d.pad); i++)
    {
        self->nn_param.conv2d.pad[i] = (uint32_t)pad[i];
    }

    w_out = vsi_nn_ComputeFilterSize(
        inputs[CONV2D_LSTM_CELL_IN_INPUT]->attr.size[0],
        p->conv2d.ksize[0],
        &p->conv2d.pad[0],
        p->conv2d.stride[0],
        p->conv2d.dilation[0],
        VSI_NN_ROUND_FLOOR
    );
    h_out = vsi_nn_ComputeFilterSize(
        inputs[CONV2D_LSTM_CELL_IN_INPUT]->attr.size[1],
        p->conv2d.ksize[1],
        &p->conv2d.pad[2],
        p->conv2d.stride[1],
        p->conv2d.dilation[1],
        VSI_NN_ROUND_FLOOR
    );

    /* setup conv2d lstm cell output tensors' shape */
    /* output */
    if(VSI_NN_DIM_AUTO == outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]->attr.dim_num)
    {
        outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]->attr.size[0] = w_out;
        outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]->attr.size[1] = h_out;
        outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]->attr.size[2] = out_channel;
        outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]->attr.size[3] = samples;
        outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]->attr.dim_num = 4;
    }

    /* hidden state output */
    if(VSI_NN_DIM_AUTO == outputs[CONV2D_LSTM_CELL_OUT_H_STATE]->attr.dim_num)
    {
        memcpy(outputs[CONV2D_LSTM_CELL_OUT_H_STATE]->attr.size, outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]->attr.size,
            sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
        outputs[CONV2D_LSTM_CELL_OUT_H_STATE]->attr.dim_num = outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]->attr.dim_num;
    }

    /* hidden state output */
    if(VSI_NN_DIM_AUTO == outputs[CONV2D_LSTM_CELL_OUT_C_STATE]->attr.dim_num)
    {
        memcpy(outputs[CONV2D_LSTM_CELL_OUT_C_STATE]->attr.size, outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]->attr.size,
            sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
        outputs[CONV2D_LSTM_CELL_OUT_C_STATE]->attr.dim_num = outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]->attr.dim_num;
    }

    return ret;
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
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_internal_optimize_node( self, direction );
} /* op_optimize() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    vsi_nn_internal_tensor_t * input_conv_outputs[CONV2D_LSTM_CELL_GATE_NUM] = { NULL };
    vsi_nn_internal_tensor_t * recurrent_conv_outputs[CONV2D_LSTM_CELL_GATE_NUM] = { NULL };
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t * reshape_cell_in = NULL;
    vsi_nn_internal_tensor_t * reshape_out = NULL;
    vsi_nn_internal_tensor_t * reshape_h_out = NULL;
    vsi_nn_internal_tensor_t * reshape_c_out = NULL;
    vsi_nn_conv2d_lstm_cell_param * p = &self->nn_param.conv2d_lstm_cell;

    vsi_nn_internal_init_node_wksp( self );

    /* compute output tensor's shapes */
    setup_op_shapes(self, inputs, outputs);

    /* create input convolution */
    for(i = 0; i < CONV2D_LSTM_CELL_GATE_NUM; i++)
    {
        input_conv_outputs[i] = create_input_conv(
            self,
            inputs[CONV2D_LSTM_CELL_IN_INPUT],
            inputs[CONV2D_LSTM_CELL_IN_KERNEL_I2I + i],
            inputs[CONV2D_LSTM_CELL_IN_BIAS_I + i]
        );
    }

    /* create recurrent convolution */
    for(i = 0; i < CONV2D_LSTM_CELL_GATE_NUM; i++)
    {
        recurrent_conv_outputs[i] = create_recurrent_conv(
            self,
            inputs[CONV2D_LSTM_CELL_IN_H_STATE],
            inputs[CONV2D_LSTM_CELL_IN_KERNEL_R2I + i]
        );
    }

    /* activations */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_LSTMUNIT_ACTIVATION, 0, 0 );
    curr->node->nn_param.lstmunit_activation.cell_clip = 0;
    curr->node->nn_param.lstmunit_activation.proj_clip = 0;
    curr->node->nn_param.lstmunit_activation.forget_bias = 0;
    curr->node->nn_param.lstmunit_activation.is_cifg = 0;
    curr->node->nn_param.lstmunit_activation.is_projection = 0;
    curr->node->nn_param.lstmunit_activation.is_layer_norm = 0;
    curr->node->nn_param.lstmunit_activation.is_peephole = FALSE;
    curr->node->nn_param.lstmunit_activation.is_hybrid = 0;
    curr->node->nn_param.lstmunit_activation.recurrent_activation = p->recurrent_activation;

    reshape_cell_in = reshape_tensor_to_act(self, inputs[CONV2D_LSTM_CELL_IN_C_STATE]);
    curr->inputs[LSTMUNIT_ACT_CSTATE_IN] = reshape_cell_in->t;
    for(i = 0; i < CONV2D_LSTM_CELL_GATE_NUM; i++)
    {
        curr->inputs[LSTMUNIT_ACT_LN_WI + i] = NULL;
        curr->inputs[LSTMUNIT_ACT_INPUT_FC_I + i] = input_conv_outputs[i]->t;
        curr->inputs[LSTMUNIT_ACT_HSTATE_FC_I + i] = recurrent_conv_outputs[i]->t;
    }
    reshape_out = reshape_tensor_to_act(self, outputs[CONV2D_LSTM_CELL_OUT_OUTPUT]);
    reshape_h_out = reshape_tensor_to_act(self, outputs[CONV2D_LSTM_CELL_OUT_H_STATE]);
    reshape_c_out = reshape_tensor_to_act(self, outputs[CONV2D_LSTM_CELL_OUT_C_STATE]);

    curr->outputs[LSTMUNIT_ACT_OUTPUT] = reshape_out->t;
    curr->outputs[LSTMUNIT_ACT_CSTATE_OUT] = reshape_c_out->t;
    curr->outputs[LSTMUNIT_ACT_HSTATE_OUT] = reshape_h_out->t;
    vsi_nn_internal_setup_node(self, curr);

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_internal_deinit_node_wksp( self );
    return status;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    return status;
} /* op_init() */

#ifdef __cpluplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CONV2D_LSTM_CELL,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ CONV2D_LSTM_CELL_IN_CNT,
    /* output_num */ CONV2D_LSTM_CELL_OUT_CNT
    );
#ifdef __cpluplus
}
#endif
