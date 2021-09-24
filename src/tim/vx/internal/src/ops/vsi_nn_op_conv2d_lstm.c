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

static vsi_nn_internal_tensor_t * reshape_cell_out
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * cell_out
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_size_t* reshape_cell_size = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    vsi_nn_internal_init_tensor_attr(&attr, &cell_out->attr.dtype, TRUE);
    output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    /* reshape cell_out [w,h,c,n] to [w,h,c,1,n] */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE, 0, 0 );
    reshape_cell_size = vsi_nn_internal_new_node_param(curr,
        VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    reshape_cell_size[0] = cell_out->attr.size[0];
    reshape_cell_size[1] = cell_out->attr.size[1];
    reshape_cell_size[2] = cell_out->attr.size[2];
    reshape_cell_size[3] = 1;
    reshape_cell_size[4] = cell_out->attr.size[3];
    curr->node->nn_param.reshape.size = reshape_cell_size;
    curr->node->nn_param.reshape.dim_num = 5;

    curr->inputs[0] = cell_out;
    curr->outputs[0] = output_tensor->t;

    vsi_nn_internal_setup_node( self, curr );
    return output_tensor;
} /* reshape_cell_out() */

static vsi_nn_internal_tensor_t * reshape_split_out
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * split_out
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_size_t *reshape_split_size = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, &split_out->attr.dtype, TRUE);
    output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    /* reshape [w,h,c,t,n] to [w,h,c,n] */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE, 0, 0 );
    reshape_split_size = vsi_nn_internal_new_node_param(curr,
        VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    reshape_split_size[0] = split_out->attr.size[0];
    reshape_split_size[1] = split_out->attr.size[1];
    reshape_split_size[2] = split_out->attr.size[2];
    reshape_split_size[3] = split_out->attr.size[4];
    curr->node->nn_param.reshape.size = reshape_split_size;
    curr->node->nn_param.reshape.dim_num = 4;

    curr->inputs[0] = split_out;
    curr->outputs[0] = output_tensor->t;
    vsi_nn_internal_setup_node( self, curr );

    return output_tensor;
} /* reshape_split_out() */

static void split_input_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t ** output,
    uint32_t time_step
    )
{
    uint32_t i;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_node_t* curr = NULL;
    uint32_t * slices = NULL;
    vsi_nn_internal_tensor_t* output_tensor = NULL;

    i = 0;
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_SPLIT, 1, time_step );
    slices = (uint32_t *)vsi_nn_internal_new_node_param(curr, time_step * sizeof(uint32_t));
    curr->node->nn_param.split.axis = 3; /* input_shape [w,h,c,t,n] */
    curr->node->nn_param.split.slices_num = time_step;
    curr->inputs[0] = input;
    curr->node->nn_param.split.slices = slices;

    for( i = 0; i < time_step; i++ )
    {
        slices[i] = 1;
        vsi_nn_internal_init_tensor_attr(&attr, &input->attr.dtype, TRUE);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        curr->outputs[i] = output_tensor->t;
        output[i] = output_tensor->t;
    }
    vsi_nn_internal_setup_node( self, curr );
} /* split_input_tensor() */

static void trans_output_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_size_t perm[VSI_NN_MAX_DIM_NUM], ID;
    vsi_nn_conv2d_lstm_param * p = &self->nn_param.conv2d_lstm;

    ID = 0;
    memset(perm, 0, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);

    // out1,out2 [w,h,c,n] --> [c,w,h,n]
    perm[0] = 2;
    perm[1] = 0;
    perm[2] = 1;
    perm[3] = 3;
    ID = CONV2D_LSTM_OUT_H_STATE;
    vsi_nn_rnn_create_permute(self, inputs[ID], outputs[ID], perm, 4, TRUE);

    ID = CONV2D_LSTM_OUT_C_STATE;
    vsi_nn_rnn_create_permute(self, inputs[ID], outputs[ID], perm, 4, TRUE);

    ID = CONV2D_LSTM_OUT_OUTPUT;
    if(p->return_sequences == TRUE)
    {
        // out0 [w,h,c,t,n] --> [c,w,h,t,n]
        perm[0] = 2;
        perm[1] = 0;
        perm[2] = 1;
        perm[3] = 3;
        perm[4] = 4;
        vsi_nn_rnn_create_permute(self, inputs[ID], outputs[ID], perm, 5, TRUE);
    }
    else
    {
        vsi_nn_rnn_create_permute(self, inputs[ID], outputs[ID], perm, 4, TRUE);
    }
} /* trans_output_tensor() */

static void trans_input_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** trans_inputs
    )
{
    vsi_size_t perm[VSI_NN_MAX_DIM_NUM];
    vsi_nn_internal_tensor_t * tmp_tensor = NULL;
    vsi_nn_conv2d_lstm_param * p = &self->nn_param.conv2d_lstm;

    memset(perm, 0, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
    if(p->data_format == CONV2D_LSTM_CHANNELS_LAST)
    {
        // [c,w,h,t,n] --> [w,h,c,t,n]
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 0;
        perm[3] = 3;
        perm[4] = 4;
        tmp_tensor = vsi_nn_rnn_create_permute(self, inputs[CONV2D_LSTM_IN_INPUT], NULL, perm, 5, TRUE);
        trans_inputs[CONV2D_LSTM_IN_INPUT] = tmp_tensor->t;

        // [c,w,h,n] --> [w,h,c,n]
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 0;
        perm[3] = 3;
        tmp_tensor = vsi_nn_rnn_create_permute(self, inputs[CONV2D_LSTM_IN_H_STATE], NULL, perm, 4, TRUE);
        trans_inputs[CONV2D_LSTM_IN_H_STATE] = tmp_tensor->t;

        tmp_tensor = vsi_nn_rnn_create_permute(self, inputs[CONV2D_LSTM_IN_C_STATE], NULL, perm, 4, TRUE);
        trans_inputs[CONV2D_LSTM_IN_C_STATE] = tmp_tensor->t;
    }
    else
    {
        trans_inputs[CONV2D_LSTM_IN_INPUT] = inputs[CONV2D_LSTM_IN_INPUT];
        trans_inputs[CONV2D_LSTM_IN_H_STATE] = inputs[CONV2D_LSTM_IN_H_STATE];
        trans_inputs[CONV2D_LSTM_IN_C_STATE] = inputs[CONV2D_LSTM_IN_C_STATE];
    }
} /* trans_input_tensor() */

static void create_state_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_size_t w_out,
    vsi_size_t h_out,
    vsi_size_t out_channel
    )
{
    vsi_size_t samples, state_shape[4];
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t * tensor = NULL;
    vsi_nn_conv2d_lstm_param * p = &self->nn_param.conv2d_lstm;

    samples = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[4];
    memset(state_shape, 0, sizeof(vsi_size_t) * 4);
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    if(p->data_format == CONV2D_LSTM_CHANNELS_LAST)
    {
        state_shape[0] = out_channel;
        state_shape[1] = w_out;
        state_shape[2] = h_out;
        state_shape[3] = samples;
    }
    else
    {
        state_shape[0] = w_out;
        state_shape[1] = h_out;
        state_shape[2] = out_channel;
        state_shape[3] = samples;
    }

    if(NULL == inputs[CONV2D_LSTM_IN_H_STATE])
    {
        attr.dim_num = 4;
        memcpy(attr.size, state_shape, sizeof(vsi_size_t) * attr.dim_num);
        memcpy(&attr.dtype, &outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ));
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        inputs[CONV2D_LSTM_IN_H_STATE] = tensor->t;
    }

    if(NULL == inputs[CONV2D_LSTM_IN_C_STATE])
    {
        attr.dim_num = 4;
        memcpy(attr.size, state_shape, sizeof(vsi_size_t) * attr.dim_num);
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        inputs[CONV2D_LSTM_IN_C_STATE] = tensor->t;
    }

    if(NULL == outputs[CONV2D_LSTM_OUT_H_STATE])
    {
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        memcpy( &attr.dtype, &outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = TRUE;
        attr.is_const = FALSE;
        tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        outputs[CONV2D_LSTM_OUT_H_STATE] = tensor->t;
    }

    if(NULL == outputs[CONV2D_LSTM_OUT_C_STATE])
    {
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        attr.vtl = TRUE;
        attr.is_const = FALSE;
        tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        outputs[CONV2D_LSTM_OUT_C_STATE] = tensor->t;
    }
} /* create_state_tensor() */

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_size_t w_out, h_out, samples, timestep, out_channel;
    vsi_size_t conv_in_shape[4];
    vsi_nn_conv2d_lstm_param * p = &self->nn_param.conv2d_lstm;
    vsi_size_t ksize[_cnt_of_array(p->conv2d.ksize)];
    vsi_size_t i, pad[_cnt_of_array(p->conv2d.pad)] = {0};

    memset(&attr, 0, sizeof(attr));
    memset(conv_in_shape, 0, sizeof(vsi_size_t) * 4);
    w_out = 0;
    h_out = 0;
    timestep = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[3];
    samples = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[4];
    out_channel = p->filters;

    // conv_in_shape is always whcn
    if(p->data_format == CONV2D_LSTM_CHANNELS_LAST)
    {
        /* input: [in_channel, w, h, time_step, batch] */
        conv_in_shape[0] = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[1];
        conv_in_shape[1] = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[2];
        conv_in_shape[2] = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[0];
        conv_in_shape[3] = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[4];
    }
    else
    {
        /* input: [w, h, in_channel, time_step, batch] */
        conv_in_shape[0] = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[0];
        conv_in_shape[1] = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[1];
        conv_in_shape[2] = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[2];
        conv_in_shape[3] = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[4];
    }

    for(i = 0; i < _cnt_of_array(p->conv2d.ksize); i++)
    {
        ksize[i] = self->nn_param.conv2d.ksize[i];
    }
    for(i = 0; i < _cnt_of_array(p->conv2d.pad); i++)
    {
        pad[i] = self->nn_param.conv2d.pad[i];
    }

    vsi_nn_compute_padding(
        conv_in_shape,
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
        conv_in_shape[0],
        p->conv2d.ksize[0],
        &p->conv2d.pad[0],
        p->conv2d.stride[0],
        p->conv2d.dilation[0],
        VSI_NN_ROUND_FLOOR
    );
    h_out = vsi_nn_ComputeFilterSize(
        conv_in_shape[1],
        p->conv2d.ksize[1],
        &p->conv2d.pad[2],
        p->conv2d.stride[1],
        p->conv2d.dilation[1],
        VSI_NN_ROUND_FLOOR
    );

    /* setup conv2d lstm output tensors' shape */
    /* output */
    if(VSI_NN_DIM_AUTO == outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.dim_num)
    {
        if(p->data_format == CONV2D_LSTM_CHANNELS_LAST)
        {
            outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.size[0] = out_channel;
            outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.size[1] = w_out;
            outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.size[2] = h_out;
        }
        else
        {
            outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.size[0] = w_out;
            outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.size[1] = h_out;
            outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.size[2] = out_channel;
        }
        outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.size[3] = timestep;
        outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.size[4] = samples;
        outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.dim_num = 5;
    }

    /* create hstate and cstate input/output if app doesn't provide them */
    create_state_tensor(self, inputs, outputs, w_out, h_out, out_channel);

    /* hidden state output */
    if(VSI_NN_DIM_AUTO == outputs[CONV2D_LSTM_OUT_H_STATE]->attr.dim_num)
    {
        if(p->data_format == CONV2D_LSTM_CHANNELS_LAST)
        {
            outputs[CONV2D_LSTM_OUT_H_STATE]->attr.size[0] = out_channel;
            outputs[CONV2D_LSTM_OUT_H_STATE]->attr.size[1] = w_out;
            outputs[CONV2D_LSTM_OUT_H_STATE]->attr.size[2] = h_out;
        }
        else
        {
            outputs[CONV2D_LSTM_OUT_H_STATE]->attr.size[0] = w_out;
            outputs[CONV2D_LSTM_OUT_H_STATE]->attr.size[1] = h_out;
            outputs[CONV2D_LSTM_OUT_H_STATE]->attr.size[2] = out_channel;
        }
        outputs[CONV2D_LSTM_OUT_H_STATE]->attr.size[3] = samples;
        outputs[CONV2D_LSTM_OUT_H_STATE]->attr.dim_num = 4;
    }

    /* cell state output */
    if(VSI_NN_DIM_AUTO == outputs[CONV2D_LSTM_OUT_C_STATE]->attr.dim_num)
    {
        if(p->data_format == CONV2D_LSTM_CHANNELS_LAST)
        {
            outputs[CONV2D_LSTM_OUT_C_STATE]->attr.size[0] = out_channel;
            outputs[CONV2D_LSTM_OUT_C_STATE]->attr.size[1] = w_out;
            outputs[CONV2D_LSTM_OUT_C_STATE]->attr.size[2] = h_out;
        }
        else
        {
            outputs[CONV2D_LSTM_OUT_C_STATE]->attr.size[0] = w_out;
            outputs[CONV2D_LSTM_OUT_C_STATE]->attr.size[1] = h_out;
            outputs[CONV2D_LSTM_OUT_C_STATE]->attr.size[2] = out_channel;
        }
        outputs[CONV2D_LSTM_OUT_C_STATE]->attr.size[3] = samples;
        outputs[CONV2D_LSTM_OUT_C_STATE]->attr.dim_num = 4;
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
    vsi_size_t i, timestep, perm[VSI_NN_MAX_DIM_NUM];
    vsi_nn_tensor_t * trans_inputs[3] = { NULL };
    vsi_nn_tensor_t * conv2dlstm_outputs[3] = { NULL };
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t * tmp_tensor = NULL;
    vsi_nn_tensor_t ** split_outputs = NULL, ** conv2dlstm_step_outputs = NULL;
    vsi_nn_tensor_t * step_h_state = NULL, * step_c_state = NULL;
    vsi_nn_tensor_t * cell_out0 = NULL, * cell_out1 = NULL, * cell_out2 = NULL;
    vsi_nn_conv2d_lstm_param * p = &self->nn_param.conv2d_lstm;
    vsi_nn_internal_node_t* curr = NULL;

    memset(&attr, 0, sizeof(attr));
    memset(perm, 0, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
    timestep = inputs[CONV2D_LSTM_IN_INPUT]->attr.size[3];

    vsi_nn_internal_init_node_wksp( self );

    setup_op_shapes(self, inputs, outputs);

    trans_input_tensor(self, inputs, trans_inputs);

    split_outputs = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *) * timestep);
    conv2dlstm_step_outputs = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *) * timestep);
    memset(split_outputs, 0, sizeof(vsi_nn_tensor_t *) * timestep);
    memset(conv2dlstm_step_outputs, 0, sizeof(vsi_nn_tensor_t *) * timestep);

    /* split input tensor by time-step */
    split_input_tensor(self, trans_inputs[CONV2D_LSTM_IN_INPUT], split_outputs, (uint32_t)timestep);

    cell_out0 = cell_out1 = cell_out2 = NULL;
    step_h_state = trans_inputs[CONV2D_LSTM_IN_H_STATE];
    step_c_state = trans_inputs[CONV2D_LSTM_IN_C_STATE];
    for( i = 0; i < timestep; i++ )
    {
        vsi_nn_tensor_t * reshape_output = NULL;

        /* reshape for split output */
        tmp_tensor = reshape_split_out(self, split_outputs[i]);
        reshape_output = tmp_tensor->t;

        if((i == timestep - 1) && p->return_sequences == FALSE && p->data_format == CONV2D_LSTM_CHANNELS_FIRST)
        {
            cell_out0 = outputs[CONV2D_LSTM_OUT_OUTPUT];
        }
        else
        {
            vsi_nn_internal_init_tensor_attr(&attr, &outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.dtype, TRUE);
            tmp_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            cell_out0 = tmp_tensor->t;
        }

        if((i == timestep - 1) && p->data_format == CONV2D_LSTM_CHANNELS_FIRST)
        {
            cell_out1 = outputs[CONV2D_LSTM_OUT_H_STATE];
            cell_out2 = outputs[CONV2D_LSTM_OUT_C_STATE];
        }
        else
        {
            /* conv2d_lstm hstate output */
            vsi_nn_internal_init_tensor_attr(&attr, &outputs[CONV2D_LSTM_OUT_H_STATE]->attr.dtype, TRUE);
            tmp_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            cell_out1 = tmp_tensor->t;

            /* conv2d_lstm cstate output */
            vsi_nn_internal_init_tensor_attr(&attr, &outputs[CONV2D_LSTM_OUT_C_STATE]->attr.dtype, TRUE);
            tmp_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            cell_out2 = tmp_tensor->t;
        }

        /* create a conv2d_lstm_cell */
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONV2D_LSTM_CELL, 0, 0 );
        curr->node->nn_param.conv2d_lstm_cell.filters = p->filters;
        curr->node->nn_param.conv2d_lstm_cell.activation = p->activation;
        curr->node->nn_param.conv2d_lstm_cell.recurrent_activation = p->recurrent_activation;
        memcpy(&curr->node->nn_param.conv2d_lstm_cell.conv2d, &p->conv2d, sizeof(p->conv2d));

        curr->inputs[CONV2D_LSTM_CELL_IN_INPUT] = reshape_output;
        curr->inputs[CONV2D_LSTM_CELL_IN_H_STATE] = step_h_state;
        curr->inputs[CONV2D_LSTM_CELL_IN_C_STATE] = step_c_state;
        curr->inputs[CONV2D_LSTM_CELL_IN_KERNEL_I2I] = inputs[CONV2D_LSTM_IN_KERNEL_I2I];
        curr->inputs[CONV2D_LSTM_CELL_IN_KERNEL_I2F] = inputs[CONV2D_LSTM_IN_KERNEL_I2F];
        curr->inputs[CONV2D_LSTM_CELL_IN_KERNEL_I2C] = inputs[CONV2D_LSTM_IN_KERNEL_I2C];
        curr->inputs[CONV2D_LSTM_CELL_IN_KERNEL_I2O] = inputs[CONV2D_LSTM_IN_KERNEL_I2O];
        curr->inputs[CONV2D_LSTM_CELL_IN_KERNEL_R2I] = inputs[CONV2D_LSTM_IN_KERNEL_R2I];
        curr->inputs[CONV2D_LSTM_CELL_IN_KERNEL_R2F] = inputs[CONV2D_LSTM_IN_KERNEL_R2F];
        curr->inputs[CONV2D_LSTM_CELL_IN_KERNEL_R2C] = inputs[CONV2D_LSTM_IN_KERNEL_R2C];
        curr->inputs[CONV2D_LSTM_CELL_IN_KERNEL_R2O] = inputs[CONV2D_LSTM_IN_KERNEL_R2O];
        curr->inputs[CONV2D_LSTM_CELL_IN_BIAS_I] = inputs[CONV2D_LSTM_IN_BIAS_I];
        curr->inputs[CONV2D_LSTM_CELL_IN_BIAS_F] = inputs[CONV2D_LSTM_IN_BIAS_F];
        curr->inputs[CONV2D_LSTM_CELL_IN_BIAS_C] = inputs[CONV2D_LSTM_IN_BIAS_C];
        curr->inputs[CONV2D_LSTM_CELL_IN_BIAS_O] = inputs[CONV2D_LSTM_IN_BIAS_O];
        curr->outputs[CONV2D_LSTM_CELL_OUT_OUTPUT] = cell_out0;
        curr->outputs[CONV2D_LSTM_CELL_OUT_H_STATE] = cell_out1;
        curr->outputs[CONV2D_LSTM_CELL_OUT_C_STATE] = cell_out2;

        vsi_nn_internal_setup_node( self, curr );

        /* update the state tensor for next time-step hstate and cstate */
        step_h_state = cell_out1;
        step_c_state = cell_out2;

        if(p->return_sequences == TRUE)
        {
            /* store step's outputs */
            tmp_tensor = reshape_cell_out(self, cell_out0);
            conv2dlstm_step_outputs[i] = tmp_tensor->t;
        }
    }

    if(p->return_sequences == TRUE)
    {
        if(p->data_format == CONV2D_LSTM_CHANNELS_LAST)
        {
            vsi_nn_internal_init_tensor_attr(&attr, &outputs[CONV2D_LSTM_OUT_OUTPUT]->attr.dtype, TRUE);
            tmp_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            conv2dlstm_outputs[CONV2D_LSTM_OUT_OUTPUT] = tmp_tensor->t;
        }
        else
        {
            conv2dlstm_outputs[CONV2D_LSTM_OUT_OUTPUT] = outputs[CONV2D_LSTM_OUT_OUTPUT];
        }
        /* concat all step's output0 data on dimension t --- cell out0 shape: [w,h,c,t,n] */
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONCAT, (uint32_t)timestep, 1 );
        curr->node->nn_param.concat.axis = 3;
        for(i = 0; i < timestep; i++)
        {
            curr->inputs[i] = conv2dlstm_step_outputs[i];
        }
        curr->outputs[0] = conv2dlstm_outputs[CONV2D_LSTM_OUT_OUTPUT];
        vsi_nn_internal_setup_node( self, curr );
    }
    else
    {
        conv2dlstm_outputs[CONV2D_LSTM_OUT_OUTPUT] = cell_out0;
    }

    conv2dlstm_outputs[CONV2D_LSTM_OUT_H_STATE] = cell_out1;
    conv2dlstm_outputs[CONV2D_LSTM_OUT_C_STATE] = cell_out2;
    if(p->data_format == CONV2D_LSTM_CHANNELS_LAST)
    {
        trans_output_tensor(self, conv2dlstm_outputs, outputs);
    }

    vsi_nn_safe_free(split_outputs);
    vsi_nn_safe_free(conv2dlstm_step_outputs)
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
    /* op_name    */ CONV2D_LSTM,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ CONV2D_LSTM_IN_CNT,
    /* output_num */ CONV2D_LSTM_OUT_CNT
    );
#ifdef __cpluplus
}
#endif
