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

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_bidirectional_sequence_lstm_param* curr_param =
        &self->nn_param.bidirectional_sequence_lstm;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t num_units =  0;
    uint32_t output_size = 0;
    uint32_t batch_size = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    if( curr_param->time_major )
    {
        batch_size = inputs[BI_LSTM_INPUT_INPUT]->attr.size[1];
    }
    else
    {
        batch_size = inputs[BI_LSTM_INPUT_INPUT]->attr.size[2];
    }

    num_units = inputs[BI_LSTM_FW_INPUT_WEIGHT_I2F]->attr.size[1];
    output_size = num_units;

    /* create h_state if app doesn't provide them */
    if( !inputs[BI_LSTM_FW_INPUT_H_STATE] )
    {
        attr.dim_num = 2;
        attr.size[1] = batch_size;
        attr.size[0] = output_size;
        memcpy(&attr.dtype, &outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ));
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        inputs[BI_LSTM_FW_INPUT_H_STATE] = output_tensor->t;
    }

    if( !inputs[BI_LSTM_BW_INPUT_H_STATE] )
    {
        attr.dim_num = 2;
        attr.size[1] = batch_size;
        attr.size[0] = output_size;
        memcpy(&attr.dtype, &outputs[BI_LSTM_BW_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ));
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        inputs[BI_LSTM_BW_INPUT_H_STATE] = output_tensor->t;
    }

    /* output */
    if( VSI_NN_DIM_AUTO == outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dim_num )
    {
        if( curr_param->merge_outputs )
        {
            outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.size[0] = output_size * 2;
            outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.size[1] = inputs[BI_LSTM_INPUT_INPUT]->attr.size[1];
            outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.size[2] = inputs[BI_LSTM_INPUT_INPUT]->attr.size[2];
            outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dim_num = 3;
        }
        else
        {
            outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.size[0] = output_size;
            outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.size[1] = inputs[BI_LSTM_INPUT_INPUT]->attr.size[1];
            outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.size[2] = inputs[BI_LSTM_INPUT_INPUT]->attr.size[2];
            outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dim_num = 3;

            outputs[BI_LSTM_BW_OUTPUT_OUTPUT]->attr.size[0] = output_size;
            outputs[BI_LSTM_BW_OUTPUT_OUTPUT]->attr.size[1] = inputs[BI_LSTM_INPUT_INPUT]->attr.size[1];
            outputs[BI_LSTM_BW_OUTPUT_OUTPUT]->attr.size[2] = inputs[BI_LSTM_INPUT_INPUT]->attr.size[2];
            outputs[BI_LSTM_BW_OUTPUT_OUTPUT]->attr.dim_num = 3;
        }
    }

    return TRUE;
}

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
    vsi_nn_bidirectional_sequence_lstm_param* curr_param =
        &self->nn_param.bidirectional_sequence_lstm;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_tensor_t** split_output_tensors = NULL;
    vsi_nn_tensor_t** lstmcell_reshape_output_tensors_fw =NULL;
    vsi_nn_tensor_t** lstmcell_reshape_output_tensors_bw =NULL;
    vsi_nn_tensor_t* last_step_h_state_fw = NULL;
    vsi_nn_tensor_t* last_step_h_state_bw = NULL;
    vsi_nn_tensor_t* last_step_c_state_fw = NULL;
    vsi_nn_tensor_t* last_step_c_state_bw = NULL;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_tensor_t* input_tensor = NULL;
    vsi_nn_tensor_t* aux_input_tensor = NULL;
    vsi_nn_tensor_t** aux_split_output_tensors = NULL;
    vsi_nn_tensor_t** reshape_output_tensors = NULL;
    vsi_nn_tensor_t** aux_reshape_output_tensors = NULL;
    vsi_bool has_aux_input = (inputs[BI_LSTM_AUX_INPUT] != NULL);
    vsi_bool use_virtual_tensor = TRUE;
    uint32_t batch_size = 0;
    uint32_t time_step = 0;
    uint32_t i = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_node_wksp( self );

    if( curr_param->time_major )
    {
        batch_size = inputs[BI_LSTM_INPUT_INPUT]->attr.size[1];
        time_step = inputs[BI_LSTM_INPUT_INPUT]->attr.size[2];
    }
    else
    {
        batch_size = inputs[BI_LSTM_INPUT_INPUT]->attr.size[2];
        time_step = inputs[BI_LSTM_INPUT_INPUT]->attr.size[1];
    }

    setup_op_shapes( self, inputs, outputs);

    /* default to input */
    input_tensor = inputs[BI_LSTM_INPUT_INPUT];
    if( !curr_param->time_major )
    {
        /* transpose to time_major */
        output_tensor = vsi_nn_rnn_transpose_time_major(self,
            inputs[BI_LSTM_INPUT_INPUT], NULL, use_virtual_tensor);
        input_tensor = output_tensor->t;
    }

    /* default to aux input */
    if(has_aux_input)
    {
        aux_input_tensor = inputs[BI_LSTM_AUX_INPUT];
        if( !curr_param->time_major )
        {
            /* transpose to time_major */
            output_tensor = vsi_nn_rnn_transpose_time_major(self,
                inputs[BI_LSTM_AUX_INPUT], NULL, use_virtual_tensor);
            aux_input_tensor = output_tensor->t;
        }
    }

    /* split input tensor */
    split_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
    memset( split_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));
    reshape_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
    memset( reshape_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));

    vsi_nn_rnn_split_input_tensor(self, input_tensor,
        split_output_tensors, time_step, use_virtual_tensor);

    vsi_nn_rnn_data_check_aligned(self, split_output_tensors, time_step, use_virtual_tensor);

    /* split aux input tensor */
    if(has_aux_input)
    {
        aux_split_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
        memset( aux_split_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));
        aux_reshape_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
        memset( aux_reshape_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));

        vsi_nn_rnn_split_input_tensor(self, aux_input_tensor,
            aux_split_output_tensors, time_step, use_virtual_tensor);

        vsi_nn_rnn_data_check_aligned(self, aux_split_output_tensors, time_step, use_virtual_tensor);
    }

    /* prepare output tensor */
    lstmcell_reshape_output_tensors_fw = (vsi_nn_tensor_t **)malloc(time_step *
        sizeof(vsi_nn_tensor_t **));
    memset( lstmcell_reshape_output_tensors_fw, 0x00, time_step * sizeof(vsi_nn_tensor_t **));
    lstmcell_reshape_output_tensors_bw = (vsi_nn_tensor_t **)malloc(time_step *
        sizeof(vsi_nn_tensor_t **));
    memset( lstmcell_reshape_output_tensors_bw, 0x00, time_step * sizeof(vsi_nn_tensor_t **));

    for( i = 0; i < time_step; i++ )
    {
        /* reshape for split output */
        output_tensor = vsi_nn_rnn_reshape_split_output(self,
            split_output_tensors[i], batch_size, use_virtual_tensor);
        reshape_output_tensors[i] = output_tensor->t;

        if (has_aux_input)
        {
            /* reshape for aux split output */
            output_tensor = vsi_nn_rnn_reshape_split_output(self,
                aux_split_output_tensors[i], batch_size, use_virtual_tensor);
            aux_reshape_output_tensors[i] = output_tensor->t;
        }
    }

    /* forward lstm op */
    last_step_h_state_fw = inputs[BI_LSTM_FW_INPUT_H_STATE];
    last_step_c_state_fw = inputs[BI_LSTM_FW_INPUT_C_STATE];
    for( i = 0; i < time_step; i++ )
    {
        vsi_nn_tensor_t* lstmcell_out0 = NULL;
        vsi_nn_tensor_t* lstmcell_out1 = NULL;
        vsi_nn_tensor_t* lstmcell_out2 = NULL;

        /* lstmcell output */
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        lstmcell_out0 = output_tensor->t;

        /* lstmcell output h_state */
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        lstmcell_out1 = output_tensor->t;

        /* lstmcell output c_state */
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        lstmcell_out2 = output_tensor->t;

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_LSTMUNIT_OVXLIB, 0, 0 );
        curr->node->nn_param.lstmunit_ovxlib.activation = curr_param->activation;
        curr->node->nn_param.lstmunit_ovxlib.cell_clip = curr_param->cell_clip;
        curr->node->nn_param.lstmunit_ovxlib.forget_bias = curr_param->forget_bias;
        curr->node->nn_param.lstmunit_ovxlib.proj_clip = curr_param->proj_clip;
        curr->node->nn_param.lstmunit_ovxlib.recurrent_activation = curr_param->recurrent_activation;
        memcpy( curr->node->nn_param.lstm_ovxlib.internal_dtype,
            curr_param->internal_dtype,
            sizeof(vsi_nn_dtype_t) * LSTMUNIT_QUANTIZE_PARAM_COUNT);
        curr->inputs[LSTMUNIT_INPUT_INPUT] = reshape_output_tensors[i];
        curr->inputs[LSTMUNIT_INPUT_H_STATE] = last_step_h_state_fw;
        curr->inputs[LSTMUNIT_INPUT_C_STATE] = last_step_c_state_fw;
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2I] = inputs[BI_LSTM_FW_INPUT_WEIGHT_I2I];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2F] = inputs[BI_LSTM_FW_INPUT_WEIGHT_I2F];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2C] = inputs[BI_LSTM_FW_INPUT_WEIGHT_I2C];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2O] = inputs[BI_LSTM_FW_INPUT_WEIGHT_I2O];

        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2I] = inputs[BI_LSTM_FW_INPUT_WEIGHT_R2I];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2F] = inputs[BI_LSTM_FW_INPUT_WEIGHT_R2F];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2C] = inputs[BI_LSTM_FW_INPUT_WEIGHT_R2C];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2O] = inputs[BI_LSTM_FW_INPUT_WEIGHT_R2O];

        curr->inputs[LSTMUNIT_INPUT_WEIGHT_C2I] = inputs[BI_LSTM_FW_INPUT_WEIGHT_C2I];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_C2F] = inputs[BI_LSTM_FW_INPUT_WEIGHT_C2F];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_C2O] = inputs[BI_LSTM_FW_INPUT_WEIGHT_C2O];

        curr->inputs[LSTMUNIT_INPUT_BIAS_I] = inputs[BI_LSTM_FW_INPUT_BIAS_I];
        curr->inputs[LSTMUNIT_INPUT_BIAS_F] = inputs[BI_LSTM_FW_INPUT_BIAS_F];
        curr->inputs[LSTMUNIT_INPUT_BIAS_C] = inputs[BI_LSTM_FW_INPUT_BIAS_C];
        curr->inputs[LSTMUNIT_INPUT_BIAS_O] = inputs[BI_LSTM_FW_INPUT_BIAS_O];

        curr->inputs[LSTMUNIT_INPUT_WEIGHT_PROJ] = inputs[BI_LSTM_FW_INPUT_WEIGHT_PROJ];
        curr->inputs[LSTMUNIT_INPUT_BIAS_PROJ] = inputs[BI_LSTM_FW_INPUT_BIAS_PROJ];

        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_I] = inputs[BI_LSTM_FW_INPUT_LAYERNORM_I];
        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_F] = inputs[BI_LSTM_FW_INPUT_LAYERNORM_F];
        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_C] = inputs[BI_LSTM_FW_INPUT_LAYERNORM_C];
        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_O] = inputs[BI_LSTM_FW_INPUT_LAYERNORM_O];

        if (has_aux_input)
        {
            curr->inputs[LSTM_INPUT_AUX_INPUT] = aux_reshape_output_tensors[i];
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2I] = inputs[BI_LSTM_FW_AUX_INPUT_WEIGHT_I2I];
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2F] = inputs[BI_LSTM_FW_AUX_INPUT_WEIGHT_I2F];
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2C] = inputs[BI_LSTM_FW_AUX_INPUT_WEIGHT_I2C];
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2O] = inputs[BI_LSTM_FW_AUX_INPUT_WEIGHT_I2O];
        }
        else
        {
            curr->inputs[LSTM_INPUT_AUX_INPUT] = NULL;
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2I] = NULL;
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2F] = NULL;
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2C] = NULL;
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2O] = NULL;
        }

        curr->outputs[LSTMUNIT_OUTPUT_OUTPUT] = lstmcell_out0;
        curr->outputs[LSTMUNIT_OUTPUT_H_STATE] = lstmcell_out1;
        curr->outputs[LSTMUNIT_OUTPUT_C_STATE] = lstmcell_out2;

        vsi_nn_internal_setup_node( self, curr );

        last_step_h_state_fw = lstmcell_out1;
        last_step_c_state_fw = lstmcell_out2;

        /* reshape output to 3-dims */
        output_tensor = vsi_nn_rnn_reshape_cell_output(self,
            lstmcell_out0, batch_size, use_virtual_tensor);
        lstmcell_reshape_output_tensors_fw[i] = output_tensor->t;
    }

    /* backward lstm op */
    last_step_h_state_bw = inputs[BI_LSTM_BW_INPUT_H_STATE];
    last_step_c_state_bw = inputs[BI_LSTM_BW_INPUT_C_STATE];
    for( i = 0; i < time_step; i++ )
    {
        vsi_nn_tensor_t* lstmcell_out0 = NULL;
        vsi_nn_tensor_t* lstmcell_out1 = NULL;
        vsi_nn_tensor_t* lstmcell_out2 = NULL;

        /* lstmcell output */
        /* if merge_outputs is true, there will be only 1 output, so use the attr
           of the fw for the bw, since they are always same as each other.*/
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        lstmcell_out0 = output_tensor->t;

        /* lstmcell output h_state */
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        lstmcell_out1 = output_tensor->t;

        /* lstmcell output c_state */
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        lstmcell_out2 = output_tensor->t;

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_LSTMUNIT_OVXLIB, 0, 0 );
        curr->node->nn_param.lstmunit_ovxlib.activation = curr_param->activation;
        curr->node->nn_param.lstmunit_ovxlib.cell_clip = curr_param->cell_clip;
        curr->node->nn_param.lstmunit_ovxlib.forget_bias = curr_param->forget_bias;
        curr->node->nn_param.lstmunit_ovxlib.proj_clip = curr_param->proj_clip;
        curr->node->nn_param.lstmunit_ovxlib.recurrent_activation = curr_param->recurrent_activation;
        memcpy( curr->node->nn_param.lstm_ovxlib.internal_dtype,
            &(curr_param->internal_dtype[LSTMUNIT_QUANTIZE_PARAM_COUNT]),
            sizeof(vsi_nn_dtype_t) * LSTMUNIT_QUANTIZE_PARAM_COUNT);
        curr->inputs[LSTMUNIT_INPUT_INPUT] = reshape_output_tensors[time_step - 1 - i];
        curr->inputs[LSTMUNIT_INPUT_H_STATE] = last_step_h_state_bw;
        curr->inputs[LSTMUNIT_INPUT_C_STATE] = last_step_c_state_bw;
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2I] = inputs[BI_LSTM_BW_INPUT_WEIGHT_I2I];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2F] = inputs[BI_LSTM_BW_INPUT_WEIGHT_I2F];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2C] = inputs[BI_LSTM_BW_INPUT_WEIGHT_I2C];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2O] = inputs[BI_LSTM_BW_INPUT_WEIGHT_I2O];

        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2I] = inputs[BI_LSTM_BW_INPUT_WEIGHT_R2I];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2F] = inputs[BI_LSTM_BW_INPUT_WEIGHT_R2F];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2C] = inputs[BI_LSTM_BW_INPUT_WEIGHT_R2C];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2O] = inputs[BI_LSTM_BW_INPUT_WEIGHT_R2O];

        curr->inputs[LSTMUNIT_INPUT_WEIGHT_C2I] = inputs[BI_LSTM_BW_INPUT_WEIGHT_C2I];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_C2F] = inputs[BI_LSTM_BW_INPUT_WEIGHT_C2F];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_C2O] = inputs[BI_LSTM_BW_INPUT_WEIGHT_C2O];

        curr->inputs[LSTMUNIT_INPUT_BIAS_I] = inputs[BI_LSTM_BW_INPUT_BIAS_I];
        curr->inputs[LSTMUNIT_INPUT_BIAS_F] = inputs[BI_LSTM_BW_INPUT_BIAS_F];
        curr->inputs[LSTMUNIT_INPUT_BIAS_C] = inputs[BI_LSTM_BW_INPUT_BIAS_C];
        curr->inputs[LSTMUNIT_INPUT_BIAS_O] = inputs[BI_LSTM_BW_INPUT_BIAS_O];

        curr->inputs[LSTMUNIT_INPUT_WEIGHT_PROJ] = inputs[BI_LSTM_BW_INPUT_WEIGHT_PROJ];
        curr->inputs[LSTMUNIT_INPUT_BIAS_PROJ] = inputs[BI_LSTM_BW_INPUT_BIAS_PROJ];

        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_I] = inputs[BI_LSTM_BW_INPUT_LAYERNORM_I];
        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_F] = inputs[BI_LSTM_BW_INPUT_LAYERNORM_F];
        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_C] = inputs[BI_LSTM_BW_INPUT_LAYERNORM_C];
        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_O] = inputs[BI_LSTM_BW_INPUT_LAYERNORM_O];

        if (has_aux_input)
        {
            curr->inputs[LSTM_INPUT_AUX_INPUT] = aux_reshape_output_tensors[time_step - 1 - i];
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2I] = inputs[BI_LSTM_BW_AUX_INPUT_WEIGHT_I2I];
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2F] = inputs[BI_LSTM_BW_AUX_INPUT_WEIGHT_I2F];
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2C] = inputs[BI_LSTM_BW_AUX_INPUT_WEIGHT_I2C];
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2O] = inputs[BI_LSTM_BW_AUX_INPUT_WEIGHT_I2O];
        }
        else
        {
            curr->inputs[LSTM_INPUT_AUX_INPUT] = NULL;
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2I] = NULL;
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2F] = NULL;
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2C] = NULL;
            curr->inputs[LSTM_INPUT_AUX_WEIGHT_I2O] = NULL;
        }

        curr->outputs[LSTMUNIT_OUTPUT_OUTPUT] = lstmcell_out0;
        curr->outputs[LSTMUNIT_OUTPUT_H_STATE] = lstmcell_out1;
        curr->outputs[LSTMUNIT_OUTPUT_C_STATE] = lstmcell_out2;

        vsi_nn_internal_setup_node( self, curr );

        last_step_h_state_bw = lstmcell_out1;
        last_step_c_state_bw = lstmcell_out2;

        /* reshape output to 3-dims */
        output_tensor = vsi_nn_rnn_reshape_cell_output(self,
            lstmcell_out0, batch_size, use_virtual_tensor);
        lstmcell_reshape_output_tensors_bw[i] = output_tensor->t;
    }

    if(curr_param->merge_outputs)
    {
        vsi_nn_tensor_t** merge_tensors = NULL;
        merge_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
        memset( merge_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));

        tensor = outputs[BI_LSTM_FW_OUTPUT_OUTPUT];
        if( !curr_param->time_major )
        {
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
            output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

            tensor = output_tensor->t;
        }

        /* concat fw & bw output, the lstm's output is 3-dims */
        for( i = 0; i < time_step; i++ )
        {
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
            output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONCAT, 2, 1 );
            curr->node->nn_param.concat.axis = 0;
            curr->inputs[0] = lstmcell_reshape_output_tensors_fw[i];
            curr->inputs[1] = lstmcell_reshape_output_tensors_bw[i];
            curr->outputs[0] = output_tensor->t;
            vsi_nn_internal_setup_node( self, curr );
            merge_tensors[i] = output_tensor->t;
        }


        /* concat lstmcell output, the lstm's output is 3-dims */
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONCAT, time_step, 1 );
        curr->node->nn_param.concat.axis = 2;
        for( i = 0; i < time_step; i++ )
        {
            curr->inputs[i] = merge_tensors[i];
        }
        curr->outputs[0] = tensor;
        vsi_nn_internal_setup_node( self, curr );

        if( !curr_param->time_major )
        {
            /* transpose time_major to batch_major*/
            vsi_nn_rnn_transpose_time_major(self,
                tensor, outputs[BI_LSTM_FW_OUTPUT_OUTPUT], use_virtual_tensor);
        }
        vsi_nn_safe_free( merge_tensors );
    }
    else
    {
        /* forward output*/
        tensor = outputs[BI_LSTM_FW_OUTPUT_OUTPUT];
        if( !curr_param->time_major )
        {
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[BI_LSTM_FW_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
            output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

            tensor = output_tensor->t;
        }

        /* concat lstmcell output, the lstm's output is 3-dims */
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONCAT, time_step, 1 );
        curr->node->nn_param.concat.axis = 2;
        for( i = 0; i < time_step; i++ )
        {
            curr->inputs[i] = lstmcell_reshape_output_tensors_fw[i];
        }
        curr->outputs[0] = tensor;
        vsi_nn_internal_setup_node( self, curr );

        if( !curr_param->time_major )
        {
            /* transpose time_major to batch_major*/
            vsi_nn_rnn_transpose_time_major(self,
                tensor, outputs[BI_LSTM_FW_OUTPUT_OUTPUT], use_virtual_tensor);
        }

        /* backward output*/
        tensor = outputs[BI_LSTM_BW_OUTPUT_OUTPUT];
        if( !curr_param->time_major )
        {
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[BI_LSTM_BW_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
            output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

            tensor = output_tensor->t;
        }

        /* concat lstmcell output, the lstm's output is 3-dims */
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONCAT, time_step, 1 );
        curr->node->nn_param.concat.axis = 2;
        for( i = 0; i < time_step; i++ )
        {
            curr->inputs[i] = lstmcell_reshape_output_tensors_bw[i];
        }
        curr->outputs[0] = tensor;
        vsi_nn_internal_setup_node( self, curr );

        if( !curr_param->time_major )
        {
            /* transpose time_major to batch_major*/
            vsi_nn_rnn_transpose_time_major(self,
                tensor, outputs[BI_LSTM_BW_OUTPUT_OUTPUT], use_virtual_tensor);
        }
    }

    vsi_nn_safe_free( split_output_tensors );
    vsi_nn_safe_free( aux_split_output_tensors )
    vsi_nn_safe_free( reshape_output_tensors );
    vsi_nn_safe_free( aux_reshape_output_tensors );
    vsi_nn_safe_free( lstmcell_reshape_output_tensors_fw );
    vsi_nn_safe_free( lstmcell_reshape_output_tensors_bw );

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_safe_free(self->nn_param.bidirectional_sequence_lstm.internal_dtype);
    vsi_nn_internal_deinit_node_wksp( self );

    return status;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.bidirectional_sequence_lstm.internal_dtype = (vsi_nn_dtype_t *)
        malloc(sizeof(vsi_nn_dtype_t) * LSTMUNIT_QUANTIZE_PARAM_COUNT * 2);
    memset(self->nn_param.bidirectional_sequence_lstm.internal_dtype, 0,
        sizeof(vsi_nn_dtype_t) * LSTMUNIT_QUANTIZE_PARAM_COUNT * 2);

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ BIDIRECTIONAL_SEQUENCE_LSTM,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ BI_LSTM_INPUT_CNT,
    /* output_num */ BI_LSTM_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif
