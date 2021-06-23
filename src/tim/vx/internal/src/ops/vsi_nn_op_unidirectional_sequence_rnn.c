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
    vsi_nn_unidirectional_sequence_rnn_param* curr_param =
        &self->nn_param.unidirectional_sequence_rnn;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t num_units =  0;
    uint32_t output_size = 0;
    uint32_t batch_size = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    if( curr_param->time_major )
    {
        batch_size = inputs[RNN_INPUT_INPUT]->attr.size[1];
    }
    else
    {
        batch_size = inputs[RNN_INPUT_INPUT]->attr.size[2];
    }

    num_units = inputs[RNN_INPUT_WEIGHT_I]->attr.size[1];
    output_size = num_units;

    /* create h_state if app doesn't provide them */
    if( !inputs[RNN_INPUT_H_STATE] )
    {
        attr.dim_num = 2;
        attr.size[1] = batch_size;
        attr.size[0] = output_size;
        memcpy(&attr.dtype, &outputs[RNN_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ));
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        inputs[RNN_INPUT_H_STATE] = output_tensor->t;
    }

    /* output */
    if( VSI_NN_DIM_AUTO == outputs[RNN_OUTPUT_OUTPUT]->attr.dim_num )
    {
        outputs[RNN_OUTPUT_OUTPUT]->attr.size[0] = output_size;
        outputs[RNN_OUTPUT_OUTPUT]->attr.size[1] = inputs[RNN_INPUT_INPUT]->attr.size[1];
        outputs[RNN_OUTPUT_OUTPUT]->attr.size[2] = inputs[RNN_INPUT_INPUT]->attr.size[2];
        outputs[RNN_OUTPUT_OUTPUT]->attr.dim_num = 3;
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
    vsi_nn_unidirectional_sequence_rnn_param* curr_param =
        &self->nn_param.unidirectional_sequence_rnn;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_tensor_t** split_output_tensors = NULL;
    vsi_nn_tensor_t** rnncell_reshape_output_tensors =NULL;
    vsi_nn_tensor_t* last_step_h_state = NULL;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_tensor_t* input_tensor = NULL;
    vsi_bool use_virtual_tensor = TRUE;
    uint32_t batch_size = 0;
    uint32_t time_step = 0;
    uint32_t i = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    vsi_nn_internal_init_node_wksp( self );

    if( curr_param->time_major )
    {
        batch_size = inputs[RNN_INPUT_INPUT]->attr.size[1];
        time_step = inputs[RNN_INPUT_INPUT]->attr.size[2];
    }
    else
    {
        batch_size = inputs[RNN_INPUT_INPUT]->attr.size[2];
        time_step = inputs[RNN_INPUT_INPUT]->attr.size[1];
    }

    setup_op_shapes( self, inputs, outputs);

    /* default to input */
    input_tensor = inputs[RNN_INPUT_INPUT];
    if( !curr_param->time_major )
    {
        /* transpose to time_major */
        output_tensor = vsi_nn_rnn_transpose_time_major(self,
            inputs[RNN_INPUT_INPUT], NULL, use_virtual_tensor);
        input_tensor = output_tensor->t;
    }

    /* split input tensor */
    split_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
    memset( split_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));
    rnncell_reshape_output_tensors = (vsi_nn_tensor_t **)malloc(time_step *
        sizeof(vsi_nn_tensor_t **));
    memset( rnncell_reshape_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));

    vsi_nn_rnn_split_input_tensor(self, input_tensor, split_output_tensors, time_step, use_virtual_tensor);

    vsi_nn_rnn_data_check_aligned(self, split_output_tensors, time_step, use_virtual_tensor);

    last_step_h_state = inputs[RNN_INPUT_H_STATE];
    for( i = 0; i < time_step; i++ )
    {
        vsi_nn_tensor_t* reshape_output = NULL;
        vsi_nn_tensor_t* rnncell_out0 = NULL;
        vsi_nn_tensor_t* rnncell_out1 = NULL;

        /* reshape for split output */
        output_tensor = vsi_nn_rnn_reshape_split_output(self,
            split_output_tensors[i], batch_size, use_virtual_tensor);
        reshape_output = output_tensor->t;

        /* rnncell output */
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[RNN_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        rnncell_out0 = output_tensor->t;

        /* rnncell output h_state */
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[RNNCELL_OUTPUT_H_STATE]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        rnncell_out1 = output_tensor->t;

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RNNCELL_OVXLIB, 0, 0 );
        curr->node->nn_param.rnncell_ovxlib.activation = curr_param->activation;
        memcpy( curr->node->nn_param.rnncell_ovxlib.internal_dtype,
            curr_param->internal_dtype, sizeof( curr_param->internal_dtype ) );
        curr->inputs[RNNCELL_INPUT_INPUT] = reshape_output;
        curr->inputs[RNNCELL_INPUT_H_STATE] = last_step_h_state;

        curr->inputs[RNNCELL_INPUT_WEIGHT_I] = inputs[RNN_INPUT_WEIGHT_I];
        curr->inputs[RNNCELL_INPUT_WEIGHT_H] = inputs[RNN_INPUT_WEIGHT_H];

        curr->inputs[RNNCELL_INPUT_BIAS] = inputs[RNN_INPUT_BIAS];

        curr->outputs[RNNCELL_OUTPUT_OUTPUT] = rnncell_out0;
        curr->outputs[RNNCELL_OUTPUT_H_STATE] = rnncell_out1;

        vsi_nn_internal_setup_node( self, curr );

        last_step_h_state = rnncell_out1;

        /* reshape output to 3-dims */
        output_tensor = vsi_nn_rnn_reshape_cell_output(self,
            rnncell_out0, batch_size, use_virtual_tensor);
        rnncell_reshape_output_tensors[i] = output_tensor->t;
    }

    tensor = outputs[RNN_OUTPUT_OUTPUT];
    if( !curr_param->time_major )
    {
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[RNN_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

        tensor = output_tensor->t;
    }

    /* concat rnncell output, the rnn's output is 3-dims */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONCAT, time_step, 1 );
    curr->node->nn_param.concat.axis = 2;
    for( i = 0; i < time_step; i++ )
    {
        curr->inputs[i] = rnncell_reshape_output_tensors[i];
    }
    curr->outputs[0] = tensor;
    vsi_nn_internal_setup_node( self, curr );

    if( !curr_param->time_major )
    {
        /* transpose time_major to batch_major*/
        vsi_nn_rnn_transpose_time_major(self,
            tensor, outputs[RNN_OUTPUT_OUTPUT], use_virtual_tensor);
    }

    vsi_nn_safe_free( split_output_tensors );
    vsi_nn_safe_free( rnncell_reshape_output_tensors );

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

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ UNIDIRECTIONAL_SEQUENCE_RNN,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ RNN_INPUT_CNT,
    /* output_num */ RNN_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif
