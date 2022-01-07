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
#include "ops/vsi_nn_op_gru.h"

typedef struct _vsi_nn_gru_local
{
    void * placeholder;
} vsi_nn_gru_local;

static void create_state_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_size_t batch_size,
    vsi_size_t hidden_size
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t * tensor = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    if(NULL == outputs[GRU_OUT_H_STATE])
    {
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(attr.size[0]));
        attr.dim_num = VSI_NN_DIM_AUTO;
        memcpy( &attr.dtype, &outputs[GRU_OUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = TRUE;
        attr.is_const = FALSE;
        tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        outputs[GRU_OUT_H_STATE] = tensor->t;
    }

    if(NULL == inputs[GRU_IN_H_STATE])
    {
        attr.dim_num = 2;
        attr.size[0] = hidden_size;
        attr.size[1] = batch_size;
        memcpy(&attr.dtype, &outputs[GRU_OUT_H_STATE]->attr.dtype, sizeof( attr.dtype ));
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        inputs[GRU_IN_H_STATE] = tensor->t;
    }

} /* create_state_tensor() */

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_gru_param * p = &self->nn_param.gru;
    vsi_size_t batch_size = 0, hidden_size = 0, timesetp = 0;

    hidden_size = p->num_units;
    if(p->time_major)
    {
        /* [input_size, batch_size, timestep] */
        batch_size = inputs[GRU_IN_INPUT]->attr.size[1];
        timesetp = inputs[GRU_IN_INPUT]->attr.size[2];
    }
    else
    {
        /* [input_size, timestep, batch_size] */
        batch_size = inputs[GRU_IN_INPUT]->attr.size[2];
        timesetp = inputs[GRU_IN_INPUT]->attr.size[1];
    }

    /* setup grucell output tensors' shape */
    /* output */
    if(VSI_NN_DIM_AUTO == outputs[GRU_OUT_OUTPUT]->attr.dim_num)
    {
        outputs[GRU_OUT_OUTPUT]->attr.size[0] = hidden_size;
        if(p->return_sequences)
        {
            outputs[GRU_OUT_OUTPUT]->attr.dim_num = 3;
            if(p->time_major)
            {
                outputs[GRU_OUT_OUTPUT]->attr.size[1] = batch_size;
                outputs[GRU_OUT_OUTPUT]->attr.size[2] = timesetp;
            }
            else
            {
                outputs[GRU_OUT_OUTPUT]->attr.size[2] = batch_size;
                outputs[GRU_OUT_OUTPUT]->attr.size[1] = timesetp;
            }
        }
        else
        {
            outputs[GRU_OUT_OUTPUT]->attr.dim_num = 2;
            outputs[GRU_OUT_OUTPUT]->attr.size[1] = batch_size;
        }

    }

    /* create hstate input/output if app doesn't provide them */
    create_state_tensor(self, inputs, outputs, batch_size, hidden_size);

    /* hstate output */
    if(VSI_NN_DIM_AUTO == outputs[GRU_OUT_H_STATE]->attr.dim_num)
    {
        outputs[GRU_OUT_H_STATE]->attr.dim_num = 2;
        outputs[GRU_OUT_H_STATE]->attr.size[0] = hidden_size;
        outputs[GRU_OUT_H_STATE]->attr.size[1] = batch_size;
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
    vsi_size_t i = 0, timestep = 0, batch_size = 0;
    vsi_bool use_virtual_tensor = TRUE;
    vsi_nn_gru_param * p = &self->nn_param.gru;
    vsi_nn_internal_node_t * curr = NULL;
    vsi_nn_tensor_t * input_tensor = NULL, * output_tensor = NULL;
    vsi_nn_tensor_t * step_h_state = NULL;
    vsi_nn_tensor_t ** split_outputs = NULL;
    vsi_nn_tensor_t ** gru_step_outputs = NULL;
    vsi_nn_internal_tensor_t * tmp_tensor = NULL;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(attr));
    vsi_nn_internal_init_node_wksp( self );

    if(p->time_major) /* [input_size, batch, timestep] */
    {
        timestep = inputs[GRU_IN_INPUT]->attr.size[2];
        batch_size = inputs[GRU_IN_INPUT]->attr.size[1];
    }
    else /* [input_size, timestep, batch] */
    {
        timestep = inputs[GRU_IN_INPUT]->attr.size[1];
        batch_size = inputs[GRU_IN_INPUT]->attr.size[2];
    }

    /* compute output shapes and initial the state tensor if needed */
    setup_op_shapes(self, inputs, outputs);

    input_tensor = inputs[GRU_IN_INPUT];
    if(FALSE == p->time_major)
    {
        /* transpose to time_major */
        tmp_tensor = vsi_nn_rnn_transpose_time_major(self,
            inputs[GRU_INPUT_INPUT], NULL, use_virtual_tensor);
        input_tensor = tmp_tensor->t;
    }

    split_outputs = (vsi_nn_tensor_t **)malloc(timestep * sizeof(vsi_nn_tensor_t *));
    memset(split_outputs, 0, timestep * sizeof(vsi_nn_tensor_t *));
    gru_step_outputs = (vsi_nn_tensor_t **)malloc(timestep * sizeof(vsi_nn_tensor_t *));
    memset(gru_step_outputs, 0, timestep * sizeof(vsi_nn_tensor_t *));

    vsi_nn_rnn_split_input_tensor(self, input_tensor, split_outputs, (uint32_t)timestep, use_virtual_tensor);

    //vsi_nn_rnn_data_check_aligned(self, split_outputs, timestep, use_virtual_tensor); ??

    step_h_state = inputs[GRU_IN_H_STATE];
    for(i = 0; i < timestep; i++)
    {
        vsi_nn_tensor_t * reshape_output = NULL;
        vsi_nn_tensor_t * cell_out0 = NULL;
        vsi_nn_tensor_t * cell_out1 = NULL;

        /* reshape split_outputs to cell_input */
        tmp_tensor = vsi_nn_rnn_reshape_split_output(
            self, split_outputs[i], (uint32_t)batch_size, use_virtual_tensor);
        reshape_output = tmp_tensor->t;

        /* grucell output */
        if ( (i == timestep - 1) && p->return_sequences == FALSE )
        {
            cell_out0 = outputs[GRU_OUT_OUTPUT];
        }
        else
        {
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[GRU_OUT_OUTPUT]->attr.dtype, use_virtual_tensor);
            tmp_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            cell_out0 = tmp_tensor->t;
        }

        /* grucell output h_state */
        if( i != timestep - 1 )
        {
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[GRU_OUTPUT_H_STATE]->attr.dtype, use_virtual_tensor);
            tmp_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            cell_out1 = tmp_tensor->t;
        }
        else
        {
            cell_out1 = outputs[GRU_OUT_H_STATE];
        }

        /* create a grucell */
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_GRUCELL, 0, 0 );
        curr->node->nn_param.grucell.num_units = p->num_units;
        curr->node->nn_param.grucell.activation = p->activation;
        curr->node->nn_param.grucell.recurrent_activation = p->recurrent_activation;
        curr->node->nn_param.grucell.reset_after = p->reset_after;
        curr->inputs[GRUCELL_IN_INPUT] = reshape_output;
        curr->inputs[GRUCELL_IN_H_STATE] = step_h_state;
        curr->inputs[GRUCELL_IN_KERNEL_I2Z] = inputs[GRU_IN_KERNEL_I2Z];
        curr->inputs[GRUCELL_IN_KERNEL_I2R] = inputs[GRU_IN_KERNEL_I2R];
        curr->inputs[GRUCELL_IN_KERNEL_I2H] = inputs[GRU_IN_KERNEL_I2H];
        curr->inputs[GRUCELL_IN_KERNEL_R2Z] = inputs[GRU_IN_KERNEL_R2Z];
        curr->inputs[GRUCELL_IN_KERNEL_R2R] = inputs[GRU_IN_KERNEL_R2R];
        curr->inputs[GRUCELL_IN_KERNEL_R2H] = inputs[GRU_IN_KERNEL_R2H];
        curr->inputs[GRUCELL_IN_BIAS_I2Z] = inputs[GRU_IN_BIAS_I2Z];
        curr->inputs[GRUCELL_IN_BIAS_I2R] = inputs[GRU_IN_BIAS_I2R];
        curr->inputs[GRUCELL_IN_BIAS_I2H] = inputs[GRU_IN_BIAS_I2H];
        curr->inputs[GRUCELL_IN_BIAS_R2Z] = inputs[GRU_IN_BIAS_R2Z];
        curr->inputs[GRUCELL_IN_BIAS_R2R] = inputs[GRU_IN_BIAS_R2R];
        curr->inputs[GRUCELL_IN_BIAS_R2H] = inputs[GRU_IN_BIAS_R2H];
        curr->outputs[GRUCELL_OUT_OUTPUT] = cell_out0;
        curr->outputs[GRUCELL_OUT_H_STATE] = cell_out1;
        vsi_nn_internal_setup_node( self, curr );

        step_h_state = cell_out1;

        if(p->return_sequences)
        {
            /* reshape every step output to 3-dims for GRU_OUTPUT */
            tmp_tensor = vsi_nn_rnn_reshape_cell_output(self,
                cell_out0, (uint32_t)batch_size, use_virtual_tensor);
            gru_step_outputs[i] = tmp_tensor->t;
        }
    } /* for(i = 0; i < timestep; i++) end */

    if(p->return_sequences)
    {
        output_tensor = outputs[GRU_OUTPUT_OUTPUT];
        if(p->time_major == FALSE)
        {
            /* create a new tensor for permute */
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
            tmp_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            output_tensor = tmp_tensor->t;
        }

        /* concat all grucell output0, the reshaped grucell output shape: [hidden_size, batch, 1] */
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONCAT, timestep, 1 );
        curr->node->nn_param.concat.axis = 2; /* concat the cell_outs in timestep */
        for( i = 0; i < timestep; i++ )
        {
            curr->inputs[i] = gru_step_outputs[i];
        }
        curr->outputs[0] = output_tensor;
        vsi_nn_internal_setup_node( self, curr );

        if(p->time_major == FALSE)
        {
            /* transpose time_major to batch_major */
            vsi_nn_rnn_transpose_time_major(self,
                output_tensor, outputs[GRU_OUTPUT_OUTPUT], use_virtual_tensor);
        }
    }

    vsi_nn_safe_free( split_outputs );
    vsi_nn_safe_free( gru_step_outputs );

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


#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GRU,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ GRU_IN_CNT,
    /* output_num */ GRU_OUT_CNT
    );
#ifdef __cplusplus
}
#endif
