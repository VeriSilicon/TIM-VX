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
#include "vsi_nn_error.h"

typedef struct _gru_ovxlib_local_data_t {
    vsi_nn_tensor_t* weights_input;
    vsi_nn_tensor_t* weights_recurrent;
    vsi_nn_tensor_t* cond_zeros;
    vsi_nn_tensor_t* bias_z;
    vsi_nn_tensor_t* bias_r;
    vsi_nn_tensor_t* bias_z_r;
    vsi_nn_tensor_t* bias_c;
} gru_ovxlib_local_data_t;

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_gru_ovxlib_param* curr_param = &self->nn_param.gru_ovxlib;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_size_t num_units =  0;
    vsi_size_t output_size = 0;
    vsi_size_t batch_size = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    if( curr_param->time_major )
    {
        batch_size = inputs[GRU_INPUT_INPUT]->attr.size[1];
    }
    else
    {
        batch_size = inputs[GRU_INPUT_INPUT]->attr.size[2];
    }

    num_units = inputs[GRU_INPUT_WEIGHT_I2R]->attr.size[1];
    if ( num_units != curr_param->num_units )
    {
        VSILOGE("The num_units not matched(GRU).\n");
        return FALSE;
    }
    output_size = num_units;

    /* create h_state input/output if app doesn't provide them */
    if( !inputs[GRU_INPUT_H_STATE] )
    {
        attr.dim_num = 2;
        attr.size[1] = batch_size;
        attr.size[0] = output_size;
        memcpy( &attr.dtype, &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
        inputs[GRU_INPUT_H_STATE] = output_tensor->t;
    }

    if( !outputs[GRU_OUTPUT_H_STATE] )
    {
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        memcpy( &attr.dtype, &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = TRUE;
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
        outputs[GRU_OUTPUT_H_STATE] = output_tensor->t;
    }

    /* output */
    if( VSI_NN_DIM_AUTO == outputs[GRU_OUTPUT_OUTPUT]->attr.dim_num )
    {
        outputs[GRU_OUTPUT_OUTPUT]->attr.size[0] = output_size;
        if ( curr_param->return_sequences )
        {
            outputs[GRU_OUTPUT_OUTPUT]->attr.size[1] = inputs[GRU_INPUT_INPUT]->attr.size[1];
            outputs[GRU_OUTPUT_OUTPUT]->attr.size[2] = inputs[GRU_INPUT_INPUT]->attr.size[2];
            outputs[GRU_OUTPUT_OUTPUT]->attr.dim_num = 3;
        }
        else
        {
            outputs[GRU_OUTPUT_OUTPUT]->attr.size[1] = batch_size;
            outputs[GRU_OUTPUT_OUTPUT]->attr.dim_num = 2;
        }
    }

    /* output_state_out */
    if( VSI_NN_DIM_AUTO == outputs[GRU_OUTPUT_H_STATE]->attr.dim_num )
    {
        outputs[GRU_OUTPUT_H_STATE]->attr.size[0] = output_size;
        outputs[GRU_OUTPUT_H_STATE]->attr.size[1] = batch_size;
        outputs[GRU_OUTPUT_H_STATE]->attr.dim_num = 2;
    }

    return TRUE;
final:
    return FALSE;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return vsi_nn_internal_compute_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    VSI_UNREFERENCED(self);
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
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
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return vsi_nn_internal_optimize_node( self, direction );
} /* op_optimize() */

static vsi_bool op_setup_default
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_gru_ovxlib_param* curr_param = &self->nn_param.gru_ovxlib;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_tensor_t** split_output_tensors = NULL;
    vsi_nn_tensor_t** grucell_reshape_output_tensors =NULL;
    vsi_nn_tensor_t* last_step_h_state = NULL;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_tensor_t* input_tensor = NULL;
    vsi_bool use_virtual_tensor = TRUE;
    vsi_size_t batch_size = 0;
    vsi_size_t time_step = 0;
    vsi_size_t i = 0;
    vsi_bool ret = FALSE;
    vsi_status status = VSI_FAILURE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_node_wksp( self );

    if( curr_param->time_major )
    {
        batch_size = inputs[GRU_INPUT_INPUT]->attr.size[1];
        time_step = inputs[GRU_INPUT_INPUT]->attr.size[2];
    }
    else
    {
        batch_size = inputs[GRU_INPUT_INPUT]->attr.size[2];
        time_step = inputs[GRU_INPUT_INPUT]->attr.size[1];
    }

    setup_op_shapes( self, inputs, outputs);

    /* default to input */
    input_tensor = inputs[GRU_INPUT_INPUT];
    if( !curr_param->time_major )
    {
        /* transpose to time_major */
        output_tensor = vsi_nn_rnn_transpose_time_major(self,
            inputs[GRU_INPUT_INPUT], NULL, use_virtual_tensor);
        CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
        input_tensor = output_tensor->t;
    }

    /* split input tensor */
    split_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
    CHECK_PTR_FAIL_GOTO( split_output_tensors, "Create buffer fail.", final );
    memset( split_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));
    grucell_reshape_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
    CHECK_PTR_FAIL_GOTO( grucell_reshape_output_tensors, "Create buffer fail.", final );
    memset( grucell_reshape_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));

    status = vsi_nn_rnn_split_input_tensor(self, input_tensor, split_output_tensors,
        (uint32_t)time_step, use_virtual_tensor);
    CHECK_STATUS_FAIL_GOTO(status, final);

    status = vsi_nn_rnn_data_check_aligned(self, split_output_tensors, (uint32_t)time_step, use_virtual_tensor);
    CHECK_STATUS_FAIL_GOTO(status, final);

    last_step_h_state = inputs[GRU_INPUT_H_STATE];
    for( i = 0; i < time_step; i++ )
    {
        vsi_nn_tensor_t* reshape_output = NULL;
        vsi_nn_tensor_t* grucell_out0 = NULL;
        vsi_nn_tensor_t* grucell_out1 = NULL;

        /* reshape for split output */
        output_tensor = vsi_nn_rnn_reshape_split_output(self,
            split_output_tensors[i], (uint32_t)batch_size, use_virtual_tensor);
        CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
        reshape_output = output_tensor->t;

        /* grucell output */
        if ( (i == time_step - 1) && !curr_param->return_sequences )
        {
            grucell_out0 = outputs[GRU_OUTPUT_OUTPUT];
        }
        else
        {
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
            output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
            grucell_out0 = output_tensor->t;
        }

        if( i != time_step - 1 )
        {
            /* grucell output h_state */
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[GRU_OUTPUT_H_STATE]->attr.dtype, use_virtual_tensor);
            output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
            grucell_out1 = output_tensor->t;
        }
        else
        {
            grucell_out1 = outputs[GRU_OUTPUT_H_STATE];
        }

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_GRUCELL_OVXLIB, 0, 0 );
        CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
        curr->node->nn_param.grucell_ovxlib.num_units = curr_param->num_units;
        curr->node->nn_param.grucell_ovxlib.activation = curr_param->activation;
        curr->node->nn_param.grucell_ovxlib.recurrent_activation = curr_param->recurrent_activation;
        curr->node->nn_param.grucell_ovxlib.linear_before_reset = curr_param->linear_before_reset;
        if ( reshape_output->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 )
        {
            size_t k = 0;
            for (k = 0; k < _cnt_of_array( curr_param->internal_dtype ); k++)
            {
                if (curr_param->internal_dtype[k].vx_type == VSI_NN_TYPE_NONE)
                {
                    curr_param->internal_dtype[k].vx_type = VSI_NN_TYPE_BFLOAT16;
                }
            }
        }
        memcpy( curr->node->nn_param.grucell_ovxlib.internal_dtype,
            curr_param->internal_dtype, sizeof( curr_param->internal_dtype ) );
        curr->node->nn_param.grucell_ovxlib.use_cudnn_implementation = curr_param->use_cudnn_implementation;
        curr->node->nn_param.grucell_ovxlib.cudnn_implementation_version = curr_param->cudnn_implementation_version;
        curr->inputs[GRUCELL_INPUT_INPUT] = reshape_output;
        curr->inputs[GRUCELL_INPUT_H_STATE] = last_step_h_state;

        curr->inputs[GRUCELL_INPUT_WEIGHT_I2R] = inputs[GRU_INPUT_WEIGHT_I2R];
        curr->inputs[GRUCELL_INPUT_WEIGHT_I2Z] = inputs[GRU_INPUT_WEIGHT_I2Z];
        curr->inputs[GRUCELL_INPUT_WEIGHT_H2R] = inputs[GRU_INPUT_WEIGHT_H2R];
        curr->inputs[GRUCELL_INPUT_WEIGHT_H2Z] = inputs[GRU_INPUT_WEIGHT_H2Z];

        curr->inputs[GRUCELL_INPUT_BIAS_I2R] = inputs[GRU_INPUT_BIAS_I2R];
        curr->inputs[GRUCELL_INPUT_BIAS_I2Z] = inputs[GRU_INPUT_BIAS_I2Z];

        curr->inputs[GRUCELL_INPUT_BIAS_H2R] = inputs[GRU_INPUT_BIAS_H2R];
        curr->inputs[GRUCELL_INPUT_BIAS_H2Z] = inputs[GRU_INPUT_BIAS_H2Z];

        curr->inputs[GRUCELL_INPUT_WEIGHT_I2C] = inputs[GRU_INPUT_WEIGHT_I2C];
        curr->inputs[GRUCELL_INPUT_WEIGHT_H2C] = inputs[GRU_INPUT_WEIGHT_H2C];

        curr->inputs[GRUCELL_INPUT_BIAS_I2C] = inputs[GRU_INPUT_BIAS_I2C];
        curr->inputs[GRUCELL_INPUT_BIAS_H2C] = inputs[GRU_INPUT_BIAS_H2C];

        curr->outputs[GRUCELL_OUTPUT_OUTPUT] = grucell_out0;
        curr->outputs[GRUCELL_OUTPUT_H_STATE] = grucell_out1;

        ret = vsi_nn_internal_setup_node( self, curr );

        last_step_h_state = grucell_out1;

        if ( curr_param->return_sequences )
        {
            /* reshape output to 3-dims */
            output_tensor = vsi_nn_rnn_reshape_cell_output(self,
                grucell_out0, (uint32_t)batch_size, use_virtual_tensor);
            CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
            grucell_reshape_output_tensors[i] = output_tensor->t;
        }
    }

    if ( curr_param->return_sequences )
    {
        tensor = outputs[GRU_OUTPUT_OUTPUT];
        if( !curr_param->time_major )
        {
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
            output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
            CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);

            tensor = output_tensor->t;
        }

        /* concat grucell output, the gru's output is 3-dims */
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONCAT, (uint32_t)time_step, 1 );
        CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
        curr->node->nn_param.concat.axis = 2;
        for( i = 0; i < time_step; i++ )
        {
            curr->inputs[i] = grucell_reshape_output_tensors[i];
        }
        curr->outputs[0] = tensor;
        vsi_nn_internal_setup_node( self, curr );

        if( !curr_param->time_major )
        {
            /* transpose time_major to batch_major*/
            vsi_nn_rnn_transpose_time_major(self,
                tensor, outputs[GRU_OUTPUT_OUTPUT], use_virtual_tensor);
        }
    }

final:
    vsi_nn_safe_free( split_output_tensors );
    vsi_nn_safe_free( grucell_reshape_output_tensors );

    return ret;
} /* op_setup_default() */

static vsi_bool op_setup_optimized
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_gru_ovxlib_param* p = &self->nn_param.gru_ovxlib;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_internal_tensor_t* tmp_tensor = NULL;
    vsi_nn_tensor_t** split_output_tensors = NULL;
    vsi_nn_tensor_t** grucell_reshape_output_tensors =NULL;
    vsi_nn_tensor_t* last_step_h_state = NULL;
    vsi_nn_tensor_t* input_tensor = NULL;
    vsi_bool use_virtual_tensor = TRUE;
    vsi_size_t batch_size = 0;
    vsi_size_t time_step = 0;
    vsi_size_t unit_nums = 0;
    vsi_size_t i = 0;
    grucell_activation_input_layout_e grucell_activation_input_layout = GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_CN;
    vsi_nn_internal_tensor_t* recurrent_weight_for_nn = NULL;
    vsi_nn_internal_tensor_t* input_weight_for_nn = NULL;
    vsi_size_t permute_in_perm[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t reshape_size[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_bool ret = FALSE;
    vsi_status status = VSI_FAILURE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_node_wksp( self );

    if( p->time_major )
    {
        batch_size = inputs[GRU_INPUT_INPUT]->attr.size[1];
        time_step = inputs[GRU_INPUT_INPUT]->attr.size[2];
    }
    else
    {
        batch_size = inputs[GRU_INPUT_INPUT]->attr.size[2];
        time_step = inputs[GRU_INPUT_INPUT]->attr.size[1];
    }

    setup_op_shapes( self, inputs, outputs);

    unit_nums = inputs[GRU_INPUT_WEIGHT_H2R]->attr.size[1];

    /* default to input */
    input_tensor = inputs[GRU_INPUT_INPUT];
    if( !p->time_major )
    {
        /* transpose to time_major */
        output_tensor = vsi_nn_rnn_transpose_time_major(self,
            inputs[GRU_INPUT_INPUT], NULL, use_virtual_tensor);
        CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
        input_tensor = output_tensor->t;
    }

    /* input FC */
    p->local->weights_input = vsi_nn_ConcatTensor(self->graph, 1, inputs[GRU_INPUT_WEIGHT_I2R],
            inputs[GRU_INPUT_WEIGHT_I2Z], inputs[GRU_INPUT_WEIGHT_I2C]);
    CHECK_PTR_FAIL_GOTO(p->local->weights_input, "Create tensor failed", final);
    p->local->weights_input->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->weights_input, VSI_NN_TENSOR_ATTR_CONST);

    p->local->weights_recurrent = vsi_nn_ConcatTensor(self->graph, 1, inputs[GRU_INPUT_WEIGHT_H2R],
                inputs[GRU_INPUT_WEIGHT_H2Z], inputs[GRU_INPUT_WEIGHT_H2C]);
    CHECK_PTR_FAIL_GOTO(p->local->weights_recurrent, "Create tensor failed", final);
    p->local->weights_recurrent->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->weights_recurrent, VSI_NN_TENSOR_ATTR_CONST);

    p->local->bias_r = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2R]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2R], inputs[GRUCELL_INPUT_BIAS_H2R]);
    CHECK_PTR_FAIL_GOTO(p->local->bias_r, "Create tensor failed", final);
    p->local->bias_r->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->bias_r, VSI_NN_TENSOR_ATTR_CONST);

    p->local->bias_z = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2Z]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2Z], inputs[GRUCELL_INPUT_BIAS_H2Z]);
    CHECK_PTR_FAIL_GOTO(p->local->bias_z, "Create tensor failed", final);
    p->local->bias_z->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->bias_z, VSI_NN_TENSOR_ATTR_CONST);

    p->local->bias_c = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2C]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2C], inputs[GRUCELL_INPUT_BIAS_H2C]);
    CHECK_PTR_FAIL_GOTO(p->local->bias_c, "Create tensor failed", final);
    p->local->bias_c->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->bias_c, VSI_NN_TENSOR_ATTR_CONST);

    /* prepare weight and bias for recurrent fc */
    recurrent_weight_for_nn = vsi_nn_rnn_prepare_weight_for_nn_fc(self, p->local->weights_recurrent, 1, 1);
    CHECK_PTR_FAIL_GOTO(recurrent_weight_for_nn, "Create internal tensor failed", final);

    /* transpose input from [T,B,D] to [D,T,B] */
    permute_in_perm[0] = 1;
    permute_in_perm[1] = 2;
    permute_in_perm[2] = 0;
    tmp_tensor = vsi_nn_rnn_create_permute(self, input_tensor, NULL, permute_in_perm, 3, use_virtual_tensor);
    CHECK_PTR_FAIL_GOTO( tmp_tensor, "Create internal tensor fail.", final );

    reshape_size[0] = tmp_tensor->t->attr.size[0];
    reshape_size[1] = tmp_tensor->t->attr.size[1];
    reshape_size[2] = tmp_tensor->t->attr.size[2];
    reshape_size[3] = 1; /* new batch dim */
    tmp_tensor = vsi_nn_rnn_create_reshape(self, tmp_tensor->t, NULL, reshape_size, 4, use_virtual_tensor);
    CHECK_PTR_FAIL_GOTO(tmp_tensor, "Create internal tensor failed", final);

    input_weight_for_nn = vsi_nn_rnn_prepare_weight_for_nn_fc(self, p->local->weights_input, 1, 1);
    CHECK_PTR_FAIL_GOTO(input_weight_for_nn, "Create internal tensor failed", final);

    vsi_nn_internal_init_tensor_attr(&attr, &p->internal_dtype[GRUCELL_CUDNN_QUANTIZE_PARAM_INPUT],
        use_virtual_tensor);
    output_tensor = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
    CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);

    curr = vsi_nn_internal_new_node(self, VSI_NN_OP_CONV2D, 0, 0 );
    CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
    curr->node->nn_param.conv2d.ksize[0] = 1;
    curr->node->nn_param.conv2d.ksize[1] = 1;
    curr->node->nn_param.conv2d.stride[0] = 1;
    curr->node->nn_param.conv2d.stride[1] = 1;
    curr->node->nn_param.conv2d.pad[0] = 0;
    curr->node->nn_param.conv2d.pad[1] = 0;
    curr->node->nn_param.conv2d.pad[2] = 0;
    curr->node->nn_param.conv2d.pad[3] = 0;
    curr->node->nn_param.conv2d.group = 1;
    curr->node->nn_param.conv2d.dilation[0] = 1;
    curr->node->nn_param.conv2d.dilation[1] = 1;
    curr->node->nn_param.conv2d.weights = (uint32_t)(input_weight_for_nn->t->attr.size[3]);

    curr->inputs[0] = tmp_tensor->t;
    curr->inputs[1] = input_weight_for_nn->t;
    curr->inputs[2] = NULL;
    curr->outputs[0] = output_tensor->t;
    vsi_nn_internal_setup_node(self, curr);

    reshape_size[0] = output_tensor->t->attr.size[0];
    reshape_size[1] = output_tensor->t->attr.size[1];
    reshape_size[2] = output_tensor->t->attr.size[2];
    output_tensor = vsi_nn_rnn_create_reshape(self, output_tensor->t, NULL, reshape_size, 3, use_virtual_tensor);
    CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);

    permute_in_perm[0] = 0;
    permute_in_perm[1] = 2;
    permute_in_perm[2] = 1;
    tmp_tensor = vsi_nn_rnn_create_permute(self, output_tensor->t, NULL, permute_in_perm, 3, use_virtual_tensor);
    CHECK_PTR_FAIL_GOTO(tmp_tensor, "Create internal tensor failed", final);

    /* split input tensor */
    split_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
    CHECK_PTR_FAIL_GOTO( split_output_tensors, "Create buffer fail.", final );
    memset( split_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));
    grucell_reshape_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
    CHECK_PTR_FAIL_GOTO( grucell_reshape_output_tensors, "Create buffer fail.", final );
    memset( grucell_reshape_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));

    status = vsi_nn_rnn_split_input_tensor(self, tmp_tensor->t, split_output_tensors,
        (uint32_t)time_step, use_virtual_tensor);
    CHECK_STATUS_FAIL_GOTO(status, final);

    status = vsi_nn_rnn_data_check_aligned(self, split_output_tensors, (uint32_t)time_step, use_virtual_tensor);
    CHECK_STATUS_FAIL_GOTO(status, final);

    memcpy(&attr, &p->local->bias_r->attr, sizeof(vsi_nn_tensor_attr_t));
    attr.size[1] = 1;
    attr.dim_num = 2;
    p->local->cond_zeros = vsi_nn_CreateTensorWithDefault(self->graph, &attr, 0.0);
    CHECK_PTR_FAIL_GOTO(p->local->cond_zeros, "Create tensor failed", final);

    last_step_h_state = inputs[GRU_INPUT_H_STATE];
    permute_in_perm[0] = 1;
    permute_in_perm[1] = 0;
    tmp_tensor = vsi_nn_rnn_create_permute(self, last_step_h_state, NULL, permute_in_perm, 2, use_virtual_tensor);
    CHECK_PTR_FAIL_GOTO(tmp_tensor, "Create internal tensor failed", final);
    last_step_h_state = tmp_tensor->t;

    for( i = 0; i < time_step; i++ )
    {
        vsi_nn_tensor_t* input_fc_output = NULL;
        vsi_nn_tensor_t* recurrent_fc_output = NULL;
        vsi_nn_tensor_t* grucell_out0 = NULL;
        vsi_nn_tensor_t* grucell_out1 = NULL;
        vsi_nn_internal_tensor_t* tmp = NULL;
        vsi_nn_internal_tensor_t** splited_input_fc_output_tensors = NULL;
        vsi_nn_internal_tensor_t** splited_recurrent_fc_output_tensors = NULL;

        /* reshape for split output */
        output_tensor = vsi_nn_rnn_reshape_split_output(self,
            split_output_tensors[i], (uint32_t)(unit_nums * 3), use_virtual_tensor);
        CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
        input_fc_output = output_tensor->t;

        /* last_step_h_state is not batch first, no need to permute */
        reshape_size[3] = 1;
        reshape_size[2] = last_step_h_state->attr.size[1] / (1/*kernel_h*/ * 1/*kernel_w*/);
        reshape_size[1] = 1/*kernel_h*/;
        reshape_size[0] = last_step_h_state->attr.size[0];
        tmp = vsi_nn_rnn_create_reshape(self, last_step_h_state, NULL, reshape_size, 4, use_virtual_tensor);
        CHECK_PTR_FAIL_GOTO(tmp, "Create internal tensor failed", final);

        vsi_nn_internal_init_tensor_attr(&attr,
            &p->internal_dtype[GRUCELL_CUDNN_QUANTIZE_PARAM_HIDDEN],
            use_virtual_tensor);
        tmp_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        CHECK_PTR_FAIL_GOTO(tmp_tensor, "Create internal tensor failed", final);

        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_CONV2D, 0, 0 );
        CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
        curr->node->nn_param.conv2d.ksize[0] = 1;
        curr->node->nn_param.conv2d.ksize[1] = 1;
        curr->node->nn_param.conv2d.stride[0] = 1;
        curr->node->nn_param.conv2d.stride[1] = 1;
        curr->node->nn_param.conv2d.pad[0] = 0;
        curr->node->nn_param.conv2d.pad[1] = 0;
        curr->node->nn_param.conv2d.pad[2] = 0;
        curr->node->nn_param.conv2d.pad[3] = 0;
        curr->node->nn_param.conv2d.group = 1;
        curr->node->nn_param.conv2d.dilation[0] = 1;
        curr->node->nn_param.conv2d.dilation[1] = 1;
        curr->node->nn_param.conv2d.weights = (uint32_t)recurrent_weight_for_nn->t->attr.size[3];

        curr->inputs[0] = tmp->t;
        curr->inputs[1] = recurrent_weight_for_nn->t;
        curr->inputs[2] = NULL;
        curr->outputs[0] = tmp_tensor->t;
        vsi_nn_internal_setup_node(self, curr);

        reshape_size[1] = recurrent_weight_for_nn->t->attr.size[3];
        reshape_size[0] = batch_size;
        tmp_tensor = vsi_nn_rnn_create_reshape(self, tmp_tensor->t, NULL, reshape_size, 2, use_virtual_tensor);
        CHECK_PTR_FAIL_GOTO(tmp_tensor, "Create internal tensor failed", final);
        recurrent_fc_output = tmp_tensor->t;

        /* grucell output */
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
        grucell_out0 = output_tensor->t;

        /* grucell output h_state */
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[GRU_OUTPUT_H_STATE]->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        CHECK_PTR_FAIL_GOTO(output_tensor, "Create internal tensor failed", final);
        grucell_out1 = output_tensor->t;

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_GRUCELL_ACTIVATION_INTERNAL, 0, 0 );
        CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
        curr->inputs[GRUCELL_ACTIVATION_INPUT_H_STATE] = last_step_h_state;
        {
            splited_input_fc_output_tensors = vsi_nn_create_split(self,
                input_fc_output, 1, 3, NULL, use_virtual_tensor);
            CHECK_PTR_FAIL_GOTO_RLS_INTERNAL_NODE(splited_input_fc_output_tensors, curr,
                "Create internal tensor failed", final);
            splited_recurrent_fc_output_tensors = vsi_nn_create_split(self,
                recurrent_fc_output, 1, 3, NULL, use_virtual_tensor);
            CHECK_PTR_FAIL_GOTO_RLS_INTERNAL_NODE(splited_recurrent_fc_output_tensors, curr,
                "Create internal tensor failed", final);
            curr->inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R] = splited_input_fc_output_tensors[0]->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_Z] = splited_input_fc_output_tensors[1]->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_C] = splited_input_fc_output_tensors[2]->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_R] = splited_recurrent_fc_output_tensors[0]->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_Z] = splited_recurrent_fc_output_tensors[1]->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_C] = splited_recurrent_fc_output_tensors[2]->t;
        }
        curr->inputs[GRUCELL_ACTIVATION_INPUT_BIAS_R] = p->local->bias_r;
        curr->inputs[GRUCELL_ACTIVATION_INPUT_BIAS_Z] = p->local->bias_z;
        curr->inputs[GRUCELL_ACTIVATION_INPUT_BIAS_C] = p->local->bias_c;
        curr->inputs[GRUCELL_ACTIVATION_INPUT_COND_R] = p->local->cond_zeros;
        curr->inputs[GRUCELL_ACTIVATION_INPUT_COND_Z] = p->local->cond_zeros;
        curr->inputs[GRUCELL_ACTIVATION_INPUT_COND_C] = p->local->cond_zeros;
        curr->outputs[0] = grucell_out0;
        curr->outputs[1] = grucell_out1;
        curr->node->nn_param.grucell_activation_internal.input_category = GRUCELL_INPUT_CATEGORY_CUDNN;
        curr->node->nn_param.grucell_activation_internal.use_cudnn_implementation = TRUE;
        curr->node->nn_param.grucell_activation_internal.input_layout = grucell_activation_input_layout;
        vsi_nn_internal_setup_node(self, curr);

        last_step_h_state = grucell_out0;

        /* reshape output to 3-dims */
        grucell_reshape_output_tensors[i] = grucell_out0;
    }

    /* concat grucell output, the gru's output is 3-dims */
    vsi_nn_internal_init_tensor_attr(&attr,
        &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, use_virtual_tensor);
    tmp_tensor = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
    CHECK_PTR_FAIL_GOTO(tmp_tensor, "Create internal tensor failed", final);

    curr = vsi_nn_internal_new_node(self, VSI_NN_OP_CONCAT, (uint32_t)time_step, 1);
    CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
    curr->node->nn_param.concat.axis = 1;
    for( i = 0; i < time_step; i++ )
    {
        curr->inputs[i] = grucell_reshape_output_tensors[i];
    }
    curr->outputs[0] = tmp_tensor->t;
    vsi_nn_internal_setup_node(self, curr);

    reshape_size[0] = batch_size;
    reshape_size[1] = (vsi_size_t)-1;
    reshape_size[2] = time_step;
    tmp_tensor = vsi_nn_rnn_create_reshape(self, tmp_tensor->t, NULL, reshape_size, 3, use_virtual_tensor);
    CHECK_PTR_FAIL_GOTO(tmp_tensor, "Create internal tensor failed", final);

    if(p->time_major)
    {
        permute_in_perm[0] = 1;
        permute_in_perm[1] = 0;
        permute_in_perm[2] = 2;
    }
    else
    {
        permute_in_perm[0] = 1;
        permute_in_perm[1] = 2;
        permute_in_perm[2] = 0;
    }
    vsi_nn_rnn_create_permute(self, tmp_tensor->t, outputs[GRU_OUTPUT_OUTPUT], permute_in_perm, 3, use_virtual_tensor);

    permute_in_perm[0] = 1;
    permute_in_perm[1] = 0;
    vsi_nn_rnn_create_permute(self, last_step_h_state, outputs[GRU_OUTPUT_H_STATE],
        permute_in_perm, 2, use_virtual_tensor);

    ret = TRUE;
final:
    vsi_nn_safe_free( split_output_tensors );
    vsi_nn_safe_free( grucell_reshape_output_tensors );

    return ret;
} /* op_setup_optimized() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if(self->nn_param.gru_ovxlib.use_cudnn_implementation)
    {
        return op_setup_optimized(self, inputs, outputs);
    }
    else
    {
        return op_setup_default(self, inputs, outputs);
    }
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_internal_deinit_node_wksp( self );

    vsi_safe_release_tensor(self->nn_param.gru_ovxlib.local->weights_input);
    vsi_safe_release_tensor(self->nn_param.gru_ovxlib.local->weights_recurrent);
    vsi_safe_release_tensor(self->nn_param.gru_ovxlib.local->cond_zeros);

    vsi_nn_safe_free(self->nn_param.gru_ovxlib.local);

    return status;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.gru_ovxlib.local = (gru_ovxlib_local_data_t *)malloc(sizeof(gru_ovxlib_local_data_t));
    memset(self->nn_param.gru_ovxlib.local, 0x00, sizeof(gru_ovxlib_local_data_t));
    self->nn_param.gru_ovxlib.time_major = TRUE;
    self->nn_param.gru_ovxlib.activation = VSI_NN_ACT_TANH;
    self->nn_param.gru_ovxlib.recurrent_activation = VSI_NN_ACT_SIGMOID;
    self->nn_param.gru_ovxlib.return_sequences = TRUE;
    self->nn_param.gru_ovxlib.linear_before_reset = 0;
    self->nn_param.gru_ovxlib.cudnn_implementation_version = 0;
    self->nn_param.gru_ovxlib.use_cudnn_implementation = FALSE;

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GRU_OVXLIB,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ GRU_INPUT_CNT,
    /* output_num */ GRU_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif
