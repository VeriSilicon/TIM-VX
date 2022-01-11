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
#include <stdarg.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "ops/vsi_nn_op_grucell_ovxlib.h"
#include "vsi_nn_internal_node.h"
#include "vsi_nn_rnn_helper.h"
#include "utils/vsi_nn_tensor_op.h"
#include "utils/vsi_nn_util.h"

#define USE_GRUCELL_ACTIVATION

typedef struct _grucell_ovxlib_local_data_t
{
    vsi_bool multi_batch;
    vsi_bool force_input_recurrent_on_NN;
    vsi_nn_activation_e gate_activation;
    vsi_nn_activation_e candidate_activation;
    vsi_nn_tensor_t* weights_update;
    vsi_nn_tensor_t* weights_reset;
    vsi_nn_tensor_t* weights_z_r;
    vsi_nn_tensor_t* weights_c;
    vsi_nn_tensor_t* weights_input;
    vsi_nn_tensor_t* weights_recurrent;
    vsi_nn_tensor_t* bias_z;
    vsi_nn_tensor_t* bias_r;
    vsi_nn_tensor_t* bias_z_r;
    vsi_nn_tensor_t* bias_c;
} grucell_ovxlib_local_data_t;

static vsi_nn_internal_tensor_t* create_multiply
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input1,
    vsi_nn_tensor_t * input2,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, output_dtype, use_virtual_tensor);
    tensor1 = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_MULTIPLY, 0, 0 );

    tmp_inode->inputs[0] = input1;
    tmp_inode->inputs[1] = input2;
    tmp_inode->node->nn_param.multiply.scale = 1.0f;
    tmp_inode->node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    tmp_inode->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    tmp_inode->outputs[0] = tensor1->t;
    vsi_nn_internal_setup_node(self, tmp_inode);

    return tensor1;
}

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_grucell_ovxlib_param* curr_param = &self->nn_param.grucell_ovxlib;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_size_t output_size = 0;
    vsi_size_t batch_size = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    batch_size = inputs[GRUCELL_INPUT_INPUT]->attr.size[1];
    output_size = inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.size[1];
    if ( output_size != curr_param->num_units )
    {
        VSILOGE("The num_units not matched(GRUCELL).\n");
        return FALSE;
    }

    /* create h_state input/output if app doesn't provide them */
    if( !inputs[GRUCELL_INPUT_H_STATE] )
    {
        attr.dim_num = 2;
        attr.size[1] = batch_size;
        attr.size[0] = output_size;
        memcpy( &attr.dtype, &outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = FALSE;
        attr.is_const = FALSE;

        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        inputs[GRUCELL_INPUT_H_STATE] = output_tensor->t;
    }

    if( !outputs[GRUCELL_OUTPUT_H_STATE] )
    {
        vsi_nn_internal_init_tensor_attr(&attr,
            &outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dtype, TRUE);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        outputs[GRUCELL_OUTPUT_H_STATE] = output_tensor->t;
    }

    /* setup grucell output tensors' shape */
    /* output */
    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dim_num)
    {
        /* num_units */
        outputs[GRUCELL_OUTPUT_OUTPUT]->attr.size[0] = output_size;
        /* batch_size */
        outputs[GRUCELL_OUTPUT_OUTPUT]->attr.size[1] = inputs[GRUCELL_INPUT_INPUT]->attr.size[1];
        outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dim_num = inputs[GRUCELL_INPUT_INPUT]->attr.dim_num;
    }

    /* output_state_out */
    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_OUTPUT_H_STATE]->attr.dim_num)
    {
        outputs[GRUCELL_OUTPUT_H_STATE]->attr.dim_num = outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dim_num;
        memcpy( outputs[GRUCELL_OUTPUT_H_STATE]->attr.size, outputs[GRUCELL_OUTPUT_OUTPUT]->attr.size,
            VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t) );
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

static vsi_bool op_setup_float
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_grucell_ovxlib_param* p = &self->nn_param.grucell_ovxlib;
    vsi_nn_dtype_t dtype;
    vsi_bool use_virtual_tensor = TRUE;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* tmp_tensor = NULL;
    vsi_nn_internal_tensor_t* tensor_rt = NULL;
    vsi_nn_internal_tensor_t* input_hstate = NULL;
    vsi_nn_internal_tensor_t** splited_tensors = NULL;

    p->local->weights_update = vsi_nn_ConcatTensor(self->graph, 0,
        inputs[GRUCELL_INPUT_WEIGHT_I2Z], inputs[GRUCELL_INPUT_WEIGHT_H2Z]);
    p->local->weights_reset = vsi_nn_ConcatTensor(self->graph, 0,
        inputs[GRUCELL_INPUT_WEIGHT_I2R], inputs[GRUCELL_INPUT_WEIGHT_H2R]);
    p->local->bias_z = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2Z]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2Z], inputs[GRUCELL_INPUT_BIAS_H2Z]);
    p->local->bias_r = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2R]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2R], inputs[GRUCELL_INPUT_BIAS_H2R]);
    p->local->bias_z_r = vsi_nn_ConcatTensor(self->graph, 0, p->local->bias_z, p->local->bias_r);
    p->local->weights_z_r = vsi_nn_ConcatTensor(self->graph, 1, p->local->weights_update, p->local->weights_reset);
    p->local->weights_c = vsi_nn_ConcatTensor(self->graph, 0,
        inputs[GRUCELL_INPUT_WEIGHT_I2C], inputs[GRUCELL_INPUT_WEIGHT_H2C]);
    p->local->bias_c = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2C]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2C], inputs[GRUCELL_INPUT_BIAS_H2C]);

    vsi_safe_release_tensor(p->local->bias_z);
    vsi_safe_release_tensor(p->local->bias_r);
    p->local->bias_z_r->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->bias_z_r, VSI_NN_TENSOR_ATTR_CONST);
    p->local->weights_z_r->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->weights_z_r, VSI_NN_TENSOR_ATTR_CONST);
    p->local->weights_c->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->weights_c, VSI_NN_TENSOR_ATTR_CONST);
    p->local->bias_c->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->bias_c, VSI_NN_TENSOR_ATTR_CONST);

    input_hstate = vsi_nn_rnn_create_concat(self, 0,
        use_virtual_tensor, inputs[GRUCELL_INPUT_INPUT], inputs[GRUCELL_INPUT_H_STATE]);

    dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if ( input_hstate->t->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 ||
         input_hstate->t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
    {
        dtype.vx_type = input_hstate->t->attr.dtype.vx_type;
    }
    else
    {
        dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    tmp_tensor = vsi_nn_rnn_create_tp_fc(self, input_hstate->t,
        p->local->weights_z_r, p->local->bias_z_r, &dtype, use_virtual_tensor);

    splited_tensors = vsi_nn_create_split(self, tmp_tensor->t, 0, 2, NULL, use_virtual_tensor);

    /* reset Gate activations */
    tensor_rt = vsi_nn_rnn_create_activation(self,
                        splited_tensors[1]->t,
                        p->local->gate_activation,
                        &splited_tensors[1]->t->attr.dtype,
                        use_virtual_tensor);

    /* if linear_before_reset=0:  ht=g(input*w_ic + (r.hstate)*w_hc + b_ic + b_hc)*/
    if ( p->linear_before_reset == 0 )
    {
        /* r{t} * h{t-1}*/
        tensor_rt = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_MULTIPLY,
            tensor_rt->t, inputs[GRUCELL_INPUT_H_STATE], &tensor_rt->t->attr.dtype, use_virtual_tensor);

        /* [x{t}, r{t}] */
        tmp_tensor = vsi_nn_rnn_create_concat(self, 0, use_virtual_tensor,
            inputs[GRUCELL_INPUT_INPUT], tensor_rt->t);

        dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        if ( tmp_tensor->t->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 ||
             tmp_tensor->t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
        {
            dtype.vx_type = input_hstate->t->attr.dtype.vx_type;
        }
        else
        {
            dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        }
        /* W{c} x [x{t}, r{t}] */
        tmp_tensor = vsi_nn_rnn_create_tp_fc(self, tmp_tensor->t, p->local->weights_c, p->local->bias_c,
            &dtype, use_virtual_tensor);
    }
    /* if linear_before_reset!=0: ht=g(input*w_ic + (r.(hstate*w_hc + b_hc)) + b_ic)*/
    else
    {
        dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        if ( inputs[GRUCELL_INPUT_H_STATE]->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 ||
             inputs[GRUCELL_INPUT_H_STATE]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
        {
            dtype.vx_type = inputs[GRUCELL_INPUT_H_STATE]->attr.dtype.vx_type;
        }
        else
        {
            dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        }
        /* r.(hstate*w_hc + b_hc) */
        tmp_tensor = vsi_nn_rnn_create_tp_fc(self, inputs[GRUCELL_INPUT_H_STATE], inputs[GRUCELL_INPUT_WEIGHT_H2C],
            inputs[GRUCELL_INPUT_BIAS_H2C], &dtype, use_virtual_tensor);
        tensor_rt = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_MULTIPLY,
            tensor_rt->t, tmp_tensor->t, &tensor_rt->t->attr.dtype, use_virtual_tensor);
        /* input*w_ic + b_ic */
        tmp_tensor = vsi_nn_rnn_create_tp_fc(self, inputs[GRUCELL_INPUT_INPUT], inputs[GRUCELL_INPUT_WEIGHT_I2C],
            inputs[GRUCELL_INPUT_BIAS_I2C], &dtype, use_virtual_tensor);

        tmp_tensor = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_ADD,
            tensor_rt->t, tmp_tensor->t, &tensor_rt->t->attr.dtype, use_virtual_tensor);
    }

#define USE_GRUCELL_ACTIVATION
#ifdef USE_GRUCELL_ACTIVATION
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_GRUCELL_ACTIVATION_INTERNAL, 0, 0 );
    curr->inputs[0] = splited_tensors[0]->t;
    curr->inputs[1] = tmp_tensor->t;
    curr->inputs[2] = inputs[GRUCELL_INPUT_H_STATE];
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
    curr->outputs[1] = outputs[GRUCELL_OUTPUT_H_STATE];
    curr->node->nn_param.grucell_activation_internal.gate_activation = p->local->gate_activation;
    curr->node->nn_param.grucell_activation_internal.candidate_activation = p->local->candidate_activation;
    curr->node->nn_param.grucell_activation_internal.use_cudnn_implementation = p->use_cudnn_implementation;
    vsi_nn_internal_setup_node(self, curr);
#else
    {
    vsi_nn_internal_tensor_t* tensor_zt = NULL;
    vsi_nn_internal_tensor_t* tensor_ht_ = NULL;
    /* z{t} */
    tensor_zt = vsi_nn_rnn_create_activation(self,
                        splited_tensors[0]->t,
                        p->local->gate_activation,
                        &splited_tensors[0]->t->attr.dtype,
                        use_virtual_tensor);
    /* h{t_} */
    tensor_ht_ = vsi_nn_rnn_create_activation(self,
                        tmp_tensor->t,
                        p->local->candidate_activation,
                        &tmp_tensor->t->attr.dtype,
                        use_virtual_tensor);
    /* z{t} * h{t-1} + (1 - z{t}) * h{t_} ==> z{t} * (h{t-1} - h{t_}) + h{t_} */
    tmp_tensor = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_SUBTRACT,
        inputs[GRUCELL_INPUT_H_STATE], tensor_ht_->t, &tmp_tensor->t->attr.dtype, use_virtual_tensor);
    tmp_tensor = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_MULTIPLY,
        tensor_zt->t, tmp_tensor->t, &tensor_ht_->t->attr.dtype, use_virtual_tensor);
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_ADD, 0, 0 );
    curr->inputs[0] = tmp_tensor->t;
    curr->inputs[1] = tensor_ht_->t;
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
    vsi_nn_internal_setup_node(self, curr);
    }

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
    curr->inputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_H_STATE];
    vsi_nn_internal_setup_node(self, curr);
#endif

    return TRUE;
}

static vsi_bool op_setup_float_cudnn
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_grucell_ovxlib_param* p = &self->nn_param.grucell_ovxlib;
    vsi_bool use_virtual_tensor = TRUE;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* input_fc_output = NULL;
    vsi_nn_internal_tensor_t* recurrent_fc_output = NULL;
    vsi_nn_internal_tensor_t** splited_input_fc_output_tensors = NULL;
    vsi_nn_internal_tensor_t** splited_recurrent_fc_output_tensors = NULL;
    uint32_t kernel_h = 1, kernel_w = 1;
    grucell_activation_input_layout_e grucell_activation_input_layout = GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_NC;
    vsi_size_t reshaped_size[2] = { 0 };

    p->local->multi_batch = inputs[GRUCELL_INPUT_INPUT]->attr.size[1] > 1;

    p->local->weights_input = vsi_nn_ConcatTensor(self->graph, 1, inputs[GRUCELL_INPUT_WEIGHT_I2R],
                inputs[GRUCELL_INPUT_WEIGHT_I2Z], inputs[GRUCELL_INPUT_WEIGHT_I2C]);
    p->local->weights_input->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->weights_input, VSI_NN_TENSOR_ATTR_CONST);

    p->local->weights_recurrent = vsi_nn_ConcatTensor(self->graph, 1, inputs[GRUCELL_INPUT_WEIGHT_H2R],
                inputs[GRUCELL_INPUT_WEIGHT_H2Z], inputs[GRUCELL_INPUT_WEIGHT_H2C]);
    p->local->weights_recurrent->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->weights_recurrent, VSI_NN_TENSOR_ATTR_CONST);

    p->local->bias_r = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2R]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2R], inputs[GRUCELL_INPUT_BIAS_H2R]);
    p->local->bias_r->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->bias_r, VSI_NN_TENSOR_ATTR_CONST);
    p->local->bias_z = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2Z]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2Z], inputs[GRUCELL_INPUT_BIAS_H2Z]);
    p->local->bias_z->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->bias_z, VSI_NN_TENSOR_ATTR_CONST);
    p->local->bias_c = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2C]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2C], inputs[GRUCELL_INPUT_BIAS_H2C]);
    p->local->bias_c->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->bias_c, VSI_NN_TENSOR_ATTR_CONST);

    if(p->local->multi_batch && p->local->force_input_recurrent_on_NN)
    {
        vsi_nn_internal_tensor_t* input_tensor = NULL;
        vsi_nn_internal_tensor_t* tmp = NULL;

        /*
        vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
            inputs[GRUCELL_INPUT_INPUT]->attr.size[0], &kernel_h, &kernel_w);
        */
        /* reshape and transpose input */
        input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, inputs[GRUCELL_INPUT_INPUT],
                                                p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);

        tmp = vsi_nn_rnn_create_nn_fc(self, input_tensor->t, p->local->weights_input,
            NULL, kernel_h, kernel_w,
            &p->internal_dtype[GRUCELL_CUDNN_QUANTIZE_PARAM_INPUT],
            use_virtual_tensor);
        /* transpose and reshape output */
        reshaped_size[0] = inputs[GRUCELL_INPUT_INPUT]->attr.size[1];
        reshaped_size[1] = p->local->weights_input->attr.size[1];
        input_fc_output = vsi_nn_rnn_create_reshape(self, tmp->t, NULL,
            reshaped_size, 2, use_virtual_tensor);

        grucell_activation_input_layout = GRUCELL_ACTIVATION_INPUT_LAYOUT_INPUT_NC_FC_CN;
    }
    else
    {
        input_fc_output = vsi_nn_rnn_create_tp_fc(self, inputs[GRUCELL_INPUT_INPUT],
            p->local->weights_input, NULL,
            &p->internal_dtype[GRUCELL_CUDNN_QUANTIZE_PARAM_INPUT], use_virtual_tensor);
        grucell_activation_input_layout = GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_NC;
    }

    if(p->local->multi_batch && p->local->force_input_recurrent_on_NN)
    {
        vsi_nn_internal_tensor_t* input_tensor = NULL;
        vsi_nn_internal_tensor_t* tmp = NULL;
        /*
        vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
            inputs[GRUCELL_INPUT_H_STATE]->attr.size[0], &kernel_h, &kernel_w);
        */
        /* reshape and transpose input */
        input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, inputs[GRUCELL_INPUT_H_STATE],
                                                p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);

        tmp = vsi_nn_rnn_create_nn_fc(self, input_tensor->t, p->local->weights_recurrent,
                                        NULL, kernel_h, kernel_w,
                                        &p->internal_dtype[GRUCELL_CUDNN_QUANTIZE_PARAM_HIDDEN], use_virtual_tensor);
        /* transpose and reshape output */
        reshaped_size[0] = inputs[GRUCELL_INPUT_H_STATE]->attr.size[1];
        reshaped_size[1] = p->local->weights_recurrent->attr.size[1];
        recurrent_fc_output = vsi_nn_rnn_create_reshape(self, tmp->t, NULL,
            reshaped_size, 2, use_virtual_tensor);
    }
    else
    {
        recurrent_fc_output = vsi_nn_rnn_create_tp_fc(self, inputs[GRUCELL_INPUT_H_STATE],
            p->local->weights_recurrent, NULL,
            &p->internal_dtype[GRUCELL_CUDNN_QUANTIZE_PARAM_HIDDEN], use_virtual_tensor);
    }

#ifdef USE_GRUCELL_ACTIVATION
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_GRUCELL_ACTIVATION_INTERNAL, 0, 0 );
    curr->inputs[GRUCELL_ACTIVATION_INPUT_H_STATE] = inputs[GRUCELL_INPUT_H_STATE];

    if(p->local->multi_batch)
    {
        if(GRUCELL_ACTIVATION_INPUT_LAYOUT_ALL_NC == grucell_activation_input_layout)
        {
            curr->inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R] = input_fc_output->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_Z] = NULL;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_C] = NULL;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_R] = recurrent_fc_output->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_Z] = NULL;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_C] = NULL;
        }
        else
        {
            splited_input_fc_output_tensors = vsi_nn_create_split(self,
                input_fc_output->t, 1, 3, NULL, use_virtual_tensor);
            splited_recurrent_fc_output_tensors = vsi_nn_create_split(self,
                recurrent_fc_output->t, 1, 3, NULL, use_virtual_tensor);
            curr->inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_R] = splited_input_fc_output_tensors[0]->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_Z] = splited_input_fc_output_tensors[1]->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_INPUT_FC_C] = splited_input_fc_output_tensors[2]->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_R] = splited_recurrent_fc_output_tensors[0]->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_Z] = splited_recurrent_fc_output_tensors[1]->t;
            curr->inputs[GRUCELL_ACTIVATION_INPUT_RECURRENT_FC_C] = splited_recurrent_fc_output_tensors[2]->t;
        }
    }
    else
    {
        splited_input_fc_output_tensors = vsi_nn_create_split(self,
            input_fc_output->t, 0, 3, NULL, use_virtual_tensor);
        splited_recurrent_fc_output_tensors = vsi_nn_create_split(self,
            recurrent_fc_output->t, 0, 3, NULL, use_virtual_tensor);
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
    curr->inputs[GRUCELL_ACTIVATION_INPUT_COND_R] = inputs[GRUCELL_INPUT_COND_RESET];
    curr->inputs[GRUCELL_ACTIVATION_INPUT_COND_Z] = inputs[GRUCELL_INPUT_COND_UPDATE];
    curr->inputs[GRUCELL_ACTIVATION_INPUT_COND_C] = inputs[GRUCELL_INPUT_COND_CANDIDATE];
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
    curr->outputs[1] = outputs[GRUCELL_OUTPUT_H_STATE];
    curr->node->nn_param.grucell_activation_internal.gate_activation = p->local->gate_activation;
    curr->node->nn_param.grucell_activation_internal.candidate_activation = p->local->candidate_activation;
    curr->node->nn_param.grucell_activation_internal.input_category = GRUCELL_INPUT_CATEGORY_CUDNN;
    curr->node->nn_param.grucell_activation_internal.use_cudnn_implementation = TRUE;
    curr->node->nn_param.grucell_activation_internal.input_layout = grucell_activation_input_layout;
    vsi_nn_internal_setup_node(self, curr);
#else
    {
        vsi_nn_internal_tensor_t* tmp_tensor = NULL;
        vsi_nn_internal_tensor_t* tensor_r = NULL;
        vsi_nn_internal_tensor_t* tensor_u = NULL;
        vsi_nn_internal_tensor_t* tensor_c = NULL;
        vsi_bool is_cond_available = FALSE;

        if(inputs[GRUCELL_INPUT_COND_RESET] && inputs[GRUCELL_INPUT_COND_UPDATE]
            && inputs[GRUCELL_INPUT_COND_CANDIDATE])
        {
            is_cond_available = TRUE;
        }
        p->local->bias_z_r = vsi_nn_ConcatTensor(self->graph, 0, p->local->bias_r,
            p->local->bias_z, p->local->bias_c);

        if(is_cond_available)
        {
            tmp_tensor = vsi_nn_rnn_create_concat(self, 0, use_virtual_tensor,
                inputs[GRUCELL_INPUT_COND_RESET], inputs[GRUCELL_INPUT_COND_UPDATE],
                inputs[GRUCELL_INPUT_COND_CANDIDATE]);

            input_fc_output = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_ADD,
                input_fc_output->t,tmp_tensor->t,
                &input_fc_output->t->attr.dtype, use_virtual_tensor);
        }
        recurrent_fc_output = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_ADD,
            recurrent_fc_output->t, p->local->bias_z_r,
            &recurrent_fc_output->t->attr.dtype, use_virtual_tensor);

        splited_input_fc_output_tensors = vsi_nn_create_split(self, input_fc_output->t, 0, 3, NULL, TRUE);
        splited_recurrent_fc_output_tensors = vsi_nn_create_split(self, recurrent_fc_output->t, 0, 3, NULL, TRUE);

        tensor_r = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_ADD,
            splited_input_fc_output_tensors[0]->t, splited_recurrent_fc_output_tensors[0]->t,
            &splited_input_fc_output_tensors[0]->t->attr.dtype, use_virtual_tensor);

        tensor_u = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_ADD,
            splited_input_fc_output_tensors[1]->t, splited_recurrent_fc_output_tensors[1]->t,
            &splited_input_fc_output_tensors[1]->t->attr.dtype, use_virtual_tensor);

        /* reset Gate activations */
        tensor_r = vsi_nn_rnn_create_activation(self,
                            tensor_r->t,
                            p->local->gate_activation,
                            &tensor_r->t->attr.dtype,
                            use_virtual_tensor);

        tensor_u = vsi_nn_rnn_create_activation(self,
                            tensor_u->t,
                            p->local->gate_activation,
                            &tensor_u->t->attr.dtype,
                            use_virtual_tensor);

        /* r{t} * h{t-1}*/
        tmp_tensor = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_MULTIPLY,
            tensor_r->t, splited_recurrent_fc_output_tensors[2]->t, &tensor_r->t->attr.dtype, use_virtual_tensor);

        tmp_tensor = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_ADD,
            tmp_tensor->t, splited_input_fc_output_tensors[2]->t, &tmp_tensor->t->attr.dtype, use_virtual_tensor);

        tmp_tensor = vsi_nn_rnn_create_activation(self,
                            tmp_tensor->t,
                            p->local->candidate_activation,
                            &tmp_tensor->t->attr.dtype,
                            use_virtual_tensor);
        tensor_c = tmp_tensor;

        /* z{t} * h{t-1} + (1 - z{t}) * h{t_} ==> z{t} * (h{t-1} - h{t_}) + h{t_} */
        tmp_tensor = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_SUBTRACT,
            inputs[GRUCELL_INPUT_H_STATE], tensor_c->t, &tensor_c->t->attr.dtype, use_virtual_tensor);

        tmp_tensor = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_MULTIPLY,
            tensor_u->t, tmp_tensor->t, &tmp_tensor->t->attr.dtype, use_virtual_tensor);

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_ADD, 0, 0 );
        curr->inputs[0] = tmp_tensor->t;
        curr->inputs[1] = tensor_c->t;
        curr->outputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
        vsi_nn_internal_setup_node(self, curr);

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
        curr->inputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
        curr->outputs[0] = outputs[GRUCELL_OUTPUT_H_STATE];
        vsi_nn_internal_setup_node(self, curr);
    }
#endif

    return TRUE;
}

/*
use TP for sigmoid and tanh, split grucell_activation to 3 parts
*/
static vsi_bool op_setup_float_cudnn_v2
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_grucell_ovxlib_param* p = &self->nn_param.grucell_ovxlib;
    vsi_nn_dtype_t dtype;
    vsi_bool use_virtual_tensor = TRUE;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* input2cand_output = NULL;
    vsi_nn_internal_tensor_t* recurrent2cand_output = NULL;
    vsi_nn_internal_tensor_t** splited_input_fc_output_tensors = NULL;
    vsi_nn_internal_tensor_t* tmp_tensor = NULL;
    vsi_nn_internal_tensor_t* tensor_r = NULL;
    vsi_nn_internal_tensor_t* concated_input = NULL;
    vsi_nn_tensor_attr_t attr;

    /* input to r,z */
    p->local->weights_update = vsi_nn_ConcatTensor(self->graph, 1/* axis */,
        inputs[GRUCELL_INPUT_WEIGHT_I2R], inputs[GRUCELL_INPUT_WEIGHT_I2Z]);
    /* recurrent to r,z */
    p->local->weights_reset = vsi_nn_ConcatTensor(self->graph, 1/* axis */,
        inputs[GRUCELL_INPUT_WEIGHT_H2R], inputs[GRUCELL_INPUT_WEIGHT_H2Z]);
    /* [input, recurrent] to r,z */
    p->local->weights_input = vsi_nn_ConcatTensor(self->graph, 0/* axis */,
        p->local->weights_update, p->local->weights_reset);
    p->local->weights_input->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->weights_input, VSI_NN_TENSOR_ATTR_CONST);
    vsi_safe_release_tensor(p->local->weights_update);
    vsi_safe_release_tensor(p->local->weights_reset);

    p->local->bias_z = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2Z]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2Z], inputs[GRUCELL_INPUT_BIAS_H2Z]);
    p->local->bias_r = vsi_nn_ConstTensorAdd(self->graph, inputs[GRUCELL_INPUT_BIAS_I2R]->attr,
        inputs[GRUCELL_INPUT_BIAS_I2R], inputs[GRUCELL_INPUT_BIAS_H2R]);
    p->local->bias_z_r = vsi_nn_ConcatTensor(self->graph, 0/* axis */,
        p->local->bias_r, p->local->bias_z);
    p->local->bias_z_r->attr.is_const = TRUE;
    vsi_nn_SetTensorAttr(p->local->bias_z_r, VSI_NN_TENSOR_ATTR_CONST);
    vsi_safe_release_tensor(p->local->bias_z);
    vsi_safe_release_tensor(p->local->bias_r);

    concated_input = vsi_nn_rnn_create_concat(self, 0/* axis */,
        use_virtual_tensor, inputs[GRUCELL_INPUT_INPUT], inputs[GRUCELL_INPUT_H_STATE]);

    dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if ( concated_input->t->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 ||
         concated_input->t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
    {
        dtype.vx_type = concated_input->t->attr.dtype.vx_type;
    }
    else
    {
        dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    tmp_tensor = vsi_nn_rnn_create_tp_fc(self, concated_input->t, p->local->weights_input,
        p->local->bias_z_r, &dtype, use_virtual_tensor);

    dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if ( splited_input_fc_output_tensors[0]->t->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 ||
         splited_input_fc_output_tensors[0]->t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
    {
        dtype.vx_type = splited_input_fc_output_tensors[0]->t->attr.dtype.vx_type;
    }
    else
    {
        dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    {
        uint32_t _slices[] = { (uint32_t)inputs[GRUCELL_INPUT_INPUT]->attr.size[0],
            (uint32_t)inputs[GRUCELL_INPUT_H_STATE]->attr.size[0] };
        splited_input_fc_output_tensors = vsi_nn_create_split(self, concated_input->t,
            0, 2, _slices, use_virtual_tensor);
    }
    input2cand_output = vsi_nn_rnn_create_tp_fc(self, splited_input_fc_output_tensors[0]->t,
        inputs[GRUCELL_INPUT_WEIGHT_I2C], inputs[GRUCELL_INPUT_BIAS_I2C], &dtype, use_virtual_tensor);

    dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if ( inputs[GRUCELL_INPUT_H_STATE]->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 ||
         inputs[GRUCELL_INPUT_H_STATE]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
    {
        dtype.vx_type = inputs[GRUCELL_INPUT_H_STATE]->attr.dtype.vx_type;
    }
    else
    {
        dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    recurrent2cand_output = vsi_nn_rnn_create_tp_fc(self, inputs[GRUCELL_INPUT_H_STATE],
        inputs[GRUCELL_INPUT_WEIGHT_H2C], inputs[GRUCELL_INPUT_BIAS_H2C], &dtype, use_virtual_tensor);

    tmp_tensor = vsi_nn_rnn_create_activation(self, tmp_tensor->t, p->local->gate_activation,
        &tmp_tensor->t->attr.dtype, use_virtual_tensor);

    /* split for combined FC outputs, r_t, z_t */
    splited_input_fc_output_tensors = vsi_nn_create_split(self, tmp_tensor->t,
        0/* axis */,
        2/* dim num */, NULL, use_virtual_tensor);

    memset( &attr, 0x00, sizeof(attr) );
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if ( splited_input_fc_output_tensors[0]->t->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 ||
         splited_input_fc_output_tensors[0]->t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
    {
        dtype.vx_type = splited_input_fc_output_tensors[0]->t->attr.dtype.vx_type;
    }
    else
    {
        dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    tmp_tensor = vsi_nn_internal_new_tensor(self, &attr, 0.0f);

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_A_TIMES_B_PLUS_C, 0, 0 );
    curr->inputs[0] = splited_input_fc_output_tensors[0]->t;
    curr->inputs[1] = recurrent2cand_output->t;
    curr->inputs[2] = input2cand_output->t;
    curr->outputs[0] = tmp_tensor->t;
    vsi_nn_internal_setup_node(self, curr);

    tensor_r = vsi_nn_rnn_create_activation(self, tmp_tensor->t,
        p->local->candidate_activation, &tmp_tensor->t->attr.dtype, use_virtual_tensor);

#define USE_GRUCELL_ACTIVATION_SMA
#ifdef USE_GRUCELL_ACTIVATION_SMA
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_GRUCELL_ACTIVATION_INTERNAL_SMA, 0, 0 );
    curr->inputs[GRUCELL_ACTIVATION_SMA_INPUT_H_STATE] = inputs[GRUCELL_INPUT_H_STATE];
    curr->inputs[GRUCELL_ACTIVATION_SMA_INPUT_H_T_] = tensor_r->t;
    curr->inputs[GRUCELL_ACTIVATION_SMA_INPUT_Z_T] = splited_input_fc_output_tensors[1]->t;
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
    curr->outputs[1] = outputs[GRUCELL_OUTPUT_H_STATE];
    curr->node->nn_param.grucell_activation_internal.gate_activation = p->local->gate_activation;
    curr->node->nn_param.grucell_activation_internal.candidate_activation = p->local->candidate_activation;
    curr->node->nn_param.grucell_activation_internal.use_cudnn_implementation = p->use_cudnn_implementation;
    vsi_nn_internal_setup_node(self, curr);
#else
    tmp_tensor = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_SUBTRACT,
        inputs[GRUCELL_INPUT_H_STATE],
        tensor_r->t, &tensor_r->t->attr.dtype, use_virtual_tensor);
    tmp_tensor = vsi_nn_rnn_create_binary_operator(self, VSI_NN_OP_MULTIPLY,
        splited_input_fc_output_tensors[1]->t,
        tmp_tensor->t, &tmp_tensor->t->attr.dtype, use_virtual_tensor);

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_ADD, 0, 0 );
    curr->inputs[0] = tmp_tensor->t;
    curr->inputs[1] = tensor_r->t;
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
    vsi_nn_internal_setup_node(self, curr);

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
    curr->inputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_H_STATE];
    vsi_nn_internal_setup_node(self, curr);
#endif

    return TRUE;
}

static vsi_bool op_setup_default
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_grucell_ovxlib_param* p = &self->nn_param.grucell_ovxlib;
    vsi_nn_tensor_attr_t attr;
    vsi_bool is_input_fc_on_tp = FALSE;
    vsi_bool is_hstate_fc_on_tp = FALSE;
    vsi_bool is_input_cand_fc_op_tp = FALSE;
    vsi_bool is_hstate_cand_fc_op_tp = FALSE;
    vsi_nn_internal_tensor_t* input_tensor = NULL;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_internal_tensor_t* tmp_tensor = NULL;
    vsi_nn_internal_tensor_t* hstate_input_tensor = NULL;
    vsi_nn_internal_tensor_t* input_gate_fc_outputs[GRUCELL_RZ_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* hstate_gate_fc_outputs[GRUCELL_RZ_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* gate_fc_outputs[GRUCELL_RZ_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* gate_act_outputs[GRUCELL_RZ_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* rh_mul_outputs = NULL;
    vsi_nn_internal_tensor_t* input_cand_fc_output = NULL;
    vsi_nn_internal_tensor_t* rh_cand_fc_output = NULL;
    vsi_nn_internal_tensor_t* r_mul_hcand_fc_output = NULL;
    vsi_nn_internal_tensor_t* cand_fc_output = NULL;
    vsi_nn_internal_tensor_t* cand_act_output = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_bool use_virtual_tensor = FALSE;
    uint32_t kernel_h = 1;
    uint32_t kernel_w = 1;
    int32_t i = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    memset( &attr, 0x00, sizeof( attr ) );

    if( inputs[GRUCELL_INPUT_INPUT]->attr.dtype.qnt_type
        != inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.dtype.qnt_type)
    {
        /* input and input weights have different qtype, only TP can do this operation */
        is_input_fc_on_tp = TRUE;
    }
    else if( inputs[GRUCELL_INPUT_INPUT]->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if input's shape is not aligned to 64-byte */
        is_input_fc_on_tp = TRUE;
    }

    if( inputs[GRUCELL_INPUT_H_STATE]->attr.dtype.qnt_type
        != inputs[GRUCELL_INPUT_WEIGHT_H2R]->attr.dtype.qnt_type)
    {
        /* recurrent and recurrent weights have different qtype, only TP can do this operation */
        is_hstate_fc_on_tp = TRUE;
    }
    else if( inputs[GRUCELL_INPUT_H_STATE]->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if inputs' shape is not aligned to 64-byte */
        is_hstate_fc_on_tp = TRUE;
    }

    /* if both input fc and recurrent fc could be executed on NN, offloads one to TP*/
    if( !is_input_fc_on_tp && !is_hstate_fc_on_tp )
    {
        is_input_fc_on_tp = TRUE;
    }
    /* TODO: now, all fc on tp because can't fetch the HW feature */
    is_input_fc_on_tp = TRUE;
    is_hstate_fc_on_tp = TRUE;

    /* Input FC */
    if( is_input_fc_on_tp )
    {
        /* tp */
        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            input_gate_fc_outputs[i] = vsi_nn_rnn_create_tp_fc(self,
                                                inputs[GRUCELL_INPUT_INPUT],
                                                inputs[GRUCELL_INPUT_WEIGHT_I2R + i],
                                                inputs[GRUCELL_INPUT_BIAS_I2R + i],
                                                &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2R + i],
                                                use_virtual_tensor);
        }
    }
    else
    {
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
            (uint32_t)inputs[GRUCELL_INPUT_INPUT]->attr.size[0], &kernel_h, &kernel_w);
        input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, inputs[GRUCELL_INPUT_INPUT],
                                                p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);

        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            vsi_nn_internal_tensor_t* tmp = vsi_nn_rnn_create_nn_fc(self,
                                                input_tensor->t,
                                                inputs[GRUCELL_INPUT_WEIGHT_I2R + i],
                                                inputs[GRUCELL_INPUT_BIAS_I2R + i],
                                                kernel_h, kernel_w,
                                                &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2R + i],
                                                use_virtual_tensor);
            /* transpose and reshape output */
            input_gate_fc_outputs[i] = vsi_nn_rnn_process_output_for_nn_fc(self,
                tmp->t, p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
        }
    }

    /* Hstate FC */
    if( is_hstate_fc_on_tp )
    {
        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            hstate_gate_fc_outputs[i] = vsi_nn_rnn_create_tp_fc(self,
                                                inputs[GRUCELL_INPUT_H_STATE],
                                                inputs[GRUCELL_INPUT_WEIGHT_H2R + i],
                                                inputs[GRUCELL_INPUT_BIAS_H2R + i],
                                                &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2R + i],
                                                use_virtual_tensor);
        }
    }
    else
    {
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
            (uint32_t)inputs[GRUCELL_INPUT_H_STATE]->attr.size[0], &kernel_h, &kernel_w);
        hstate_input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self,
            inputs[GRUCELL_INPUT_H_STATE], p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);

        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            vsi_nn_internal_tensor_t* tmp = vsi_nn_rnn_create_nn_fc(self,
                                                hstate_input_tensor->t,
                                                inputs[GRUCELL_INPUT_WEIGHT_H2R + i],
                                                inputs[GRUCELL_INPUT_BIAS_H2R + i],
                                                kernel_h, kernel_w,
                                                &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2R + i],
                                                use_virtual_tensor);
            /* transpose and reshape output */
            hstate_gate_fc_outputs[i] = vsi_nn_rnn_process_output_for_nn_fc(self,
                tmp->t, p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
        }
    }

    /* Gate Input FC add Hstate FC */
    for ( i = 0;  i < GRUCELL_RZ_GATE_COUNT;  i++)
    {
        gate_fc_outputs[i] = vsi_nn_rnn_create_tensor_add(self,
                                 input_gate_fc_outputs[i]->t,
                                 hstate_gate_fc_outputs[i]->t,
                                 &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2R + i],
                                 use_virtual_tensor);
    }

    /* Gate activations */
    for ( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
    {
        gate_act_outputs[i] = vsi_nn_rnn_create_activation(self,
                                  gate_fc_outputs[i]->t,
                                  p->local->gate_activation,
                                  &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2R + i],
                                  use_virtual_tensor);
    }

    /* Candidate FC */
    /* if linear_before_reset=0:  ht=g(input*w_ic + (r.hstate)*w_hc + b_ic + b_hc)*/
    /* if linear_before_reset!=0: ht=g(input*w_ic + (r.(hstate*w_hc + b_hc)) + b_ic)*/
    if ( p->linear_before_reset == 0 )
    {
        rh_mul_outputs = create_multiply(self,
                             gate_act_outputs[GRUCELL_GATE_R]->t,
                             inputs[GRUCELL_INPUT_H_STATE],
                             &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2R],
                             use_virtual_tensor);
    }
    else
    {
        rh_mul_outputs = vsi_nn_rnn_create_reshape(self,
            inputs[GRUCELL_INPUT_H_STATE],
            NULL,
            inputs[GRUCELL_INPUT_H_STATE]->attr.size,
            inputs[GRUCELL_INPUT_H_STATE]->attr.dim_num,
            use_virtual_tensor);
    }

    if( inputs[GRUCELL_INPUT_INPUT]->attr.dtype.qnt_type
        != inputs[GRUCELL_INPUT_WEIGHT_I2C]->attr.dtype.qnt_type)
    {
        /* input and input weights have different qtype, only TP can do this operation */
        is_input_cand_fc_op_tp = TRUE;
    }
    else if( inputs[GRUCELL_INPUT_INPUT]->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if input's shape is not aligned to 64-byte */
        is_input_cand_fc_op_tp = TRUE;
    }

    if( rh_mul_outputs->t->attr.dtype.qnt_type
        != inputs[GRUCELL_INPUT_WEIGHT_H2C]->attr.dtype.qnt_type)
    {
        /* recurrent and recurrent weights have different qtype, only TP can do this operation */
        is_hstate_cand_fc_op_tp = TRUE;
    }
    else if( rh_mul_outputs->t->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if inputs' shape is not aligned to 64-byte */
        is_hstate_cand_fc_op_tp = TRUE;
    }
    /* if both input fc and recurrent fc could be executed on NN, offloads one to TP*/
    if( !is_input_cand_fc_op_tp && !is_hstate_cand_fc_op_tp )
    {
        is_input_cand_fc_op_tp = TRUE;
    }
    /* TODO: now, all fc on tp because can't fetch the HW feature */
    is_input_cand_fc_op_tp = TRUE;
    is_hstate_cand_fc_op_tp = TRUE;

    if ( is_input_cand_fc_op_tp )
    {
        input_cand_fc_output = vsi_nn_rnn_create_tp_fc(self,
                                   inputs[GRUCELL_INPUT_INPUT],
                                   inputs[GRUCELL_INPUT_WEIGHT_I2C],
                                   inputs[GRUCELL_INPUT_BIAS_I2C],
                                   &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                                   use_virtual_tensor);
    }
    else
    {
        vsi_nn_internal_tensor_t* tmp = NULL;
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
            (uint32_t)inputs[GRUCELL_INPUT_INPUT]->attr.size[0], &kernel_h, &kernel_w);
        input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, inputs[GRUCELL_INPUT_INPUT],
                                                p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
        tmp = vsi_nn_rnn_create_nn_fc(self,
                  input_tensor->t,
                  inputs[GRUCELL_INPUT_WEIGHT_I2C],
                  inputs[GRUCELL_INPUT_BIAS_I2C],
                  kernel_h, kernel_w,
                  &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                  use_virtual_tensor);
        /* transpose and reshape output */
        input_cand_fc_output = vsi_nn_rnn_process_output_for_nn_fc(self,
            tmp->t, p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
    }
    if ( is_hstate_cand_fc_op_tp )
    {
        /* if the tp support in:fp16,weight:u8,bias:fp32 batch>1, remove this. */
        if ((rh_mul_outputs->t->attr.dtype.vx_type) != (inputs[GRUCELL_INPUT_WEIGHT_H2C]->attr.dtype.vx_type)
            && (p->local->multi_batch))
        {
            vsi_nn_tensor_t* wei_r2c_tensor = NULL;
            vsi_nn_tensor_t* bias_r2c_tensor = NULL;

            memcpy(&attr, &(inputs[GRUCELL_INPUT_WEIGHT_H2C]->attr), sizeof(attr));
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
            if ( rh_mul_outputs->t->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 ||
                 rh_mul_outputs->t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
            {
                attr.dtype.vx_type = rh_mul_outputs->t->attr.dtype.vx_type;
            }
            else
            {
                attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
            }

            wei_r2c_tensor = vsi_nn_ConvertTensorDtype(self->graph, inputs[GRUCELL_INPUT_WEIGHT_H2C], &(attr.dtype));
            attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
            bias_r2c_tensor = vsi_nn_ConvertTensorDtype(self->graph, inputs[GRUCELL_INPUT_BIAS_H2C], &(attr.dtype));
            rh_cand_fc_output = vsi_nn_rnn_create_tp_fc(self,
                                    rh_mul_outputs->t,
                                    wei_r2c_tensor,
                                    bias_r2c_tensor,
                                    &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2C],
                                    use_virtual_tensor);
        }
        else
        {
            rh_cand_fc_output = vsi_nn_rnn_create_tp_fc(self,
                                    rh_mul_outputs->t,
                                    inputs[GRUCELL_INPUT_WEIGHT_H2C],
                                    inputs[GRUCELL_INPUT_BIAS_H2C],
                                    &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2C],
                                    use_virtual_tensor);
        }
    }
    else
    {
        vsi_nn_internal_tensor_t* tmp = NULL;
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
            (uint32_t)rh_mul_outputs->t->attr.size[0], &kernel_h, &kernel_w);
        hstate_input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, rh_mul_outputs->t,
                                                p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
        tmp = vsi_nn_rnn_create_nn_fc(self,
                  hstate_input_tensor->t,
                  inputs[GRUCELL_INPUT_WEIGHT_H2C],
                  inputs[GRUCELL_INPUT_BIAS_H2C],
                  kernel_h, kernel_w,
                  &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2C],
                  use_virtual_tensor);
        /* transpose and reshape output */
        rh_cand_fc_output = vsi_nn_rnn_process_output_for_nn_fc(self,
            tmp->t, p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
    }

    if ( p->linear_before_reset == 0 )
    {
        r_mul_hcand_fc_output = rh_cand_fc_output;
    }
    else
    {
        r_mul_hcand_fc_output = create_multiply(self,
                                    gate_act_outputs[GRUCELL_GATE_R]->t,
                                    rh_cand_fc_output->t,
                                    &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2C],
                                    use_virtual_tensor);
    }
    /* Candidate input FC add r*h FC */
    cand_fc_output = vsi_nn_rnn_create_tensor_add(self,
                         input_cand_fc_output->t,
                         r_mul_hcand_fc_output->t,
                         &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                         use_virtual_tensor);

    /* Candidate activation */
    cand_act_output = vsi_nn_rnn_create_activation(self,
                                  cand_fc_output->t,
                                  p->local->candidate_activation,
                                  &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                                  use_virtual_tensor);

    /* GRU cell output */
    memcpy( &attr.dtype, &gate_act_outputs[GRUCELL_GATE_Z]->t->attr.dtype, sizeof( attr.dtype ) );
    memcpy( &attr.size, &gate_act_outputs[GRUCELL_GATE_Z]->t->attr.size, sizeof( attr.size ) );
    attr.dim_num = gate_act_outputs[GRUCELL_GATE_Z]->t->attr.dim_num;
    attr.vtl = use_virtual_tensor;
    attr.is_const = TRUE;
    input_tensor = vsi_nn_internal_new_tensor(self, &attr, 1.0f);

    memset( &attr, 0x00, sizeof(attr) );
    //memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if ( input_tensor->t->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 ||
         input_tensor->t->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
    {
        attr.dtype.vx_type = input_tensor->t->attr.dtype.vx_type;
    }
    else
    {
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }

    tmp_tensor = vsi_nn_internal_new_tensor(self, &attr, 0.0f);

    /* create internal tensor sub node (1-zt)*c */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_SUBTRACT, 0, 0 );
    curr->inputs[0] = input_tensor->t;
    curr->inputs[1] = gate_act_outputs[GRUCELL_GATE_Z]->t;
    curr->outputs[0] = tmp_tensor->t;

    vsi_nn_internal_setup_node(self, curr);

    /* create internal multiply node (1-zt)*c */
    output_tensor = create_multiply(self,
                        tmp_tensor->t,
                        cand_act_output->t,
                        &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                        use_virtual_tensor);

    /* create internal multiply node zt*hstate */
    tmp_tensor = create_multiply(self,
                     gate_act_outputs[GRUCELL_GATE_Z]->t,
                     inputs[GRUCELL_INPUT_H_STATE],
                     &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2Z],
                     use_virtual_tensor);

     /* create internal tensor add node (1-zt)*c + zt*hstate */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_ADD, 0, 0 );
    curr->inputs[0] = output_tensor->t;
    curr->inputs[1] = tmp_tensor->t;
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];

    vsi_nn_internal_setup_node(self, curr);

    /* copy output to h_state  */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
    curr->inputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_H_STATE];
    vsi_nn_internal_setup_node(self, curr);

    return TRUE;
} /* op_setup() */

static vsi_bool op_setup
        (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_grucell_ovxlib_param* p = &self->nn_param.grucell_ovxlib;
    vsi_bool is_all_inputs_fp16 = FALSE;
    vsi_bool is_all_inputs_u8 = FALSE;

    p->local->multi_batch = (inputs[GRUCELL_INPUT_INPUT]->attr.size[1] > 1);
    p->local->gate_activation = p->recurrent_activation;
    p->local->candidate_activation = p->activation;

    is_all_inputs_fp16 = inputs[GRUCELL_INPUT_INPUT]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
        && inputs[GRUCELL_INPUT_H_STATE]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
        && inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
        && inputs[GRUCELL_INPUT_WEIGHT_I2Z]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
        && inputs[GRUCELL_INPUT_WEIGHT_H2R]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
        && inputs[GRUCELL_INPUT_WEIGHT_H2Z]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
        && inputs[GRUCELL_INPUT_WEIGHT_I2C]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
        && inputs[GRUCELL_INPUT_WEIGHT_H2C]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16;

    is_all_inputs_u8 = inputs[GRUCELL_INPUT_INPUT]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8
        && inputs[GRUCELL_INPUT_H_STATE]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8
        && inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8
        && inputs[GRUCELL_INPUT_WEIGHT_I2Z]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8
        && inputs[GRUCELL_INPUT_WEIGHT_H2R]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8
        && inputs[GRUCELL_INPUT_WEIGHT_H2Z]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8
        && inputs[GRUCELL_INPUT_WEIGHT_I2C]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8
        && inputs[GRUCELL_INPUT_WEIGHT_H2C]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8;

    if(is_all_inputs_u8)
    {
        vsi_nn_qnt_type_e qnt_type = inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.dtype.qnt_type;
        float scale = inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.dtype.scale;
        int32_t zero_point = inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.dtype.zero_point;

        is_all_inputs_u8 &= inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.dtype.qnt_type == qnt_type
            && inputs[GRUCELL_INPUT_WEIGHT_I2Z]->attr.dtype.qnt_type == qnt_type
            && inputs[GRUCELL_INPUT_WEIGHT_H2R]->attr.dtype.qnt_type == qnt_type
            && inputs[GRUCELL_INPUT_WEIGHT_H2Z]->attr.dtype.qnt_type == qnt_type
            && inputs[GRUCELL_INPUT_WEIGHT_I2C]->attr.dtype.qnt_type == qnt_type
            && inputs[GRUCELL_INPUT_WEIGHT_H2C]->attr.dtype.qnt_type == qnt_type;

        is_all_inputs_u8 &= inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.dtype.scale == scale
            && inputs[GRUCELL_INPUT_WEIGHT_I2Z]->attr.dtype.scale == scale
            && inputs[GRUCELL_INPUT_WEIGHT_H2R]->attr.dtype.scale == scale
            && inputs[GRUCELL_INPUT_WEIGHT_H2Z]->attr.dtype.scale == scale
            && inputs[GRUCELL_INPUT_WEIGHT_I2C]->attr.dtype.scale == scale
            && inputs[GRUCELL_INPUT_WEIGHT_H2C]->attr.dtype.scale == scale;

        is_all_inputs_u8 &= inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.dtype.zero_point == zero_point
            && inputs[GRUCELL_INPUT_WEIGHT_I2Z]->attr.dtype.zero_point == zero_point
            && inputs[GRUCELL_INPUT_WEIGHT_H2R]->attr.dtype.zero_point == zero_point
            && inputs[GRUCELL_INPUT_WEIGHT_H2Z]->attr.dtype.zero_point == zero_point
            && inputs[GRUCELL_INPUT_WEIGHT_I2C]->attr.dtype.zero_point == zero_point
            && inputs[GRUCELL_INPUT_WEIGHT_H2C]->attr.dtype.zero_point == zero_point;
    }

    setup_op_shapes(self, inputs, outputs);

    if(is_all_inputs_fp16 || is_all_inputs_u8 )
    {
        if(p->use_cudnn_implementation && p->linear_before_reset == 0)
        {
            switch(p->cudnn_implementation_version)
            {
                default:
                case 0:
                case 1:
                    return op_setup_float_cudnn(self, inputs, outputs);
                    /* break; */
                case 2:
                    return op_setup_float_cudnn_v2(self, inputs, outputs);
                    /* break; */
                case 3:
                    p->local->force_input_recurrent_on_NN = TRUE;
                    return op_setup_float_cudnn(self, inputs, outputs);
                    /* break; */
            }
        }
        else
        {
            return op_setup_float(self, inputs, outputs);
        }
    }
    else
    {
        return op_setup_default(self, inputs, outputs);
    }
}

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    int i = 0;

    vsi_nn_internal_init_node_wksp( self );

    self->nn_param.grucell_ovxlib.local = \
        (grucell_ovxlib_local_data_t*)malloc(sizeof(grucell_ovxlib_local_data_t));
    if(self->nn_param.grucell_ovxlib.local)
    {
        memset(self->nn_param.grucell_ovxlib.local, 0x00,
            sizeof(grucell_ovxlib_local_data_t));
        self->nn_param.grucell_ovxlib.local->candidate_activation = VSI_NN_ACT_TANH;
        self->nn_param.grucell_ovxlib.local->gate_activation = VSI_NN_ACT_SIGMOID;
        self->nn_param.grucell_ovxlib.local->force_input_recurrent_on_NN = FALSE;
    }
    else
    {
        status = VSI_FAILURE;
    }

    for(i = 0; i < GRUCELL_QUANTIZE_PARAM_COUNT; i++)
    {
        memset(&self->nn_param.grucell_ovxlib.internal_dtype[i], 0x00,
            sizeof(self->nn_param.grucell_ovxlib.internal_dtype[i]));
        self->nn_param.grucell_ovxlib.internal_dtype[i].qnt_type = VSI_NN_QNT_TYPE_NONE;
        self->nn_param.grucell_ovxlib.internal_dtype[i].vx_type = VSI_NN_TYPE_FLOAT16;
    }

    self->nn_param.grucell_ovxlib.activation = VSI_NN_ACT_TANH;
    self->nn_param.grucell_ovxlib.recurrent_activation = VSI_NN_ACT_SIGMOID;
    self->nn_param.grucell_ovxlib.use_cudnn_implementation = FALSE;
    self->nn_param.grucell_ovxlib.cudnn_implementation_version = 0;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_grucell_ovxlib_param* p = &self->nn_param.grucell_ovxlib;

    vsi_safe_release_tensor(p->local->weights_update);
    vsi_safe_release_tensor(p->local->weights_reset);
    vsi_safe_release_tensor(p->local->weights_z_r);
    vsi_safe_release_tensor(p->local->weights_c);
    vsi_safe_release_tensor(p->local->weights_input);
    vsi_safe_release_tensor(p->local->weights_recurrent);
    vsi_safe_release_tensor(p->local->bias_z);
    vsi_safe_release_tensor(p->local->bias_r);
    vsi_safe_release_tensor(p->local->bias_z_r);
    vsi_safe_release_tensor(p->local->bias_c);
    vsi_nn_internal_deinit_node_wksp( self );
    vsi_nn_safe_free(self->nn_param.grucell_ovxlib.local);

    return status;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GRUCELL_OVXLIB,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ GRUCELL_INPUT_CNT,
    /* output_num */ GRUCELL_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif
