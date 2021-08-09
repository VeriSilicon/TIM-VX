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
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "ops/vsi_nn_op_lstmunit_ovxlib.h"
#include "vsi_nn_internal_node.h"
#include "vsi_nn_rnn_helper.h"
#include "utils/vsi_nn_dtype_util.h"


static vsi_nn_internal_tensor_t* create_tp_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_lstmunit_ovxlib_param* p = &self->nn_param.lstmunit_ovxlib;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    tensor = bias;
    if( !bias || p->local->use_layer_norm || p->local->use_hybrid )
    {
        /* create zero bias for NN/TP */
        tensor1 = vsi_nn_internal_create_zero_bias_tensor(self, &input->attr, &weight->attr, FALSE);
        tensor = tensor1->t;
    }

    vsi_nn_internal_init_tensor_attr(&attr, output_dtype, use_virtual_tensor);
    tensor2 = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_FCL, 0, 0 );
    tmp_inode->node->nn_param.fcl.axis = 0;
    tmp_inode->node->nn_param.fcl.weights = weight->attr.size[1];

    tmp_inode->inputs[0] = input;
    tmp_inode->inputs[1] = weight;
    tmp_inode->inputs[2] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    vsi_nn_internal_setup_node(self, tmp_inode);

    return tensor2;
}

static vsi_nn_internal_tensor_t* create_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias,
    uint32_t kernel_h,
    uint32_t kernel_w,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_lstmunit_ovxlib_param* p = &self->nn_param.lstmunit_ovxlib;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    vsi_nn_internal_tensor_t* reshaped_weight_tensor = NULL;
    uint32_t reshaped_weight_shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_nn_internal_node_t* tmp_inode = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    tensor = bias;
    if( !bias || p->local->use_layer_norm || p->local->use_hybrid )
    {
        /* create zero bias for NN/TP */
        tensor1 = vsi_nn_internal_create_zero_bias_tensor(self, &input->attr, &weight->attr, FALSE);
        tensor = tensor1->t;
    }

    vsi_nn_internal_init_tensor_attr(&attr, output_dtype, use_virtual_tensor);
    tensor2 = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    reshaped_weight_shape[3] = weight->attr.size[1];
    reshaped_weight_shape[2] = weight->attr.size[0] / ( kernel_h * kernel_w );
    reshaped_weight_shape[1] = kernel_h;
    reshaped_weight_shape[0] = kernel_w;

    memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = weight->attr.vtl;
    attr.is_const = FALSE; //weight->attr.is_const;
    memcpy( &attr.dtype, &weight->attr.dtype, sizeof(attr.dtype) );
    memcpy( &attr.size, &reshaped_weight_shape, sizeof(attr.size));
    reshaped_weight_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    vsi_nn_ReshapeTensor( self->graph, weight, reshaped_weight_tensor->t, reshaped_weight_shape, 4 );

    tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_CONV2D, 0, 0 );
    tmp_inode->node->nn_param.conv2d.ksize[0] = kernel_w;
    tmp_inode->node->nn_param.conv2d.ksize[1] = kernel_h;
    tmp_inode->node->nn_param.conv2d.stride[0] = 1;
    tmp_inode->node->nn_param.conv2d.stride[1] = 1;
    tmp_inode->node->nn_param.conv2d.pad[0] = 0;
    tmp_inode->node->nn_param.conv2d.pad[1] = 0;
    tmp_inode->node->nn_param.conv2d.pad[2] = 0;
    tmp_inode->node->nn_param.conv2d.pad[3] = 0;
    tmp_inode->node->nn_param.conv2d.group = 1;
    tmp_inode->node->nn_param.conv2d.dilation[0] = 1;
    tmp_inode->node->nn_param.conv2d.dilation[1] = 1;
    tmp_inode->node->nn_param.conv2d.weights = weight->attr.size[1];

    tmp_inode->inputs[0] = input;
    tmp_inode->inputs[1] = reshaped_weight_tensor->t;
    tmp_inode->inputs[2] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    vsi_nn_internal_setup_node(self, tmp_inode);

    return tensor2;
}

static void create_peephole
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_internal_tensor_t ** input_fc,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* input_tensor0 = NULL;
    vsi_nn_internal_tensor_t* input_tensor1 = NULL;
    vsi_nn_internal_node_t* curr = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;
    memcpy(&(attr.dtype), &((*input_fc)->t->attr.dtype), sizeof(vsi_nn_dtype_t));
    input_tensor0 = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
    /* create internal nodes */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_MULTIPLY, 0, 0 );
    curr->node->nn_param.multiply.scale = 1.0f;
    curr->node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    curr->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    curr->inputs[0] = input;
    curr->inputs[1] = weight;
    curr->outputs[0] = input_tensor0->t;
    vsi_nn_internal_setup_node(self, curr);
    input_tensor1 = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
    /* create internal nodes */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_ADD, 0, 0 );
    curr->inputs[0] = (*input_fc)->t;
    curr->inputs[1] = input_tensor0->t;
    curr->outputs[0] = input_tensor1->t;
    vsi_nn_internal_setup_node(self, curr);
    *input_fc = input_tensor1;
}

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_lstmunit_ovxlib_param* p = &self->nn_param.lstmunit_ovxlib;

    /* setup lstmunit output tensors' shape */
    /* output */
    if(VSI_NN_DIM_AUTO == outputs[LSTMUNIT_OUTPUT_OUTPUT]->attr.dim_num)
    {
        if(p->local->use_projection) /* enable projection_weight */
        {
            /* output_size */
            outputs[LSTMUNIT_OUTPUT_OUTPUT]->attr.size[0] = inputs[LSTMUNIT_INPUT_WEIGHT_PROJ]->attr.size[1];
        }
        else /* disable projection_weight */
        {
            /* num_units */
            outputs[LSTMUNIT_OUTPUT_OUTPUT]->attr.size[0] = inputs[LSTMUNIT_INPUT_WEIGHT_I2F]->attr.size[1];
        }
        /* batch_size */
        outputs[LSTMUNIT_OUTPUT_OUTPUT]->attr.size[1] = inputs[LSTMUNIT_INPUT_INPUT]->attr.size[1];
        outputs[LSTMUNIT_OUTPUT_OUTPUT]->attr.dim_num = inputs[LSTMUNIT_INPUT_INPUT]->attr.dim_num;
    }

    /* output_state_out */
    if(VSI_NN_DIM_AUTO == outputs[LSTMUNIT_OUTPUT_H_STATE]->attr.dim_num)
    {
        outputs[LSTMUNIT_OUTPUT_H_STATE]->attr.dim_num = outputs[LSTMUNIT_OUTPUT_OUTPUT]->attr.dim_num;
        memcpy( outputs[LSTMUNIT_OUTPUT_H_STATE]->attr.size, outputs[LSTMUNIT_OUTPUT_OUTPUT]->attr.size,
            VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ) );
    }

    /* cell_state_out */
    if(VSI_NN_DIM_AUTO == outputs[LSTMUNIT_OUTPUT_C_STATE]->attr.dim_num)
    {
        outputs[LSTMUNIT_OUTPUT_C_STATE]->attr.dim_num = outputs[LSTMUNIT_OUTPUT_OUTPUT]->attr.dim_num;
        outputs[LSTMUNIT_OUTPUT_C_STATE]->attr.size[0] = inputs[LSTMUNIT_INPUT_WEIGHT_I2F]->attr.size[1];
        outputs[LSTMUNIT_OUTPUT_C_STATE]->attr.size[1] = inputs[LSTMUNIT_INPUT_INPUT]->attr.size[1];
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
    vsi_nn_lstmunit_ovxlib_param* p = &self->nn_param.lstmunit_ovxlib;
    vsi_nn_tensor_attr_t attr;
    vsi_bool is_input_fc_on_tp = FALSE;
    vsi_bool is_recurrent_fc_on_tp = FALSE;
    vsi_nn_internal_tensor_t* input_tensor = NULL;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_internal_tensor_t* recurrent_input_tensor = NULL;
    vsi_nn_internal_tensor_t* input_fc_outputs[LSTMUNIT_IFCO_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* aux_input_fc_outputs[LSTMUNIT_IFCO_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* input_add_aux_input_fc_outputs[LSTMUNIT_IFCO_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* recurrent_fc_outputs[LSTMUNIT_IFCO_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* layernorm_outputs[LSTMUNIT_IFCO_GATE_COUNT] = { NULL };
    vsi_nn_tensor_t* bias_tensors[LSTMUNIT_IFCO_GATE_COUNT] = { NULL };
    vsi_nn_tensor_t* zero_bias_tensor = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    int32_t ifco_start_index = 0;
    uint32_t kernel_h = 1;
    uint32_t kernel_w = 1;
    int32_t i = 0;
    vsi_bool use_virtual_tensor = TRUE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_node_wksp( self );

    memset( p->local, 0x00, sizeof(vsi_nn_lstmunit_ovxlib_lcl_data_t) );
    memset( &attr, 0x00, sizeof( attr ) );
    p->local->use_cifg = ( NULL == inputs[LSTMUNIT_INPUT_WEIGHT_I2I] );
    p->local->use_layer_norm = ( NULL != inputs[LSTMUNIT_INPUT_LAYERNORM_F] );
    p->local->use_projection = ( NULL != inputs[LSTMUNIT_INPUT_WEIGHT_PROJ] );
    p->local->use_projection_bias = ( NULL != inputs[LSTMUNIT_INPUT_BIAS_PROJ] );
    p->local->multi_batch = ( inputs[LSTMUNIT_INPUT_INPUT]->attr.size[1] > 1 );
    p->local->use_peephole = ( NULL != inputs[LSTMUNIT_INPUT_WEIGHT_C2O] );
    ifco_start_index = p->local->use_cifg ? 1 : 0;
    if( inputs[LSTMUNIT_INPUT_WEIGHT_I2F]->attr.dtype.qnt_type
        != inputs[LSTMUNIT_INPUT_BIAS_F]->attr.dtype.qnt_type )
    {
        p->local->use_hybrid = TRUE;
    }

    if( inputs[LSTMUNIT_INPUT_INPUT]->attr.dtype.qnt_type
        != inputs[LSTMUNIT_INPUT_WEIGHT_I2F]->attr.dtype.qnt_type)
    {
        /* input and input weights have different qtype, only TP can do this operation */
        is_input_fc_on_tp = TRUE;
    }
    else if( inputs[LSTMUNIT_INPUT_INPUT]->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if input's shape is not aligned to 64-byte */
        is_input_fc_on_tp = TRUE;
    }

    if( inputs[LSTMUNIT_INPUT_H_STATE]->attr.dtype.qnt_type
        != inputs[LSTMUNIT_INPUT_WEIGHT_R2F]->attr.dtype.qnt_type)
    {
        /* recurrent and recurrent weights have different qtype, only TP can do this operation */
        is_recurrent_fc_on_tp = TRUE;
    }
    else if( inputs[LSTMUNIT_INPUT_H_STATE]->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if inputs' shape is not aligned to 64-byte */
        is_recurrent_fc_on_tp = TRUE;
    }

    /* if both input fc and recurrent fc could be executed on NN, offloads one to TP*/
    if( !is_input_fc_on_tp && !is_recurrent_fc_on_tp )
    {
        is_input_fc_on_tp = TRUE;
    }

    setup_op_shapes(self, inputs, outputs);

    for( i = 0; i < LSTMUNIT_IFCO_GATE_COUNT; i++)
    {
        if( p->local->use_layer_norm || p->local->use_hybrid )
        {
            bias_tensors[i] = NULL;
        }
        else
        {
            bias_tensors[i] = inputs[LSTMUNIT_INPUT_BIAS_I + i];
        }
    }

    /* Input FC */
    if( is_input_fc_on_tp )
    {
        /* tp */
        for( i = ifco_start_index; i < LSTMUNIT_IFCO_GATE_COUNT; i++)
        {
            input_fc_outputs[i] = create_tp_fc(self,
                                                inputs[LSTMUNIT_INPUT_INPUT],
                                                inputs[LSTMUNIT_INPUT_WEIGHT_I2I + i],
                                                bias_tensors[i],
                                                &p->internal_dtype[LSTMUNIT_QUANTIZE_PARAM_I2I + i],
                                                use_virtual_tensor);
        }
        if (inputs[LSTMUNIT_INPUT_AUX_INPUT] != NULL)
        {
            for( i = ifco_start_index; i < LSTMUNIT_IFCO_GATE_COUNT; i++)
            {
                aux_input_fc_outputs[i] = create_tp_fc(self,
                                                    inputs[LSTMUNIT_INPUT_AUX_INPUT],
                                                    inputs[LSTMUNIT_INPUT_AUX_WEIGHT_I2I + i],
                                                    NULL,
                                                    &p->internal_dtype_aux[LSTMUNIT_QUANTIZE_PARAM_AUX_I2I + i],
                                                    use_virtual_tensor);
            }
        }
    }
    else
    {
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
            inputs[LSTMUNIT_INPUT_INPUT]->attr.size[0], &kernel_h, &kernel_w);
        input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, inputs[LSTMUNIT_INPUT_INPUT],
                                                p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);

        for( i = ifco_start_index; i < LSTMUNIT_IFCO_GATE_COUNT; i++)
        {
            vsi_nn_internal_tensor_t* tmp = create_nn_fc(self,
                                                input_tensor->t,
                                                inputs[LSTMUNIT_INPUT_WEIGHT_I2I + i],
                                                bias_tensors[i],
                                                kernel_h, kernel_w,
                                                &p->internal_dtype[LSTMUNIT_QUANTIZE_PARAM_I2I + i],
                                                use_virtual_tensor);
            /* transpose and reshape output */
            input_fc_outputs[i] = vsi_nn_rnn_process_output_for_nn_fc(self,
                tmp->t, p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
        }
        if (inputs[LSTMUNIT_INPUT_AUX_INPUT] != NULL)
        {
            /* reshape and transpose input */
            vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
                inputs[LSTMUNIT_INPUT_AUX_INPUT]->attr.size[0], &kernel_h, &kernel_w);
            input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, inputs[LSTMUNIT_INPUT_AUX_INPUT],
                                                    p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);

            for( i = ifco_start_index; i < LSTMUNIT_IFCO_GATE_COUNT; i++)
            {
                vsi_nn_internal_tensor_t* tmp = create_nn_fc(self,
                                                    input_tensor->t,
                                                    inputs[LSTMUNIT_INPUT_AUX_WEIGHT_I2I + i],
                                                    NULL,
                                                    kernel_h, kernel_w,
                                                    &p->internal_dtype_aux[LSTMUNIT_QUANTIZE_PARAM_AUX_I2I + i],
                                                    use_virtual_tensor);
                /* transpose and reshape output */
                aux_input_fc_outputs[i] = vsi_nn_rnn_process_output_for_nn_fc(self,
                    tmp->t, p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
            }
        }
    }

    if (inputs[LSTMUNIT_INPUT_AUX_INPUT] != NULL)
    {
        for( i = ifco_start_index; i < LSTMUNIT_IFCO_GATE_COUNT; i++)
        {
            input_add_aux_input_fc_outputs[i] = vsi_nn_rnn_create_tensor_add(self,
                                                    input_fc_outputs[i]->t,
                                                    aux_input_fc_outputs[i]->t,
                                                    &p->internal_dtype[LSTMUNIT_QUANTIZE_PARAM_I2I],
                                                    use_virtual_tensor);
            input_fc_outputs[i] = input_add_aux_input_fc_outputs[i];
        }
    }

    /* Recurrent FC */
    if( is_recurrent_fc_on_tp )
    {
        for( i = ifco_start_index; i < LSTMUNIT_IFCO_GATE_COUNT; i++)
        {
            recurrent_fc_outputs[i] = create_tp_fc(self,
                                                inputs[LSTMUNIT_INPUT_H_STATE],
                                                inputs[LSTMUNIT_INPUT_WEIGHT_R2I + i],
                                                NULL,
                                                &p->internal_dtype[LSTMUNIT_QUANTIZE_PARAM_R2I + i],
                                                use_virtual_tensor);
        }
    }
    else
    {
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local->multi_batch,
            inputs[LSTMUNIT_INPUT_H_STATE]->attr.size[0], &kernel_h, &kernel_w);
        recurrent_input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self,
            inputs[LSTMUNIT_INPUT_H_STATE], p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);

        for( i = ifco_start_index; i < LSTMUNIT_IFCO_GATE_COUNT; i++)
        {
            vsi_nn_internal_tensor_t* tmp = create_nn_fc(self,
                                                recurrent_input_tensor->t,
                                                inputs[LSTMUNIT_INPUT_WEIGHT_R2I + i],
                                                NULL,
                                                kernel_h, kernel_w,
                                                &p->internal_dtype[LSTMUNIT_QUANTIZE_PARAM_R2I + i],
                                                use_virtual_tensor);
            /* transpose and reshape output */
            recurrent_fc_outputs[i] = vsi_nn_rnn_process_output_for_nn_fc(self,
                tmp->t, p->local->multi_batch, kernel_h, kernel_w, use_virtual_tensor);
        }
    }

    if (p->local->use_peephole)
    {
        /* update input gate */
        if (!p->local->use_cifg)
        {
            create_peephole(self, inputs[LSTMUNIT_INPUT_C_STATE],
                inputs[LSTMUNIT_INPUT_WEIGHT_C2I], &(input_fc_outputs[0]),
                use_virtual_tensor);
        }

        /* update forget gate */
        create_peephole(self, inputs[LSTMUNIT_INPUT_C_STATE],
                inputs[LSTMUNIT_INPUT_WEIGHT_C2F], &(input_fc_outputs[1]),
                use_virtual_tensor);

        /* update output gate */
        create_peephole(self, inputs[LSTMUNIT_INPUT_C_STATE],
                inputs[LSTMUNIT_INPUT_WEIGHT_C2O], &(input_fc_outputs[3]),
                use_virtual_tensor);
    }

    /* layernorm */
    if( p->local->use_layer_norm )
    {
        for( i = ifco_start_index; i < LSTMUNIT_IFCO_GATE_COUNT; i++ )
        {
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
            attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
            input_tensor = vsi_nn_internal_new_tensor(self, &attr, 0.0f);

            /* create internal nodes */
            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_TENSOR_ADD_MEAN_STDDEV_NORM, 0, 0 );
            curr->node->nn_param.tensor_add_mean_stddev_norm.eps = (float)1e-8;
            curr->inputs[0] = input_fc_outputs[i]->t;
            curr->inputs[1] = recurrent_fc_outputs[i]->t;
            curr->outputs[0] = input_tensor->t;
            vsi_nn_internal_setup_node(self, curr);

            layernorm_outputs[i] = input_tensor;
        }
    }

    /* activations */
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_LSTMUNIT_ACTIVATION, 0, 0 );
    curr->node->nn_param.lstmunit_activation.cell_clip = p->cell_clip;
    curr->node->nn_param.lstmunit_activation.proj_clip = p->proj_clip;
    curr->node->nn_param.lstmunit_activation.forget_bias = p->forget_bias;
    curr->node->nn_param.lstmunit_activation.is_cifg = (uint8_t)p->local->use_cifg;
    curr->node->nn_param.lstmunit_activation.is_projection = (uint8_t)p->local->use_projection;
    curr->node->nn_param.lstmunit_activation.is_layer_norm = (uint8_t)p->local->use_layer_norm;
    curr->node->nn_param.lstmunit_activation.is_peephole = FALSE;
    curr->node->nn_param.lstmunit_activation.is_hybrid = (uint8_t)p->local->use_hybrid;
    curr->node->nn_param.lstmunit_activation.recurrent_activation = p->recurrent_activation;

    curr->inputs[LSTMUNIT_ACT_CSTATE_IN] = inputs[LSTMUNIT_INPUT_C_STATE];
    for( i = ifco_start_index; i < LSTMUNIT_IFCO_GATE_COUNT; i++ )
    {
        if( p->local->use_layer_norm || p->local->use_hybrid )
        {
            curr->inputs[LSTMUNIT_ACT_DATA_BI + i] = inputs[LSTMUNIT_INPUT_BIAS_I + i];
        }

        if( p->local->use_layer_norm )
        {
            /* Pass layernorm weights to VSI_NN_OP_LSTMUNIT_ACTIVATION */
            curr->inputs[LSTMUNIT_ACT_LN_WI + i] = inputs[LSTMUNIT_INPUT_LAYERNORM_I + i];
            curr->inputs[LSTMUNIT_ACT_INPUT_FC_I + i] = layernorm_outputs[i]->t;
            curr->inputs[LSTMUNIT_ACT_HSTATE_FC_I + i] = NULL;
        }
        else
        {
            curr->inputs[LSTMUNIT_ACT_LN_WI + i] = NULL;
            curr->inputs[LSTMUNIT_ACT_INPUT_FC_I + i] = input_fc_outputs[i]->t;
            curr->inputs[LSTMUNIT_ACT_HSTATE_FC_I + i] = recurrent_fc_outputs[i]->t;
        }
    }

    if( p->local->use_projection )
    {
        /* create virtual tensor for activations' output0 */
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));

        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;

        if( p->local->multi_batch &&
            inputs[LSTMUNIT_INPUT_WEIGHT_PROJ]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
        {
            /* projection FC on NN requires quantized input */
            attr.dtype.scale = (float)0.007866097716834601;
            attr.dtype.zero_point = 128;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
            attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
        }
        else
        {
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
            attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        }
        output_tensor = vsi_nn_internal_new_tensor(self, &attr, 0.0f);

        curr->outputs[LSTMUNIT_ACT_OUTPUT] = output_tensor->t;
        curr->outputs[LSTMUNIT_ACT_CSTATE_OUT] = outputs[LSTMUNIT_OUTPUT_C_STATE];
        curr->outputs[LSTMUNIT_ACT_HSTATE_OUT] = NULL;
    }
    else
    {
        /* kernel VSI_NN_OP_LSTMUNIT_ACTIVATION has 3 outputs if no projection layer behind */
        curr->outputs[LSTMUNIT_ACT_OUTPUT] = outputs[LSTMUNIT_OUTPUT_OUTPUT];
        curr->outputs[LSTMUNIT_ACT_CSTATE_OUT] = outputs[LSTMUNIT_OUTPUT_C_STATE];
        curr->outputs[LSTMUNIT_ACT_HSTATE_OUT] = outputs[LSTMUNIT_OUTPUT_H_STATE];
    }
    vsi_nn_internal_setup_node(self, curr); /* setup for VSI_NN_OP_LSTMUNIT_ACTIVATION */

    if( p->local->use_projection )
    {
        if ( p->local->use_hybrid && p->local->use_projection_bias )
        {
            vsi_bool use_virtual_tensor = inputs[LSTMUNIT_INPUT_BIAS_PROJ]->attr.vtl;
            input_tensor = vsi_nn_internal_create_zero_bias_tensor(self, &output_tensor->t->attr,
                &inputs[LSTMUNIT_INPUT_WEIGHT_PROJ]->attr, FALSE);
            zero_bias_tensor = input_tensor->t;

            if (use_virtual_tensor)
            {
                curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
                curr->inputs[0] = inputs[LSTMUNIT_INPUT_BIAS_PROJ];
                curr->outputs[0] = zero_bias_tensor;

                vsi_nn_internal_setup_node(self, curr);
            }
            else
            {
                vsi_nn_ConvertTensor(self->graph, inputs[1], zero_bias_tensor);
            }
        }
        else if ( p->local->use_hybrid || !p->local->use_projection_bias )
        {
            input_tensor = vsi_nn_internal_create_zero_bias_tensor(self, &output_tensor->t->attr,
                &inputs[LSTMUNIT_INPUT_WEIGHT_PROJ]->attr, FALSE);
            zero_bias_tensor = input_tensor->t;
        }
        else
        {
            zero_bias_tensor = inputs[LSTMUNIT_INPUT_BIAS_PROJ];
        }

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_FCL, 0, 0 );
        curr->node->nn_param.fcl.axis = 0;
        curr->node->nn_param.fcl.weights = inputs[LSTMUNIT_INPUT_WEIGHT_PROJ]->attr.size[1];

        curr->inputs[0] = output_tensor->t;
        curr->inputs[1] = inputs[LSTMUNIT_INPUT_WEIGHT_PROJ];
        curr->inputs[2] = zero_bias_tensor;

        /* Save output to h_state first and copy to output */
        if( p->local->use_hybrid && p->local->use_projection_bias )
        {
            vsi_nn_internal_init_tensor_attr(&attr,
                &outputs[LSTMUNIT_OUTPUT_H_STATE]->attr.dtype, use_virtual_tensor);
            output_tensor = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
            curr->outputs[0] = output_tensor->t;
        }
        else
        {
            curr->outputs[0] = outputs[LSTMUNIT_OUTPUT_H_STATE];
        }

        vsi_nn_internal_setup_node(self, curr);

        /* copy h_state to output */
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
        curr->inputs[0] = outputs[LSTMUNIT_OUTPUT_H_STATE];
        curr->outputs[0] = outputs[LSTMUNIT_OUTPUT_OUTPUT];
        vsi_nn_internal_setup_node(self, curr);
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_safe_free(self->nn_param.lstmunit_ovxlib.local);
    vsi_nn_safe_free(self->nn_param.lstmunit_ovxlib.internal_dtype_aux);
    vsi_nn_internal_deinit_node_wksp( self );

    return status;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.lstmunit_ovxlib.activation = VSI_NN_ACT_TANH;
    self->nn_param.lstmunit_ovxlib.recurrent_activation = VSI_NN_ACT_SIGMOID;
    self->nn_param.lstmunit_ovxlib.local = (vsi_nn_lstmunit_ovxlib_lcl_data_t *)
        malloc(sizeof(vsi_nn_lstmunit_ovxlib_lcl_data_t));
    memset(self->nn_param.lstmunit_ovxlib.local, 0,
        sizeof(vsi_nn_lstmunit_ovxlib_lcl_data_t));
    self->nn_param.lstmunit_ovxlib.internal_dtype_aux = (vsi_nn_dtype_t *)
        malloc(sizeof(vsi_nn_dtype_t) * LSTMUNIT_QUANTIZE_PARAM_AUX_COUNT);
    memset(self->nn_param.lstmunit_ovxlib.internal_dtype_aux, 0,
        sizeof(vsi_nn_dtype_t) * LSTMUNIT_QUANTIZE_PARAM_AUX_COUNT);

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LSTMUNIT_OVXLIB,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ LSTMUNIT_INPUT_CNT,
    /* output_num */ LSTMUNIT_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif
