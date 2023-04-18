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
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_rnn_prv.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_rnn_helper.h"

vsi_bool vsi_nn_rnn_find_best_kernel_size
    (
    vsi_bool multi_batch,
    uint32_t input_size,
    uint32_t* p_kernel_h,
    uint32_t* p_kernel_w
    )
{
    uint32_t kernel_h = 1;
    uint32_t kernel_w = 1;

    if( multi_batch)
    {
        /* batch FC only be converted to 1x1 or 1xN conv */
        /* try 1xN */
        kernel_h = 7;
        while( input_size % kernel_h != 0 )
        {
            kernel_h--;
        }
    }
    else
    {
        /* try NxN */
        if( !multi_batch )
        {
            #if( !(defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32)) )
            /* try NxN conv */
            kernel_h = 8;
            while( input_size % (kernel_h * kernel_h) != 0 )
            {
                kernel_h--;
            }
            #endif
        }

        if( kernel_h > 1 )
        {
            kernel_w = kernel_h;
        }
        else
        {
            /* Only 1x1 found, try 1xN */
            kernel_h = 7;
            while( input_size % kernel_h != 0 )
            {
                kernel_h--;
            }
            kernel_w = 1;
        }
    }

    VSILOGD("Use kernel_h: %d, kernel_w: %d to convert FC", kernel_h, kernel_w);
    if( p_kernel_h )
    {
        *p_kernel_h = kernel_h;
    }

    if( p_kernel_w )
    {
        *p_kernel_w = kernel_w;
    }

    return TRUE;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_process_input_for_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_bool multi_batch,
    uint32_t kernel_h,
    uint32_t kernel_w,
    int32_t use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    vsi_size_t* reshape_in_size = NULL;
    uint32_t* permute_in_perm = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, &input->attr.dtype, use_virtual_tensor);
    tensor1 = vsi_nn_internal_new_tensor(self, &attr, 0.0f);

    tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_RESHAPE2, 0, 0 );
    reshape_in_size = (vsi_size_t*)vsi_nn_internal_new_node_param(tmp_inode, 4 * sizeof(vsi_size_t));

    reshape_in_size[3] = input->attr.size[1];
    reshape_in_size[2] = input->attr.size[0] / (kernel_h * kernel_w);
    reshape_in_size[1] = kernel_h;
    reshape_in_size[0] = kernel_w;

    tmp_inode->node->nn_param.reshape2.size = reshape_in_size;
    tmp_inode->node->nn_param.reshape2.dim_num = 4;
    tmp_inode->inputs[0] = input;
    tmp_inode->outputs[0] = tensor1->t;
    ret = vsi_nn_internal_setup_node(self, tmp_inode);

    if( multi_batch )
    {
        vsi_size_t reshape_size[4] = { 0 };
        vsi_size_t c = 0, h = 0;
        vsi_nn_internal_tensor_t* tensor0 = NULL;
        h = tensor1->t->attr.size[2];
        c = tensor1->t->attr.size[1];

        reshape_size[2] = tensor1->t->attr.size[3];
        reshape_size[1] = -1;
        reshape_size[0] = tensor1->t->attr.size[0];
        tensor0 = vsi_nn_rnn_create_reshape(self, tensor1->t, NULL, reshape_size, 3, use_virtual_tensor);

        tensor2 = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
        tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_PERMUTE, 0, 0 );
        permute_in_perm = (uint32_t *)vsi_nn_internal_new_node_param(tmp_inode, 3 * sizeof(uint32_t));
        permute_in_perm[0] = 2;
        permute_in_perm[1] = 1;
        permute_in_perm[2] = 0;
        tmp_inode->node->nn_param.permute.perm = permute_in_perm;
        tmp_inode->node->nn_param.permute.dim_num = 3;
        tmp_inode->inputs[0] = tensor0->t;
        tmp_inode->outputs[0] = tensor2->t;
        vsi_nn_internal_setup_node(self, tmp_inode);

        reshape_size[3] = tensor2->t->attr.size[2];
        reshape_size[2] = h;
        reshape_size[1] = c;
        reshape_size[0] = tensor2->t->attr.size[0];
        tensor0 = vsi_nn_rnn_create_reshape(self, tensor2->t, NULL, reshape_size, 4, use_virtual_tensor);

        tensor1 = tensor0;
    }
    if (!ret)
    {
        tensor1 = NULL;
    }

    return tensor1;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_process_output_for_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_bool multi_batch,
    uint32_t kernel_h,
    uint32_t kernel_w,
    int32_t use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    vsi_size_t* reshape_in_size = NULL;
    uint32_t* permute_in_perm = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;
    vsi_nn_tensor_t* tensor = input;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, &input->attr.dtype, use_virtual_tensor);

    if( multi_batch )
    {
        vsi_size_t reshape_size[4] = { 0 };
        vsi_size_t c = 0, h = 0;
        vsi_nn_internal_tensor_t* tensor0 = NULL;
        h = tensor->attr.size[2];
        c = tensor->attr.size[1];

        reshape_size[2] = tensor->attr.size[3];
        reshape_size[1] = -1;
        reshape_size[0] = tensor->attr.size[0];
        tensor0 = vsi_nn_rnn_create_reshape(self, tensor, NULL, reshape_size, 3, use_virtual_tensor);

        tensor1 = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
        tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_PERMUTE, 0, 0 );
        permute_in_perm = (uint32_t *)vsi_nn_internal_new_node_param(tmp_inode, 3 * sizeof(uint32_t));

        permute_in_perm[0] = 2;
        permute_in_perm[1] = 1;
        permute_in_perm[2] = 0;

        tmp_inode->node->nn_param.permute.perm = permute_in_perm;
        tmp_inode->node->nn_param.permute.dim_num = 3;
        tmp_inode->inputs[0] = tensor0->t;
        tmp_inode->outputs[0] = tensor1->t;
        vsi_nn_internal_setup_node(self, tmp_inode);

        reshape_size[3] = tensor1->t->attr.size[2];
        reshape_size[2] = h;
        reshape_size[1] = c;
        reshape_size[0] = tensor1->t->attr.size[0];
        tensor0 = vsi_nn_rnn_create_reshape(self, tensor1->t, NULL, reshape_size, 4, use_virtual_tensor);

        tensor = tensor0->t;
    }

    tensor2 = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
    tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_RESHAPE2, 0, 0 );
    reshape_in_size = (vsi_size_t *)vsi_nn_internal_new_node_param(tmp_inode, 4 * sizeof(vsi_size_t));

    reshape_in_size[1] = tensor->attr.size[3];
    reshape_in_size[0] = tensor->attr.size[2];

    tmp_inode->node->nn_param.reshape2.size = reshape_in_size;
    tmp_inode->node->nn_param.reshape2.dim_num = 2;
    tmp_inode->inputs[0] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    ret = vsi_nn_internal_setup_node(self, tmp_inode);
    if (!ret)
    {
        tensor2 = NULL;
    }

    return tensor2;
}

vsi_bool vsi_nn_rnn_process_output_for_nn_fc2
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    vsi_bool multi_batch,
    uint32_t kernel_h,
    uint32_t kernel_w,
    int32_t use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_size_t* reshape_in_size = NULL;
    uint32_t* permute_in_perm = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;
    vsi_nn_tensor_t* tensor = input;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, &input->attr.dtype, use_virtual_tensor);

    if( multi_batch )
    {
        vsi_size_t reshape_size[4] = { 0 };
        vsi_size_t c = 0, h = 0;
        vsi_nn_internal_tensor_t* tensor0 = NULL;
        h = tensor->attr.size[2];
        c = tensor->attr.size[1];

        reshape_size[2] = tensor->attr.size[3];
        reshape_size[1] = -1;
        reshape_size[0] = tensor->attr.size[0];
        tensor0 = vsi_nn_rnn_create_reshape(self, tensor, NULL, reshape_size, 3, use_virtual_tensor);

        tensor1 = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
        tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_PERMUTE, 0, 0 );
        permute_in_perm = (uint32_t *)vsi_nn_internal_new_node_param(tmp_inode, 3 * sizeof(uint32_t));

        permute_in_perm[0] = 2;
        permute_in_perm[1] = 1;
        permute_in_perm[2] = 0;

        tmp_inode->node->nn_param.permute.perm = permute_in_perm;
        tmp_inode->node->nn_param.permute.dim_num = 3;
        tmp_inode->inputs[0] = tensor0->t;
        tmp_inode->outputs[0] = tensor1->t;
        vsi_nn_internal_setup_node(self, tmp_inode);

        reshape_size[3] = tensor1->t->attr.size[2];
        reshape_size[2] = h;
        reshape_size[1] = c;
        reshape_size[0] = tensor1->t->attr.size[0];
        tensor0 = vsi_nn_rnn_create_reshape(self, tensor1->t, NULL, reshape_size, 4, use_virtual_tensor);

        tensor = tensor0->t;
    }

    tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_RESHAPE2, 0, 0 );
    reshape_in_size = (vsi_size_t*)vsi_nn_internal_new_node_param(tmp_inode, 4 * sizeof(vsi_size_t));

    reshape_in_size[1] = tensor->attr.size[3];
    reshape_in_size[0] = tensor->attr.size[2];

    tmp_inode->node->nn_param.reshape2.size = reshape_in_size;
    tmp_inode->node->nn_param.reshape2.dim_num = 2;
    tmp_inode->inputs[0] = tensor;
    tmp_inode->outputs[0] = output;
    vsi_nn_internal_setup_node(self, tmp_inode);

    return TRUE;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_tp_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    tensor = bias;
    if( !bias )
    {
        /* create zero bias for NN/TP */
        tensor1 = vsi_nn_internal_create_zero_bias_tensor(
            self, &input->attr, &weight->attr, VSI_NN_OP_FCL, FALSE);
        tensor = tensor1->t;
    }
    vsi_nn_internal_init_tensor_attr(&attr, output_dtype, use_virtual_tensor);
    tensor2 = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_FCL, 0, 0 );
    tmp_inode->node->nn_param.fcl.axis = 0;
    tmp_inode->node->nn_param.fcl.weights = (uint32_t)weight->attr.size[1];

    tmp_inode->inputs[0] = input;
    tmp_inode->inputs[1] = weight;
    tmp_inode->inputs[2] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    ret = vsi_nn_internal_setup_node(self, tmp_inode);
    if (!ret)
    {
        tensor2 = NULL;
    }

    return tensor2;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_nn_fc
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
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    vsi_nn_internal_tensor_t* reshaped_weight_tensor = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    tensor = bias;
    if( !bias )
    {
        /* create zero bias for NN/TP */
        tensor1 = vsi_nn_internal_create_zero_bias_tensor(
            self, &input->attr, &weight->attr, VSI_NN_OP_FCL, FALSE);
        tensor = tensor1->t;
    }

    vsi_nn_internal_init_tensor_attr(&attr, output_dtype, use_virtual_tensor);
    tensor2 = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    reshaped_weight_tensor = vsi_nn_rnn_prepare_weight_for_nn_fc(self, weight, kernel_h, kernel_w);

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
    tmp_inode->node->nn_param.conv2d.weights = (uint32_t)weight->attr.size[1];

    tmp_inode->inputs[0] = input;
    tmp_inode->inputs[1] = reshaped_weight_tensor->t;
    tmp_inode->inputs[2] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    ret = vsi_nn_internal_setup_node(self, tmp_inode);
    if (!ret)
    {
        tensor2 = NULL;
    }

    return tensor2;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_prepare_weight_for_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * weight,
    uint32_t kernel_h,
    uint32_t kernel_w
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* reshaped_weight_tensor = NULL;
    vsi_size_t reshaped_weight_shape[VSI_NN_MAX_DIM_NUM] = { 0 };

    reshaped_weight_shape[3] = weight->attr.size[1];
    reshaped_weight_shape[2] = weight->attr.size[0] / ( kernel_h * kernel_w );
    reshaped_weight_shape[1] = kernel_h;
    reshaped_weight_shape[0] = kernel_w;

    memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = weight->attr.vtl;
    attr.is_const = FALSE;
    memcpy( &attr.dtype, &weight->attr.dtype, sizeof(attr.dtype));
    memcpy( &attr.size, &reshaped_weight_shape, sizeof(attr.size));
    reshaped_weight_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    vsi_nn_ReshapeTensor( self->graph, weight, reshaped_weight_tensor->t, reshaped_weight_shape, 4 );

    reshaped_weight_tensor->t->attr.is_const = weight->attr.is_const;
    if(reshaped_weight_tensor->t->attr.is_const)
    {
        vsi_nn_SetTensorAttr(reshaped_weight_tensor->t, VSI_NN_TENSOR_ATTR_CONST);
    }

    return reshaped_weight_tensor;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_nn_fc_relu
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias,
    uint32_t kernel_h,
    uint32_t kernel_w,
    vsi_bool has_relu,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    vsi_nn_internal_tensor_t* reshaped_weight_tensor = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    tensor = bias;
    if( !bias )
    {
        /* create zero bias for NN/TP */
        tensor1 = vsi_nn_internal_create_zero_bias_tensor(
            self, &input->attr, &weight->attr, VSI_NN_OP_FCL, FALSE);
        tensor = tensor1->t;
    }

    vsi_nn_internal_init_tensor_attr(&attr, output_dtype, use_virtual_tensor);
    tensor2 = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    reshaped_weight_tensor = vsi_nn_rnn_prepare_weight_for_nn_fc(self, weight, kernel_h, kernel_w);

    tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_CONV_RELU, 0, 0 );
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
    tmp_inode->node->nn_param.conv2d.weights = (uint32_t)weight->attr.size[1];
    tmp_inode->node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    tmp_inode->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    tmp_inode->node->vx_param.has_relu = has_relu;
    tmp_inode->node->vx_param.down_scale_size_rounding =
        VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    tmp_inode->inputs[0] = input;
    tmp_inode->inputs[1] = reshaped_weight_tensor->t;
    tmp_inode->inputs[2] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    ret = vsi_nn_internal_setup_node(self, tmp_inode);
    if (!ret)
    {
        tensor2 = NULL;
    }

    return tensor2;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_tensor_add
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
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, output_dtype, use_virtual_tensor);
    tensor1 = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_internal_new_node(self, VSI_NN_OP_ADD, 0, 0 );

    tmp_inode->inputs[0] = input1;
    tmp_inode->inputs[1] = input2;
    tmp_inode->outputs[0] = tensor1->t;
    ret = vsi_nn_internal_setup_node(self, tmp_inode);
    if (!ret)
    {
        tensor1 = NULL;
    }
    return tensor1;
}

vsi_nn_op_t vsi_nn_rnn_get_act_op_type
    (
    vsi_nn_activation_e type
    )
{
    switch (type)
    {
    case VSI_NN_ACT_RELU:
        return VSI_NN_OP_RELU;
    case VSI_NN_ACT_RELU6:
        return VSI_NN_OP_RELU6;
    case VSI_NN_ACT_TANH:
        return VSI_NN_OP_TANH;
    case VSI_NN_ACT_SIGMOID:
        return VSI_NN_OP_SIGMOID;
    case VSI_NN_ACT_HARD_SIGMOID:
        return VSI_NN_OP_HARD_SIGMOID;
    default:
        VSILOGE("error activation type %d", type);
        break;
    }

    return VSI_NN_OP_TANH;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_activation
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_activation_e act_type,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, output_dtype, use_virtual_tensor);
    tensor1 = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_internal_new_node(self, vsi_nn_rnn_get_act_op_type(act_type), 0, 0 );

    tmp_inode->inputs[0] = input;
    tmp_inode->node->nn_param.tanh.scale_a = 1.0f;
    tmp_inode->node->nn_param.tanh.scale_b = 1.0f;
    tmp_inode->outputs[0] = tensor1->t;
    ret = vsi_nn_internal_setup_node(self, tmp_inode);
    if (!ret)
    {
        tensor1 = NULL;
    }

    return tensor1;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_transpose_time_major
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    uint32_t* permute_in_perm = NULL;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    if (output == NULL)
    {
        vsi_nn_internal_init_tensor_attr(&attr,
            &input->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
    }

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_PERMUTE, 0, 0 );
    permute_in_perm = (uint32_t *)vsi_nn_internal_new_node_param(curr,
        VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    permute_in_perm[0] = 0;
    permute_in_perm[1] = 2;
    permute_in_perm[2] = 1;

    curr->node->nn_param.permute.perm = permute_in_perm;
    curr->node->nn_param.permute.dim_num = 3;
    curr->inputs[0] = input;

    if (output == NULL)
    {
        curr->outputs[0] = output_tensor->t;
    }
    else
    {
        curr->outputs[0] = output;
    }
    ret = vsi_nn_internal_setup_node(self, curr);
    if (!ret)
    {
        output_tensor = NULL;
    }

    return output_tensor;
}

void vsi_nn_rnn_split_input_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t ** output,
    uint32_t time_step,
    vsi_bool use_virtual_tensor
    )
{
    uint32_t* slices = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t i = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_SPLIT, 1, time_step );
    slices = (uint32_t *)vsi_nn_internal_new_node_param(curr, time_step * sizeof(uint32_t));
    curr->node->nn_param.split.axis = 2; /* timestep axis */
    curr->node->nn_param.split.slices_num = time_step;
    curr->inputs[0] = input;

    curr->node->nn_param.split.slices = slices;
    for( i = 0; i < time_step; i++ )
    {
        slices[i] = 1;
        vsi_nn_internal_init_tensor_attr(&attr, &input->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        curr->outputs[i] = output_tensor->t;
        output[i] = output_tensor->t;
    }
    vsi_nn_internal_setup_node( self, curr );
}

void vsi_nn_rnn_data_check_aligned
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** input,
    uint32_t time_step,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t i = 0;
    vsi_size_t ofst = 0;
    ofst = 0;
    for( i = 0; i < time_step; i++ )
    {
        vsi_size_t tensor_size = vsi_nn_GetTensorSize( input[i]->attr.size,
            input[i]->attr.dim_num, input[i]->attr.dtype.vx_type );

        if( ofst & 0x3f && !self->graph->ctx->config.support_stream_processor)
        {
            vsi_nn_internal_init_tensor_attr(&attr, &input[i]->attr.dtype, use_virtual_tensor);
            output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

            curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
            curr->inputs[0] = input[i];
            curr->outputs[0] = output_tensor->t;
            vsi_nn_internal_setup_node( self, curr );

            input[i] = output_tensor->t;
        }

        ofst += tensor_size;
    }
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_reshape_split_output
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    uint32_t batch_size,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_size_t *reshape_split_size = NULL;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    /* reshape for split output */
    vsi_nn_internal_init_tensor_attr(&attr, &input->attr.dtype, use_virtual_tensor);
    output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE2, 0, 0 );
    reshape_split_size = (vsi_size_t *)vsi_nn_internal_new_node_param(curr,
        VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    reshape_split_size[0] = -1;
    reshape_split_size[1] = batch_size;

    curr->node->nn_param.reshape2.size = reshape_split_size;
    curr->node->nn_param.reshape2.dim_num = 2;
    curr->inputs[0] = input;
    curr->outputs[0] = output_tensor->t;
    ret = vsi_nn_internal_setup_node( self, curr );
    if (!ret)
    {
        output_tensor = NULL;
    }

    return output_tensor;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_reshape_cell_output
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    uint32_t batch_size,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_size_t* reshape_grucell_output_size = NULL;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    /* reshape output to 3-dims */
    vsi_nn_internal_init_tensor_attr(&attr, &input->attr.dtype, use_virtual_tensor);
    output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE2, 0, 0 );
    reshape_grucell_output_size = (vsi_size_t*)vsi_nn_internal_new_node_param(curr,
        VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    reshape_grucell_output_size[0] = -1;
    reshape_grucell_output_size[1] = batch_size;
    reshape_grucell_output_size[2] = 1;

    curr->node->nn_param.reshape2.size = reshape_grucell_output_size;
    curr->node->nn_param.reshape2.dim_num = 3;
    curr->inputs[0] = input;
    curr->outputs[0] = output_tensor->t;
    ret = vsi_nn_internal_setup_node( self, curr );
    if (!ret)
    {
        output_tensor = NULL;
    }

    return output_tensor;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_binary_operator
    (
    vsi_nn_node_t* self,
    vsi_nn_op_t op,
    vsi_nn_tensor_t* operand1,
    vsi_nn_tensor_t* operand2,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, output_dtype, use_virtual_tensor);
    output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_internal_new_node(self, op, 0, 0 );
    tmp_inode->node->nn_param.multiply.scale = 1.0f;
    tmp_inode->node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    tmp_inode->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    tmp_inode->inputs[0] = operand1;
    tmp_inode->inputs[1] = operand2;
    tmp_inode->outputs[0] = output_tensor->t;
    ret = vsi_nn_internal_setup_node(self, tmp_inode);
    if (!ret)
    {
        output_tensor = NULL;
    }

    return output_tensor;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_concat_impl
    (
    vsi_nn_node_t* self,
    uint32_t axis,
    vsi_bool use_virtual_tensor,
    vsi_nn_tensor_t* tensor,
    ...
    )
{
    va_list args;
    vsi_nn_tensor_t* next;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* tmp_tensor = NULL;
    vsi_nn_internal_node_t* inode = NULL;
    int tensor_count = 1;
    vsi_bool ret = FALSE;

    va_start(args, tensor);

    FOREACH_ARGS(args, next, vsi_nn_tensor_t*)
    {
        tensor_count++;
    }
    va_end(args);

    memset(&attr, 0x00, sizeof(attr));
    memcpy(&attr.dtype, &tensor->attr.dtype, sizeof(attr.dtype));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;
    tmp_tensor = vsi_nn_internal_new_tensor(self, &attr, 0.0f);

    inode = vsi_nn_internal_new_node(self, VSI_NN_OP_CONCAT, tensor_count, 1);
    inode->inputs[0] = tensor;
    tensor_count = 0;
    va_start(args, tensor);

    FOREACH_ARGS(args, next, vsi_nn_tensor_t*)
    {
        inode->inputs[1 + tensor_count++] = next;
    }
    va_end(args);
    inode->outputs[0] = tmp_tensor->t;

    ret = vsi_nn_internal_setup_node(self, inode);
    if (!ret)
    {
        tmp_tensor = NULL;
    }

    return tmp_tensor;
}

vsi_nn_internal_tensor_t** vsi_nn_create_split
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t* tensor,
    uint32_t axis,
    uint32_t slices_num,
    uint32_t* slices,
    vsi_bool use_virtual_tensor
    )
{
    uint32_t i = 0;
    uint32_t num_per_output = 0;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t** output_tensors = NULL;

    if(!slices_num)
    {
        VSILOGE("slices_num must be set!");
        return NULL;
    }

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_SPLIT, 1, slices_num );
    if(!slices)
    {
        slices = (uint32_t *)vsi_nn_internal_new_node_param(curr, slices_num * sizeof(uint32_t));
        num_per_output = (uint32_t)(tensor->attr.size[axis] / slices_num);
        for( i = 0; i < slices_num; i++ )
        {
            slices[i] = num_per_output;
        }
    }
    output_tensors = (vsi_nn_internal_tensor_t**)vsi_nn_internal_new_node_param(curr,
        slices_num * sizeof(vsi_nn_internal_tensor_t*));
    curr->node->nn_param.split.axis = axis;
    curr->node->nn_param.split.slices_num = slices_num;
    curr->node->nn_param.split.slices = slices;
    curr->inputs[0] = tensor;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, &tensor->attr.dtype, use_virtual_tensor);
    for( i = 0; i < slices_num; i++ )
    {
        output_tensors[i] = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        curr->outputs[i] = output_tensors[i]->t;
    }
    vsi_nn_internal_setup_node( self, curr );

    return output_tensors;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_reshape
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t* input_tensor,
    vsi_nn_tensor_t* output_tensor,
    vsi_size_t* size,
    vsi_size_t dim_num,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* tensor0 = NULL;
    vsi_size_t* reshape_in_size = NULL;
    vsi_bool ret = FALSE;

    curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RESHAPE2, 0, 0 );
    reshape_in_size = (vsi_size_t*)vsi_nn_internal_new_node_param(curr, dim_num * sizeof(vsi_size_t));
    memcpy(reshape_in_size, size, dim_num * sizeof(vsi_size_t));
    curr->node->nn_param.reshape2.size = reshape_in_size;
    curr->node->nn_param.reshape2.dim_num = (uint32_t)dim_num;
    curr->inputs[0] = input_tensor;
    curr->outputs[0] = output_tensor;

    if(output_tensor)
    {
        curr->outputs[0] = output_tensor;
    }
    else
    {
        vsi_nn_tensor_attr_t attr;
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        vsi_nn_internal_init_tensor_attr(&attr, &input_tensor->attr.dtype, use_virtual_tensor);
        tensor0 = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
        curr->outputs[0] = tensor0->t;
    }
    ret = vsi_nn_internal_setup_node(self, curr);
    if (!ret)
    {
        tensor0 = NULL;
    }


    return tensor0;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_permute
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t* input_tensor,
    vsi_nn_tensor_t* output_tensor,
    vsi_size_t* perm,
    vsi_size_t dim_num,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* tensor0 = NULL;
    uint32_t i = 0, * permute_in_perm = NULL;
    vsi_bool ret = FALSE;

    curr = vsi_nn_internal_new_node(self, VSI_NN_OP_PERMUTE, 0, 0);
    permute_in_perm = (uint32_t *)vsi_nn_internal_new_node_param(curr,
        dim_num * sizeof(uint32_t));

    for (i = 0; i < dim_num; i++)
    {
        permute_in_perm[i] = (uint32_t)perm[i];
    }
    curr->node->nn_param.permute.perm = permute_in_perm;
    curr->node->nn_param.permute.dim_num = (uint32_t)dim_num;
    curr->inputs[0] = input_tensor;

    if(output_tensor)
    {
        curr->outputs[0] = output_tensor;
    }
    else
    {
        vsi_nn_tensor_attr_t attr;
        vsi_nn_internal_init_tensor_attr(&attr, &input_tensor->attr.dtype, use_virtual_tensor);
        tensor0 = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
        curr->outputs[0] = tensor0->t;
    }
    ret = vsi_nn_internal_setup_node(self, curr);
    if (!ret)
    {
        tensor0 = NULL;
    }

    return tensor0;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_tensor_copy
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t* input_tensor,
    vsi_nn_tensor_t* output_tensor,
    vsi_nn_dtype_t* dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* tensor0 = NULL;
    vsi_bool ret = FALSE;

    curr = vsi_nn_internal_new_node(self, VSI_NN_OP_DATACONVERT, 0, 0);
    curr->inputs[0] = input_tensor;
    if(!dtype)
    {
        dtype = &input_tensor->attr.dtype;
    }

    if(output_tensor)
    {
        curr->outputs[0] = output_tensor;
    }
    else
    {
        vsi_nn_tensor_attr_t attr;
        vsi_nn_internal_init_tensor_attr(&attr, dtype, use_virtual_tensor);
        tensor0 = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
        curr->outputs[0] = tensor0->t;
    }
    ret = vsi_nn_internal_setup_node(self, curr);
    if (!ret)
    {
        tensor0 = NULL;
    }

    return tensor0;
}