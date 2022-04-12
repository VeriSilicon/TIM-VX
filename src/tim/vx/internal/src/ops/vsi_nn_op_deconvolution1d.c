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
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_kernel.h"

#define COMPUTE_DECONV_SZ( in, ksize, pad_1, pad_2, stride, output_padding )\
    (( in - 1 ) * stride + ksize - pad_1 - pad_2 + output_padding)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_size_t perm[] = { 0, 1, 3, 2 };
    vsi_nn_tensor_attr_t weight_attr;
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_tensor_t* weight_tensor = NULL;
    vsi_nn_tensor_t* weights_tensor = NULL;
    vsi_nn_tensor_t* new_inputs[3] = {NULL};
    vsi_nn_tensor_t *permute_tensor = NULL;
#ifdef VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS
    vsi_nn_tensor_t *reverse_tensor = NULL;
#endif

    memcpy(&weight_attr, &(inputs[1]->attr), sizeof(vsi_nn_tensor_attr_t));
    weight_attr.size[3] = weight_attr.size[2];
    weight_attr.size[2] = weight_attr.size[1];
    weight_attr.size[1] = 1;
    weight_attr.dim_num = 4;
    if (inputs[1]->attr.dtype.qnt_type != VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC)
    {
        weight_tensor = vsi_nn_reshape_tensor( self->graph, inputs[1], weight_attr.size, 4 );
    }
    else
    {
        uint8_t * data = NULL;
        data = vsi_nn_ConvertTensorToData( self->graph, inputs[1] );
        if (NULL == data)
        {
            VSILOGE("Convert data fail.\n");
            status = VSI_FAILURE;
            return status;
        }
        weight_attr.dtype.channel_dim = inputs[1]->attr.dtype.channel_dim + 1;
        weight_tensor = vsi_nn_CreateTensorFromData(self->graph, data, &weight_attr);
        vsi_nn_safe_free( data );
    }

#ifdef VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 && TRUE == weight_tensor->attr.is_const )
    {
        /* whnc->whcn */
        vsi_nn_PermuteTensor( self->graph, weight_tensor, perm, 4 );
    }

    /* Rotate 180 degrees for weights data */
    if ( TRUE == weight_tensor->attr.is_const )
    {
        vsi_nn_reshuffle_weight_data( self->graph, weight_tensor );
    }
#else
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) >= 0 && TRUE == weight_tensor->attr.is_const)
    {
        /* whcn->whnc */
        vsi_nn_PermuteTensor( self->graph, weight_tensor, perm, 4 );
    }
#endif

#ifdef VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS
    if (FALSE == inputs[1]->attr.is_const)
    {
        vsi_nn_tensor_t *tmp_in_tensor = NULL;
        vx_nn_tensor_reverse_params_t para;
        vx_int32 axis_reverse[4] = {0, 1, 0, 0};
        vsi_nn_tensor_attr_t attr_reverse;

        if (vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1)
        {
            uint32_t perm_array[] = { 0, 1, 3, 2 };
            vsi_nn_tensor_attr_t attr;
            memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
            memcpy( &attr, &weight_tensor->attr, sizeof(vsi_nn_tensor_attr_t) );
            attr.size[2] = weight_tensor->attr.size[3];
            attr.size[3] = weight_tensor->attr.size[2];
            permute_tensor = vsi_nn_CreateTensor(self->graph, &attr);
            self->n = vxTensorPermuteNode( self->graph->g, weight_tensor->t,
                        permute_tensor->t, perm_array, 4);
            if ( NULL == self->n )
            {
                status = VSI_FAILURE;
                goto final;
            }
            tmp_in_tensor = permute_tensor;
        }
        else
        {
            tmp_in_tensor = weight_tensor;
        }

        memset(&attr_reverse, 0, sizeof(vsi_nn_tensor_attr_t));
        memcpy(&attr_reverse, &tmp_in_tensor->attr, sizeof(vsi_nn_tensor_attr_t) );
        reverse_tensor = vsi_nn_CreateTensor(self->graph, &attr_reverse);
        para.axis = axis_reverse;
        para.numberOfAxis = 2;

        self->n = vxTensorReverse( self->graph->g, tmp_in_tensor->t, &para,
            sizeof(vx_nn_tensor_reverse_params_t), reverse_tensor->t );
        if ( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto final;
        }

        weights_tensor  = reverse_tensor;
    }
    else
    {
        weights_tensor = weight_tensor;
    }

#else
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 && FALSE == inputs[1]->attr.is_const)
    {
        uint32_t perm_array[] = { 0, 1, 3, 2 };
        vsi_nn_tensor_attr_t attr;
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        memcpy( &attr, &weight_tensor->attr, sizeof(vsi_nn_tensor_attr_t) );
        attr.size[2] = weight_tensor->attr.size[3];
        attr.size[3] = weight_tensor->attr.size[2];
        permute_tensor = vsi_nn_CreateTensor(self->graph, &attr);
        self->n = vxTensorPermuteNode( self->graph->g, weight_tensor->t,
                    permute_tensor->t, perm_array, 4);
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto final;
        }
        weights_tensor  = permute_tensor;
    }
    else
    {
        weights_tensor = weight_tensor;
    }
#endif

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "stride", self->nn_param.deconvolution1d.stride );
    vsi_nn_kernel_param_add_int32( param, "pad_front", self->nn_param.deconvolution1d.pad[0] );
    vsi_nn_kernel_param_add_int32( param, "pad_end", self->nn_param.deconvolution1d.pad[1] );
    vsi_nn_kernel_param_add_int32( param, "group", self->nn_param.deconvolution1d.group );
    vsi_nn_kernel_param_add_int32( param, "overflow_policy", self->vx_param.overflow_policy );
    vsi_nn_kernel_param_add_int32( param, "rounding_policy", self->vx_param.rounding_policy );
    vsi_nn_kernel_param_add_int32( param,
            "down_scale_size_rounding", self->vx_param.down_scale_size_rounding );

    new_inputs[0] = inputs[0];
    new_inputs[1] = weights_tensor;
    new_inputs[2] = inputs[2];

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "deconvolution1d",
            new_inputs, 3, outputs, 1, param );

    if( self->n )
    {
        status = VSI_SUCCESS;
    }
    vsi_nn_kernel_param_release( &param );

final:
    vsi_safe_release_tensor(weight_tensor);
    vsi_safe_release_tensor(permute_tensor);
#ifdef VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS
    vsi_safe_release_tensor(reverse_tensor);
#endif
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;

    ret = vsi_nn_OpCheck(VSI_NN_OP_DECONVOLUTION, self, inputs, outputs);

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_deconvolution1d_param *nn_param = &self->nn_param.deconvolution1d;

#ifdef VX_CONVERT_POLICY_WRAP_ENABLE
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }
#endif

    nn_param->group = ( 0 == nn_param->group ) ? 1 : nn_param->group;
    nn_param->ksize = (uint32_t)inputs[1]->attr.size[0];

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = COMPUTE_DECONV_SZ(
            inputs[0]->attr.size[0],
            nn_param->ksize,
            nn_param->pad[0],
            nn_param->pad[1],
            nn_param->stride,
            nn_param->output_padding
        );

        if( nn_param->weights > 0 )
        {
            outputs[0]->attr.size[1] = nn_param->weights;
        }
        else
        {
            outputs[0]->attr.size[1] = inputs[1]->attr.size[3];
        }
        outputs[0]->attr.size[1] = nn_param->weights;
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_op_common_deinit( self );
    return VSI_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ DECONVOLUTION1D,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
