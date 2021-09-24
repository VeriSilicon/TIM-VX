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

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "ops/vsi_nn_op_conv_relu.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_convolution_relu_pooling_params_ext2_t p;
    status = VSI_FAILURE;

    if(vsi_nn_InitConvReluPoolParameter(self, &p, FALSE) != VSI_SUCCESS)
    {
        VSILOGE("SetConvReluParameter fail\n");
        return VSI_FAILURE;
    }

    self->n = vxConvolutionReluPoolingLayer2(
        self->graph->g,
        inputs[0]->t,
        inputs[1]->wb,
        (vx_nn_convolution_relu_pooling_params_t *)&p,
        sizeof(p),
        outputs[0]->t
        );

    vsi_nn_DeinitConvReluPoolParameter( &p );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
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
    BEGIN_IO_TYPE_DECL(CONV_RELU, 3, 1)
        IO_TYPE(D_F16,  D_F16,  D_NONE, D_F16)
        IO_TYPE(D_F16,  D_F16,  D_F32, D_F16)
        IO_TYPE(D_F16,  D_F16,  D_F16, D_F16)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_I32|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_I32|Q_DFP, D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_I32|Q_DFP, D_F16)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_I32|Q_DFP, D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP,  D_NONE, D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_NONE, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_I32|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_I64|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP,  D_I16|Q_DFP, D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_I8|Q_DFP,  D_I32|Q_DFP, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_NONE, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_F16)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM, D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM, D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_U8|Q_ASYM, D_F16)
        IO_TYPE(D_BF16,  D_BF16,  D_F32, D_BF16)
        IO_TYPE(D_BF16,  D_BF16,  D_F32, D_F32)
        IO_TYPE(D_BF16,  D_BF16,  D_NONE, D_BF16)
        IO_TYPE(D_F32,  D_F32,  D_F32, D_F32)
        IO_TYPE(D_F32,  D_F32,  D_F32, D_BF16)
        IO_TYPE(D_F32,  D_F32,  D_NONE, D_F32)
    END_IO_TYPE_DECL(CONV_RELU)
    if (!VALIDATE_OP_IO_TYPES(CONV_RELU, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    /* Check fl and scale*/
    ret = vsi_nn_QuantCheck(inputs[0], inputs[1], inputs[2]);

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;

#ifdef VX_CONVERT_POLICY_WRAP_ENABLE
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }
#endif

    ret = vsi_nn_OpSetup( VSI_NN_OP_CONV2D, self, inputs, outputs );

    return ret;
} /* op_setup() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status status;
    vx_nn_convolution_relu_pooling_params_ext2_t p;
    vx_weights_biases_parameter_optimizations_t opt;
    vx_weights_biases_parameter_optimizations_t * p_opt;

    status = VSI_SUCCESS;

    if(direction == VSI_NN_OPTIMIZE_BACKWARD)
    {
        return VSI_SUCCESS;
    }

    VSILOGD("Optimize %s", vsi_nn_OpGetName(self->op));
    /* Prepare weight_bias */
    if(inputs[1]->wb == NULL)
    {
        if(vsi_nn_InitConvReluPoolParameter(self, &p, FALSE) != VSI_SUCCESS)
        {
            VSILOGE("SetConvReluParameter fail\n");
            return VSI_FAILURE;
        }

        p_opt = NULL;
        if( outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC
         || inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
        {
            memset( &opt, 0, sizeof( opt ) );
            opt.inputZeroPoint = inputs[0]->attr.dtype.zero_point;
            opt.zrl = -1;
            opt.outputFormat = outputs[0]->attr.dtype.vx_type;
            p_opt = &opt;
        }

        inputs[1]->wb = vxCreateWeightsBiasesParameterFromTensors2(
            VX_CONVOLUTIONAL_NETWORK_CONVOLUTION_LAYER,
            4,
#ifdef VSI_40BIT_VA_SUPPORT
            inputs[0]->attr.size,
            outputs[0]->attr.size,
            outputs[0]->attr.size,
#else
            (vx_uint32*)inputs[0]->attr.size,
            (vx_uint32*)outputs[0]->attr.size,
            (vx_uint32*)outputs[0]->attr.size,
#endif
            outputs[0]->attr.dtype.vx_type,
            (vx_nn_convolution_relu_pooling_params_t *)&p,
            sizeof(p),
            p_opt,
            inputs[1]->t, inputs[2]->t
            );
        vsi_nn_DeinitConvReluPoolParameter( &p );
    }

    if( NULL == inputs[1]->wb )
    {
        VSILOGE( "Create weight bias fail." );
        status = VSI_FAILURE;
    }

    return status;
} /* op_optimize() */

vsi_status vsi_nn_InitConvReluPoolParameter
    (
    vsi_nn_node_t * node,
    vx_nn_convolution_relu_pooling_params_ext2_t * param_ext2,
    vsi_bool has_pool
    )
{
    int32_t pad_const_val;
    vx_scalar pad_const;
    vx_nn_convolution_relu_pooling_params_t *param;
    vx_nn_convolution_relu_pooling_params_ext_t *param_ext;

    pad_const_val = 0;
    pad_const = NULL;
    param = NULL;

    if( NULL == node || NULL == param_ext2 )
    {
        VSILOGE("Set param fail\n");
        return VSI_FAILURE;
    }
    memset( param_ext2, 0, sizeof( vx_nn_convolution_relu_pooling_params_ext2_t ) );
    param_ext = &param_ext2->ext;
    param = &param_ext->base;

    pad_const = vxCreateScalar( node->graph->ctx->c, VX_TYPE_INT32, &pad_const_val );
    if( NULL == pad_const )
    {
        VSILOGE("Create scalar fail\n");
        return VSI_FAILURE;
    }

    if( node->nn_param.conv2d.dilation[0] > 0 )
    {
        param->dilation_x = node->nn_param.conv2d.dilation[0] - 1;
    }
    if( node->nn_param.conv2d.dilation[1] > 0 )
    {
        param->dilation_y = node->nn_param.conv2d.dilation[1] - 1;
    }
    param->pad_x_left    = node->nn_param.conv2d.pad[0];
    param->pad_x_right   = node->nn_param.conv2d.pad[1];
    param->pad_y_top     = node->nn_param.conv2d.pad[2];
    param->pad_y_bottom  = node->nn_param.conv2d.pad[3];
    param->accumulator_bits = (vx_uint8)node->vx_param.accumulator_bits;
    param->overflow_policy = node->vx_param.overflow_policy;
    param->rounding_policy = node->vx_param.rounding_policy;
    param->down_scale_size_rounding = node->vx_param.down_scale_size_rounding;
    param->enable_relu = (vx_bool)node->vx_param.has_relu;
    param->pad_mode = VX_PAD_CONSTANT;
    param->pad_const = pad_const;
    if( TRUE == has_pool )
    {
        param->pool_type = node->nn_param.pool.type;
        param->pool_size_x = node->nn_param.pool.ksize[0];
        param->pool_size_y = node->nn_param.pool.ksize[1];
    }
    param_ext->stride_x = node->nn_param.conv2d.stride[0];
    param_ext->stride_y = node->nn_param.conv2d.stride[1];

    param_ext2->depth_multiplier = node->nn_param.conv2d.multiplier;

    return VSI_SUCCESS;
} /* vsi_nn_InitConvReluPoolParameter() */

void vsi_nn_DeinitConvReluPoolParameter
    (
    vx_nn_convolution_relu_pooling_params_ext2_t * param
    )
{
    if( NULL != param )
    {
        if( NULL != param->ext.base.pad_const )
        {
            vxReleaseScalar( &param->ext.base.pad_const );
        }
    }
} /* vsi_nn_DeinitConvReluPoolParameter() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CONV_RELU,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

