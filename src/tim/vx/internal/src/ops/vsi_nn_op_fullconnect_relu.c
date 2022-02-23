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
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status _set_fc_relu_parameter
    (
    vsi_nn_node_t * self,
    vx_nn_convolution_relu_pooling_params_t * param
    );

static vsi_status _set_fc_relu_parameter
    (
    vsi_nn_node_t * self,
    vx_nn_convolution_relu_pooling_params_t * param
    )
{
    vx_scalar pad_const;
    int32_t pad_const_val;

    pad_const_val = 0;
    memset( param, 0, sizeof(vx_nn_convolution_relu_pooling_params_t) );
    pad_const = vxCreateScalar(self->graph->ctx->c, VX_TYPE_INT32, &pad_const_val);
    if( !pad_const )
    {
        VSILOGE("Create scalar fail\n");
        return VSI_FAILURE;
    }

    param->pad_x_left    = 0;
    param->pad_x_right   = 0;
    param->pad_y_top     = 0;
    param->pad_y_bottom  = 0;
    param->dilation_x    = 0;
    param->dilation_y    = 0;
    param->accumulator_bits = (vx_uint8)self->vx_param.accumulator_bits;
    param->overflow_policy = self->vx_param.overflow_policy;
    param->rounding_policy = self->vx_param.rounding_policy;
    param->down_scale_size_rounding = self->vx_param.down_scale_size_rounding;
    param->enable_relu = self->vx_param.has_relu;
    param->pool_type = 0;
    param->pool_size_x = 0;
    param->pool_size_y = 0;
    param->pad_mode = VX_PAD_CONSTANT;
    param->pad_const = pad_const;

    return VSI_SUCCESS;
} /* _set_fc_relu_parameter() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    status = VSI_FAILURE;

    self->n = vxFullyConnectedReluLayer(
        self->graph->g,
        inputs[0]->t,
        inputs[1]->wb,
        0,
        0,
        self->vx_param.overflow_policy,
        self->vx_param.rounding_policy,
        self->vx_param.down_scale_size_rounding,
        self->vx_param.has_relu,
        outputs[0]->t
        );

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

    /* Check fl and scale*/
    ret = vsi_nn_QuantCheck(inputs[0], inputs[1], inputs[2]);

    if(ret) {
        /* check inputs outputs data type */
        /* NN Support */
        BEGIN_IO_TYPE_DECL(FCL_RELU, 3, 1)
            /* IO_TYPE(INPUT, WEIGHT, BIAS, OUTPUT) */
            /* NN Support - I8 */
            IO_TYPE(D_I8|Q_SYM_PC, D_I8|Q_SYM_PC, D_I32|Q_SYM_PC, D_I8|Q_SYM_PC)
            IO_TYPE(D_I8|Q_ASYM, D_I8|Q_SYM_PC, D_I32|Q_SYM_PC, D_I8|Q_ASYM)

            IO_TYPE(D_I8|Q_ASYM, D_I8|Q_SYM_PC, D_I32|Q_SYM_PC, D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_ASYM, D_I8|Q_SYM_PC, D_I32|Q_SYM_PC, D_F16)

            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_I32|Q_DFP, D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP, D_I8|Q_DFP, D_I32|Q_DFP, D_F16)

            /* NN Support - U8 */
            IO_TYPE(D_U8|Q_SYM_PC, D_U8|Q_SYM_PC, D_I32|Q_SYM_PC, D_U8|Q_SYM_PC)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC, D_I32|Q_SYM_PC, D_U8|Q_ASYM)

            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC, D_I32|Q_SYM_PC, D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_SYM_PC, D_I32|Q_SYM_PC, D_F16)

            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_I32|Q_ASYM, D_F16)

            /* NN Support - I16 */
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I32|Q_DFP, D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I32|Q_DFP, D_U8|Q_ASYM)
            IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_I32|Q_DFP, D_I16|Q_DFP)

            /* NN Support - F16 */
            IO_TYPE(D_F16, D_F16, D_F16, D_F16)
            IO_TYPE(D_F16, D_F16, D_F16, D_I8|Q_DFP)
            IO_TYPE(D_F16, D_F16, D_F16, D_U8|Q_ASYM)

            /* NN Support - BF16 */
            IO_TYPE(D_BF16, D_BF16, D_F32, D_BF16)
            IO_TYPE(D_BF16, D_BF16, D_F32, D_F32)

            /* NN Support - F32 */
            IO_TYPE(D_F32, D_BF16, D_F32, D_F32)
            IO_TYPE(D_F32, D_BF16, D_F32, D_BF16)
            /* HW 9.0.1 */
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE,          D_F32)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I8|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC,  D_F32)

        END_IO_TYPE_DECL(FCL_RELU)
        ret = VALIDATE_OP_IO_TYPES(FCL_RELU, self, inputs, self->input.num, outputs, self->output.num);

        /* TP Support */
        if (!ret ) {
            uint32_t valid_dtypes[] = {
                D_F16, D_BF16, D_F32, D_I16|Q_DFP, D_I8|Q_DFP, D_I8|Q_ASYM, D_U8|Q_DFP, D_U8|Q_ASYM
            };

            uint32_t weight_type = inputs[1]->attr.dtype.vx_type | inputs[1]->attr.dtype.qnt_type << Q_SHIFT;
            uint32_t inputs_types[3] = { 0 };
            vsi_bool supported[3] = { FALSE, FALSE, FALSE };
            int i = 0;

            inputs_types[0] = inputs[0]->attr.dtype.vx_type | inputs[0]->attr.dtype.qnt_type << Q_SHIFT;
            inputs_types[2] = outputs[0]->attr.dtype.vx_type | outputs[0]->attr.dtype.qnt_type << Q_SHIFT;
            if (inputs[2]) {
                switch(inputs[1]->attr.dtype.vx_type) {
                    case D_F16:
                    case D_BF16:
                    case D_F32:
                        if(inputs[2]->attr.dtype.vx_type == (vsi_nn_type_e)D_F32) {
                            inputs_types[1] = weight_type;
                        }
                        break;
                    case D_I16:
                    case D_I8:
                    case D_U8:
                        if (inputs[2]->attr.dtype.vx_type == (vsi_nn_type_e)D_I32 ||
                                inputs[2]->attr.dtype.vx_type == (vsi_nn_type_e)D_I64) {
                            inputs_types[1] = weight_type;
                        }
                        break;
                    default:
                        break;
                }
            } else {
                inputs_types[1] = weight_type;
            }

            for (i = 0; i < 3; i++) {
                supported[i] = is_item_in_array(&inputs_types[i], valid_dtypes,
                        sizeof(uint32_t), _cnt_of_array(valid_dtypes));
            }

            ret = supported[0] && supported[1] && supported[2];
        }

        if(!ret) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            vsi_nn_safe_free(desc);
            return FALSE;
        }
    }
    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret;
    vx_nn_convolution_relu_pooling_params_t p;
    vx_weights_biases_parameter_optimizations_ext_t opt;
    vx_weights_biases_parameter_optimizations_ext_t * p_opt;

#ifdef VX_CONVERT_POLICY_WRAP_ENABLE
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }
#endif

    ret = vsi_nn_OpSetup( VSI_NN_OP_FCL, self, inputs, outputs );

    /* Prepare weight_bias */
    if(inputs[1]->wb == NULL)
    {
        if( _set_fc_relu_parameter( self, &p ) != VSI_SUCCESS )
        {
            VSILOGE("set fc_relu weightbias parameter fail\n");
            return FALSE;
        }

        p_opt = NULL;
        memset( &opt, 0, sizeof( opt ) );
        if( outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC
         || inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
        {
            opt.inputZeroPoint = inputs[0]->attr.dtype.zero_point;
        }
        opt.zrl = -1;
        opt.outputFormat = outputs[0]->attr.dtype.vx_type;
        opt.num_of_input_dims = inputs[0]->attr.dim_num;
        opt.num_of_output_dims = outputs[0]->attr.dim_num;
        p_opt = &opt;

#ifdef VSI_40BIT_VA_SUPPORT
        {
            vx_size size_input0[VSI_NN_MAX_DIM_NUM];
            vx_size size_output0[VSI_NN_MAX_DIM_NUM];
            size_t i = 0;
            for(i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
            {
                size_input0[i] = (vx_size)inputs[0]->attr.size[i];
                size_output0[i] = (vx_size)outputs[0]->attr.size[i];
            }
            inputs[1]->wb = vxCreateWeightsBiasesParameterFromTensors3(
                VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER,
                size_input0,
                size_output0,
                size_output0,
                &p,
                sizeof(p),
                (vx_weights_biases_parameter_optimizations_t *)p_opt,
                sizeof(opt),
                inputs[1]->t, inputs[2]->t
                );
        }
#else
        {
            uint32_t size_u32_input0[VSI_NN_MAX_DIM_NUM];
            uint32_t size_u32_output0[VSI_NN_MAX_DIM_NUM];
            size_t i = 0;
            for(i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
            {
                size_u32_input0[i] = (uint32_t)inputs[0]->attr.size[i];
                size_u32_output0[i] = (uint32_t)outputs[0]->attr.size[i];
            }
            inputs[1]->wb = vxCreateWeightsBiasesParameterFromTensors3(
                VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER,
                size_u32_input0,
                size_u32_output0,
                size_u32_output0,
                &p,
                sizeof(p),
                (vx_weights_biases_parameter_optimizations_t *)p_opt,
                sizeof(opt),
                inputs[1]->t, inputs[2]->t
                );
        }
#endif
        if( p.pad_const )
        {
            vxReleaseScalar( &p.pad_const );
        }
    }


    if( NULL == inputs[1]->wb )
    {
        VSILOGE( "Create weight bias fail." );
        ret = FALSE;
    }

    return ret;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ FCL_RELU,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
