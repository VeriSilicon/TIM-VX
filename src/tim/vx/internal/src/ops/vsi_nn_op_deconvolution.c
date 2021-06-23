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
#include "utils/vsi_nn_constraint_check.h"

#define COMPUTE_DECONV_SZ( in, ksize, pad_1, pad_2, stride, output_padding )\
    (( in - 1 ) * stride + ksize - pad_1 - pad_2 + output_padding)
static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_deconvolution_params_ext2_t param;
    vsi_nn_tensor_t *permute_tensor = NULL;
#ifdef VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS
    vsi_nn_tensor_t *reverse_tensor = NULL;
#endif
    vsi_nn_tensor_t *weight_tensor  = NULL;

    status = VSI_FAILURE;
#ifdef VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS
    if (FALSE == inputs[1]->attr.is_const)
    {
        vsi_nn_tensor_t *tmp_in_tensor = NULL;
        vx_nn_tensor_reverse_params_t para;
        vx_int32 axis_reverse[4] = {0, 1, 0, 0};
        vsi_nn_tensor_attr_t attr_reverse;

        if (vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1)
        {
            uint32_t perm[] = { 0, 1, 3, 2 };
            vsi_nn_tensor_attr_t attr;
            memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
            memcpy( &attr, &inputs[1]->attr, sizeof(vsi_nn_tensor_attr_t) );
            attr.size[2] = inputs[1]->attr.size[3];
            attr.size[3] = inputs[1]->attr.size[2];
            permute_tensor = vsi_nn_CreateTensor(self->graph, &attr);
            self->n = vxTensorPermuteNode( self->graph->g, inputs[1]->t,
                        permute_tensor->t, perm, 4);
            if( NULL == self->n )
            {
                status = VSI_FAILURE;
                goto final;
            }
            tmp_in_tensor = permute_tensor;
        }
        else
        {
            tmp_in_tensor = inputs[1];
        }

        memset(&attr_reverse, 0, sizeof(vsi_nn_tensor_attr_t));
        memcpy(&attr_reverse, &tmp_in_tensor->attr, sizeof(vsi_nn_tensor_attr_t) );
        reverse_tensor = vsi_nn_CreateTensor(self->graph, &attr_reverse);
        para.axis = axis_reverse;
        para.numberOfAxis = 2;

        self->n = vxTensorReverse( self->graph->g, tmp_in_tensor->t, &para,
            sizeof(vx_nn_tensor_reverse_params_t), reverse_tensor->t );
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto final;
        }

        weight_tensor  = reverse_tensor;
    }
    else
    {
        weight_tensor = inputs[1];
    }

#else
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 && FALSE == inputs[1]->attr.is_const)
    {
        uint32_t perm[] = { 0, 1, 3, 2 };
        vsi_nn_tensor_attr_t attr;
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        memcpy( &attr, &inputs[1]->attr, sizeof(vsi_nn_tensor_attr_t) );
        attr.size[2] = inputs[1]->attr.size[3];
        attr.size[3] = inputs[1]->attr.size[2];
        permute_tensor = vsi_nn_CreateTensor(self->graph, &attr);
        self->n = vxTensorPermuteNode( self->graph->g, inputs[1]->t,
                    permute_tensor->t, perm, 4);
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto final;
        }
        weight_tensor  = permute_tensor;
    }
    else
    {
        weight_tensor = inputs[1];
    }
#endif
    // param.a_x = self->nn_param.deconv.dilation;
    // param.a_y = self->nn_param.deconv.dilation;
    param.ext.khr.a_x = 1;
    param.ext.khr.a_y = 1;
    param.ext.khr.padding_x = (size_t)self->nn_param.deconv.pad[0];
    param.ext.khr.padding_y = (size_t)self->nn_param.deconv.pad[2];
    param.ext.khr.overflow_policy = self->vx_param.overflow_policy;
    param.ext.khr.rounding_policy = self->vx_param.rounding_policy;
    param.ext.padding_x_right = (size_t)self->nn_param.deconv.pad[1];
    param.ext.padding_y_bottom = (size_t)self->nn_param.deconv.pad[3];
    param.ext.channel_group = self->nn_param.deconv.group;
    param.stride_x = self->nn_param.deconv.stride[0];
    param.stride_y = self->nn_param.deconv.stride[1];
    //param.border_mode;
    //param.border_const;

    self->n = vxDeconvolutionLayer(
        self->graph->g,
        inputs[0]->t,
        weight_tensor->t,
        (NULL == inputs[2]) ? NULL : inputs[2]->t,
        (vx_nn_deconvolution_params_t *)&param,
        sizeof( vx_nn_deconvolution_params_ext2_t ),
        outputs[0]->t
        );
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

final:
    if (permute_tensor)
    {
        vsi_nn_ReleaseTensor(&permute_tensor);
    }
#ifdef VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS
    if (reverse_tensor)
    {
        vsi_nn_ReleaseTensor(&reverse_tensor);
    }
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

    BEGIN_IO_TYPE_DECL(DECONVOLUTION, 3, 1)
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
        IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC, D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM, D_I8|Q_SYM_PC,  D_I32|Q_SYM_PC, D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE, D_I8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM, D_I8|Q_SYM_PC,  D_NONE, D_U8|Q_ASYM)
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
        IO_TYPE(D_F32,  D_F32,  D_F32, D_F32)
        IO_TYPE(D_F32,  D_F32,  D_F32, D_BF16)
        IO_TYPE(D_F32,  D_F32,  D_NONE, D_F32)

        /* HW 9.0 */
        IO_TYPE(D_F32,  D_BF16,  D_F32, D_BF16)
        IO_TYPE(D_F32,  D_BF16,  D_NONE, D_BF16)
    END_IO_TYPE_DECL(DECONVOLUTION)
    if (!VALIDATE_OP_IO_TYPES(DECONVOLUTION, self, inputs, self->input.num, outputs, self->output.num))
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
    vsi_nn_deconv_param *nn_param;
    uint32_t perm[] = { 3, 2, 0, 1 };
    uint32_t perm1[] = { 0, 1, 3, 2 };

    /* TODO: Driver should handle this,
    * Check transpose
    * TODO: remove this
    * */
    if( VSI_NN_DIM_FMT_NHWC == inputs[1]->attr.dtype.fmt )
    {
        vsi_nn_TransposeTensor( self->graph, inputs[1], perm, 4, NULL );
        inputs[1]->attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;
    }

#ifdef VX_CONVERT_POLICY_WRAP_ENABLE
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }
#endif

#ifdef VX_DECONVOLUTION_WEIGHT_LAYOUT_COMPATIBLE_KHRONOS
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 && TRUE == inputs[1]->attr.is_const)
    {
        /* whnc->whcn */
        vsi_nn_PermuteTensor( self->graph, inputs[1], perm1, 4 );
    }
    /* Rotate 180 degrees for weights data */
    if (TRUE == inputs[1]->attr.is_const)
    {
        vsi_nn_reshuffle_weight_data(self->graph, inputs[1]);
    }
#else
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) >= 0 && TRUE == inputs[1]->attr.is_const)
    {
        /* whcn->whnc */
        vsi_nn_PermuteTensor( self->graph, inputs[1], perm1, 4 );
    }
#endif

    nn_param = &self->nn_param.deconv;

    nn_param->group = ( 0 == nn_param->group ) ? 1 : nn_param->group;

    nn_param->ksize[0] = inputs[1]->attr.size[0];
    nn_param->ksize[1] = inputs[1]->attr.size[1];


    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = COMPUTE_DECONV_SZ(
            inputs[0]->attr.size[0],
            nn_param->ksize[0],
            nn_param->pad[0],
            nn_param->pad[1],
            nn_param->stride[0],
            nn_param->output_padding[0]
        );

        outputs[0]->attr.size[1] = COMPUTE_DECONV_SZ(
            inputs[0]->attr.size[1],
            nn_param->ksize[1],
            nn_param->pad[2],
            nn_param->pad[3],
            nn_param->stride[1],
            nn_param->output_padding[1]
        );

        if(self->nn_param.deconv.weights > 0)
        {
            outputs[0]->attr.size[2] = self->nn_param.deconv.weights;
        }
        else
        {
            outputs[0]->attr.size[2] = inputs[1]->attr.size[3];
        }
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ DECONVOLUTION,
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

