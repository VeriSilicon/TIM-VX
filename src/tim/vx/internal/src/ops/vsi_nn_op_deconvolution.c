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

#define LOCAL() (local)

typedef struct _vsi_nn_grouped_deconv2d_param_local_data {
    vsi_nn_tensor_t ** input_tensor_group;
    vsi_nn_tensor_t ** weight_tensor_group;
    vsi_nn_tensor_t ** bias_tensor_group;
    vsi_nn_tensor_t ** output_tensor_group;
} vsi_nn_grouped_deconv2d_param_local_data;

static vsi_status op_grouped_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * inputs[3],
    vsi_nn_tensor_t ** outputs,
    vx_nn_deconvolution_params_ext2_t param
    )
{
    vsi_bool res;
    uint32_t i;
    vsi_status status = VSI_FAILURE;
    vsi_nn_deconv_param *nn_param = &self->nn_param.deconv;
    uint32_t group = nn_param->group;
    vsi_nn_grouped_deconv2d_param_local_data *local =
        (vsi_nn_grouped_deconv2d_param_local_data*)malloc(sizeof(vsi_nn_grouped_deconv2d_param_local_data));
    if (NULL == local)
    {
        VSILOGE("Malloc fail, (GROUPED_DECONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        goto final;
    }
    memset(local, 0, sizeof(vsi_nn_grouped_deconv2d_param_local_data));
    /* TODO */
    LOCAL()->input_tensor_group = (vsi_nn_tensor_t **)malloc(
        group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->input_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_DECONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        goto final;
    }
    memset(LOCAL()->input_tensor_group, 0, group * sizeof(vsi_nn_tensor_t *));
    res = vsi_nn_CreateTensorGroup(self->graph, inputs[0], 2,
        LOCAL()->input_tensor_group, group);
    if (res == FALSE)
    {
        VSILOGE("CreateTensorGroup fail, (GROUPED_DECONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        goto final;
    }

    LOCAL()->weight_tensor_group = (vsi_nn_tensor_t **)malloc(
        group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->weight_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_DECONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(LOCAL()->weight_tensor_group, 0, group * sizeof(vsi_nn_tensor_t *));
    res = vsi_nn_CreateTensorGroup(self->graph, inputs[1], 2,
        LOCAL()->weight_tensor_group, group);
    if (res == FALSE)
    {
        VSILOGE("CreateTensorGroup fail, (GROUPED_DECONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        goto final;
    }

    LOCAL()->bias_tensor_group = (vsi_nn_tensor_t **)malloc(
        group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->bias_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        goto final;
    }
    memset(LOCAL()->bias_tensor_group, 0, group * sizeof(vsi_nn_tensor_t *));
    if (inputs[2] != NULL)
    {
        res = vsi_nn_CreateTensorGroup(self->graph, inputs[2], 0,
            LOCAL()->bias_tensor_group, group);
        if (res == FALSE)
        {
            VSILOGE("CreateTensorGroup fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
            goto final;
        }
    }

    LOCAL()->output_tensor_group = (vsi_nn_tensor_t **)malloc(
        group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->output_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        goto final;
    }
    memset(LOCAL()->output_tensor_group, 0, group * sizeof(vsi_nn_tensor_t *));
    res = vsi_nn_CreateTensorGroup(self->graph, outputs[0], 2,
        LOCAL()->output_tensor_group, group);
    if (res == FALSE)
    {
        VSILOGE("CreateTensorGroup fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        goto final;
    }

    param.ext.channel_group = 1;
    for (i = 0; i < group; i++)
    {
        vx_tensor bias;

        if ( inputs[2] == NULL )
        {
            bias = NULL;
        }
        else
        {
            bias = LOCAL()->bias_tensor_group[i]->t;
        }

        self->n = vxDeconvolutionLayer(
            self->graph->g,
            LOCAL()->input_tensor_group[i]->t,
            LOCAL()->weight_tensor_group[i]->t,
            bias,
            (vx_nn_deconvolution_params_t *)&param,
            sizeof( vx_nn_deconvolution_params_ext2_t ),
            LOCAL()->output_tensor_group[i]->t
            );
        if ( NULL == self->n )
        {
            VSILOGE("Add vxConvolutionLayer fail, (GROUPED_DECONV2D) at [%s : %d]\n", __FILE__, __LINE__);
            status = VSI_FAILURE;
            goto final;
        }
        else
        {
            // no need to maintain self->n
            vxReleaseNode( &self->n );
            status = VSI_SUCCESS;
            self->n = NULL;
        }
    }

final:
    if (LOCAL())
    {
        if (LOCAL()->input_tensor_group)
        {
            for (i = 0; i < group; i++)
            {
                vsi_safe_release_tensor((LOCAL()->input_tensor_group[i]));
            }
            vsi_nn_safe_free(LOCAL()->input_tensor_group);
        }
        if (LOCAL()->weight_tensor_group)
        {
            for (i = 0; i < group; i++)
            {
                vsi_safe_release_tensor((LOCAL()->weight_tensor_group[i]));
            }
            vsi_nn_safe_free(LOCAL()->weight_tensor_group);
        }
        if (LOCAL()->bias_tensor_group != NULL)
        {
            for (i = 0; i < group; i++)
            {
                vsi_safe_release_tensor((LOCAL()->bias_tensor_group[i]));
            }
            vsi_nn_safe_free(LOCAL()->bias_tensor_group);
        }
        if (LOCAL()->output_tensor_group != NULL)
        {
            for (i = 0; i < group; i++)
            {
                vsi_safe_release_tensor((LOCAL()->output_tensor_group[i]));
            }
            vsi_nn_safe_free(LOCAL()->output_tensor_group);
        }

        vsi_nn_safe_free(LOCAL());
    }
    return status;
} /* op_compute() */

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
            if ( NULL == permute_tensor )
            {
                status = VSI_FAILURE;
                goto final;
            }
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
        if ( NULL == reverse_tensor )
        {
            status = VSI_FAILURE;
            goto final;
        }
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
        if ( NULL == permute_tensor )
        {
            status = VSI_FAILURE;
            goto final;
        }
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

    if (self->nn_param.deconv.group > 1 &&
        self->nn_param.deconv.group < inputs[0]->attr.size[2])
    {
        vsi_nn_tensor_t *inputs_tensors[3] = {NULL};

        inputs_tensors[0] = inputs[0];
        inputs_tensors[1] = weight_tensor;
        inputs_tensors[2] = inputs[2];
        status = op_grouped_compute(self, inputs_tensors, outputs, param );
    }
    else
    {
        self->n = vxDeconvolutionLayer(
            self->graph->g,
            inputs[0]->t,
            weight_tensor->t,
            (NULL == inputs[2]) ? NULL : inputs[2]->t,
            (vx_nn_deconvolution_params_t *)&param,
            sizeof( vx_nn_deconvolution_params_ext2_t ),
            outputs[0]->t
            );
        if ( NULL != self->n )
        {
            status = VSI_SUCCESS;
        }
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

    ret = vsi_nn_OpCheck(VSI_NN_OP_CONV2D, self, inputs, outputs);

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
    vsi_size_t perm[] = { 3, 2, 0, 1 };
    vsi_size_t perm1[] = { 0, 1, 3, 2 };

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

    nn_param->ksize[0] = (uint32_t)inputs[1]->attr.size[0];
    nn_param->ksize[1] = (uint32_t)inputs[1]->attr.size[1];

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
