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
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _ARG_NUM            (1)
#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define LOCAL() ((vsi_nn_grouped_conv2d_param_local_data *)nn_param->local)

typedef struct _vsi_nn_grouped_conv2d_param_local_data {
    vsi_nn_tensor_t ** input_tensor_group;
    vsi_nn_tensor_t ** weight_tensor_group;
    vsi_nn_tensor_t ** bias_tensor_group;
    vsi_nn_tensor_t ** output_tensor_group;
} vsi_nn_grouped_conv2d_param_local_data;

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool res;
    uint32_t i;
    vsi_nn_grouped_conv2d_param *nn_param = &self->nn_param.grouped_conv2d;
    nn_param->local = (vsi_nn_grouped_conv2d_param_local_data*)malloc(
        sizeof(vsi_nn_grouped_conv2d_param_local_data));
    if (NULL == nn_param->local)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(nn_param->local, 0, sizeof(vsi_nn_grouped_conv2d_param_local_data));
    /* TODO */
    /* example code : add op */
    /*
    self->n = vxTensorAddNode( self->graph->g, inputs[0]->t, inputs[1]->t,
        VX_CONVERT_POLICY_SATURATE, outputs[0]->t );
    */
    LOCAL()->input_tensor_group = (vsi_nn_tensor_t **)malloc(
        nn_param->group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->input_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(LOCAL()->input_tensor_group, 0, nn_param->group * sizeof(vsi_nn_tensor_t *));
    res = vsi_nn_CreateTensorGroup(self->graph, inputs[0], 2,
        LOCAL()->input_tensor_group, nn_param->group);
    if (res == FALSE)
    {
        VSILOGE("CreateTensorGroup fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }

    LOCAL()->weight_tensor_group = (vsi_nn_tensor_t **)malloc(
        nn_param->group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->weight_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(LOCAL()->weight_tensor_group, 0, nn_param->group * sizeof(vsi_nn_tensor_t *));
    res = vsi_nn_CreateTensorGroup(self->graph, inputs[1], 3,
        LOCAL()->weight_tensor_group, nn_param->group);
    if (res == FALSE)
    {
        VSILOGE("CreateTensorGroup fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }

    LOCAL()->bias_tensor_group = (vsi_nn_tensor_t **)malloc(
        nn_param->group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->bias_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(LOCAL()->bias_tensor_group, 0, nn_param->group * sizeof(vsi_nn_tensor_t *));
    if (inputs[2] != NULL)
    {
        res = vsi_nn_CreateTensorGroup(self->graph, inputs[2], 0,
            LOCAL()->bias_tensor_group, nn_param->group);
        if (res == FALSE)
        {
            VSILOGE("CreateTensorGroup fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
            return VSI_FAILURE;
        }
    }

    LOCAL()->output_tensor_group = (vsi_nn_tensor_t **)malloc(
        nn_param->group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->output_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(LOCAL()->output_tensor_group, 0, nn_param->group * sizeof(vsi_nn_tensor_t *));
    res = vsi_nn_CreateTensorGroup(self->graph, outputs[0], 2,
        LOCAL()->output_tensor_group, nn_param->group);
    if (res == FALSE)
    {
        VSILOGE("CreateTensorGroup fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }

    for (i = 0; i < nn_param->group; i++)
    {
        vx_tensor bias;
        vx_nn_convolution_params_ext_t *p_ext = NULL;
        vx_nn_convolution_params_ext2_t *p_ext2 = NULL;
        vx_nn_convolution_params_ext2_t param_ext2;
        memset( &param_ext2, 0, sizeof( vx_nn_convolution_params_ext2_t ) );
        p_ext2 = &param_ext2;
        p_ext = &p_ext2->ext;

        //set ext relative parameters
        p_ext->khr.padding_x = self->nn_param.conv2d.pad[0];
        p_ext->khr.padding_y = self->nn_param.conv2d.pad[2];
        if (self->nn_param.conv2d.dilation[0] > 0)
        {
            p_ext->khr.dilation_x = self->nn_param.conv2d.dilation[0] - 1;
        }
        if (self->nn_param.conv2d.dilation[1] > 0)
        {
            p_ext->khr.dilation_y = self->nn_param.conv2d.dilation[1] - 1;
        }
        p_ext->khr.overflow_policy = self->vx_param.overflow_policy;
        p_ext->khr.rounding_policy =  self->vx_param.rounding_policy;
        p_ext->khr.down_scale_size_rounding = self->vx_param.down_scale_size_rounding;

        p_ext->padding_x_right = self->nn_param.conv2d.pad[1];
        p_ext->padding_y_bottom = self->nn_param.conv2d.pad[3];
        p_ext->pad_mode = vsi_nn_get_vx_pad_mode(nn_param->pad_mode);

        //set ext2 relative parameters
        p_ext2->depth_multiplier = self->nn_param.conv2d.multiplier;
        p_ext2->stride_x = self->nn_param.conv2d.stride[0];
        p_ext2->stride_y = self->nn_param.conv2d.stride[1];

        if( inputs[2] == NULL )
        {
            bias = NULL;
        }
        else
        {
            bias = LOCAL()->bias_tensor_group[i]->t;
        }

        self->n = vxConvolutionLayer(
            self->graph->g,
            LOCAL()->input_tensor_group[i]->t,
            LOCAL()->weight_tensor_group[i]->t,
            bias,
            (vx_nn_convolution_params_t *)p_ext2,
            sizeof(vx_nn_convolution_params_ext2_t),
            LOCAL()->output_tensor_group[i]->t
            );
        if( NULL == self->n )
        {
            VSILOGE("Add vxConvolutionLayer fail, (GROUPED_CONV2D) at [%s : %d]\n", __FILE__, __LINE__);
            return VSI_FAILURE;
        }
        else
        {
            // no need to maintain self->n
            vxReleaseNode( &self->n );
            self->n = NULL;
        }
    }
    return VSI_SUCCESS;
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
    /* TODO: Add code to comput outputs' shape. */
    vsi_nn_grouped_conv2d_param *nn_param;
    vsi_size_t perm[] = { 3, 2, 0, 1 };

    /* TODO: Driver should handle this,
    * Check transpose
    * */
#ifdef VX_CONVERT_POLICY_WRAP_ENABLE
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }
#endif

    if( VSI_NN_DIM_FMT_NHWC == inputs[1]->attr.dtype.fmt &&
        VSI_NN_TYPE_VDATA != inputs[1]->attr.dtype.vx_type )
    {
        vsi_nn_TransposeTensor( self->graph, inputs[1], perm, 4, NULL );
        inputs[1]->attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;
    }

    nn_param = &self->nn_param.grouped_conv2d;
    {
        vsi_size_t i, pad[_cnt_of_array(nn_param->pad)] = {0};
        for(i = 0; i < _cnt_of_array(nn_param->pad); i++)
        {
            pad[i] = self->nn_param.conv2d.pad[i];
        }
        vsi_nn_compute_padding(
            inputs[0]->attr.size,
            inputs[1]->attr.size,
            nn_param->stride,
            nn_param->dilation,
            nn_param->pad_type,
            pad
        );
        for(i = 0; i < _cnt_of_array(nn_param->pad); i++)
        {
            self->nn_param.conv2d.pad[i] = (uint32_t)pad[i];
        }
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[0],
            inputs[1]->attr.size[0],
            &nn_param->pad[0],
            nn_param->stride[0],
            nn_param->dilation[0],
            VSI_NN_ROUND_FLOOR
            );
        outputs[0]->attr.size[1] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[1],
            inputs[1]->attr.size[1],
            &nn_param->pad[2],
            nn_param->stride[1],
            nn_param->dilation[1],
            VSI_NN_ROUND_FLOOR
            );
        if(self->nn_param.conv2d.weights > 0)
        {
            outputs[0]->attr.size[2] = self->nn_param.conv2d.weights;
        }
        else if(self->nn_param.conv2d.multiplier > 0)
        {
            outputs[0]->attr.size[2] = inputs[0]->attr.size[2] * self->nn_param.conv2d.multiplier;
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

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_grouped_conv2d_param *nn_param = &(self->nn_param.grouped_conv2d);
    uint32_t i;
    if (LOCAL())
    {
        if (LOCAL()->input_tensor_group)
        {
            for (i = 0; i < nn_param->group; i++)
            {
                vsi_nn_ReleaseTensor(&(LOCAL()->input_tensor_group[i]));
            }
            free(LOCAL()->input_tensor_group);
        }
        if (LOCAL()->weight_tensor_group)
        {
            for (i = 0; i < nn_param->group; i++)
            {
                vsi_nn_ReleaseTensor(&(LOCAL()->weight_tensor_group[i]));
            }
            free(LOCAL()->weight_tensor_group);
        }
        if (LOCAL()->bias_tensor_group != NULL)
        {
            for (i = 0; i < nn_param->group; i++)
            {
                vsi_nn_ReleaseTensor(&(LOCAL()->bias_tensor_group[i]));
            }
            free(LOCAL()->bias_tensor_group);
        }
        if (LOCAL()->output_tensor_group != NULL)
        {
            for (i = 0; i < nn_param->group; i++)
            {
                vsi_nn_ReleaseTensor(&(LOCAL()->output_tensor_group[i]));
            }
            free(LOCAL()->output_tensor_group);
        }

        free(LOCAL());
    }
    vsi_nn_op_common_deinit(self);
    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GROUPED_CONV2D,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
