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
#include "utils/vsi_nn_constraint_check.h"


/*
 Declare number of input and output.
 */
#define _ARG_NUM            (1)
#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define LOCAL() ((vsi_nn_grouped_conv3d_param_local_data *)nn_param->local)

typedef struct _vsi_nn_grouped_conv3d_param_local_data {
    vsi_nn_tensor_t ** input_tensor_group;
    vsi_nn_tensor_t ** weight_tensor_group;
    vsi_nn_tensor_t ** bias_tensor_group;
    vsi_nn_tensor_t ** output_tensor_group;
} vsi_nn_grouped_conv3d_param_local_data;

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
#if VX_CONV_3D_API_SUPPORT
#define _TENSOR_LEN 64
    vsi_bool res;
    uint32_t i;
    char tensor_name[_TENSOR_LEN];
    vsi_nn_grouped_conv3d_param *nn_param = &self->nn_param.grouped_conv3d;
    nn_param->local = (vsi_nn_grouped_conv3d_param_local_data*)malloc(
        sizeof(vsi_nn_grouped_conv3d_param_local_data));
    if (NULL == nn_param->local)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV3D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(nn_param->local, 0, sizeof(vsi_nn_grouped_conv3d_param_local_data));
    LOCAL()->input_tensor_group = (vsi_nn_tensor_t **)malloc(
        nn_param->group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->input_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV3D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(LOCAL()->input_tensor_group, 0, nn_param->group * sizeof(vsi_nn_tensor_t *));
    res = vsi_nn_CreateTensorGroup(self->graph, inputs[0], 3,
        LOCAL()->input_tensor_group, nn_param->group);
    if (res == FALSE)
    {
        VSILOGE("CreateTensorGroup fail, (GROUPED_CONV3D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }

    LOCAL()->weight_tensor_group = (vsi_nn_tensor_t **)malloc(
        nn_param->group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->weight_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV3D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(LOCAL()->weight_tensor_group, 0, nn_param->group * sizeof(vsi_nn_tensor_t *));
    res = vsi_nn_CreateTensorGroup(self->graph, inputs[1], 4,
        LOCAL()->weight_tensor_group, nn_param->group);
    if (res == FALSE)
    {
        VSILOGE("CreateTensorGroup fail, (GROUPED_CONV3D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }

    LOCAL()->bias_tensor_group = (vsi_nn_tensor_t **)malloc(
        nn_param->group * sizeof(vsi_nn_tensor_t *));
    if (NULL == LOCAL()->bias_tensor_group)
    {
        VSILOGE("Malloc fail, (GROUPED_CONV3D) at [%s : %d]\n", __FILE__, __LINE__);
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
        VSILOGE("Malloc fail, (GROUPED_CONV3D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(LOCAL()->output_tensor_group, 0, nn_param->group * sizeof(vsi_nn_tensor_t *));
    res = vsi_nn_CreateTensorGroup(self->graph, outputs[0], 3,
        LOCAL()->output_tensor_group, nn_param->group);
    if (res == FALSE)
    {
        VSILOGE("CreateTensorGroup fail, (GROUPED_CONV3D) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }

    for (i = 0; i < nn_param->group; i++)
    {
        vx_tensor bias;
        vx_nn_convolution_3d_params_t *param = NULL;
        vx_nn_convolution_3d_params_t param_;
        memset( &param_, 0, sizeof( vx_nn_convolution_3d_params_t ) );
        param = &param_;
        param->padding_w_left = self->nn_param.grouped_conv3d.pad[0];
        param->padding_w_right = self->nn_param.grouped_conv3d.pad[1];
        param->padding_h_top = self->nn_param.grouped_conv3d.pad[2];
        param->padding_h_bottom = self->nn_param.grouped_conv3d.pad[3];
        param->padding_d_front = self->nn_param.grouped_conv3d.pad[4];
        param->padding_d_rear = self->nn_param.grouped_conv3d.pad[5];

        param->stride_w = self->nn_param.grouped_conv3d.stride[0];
        param->stride_h = self->nn_param.grouped_conv3d.stride[1];
        param->stride_d = self->nn_param.grouped_conv3d.stride[2];

        if (self->nn_param.grouped_conv3d.dilation[0] *
            self->nn_param.grouped_conv3d.dilation[1] *
            self->nn_param.grouped_conv3d.dilation[2] > 1)
        {
            VSILOGE("conv3d could not support dilation > 1\n");
            return VSI_FAILURE;
        }
        if ( self->nn_param.grouped_conv3d.dilation[0] > 0 )
        {
            param->dilation_w = self->nn_param.grouped_conv3d.dilation[0] - 1;
        }
        if ( self->nn_param.grouped_conv3d.dilation[1] > 0 )
        {
            param->dilation_h = self->nn_param.grouped_conv3d.dilation[1] - 1;
        }
        if ( self->nn_param.grouped_conv3d.dilation[2] > 0 )
        {
            param->dilation_d = self->nn_param.grouped_conv3d.dilation[2] - 1;
        }
        param->pad_mode = vsi_nn_get_vx_pad_mode(nn_param->pad_mode);
        param->depth_multiplier = self->nn_param.grouped_conv3d.multiplier;
        param->overflow_policy = self->vx_param.overflow_policy;
        param->rounding_policy = self->vx_param.rounding_policy;
        param->down_scale_size_rounding = self->vx_param.down_scale_size_rounding;

        if ( inputs[2] == NULL )
        {
            bias = NULL;
        }
        else
        {
            bias = LOCAL()->bias_tensor_group[i]->t;
        }

        self->n = vxConv3dLayer(
            self->graph->g,
            LOCAL()->input_tensor_group[i]->t,
            LOCAL()->weight_tensor_group[i]->t,
            bias,
            (vx_nn_convolution_3d_params_t* )param,
            sizeof( vx_nn_convolution_3d_params_t),
            LOCAL()->output_tensor_group[i]->t
            );

        memset(tensor_name, 0, sizeof(tensor_name));
        snprintf(tensor_name, sizeof(tensor_name), "uid_%u_sub_uid_%u_out_0", self->uid, i);
        if (vxSetReferenceName((vx_reference)LOCAL()->output_tensor_group[i]->t, tensor_name) == VSI_FAILURE)
        {
            VSILOGW("Set uid %u copy node output name fail", self->uid);
            return VSI_FAILURE;
        }
        if ( NULL == self->n )
        {
            VSILOGE("Add vxConvolutionLayer fail, (GROUPED_CONV3D) at [%s : %d]\n", __FILE__, __LINE__);
            return VSI_FAILURE;
        }
        else
        {
            // no need to maintain self->n
            vxReleaseNode( &self->n );
            self->n = NULL;
        }
    }
#else
    VSI_UNREFERENCED(self);
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
#endif
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

    ret = vsi_nn_OpCheck(VSI_NN_OP_CONV3D, self, inputs, outputs);

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
    vsi_nn_grouped_conv3d_param *nn_param;
    vsi_size_t perm[] = { 3, 2, 0, 1 };

#ifdef VX_CONVERT_POLICY_WRAP_ENABLE
    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }
#endif

    if ( VSI_NN_DIM_FMT_NHWC == inputs[1]->attr.dtype.fmt &&
        VSI_NN_TYPE_VDATA != inputs[1]->attr.dtype.vx_type )
    {
        vsi_nn_TransposeTensor( self->graph, inputs[1], perm, 4, NULL );
        inputs[1]->attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;
    }

    nn_param = &self->nn_param.grouped_conv3d;
    {
        vsi_size_t i, pad[_cnt_of_array(nn_param->pad)] = {0};
        for (i = 0; i < _cnt_of_array(nn_param->pad); i++)
        {
            pad[i] = self->nn_param.grouped_conv3d.pad[i];
        }
        vsi_nn_compute_padding_3d(
            inputs[0]->attr.size,
            inputs[1]->attr.size,
            nn_param->stride,
            nn_param->dilation,
            nn_param->pad_type,
            pad
        );
        for (i = 0; i < _cnt_of_array(nn_param->pad); i++)
        {
            self->nn_param.grouped_conv3d.pad[i] = (uint32_t)pad[i];
        }
    }

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
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
        outputs[0]->attr.size[2] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[2],
            inputs[1]->attr.size[2],
            &nn_param->pad[4],
            nn_param->stride[2],
            nn_param->dilation[2],
            VSI_NN_ROUND_FLOOR
            );
        if (self->nn_param.grouped_conv3d.weights > 0)
        {
            outputs[0]->attr.size[3] = self->nn_param.grouped_conv3d.weights;
        }
        else if (self->nn_param.grouped_conv3d.multiplier > 0)
        {
            outputs[0]->attr.size[3] = inputs[0]->attr.size[3] * self->nn_param.grouped_conv3d.multiplier;
        }
        else
        {
            outputs[0]->attr.size[3] = inputs[1]->attr.size[4];
        }
        outputs[0]->attr.size[4] = inputs[0]->attr.size[4];
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    return TRUE;
} /* op_setup() */


static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_nn_grouped_conv3d_param *nn_param = &(self->nn_param.grouped_conv3d);
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

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GROUPED_CONV3D,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

