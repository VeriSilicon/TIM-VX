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
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

typedef struct _deconv3d_local_data_t {
    int32_t placeholder;
} deconv3d_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)

#define COMPUTE_DECONV_SZ( in, ksize, pad_1, pad_2, stride, output_padding )\
    (( in - 1 ) * stride + ksize - pad_1 - pad_2 + output_padding)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;

    // Create kernel param
    vsi_nn_kernel_param_t * param;
    //vsi_nn_kernel_node_t    n;
    param = vsi_nn_kernel_param_create();

    // Add params
#define MAP_PARAM(type_name, value) {\
    vsi_nn_kernel_param_add_int32( param, type_name, value); \
    }

    MAP_PARAM("stride_w",self->nn_param.deconv3d.stride[0]);
    MAP_PARAM("stride_h",self->nn_param.deconv3d.stride[1]);
    MAP_PARAM("stride_d",self->nn_param.deconv3d.stride[2]);

    MAP_PARAM("outpadding_w",self->nn_param.deconv3d.output_padding[0]);
    MAP_PARAM("outpadding_h",self->nn_param.deconv3d.output_padding[1]);
    MAP_PARAM("outpadding_d",self->nn_param.deconv3d.output_padding[2]);

    MAP_PARAM("pad_left",self->nn_param.deconv3d.pad[0]);
    MAP_PARAM("pad_right",self->nn_param.deconv3d.pad[1]);
    MAP_PARAM("pad_top",self->nn_param.deconv3d.pad[2]);
    MAP_PARAM("pad_bottom",self->nn_param.deconv3d.pad[3]);
    MAP_PARAM("pad_front",self->nn_param.deconv3d.pad[4]);
    MAP_PARAM("pad_end",self->nn_param.deconv3d.pad[5]);

    MAP_PARAM("weights",self->nn_param.deconv3d.weights);
    MAP_PARAM("group",self->nn_param.deconv3d.group);

    MAP_PARAM("overflow_policy",self->vx_param.overflow_policy);
    MAP_PARAM("rounding_policy",self->vx_param.rounding_policy);
    MAP_PARAM("down_scale_size_rounding",self->vx_param.down_scale_size_rounding);

#undef MAP_PARAM
    self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "deconv3d",
            inputs, 3, outputs, 1, param );
    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );
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

void _rotate_weight_data(
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * weights)
{
    vsi_ssize_t oc = 0, ic = 0;
    uint8_t* weight_data = NULL;
    uint8_t* buffer = NULL;
    vsi_ssize_t kernel_size_w = weights->attr.size[0];
    vsi_ssize_t kernel_size_h = weights->attr.size[1];
    vsi_ssize_t kernel_size_d = weights->attr.size[2];
    vsi_ssize_t weight_ic = weights->attr.size[3];
    vsi_ssize_t weight_oc = weights->attr.size[4];
    vsi_ssize_t slice_size = kernel_size_w * kernel_size_h;
    vsi_ssize_t depth_size = slice_size * kernel_size_d;
    int32_t item_size = vsi_nn_TypeGetBytes(weights->attr.dtype.vx_type);

    weight_data = vsi_nn_ConvertTensorToData(graph, weights);
    buffer = (uint8_t*)malloc(item_size * depth_size * weight_ic * weight_oc);
    memset(buffer, 0x00, item_size * depth_size * weight_ic * weight_oc);
    //memcpy(buffer, weight_data, item_size * slice_size * weight_ic * weight_oc);
    for(oc = 0; oc < weight_oc; oc++)
    {
        for(ic = 0; ic < weight_ic; ic++)
        {
            vsi_ssize_t d, h, w;
            vsi_ssize_t offset = item_size * depth_size * (oc * weight_ic + ic);
            for(d = 0; d < kernel_size_d; d++)
            {
                uint8_t *src_depth = weight_data + offset +  (kernel_size_d - d - 1) * item_size * slice_size;
                uint8_t *dst_depth = buffer + offset + d * item_size * slice_size;
                for(h = 0; h < kernel_size_h; h ++)
                {
                    uint8_t *dst_height = dst_depth + h * kernel_size_w * item_size;
                    uint8_t *src_height = src_depth + (kernel_size_h - 1 - h) * kernel_size_w * item_size;
                    for(w = 0; w < kernel_size_w; w++)
                    {
                        memcpy(dst_height + w * item_size,
                            src_height + (kernel_size_w - 1 - w) * item_size,
                            item_size);
                    }
                }
            }
        }
    }

    vsi_nn_CopyDataToTensor( graph, weights, buffer );
    vsi_nn_Free( buffer );
    vsi_nn_safe_free( weight_data );
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_deconv3d_param *nn_param;

    if ( vsi_nn_compareVersion(self->graph, 1, 1, 21) == -1 )
    {
        self->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    }

    /* Rotate 180 degrees for weights data */
    if (TRUE == inputs[1]->attr.is_const)
    {
        _rotate_weight_data(self->graph, inputs[1]);
    }
    else
    {
         VSILOGE("deconv3d: do not support dynamic weight");
    }

    nn_param = &self->nn_param.deconv3d;

    nn_param->group = ( 0 == nn_param->group ) ? 1 : nn_param->group;
    nn_param->ksize[0] = (uint32_t)inputs[1]->attr.size[0];
    nn_param->ksize[1] = (uint32_t)inputs[1]->attr.size[1];
    nn_param->ksize[2] = (uint32_t)inputs[1]->attr.size[2];

    if(nn_param->group != 1)
    {
        VSILOGE("deconv3d: only support group == 1, but group is %d", nn_param->group);
        return FALSE;
    }

    if(nn_param->ksize[2] < nn_param->stride[2])
    {
        VSILOGE("deconv3d: only support kernel_depth < stride_depth,but \
            kernel_depth = %d, stried_depth = %d", nn_param->ksize[2], nn_param->stride[2]);
        return FALSE;
    }

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
        outputs[0]->attr.size[2] = COMPUTE_DECONV_SZ(
            inputs[0]->attr.size[2],
            nn_param->ksize[2],
            nn_param->pad[4],
            nn_param->pad[5],
            nn_param->stride[2],
            nn_param->output_padding[2]
        );
        if(self->nn_param.deconv3d.weights > 0)
        {
            outputs[0]->attr.size[3] = self->nn_param.deconv3d.weights;
        }
        else
        {
            outputs[0]->attr.size[3] = inputs[1]->attr.size[3];
        }
        outputs[0]->attr.size[4] = inputs[0]->attr.size[4];
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    /* TODO
    //self->nn_param.deconv3d.local = \
    //    (deconv3d_local_data_t*)malloc(sizeof(deconv3d_local_data_t));
    */

    return VSI_SUCCESS;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_op_common_deinit(self);

    /* TODO
    //vsi_nn_safe_free(self->nn_param.deconv3d.local);
    */

    return status;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ DECONV3D,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS