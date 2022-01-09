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

/****************************************************************************
*  This operation originally come from:
*  https://github.com/pjreddie/darknet/tree/master/src/upsample_layer.c
*  which is used by YOLOv3
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
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "vsi_nn_internal_node.h"

#define _ARG_NUM            (1)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)


static vsi_bool _is_same_shape
    (
    vsi_nn_tensor_t * inputs,
    vsi_size_t *sizes,
    uint32_t dims
    )
{
    uint32_t i = 0;

    if (inputs->attr.dim_num != dims)
        return FALSE;

    for (i = 0; i < dims; i++)
    {
        if (sizes[i] != inputs->attr.size[i])
            return FALSE;
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
    vsi_status status = VSI_FAILURE;

    if ( self->nn_param.resize.lcl_data->use_internal_node )
    {
        status = vsi_nn_internal_compute_node( self );
    }
    else
    {
        vx_nn_scale_params_t para;
        switch (self->nn_param.resize.type)
        {
            case VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR:
                para.type = VX_INTERPOLATION_NEAREST_NEIGHBOR; break;
            case VSI_NN_INTERPOLATION_BILINEAR:
                para.type = VX_INTERPOLATION_BILINEAR; break;
            case VSI_NN_INTERPOLATION_AREA:
                para.type = VX_INTERPOLATION_AREA; break;
            default:
                para.type = VX_INTERPOLATION_NEAREST_NEIGHBOR;
        }
        self->n = vxTensorScaleNode( self->graph->g, inputs[0]->t, &para,
            sizeof(vx_nn_scale_params_t), outputs[0]->t );
        if( NULL != self->n )
        {
            status = VSI_SUCCESS;
        }
    }

    return status;
} /* op_compute() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    if ( self->nn_param.resize.lcl_data->use_internal_node )
    {
        return vsi_nn_internal_optimize_node(self, direction );
    }
    else
    {
        return VSI_SUCCESS;
    }
} /* op_optimize() */

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

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* TODO: Add code to comput outputs' shape. */
    float factor = self->nn_param.resize.factor;
    vsi_enum layout = self->nn_param.resize.layout;
    vsi_nn_internal_node_t* curr = NULL;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        if (factor != 0)
        {
            if (layout == VSI_NN_RESIZE_LAYOUT_NCHW)
            {
                outputs[0]->attr.size[0] = (uint32_t)(inputs[0]->attr.size[0] * factor);
                outputs[0]->attr.size[1] = (uint32_t)(inputs[0]->attr.size[1] * factor);
            }
            else
            {
                outputs[0]->attr.size[1] = (uint32_t)(inputs[0]->attr.size[1] * factor);
                outputs[0]->attr.size[2] = (uint32_t)(inputs[0]->attr.size[2] * factor);
            }
        }
        else
        {
            if (layout == VSI_NN_RESIZE_LAYOUT_NCHW)
            {
                outputs[0]->attr.size[0] = self->nn_param.resize.size[0];
                outputs[0]->attr.size[1] = self->nn_param.resize.size[1];
            }
            else
            {
                outputs[0]->attr.size[1] = self->nn_param.resize.size[0];
                outputs[0]->attr.size[2] = self->nn_param.resize.size[1];
            }
        }
        if (layout == VSI_NN_RESIZE_LAYOUT_NCHW)
        {
            outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
            outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
        }
        else
        {
            outputs[0]->attr.size[0] = inputs[0]->attr.size[0];
            outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
        }
    }

    if ( ( self->nn_param.resize.align_corners ||
           self->nn_param.resize.half_pixel_centers ||
           layout == VSI_NN_RESIZE_LAYOUT_NHWC )
       && ( VSI_NN_INTERPOLATION_BILINEAR == self->nn_param.resize.type ) )
    {
        self->nn_param.resize.lcl_data->use_internal_node = TRUE;

        vsi_nn_internal_init_node_wksp( self );
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESIZE_INTERNAL, 0, 0 );
        curr->node->nn_param.resize_internal.align_corners = self->nn_param.resize.align_corners;
        curr->node->nn_param.resize_internal.factor = self->nn_param.resize.factor;
        curr->node->nn_param.resize_internal.half_pixel_centers = self->nn_param.resize.half_pixel_centers;
        curr->node->nn_param.resize_internal.layout = self->nn_param.resize.layout;
        curr->inputs[0]  = inputs[0];
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
    }
    else if ((self->nn_param.resize.align_corners || self->nn_param.resize.half_pixel_centers)
            && (VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR == self->nn_param.resize.type))
    {
        self->nn_param.resize.lcl_data->use_internal_node = TRUE;

        vsi_nn_internal_init_node_wksp( self );
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESIZE_NEAREST_INTERNAL, 0, 0 );
        curr->node->nn_param.resize_nearest_internal.align_corners = self->nn_param.resize.align_corners;
        curr->node->nn_param.resize_nearest_internal.factor = self->nn_param.resize.factor;
        curr->node->nn_param.resize_nearest_internal.half_pixel_centers = self->nn_param.resize.half_pixel_centers;
        curr->inputs[0]  = inputs[0];
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
    }
    else if (_is_same_shape(inputs[0], outputs[0]->attr.size, outputs[0]->attr.dim_num))
    {
        self->nn_param.resize.lcl_data->use_internal_node = TRUE;

        vsi_nn_internal_init_node_wksp( self );
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
        curr->inputs[0]  = inputs[0];
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{

    if (self->nn_param.resize.lcl_data->use_internal_node)
    {
        vsi_nn_safe_free(self->nn_param.resize.lcl_data);
        vsi_nn_internal_deinit_node_wksp(self);
    }
    else
    {
        vsi_nn_safe_free(self->nn_param.resize.lcl_data);
        vsi_nn_op_common_deinit(self);
    }

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.resize.lcl_data =
        (vsi_nn_resize_local_data *)malloc( sizeof(vsi_nn_resize_local_data) );
    if( NULL == self->nn_param.resize.lcl_data )
    {
        VSILOGE( "Create resize local data fail." );
        status = VSI_FAILURE;
        goto final;
    }
    memset( self->nn_param.resize.lcl_data, 0, sizeof(vsi_nn_resize_local_data) );

    if (vsi_nn_compareVersion(self->graph, 1, 1, 14) == -1)
    {
        self->nn_param.resize.align_corners      = FALSE;
        self->nn_param.resize.half_pixel_centers = FALSE;
    }

    self->nn_param.resize.layout = VSI_NN_RESIZE_LAYOUT_NCHW;

final:
    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RESIZE,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
