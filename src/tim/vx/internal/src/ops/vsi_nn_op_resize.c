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
#include "client/vsi_nn_vxkernel.h"
#include "vsi_nn_internal_node.h"

#define _ARG_NUM            (1)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define USE_OVX_API TRUE

#if (USE_OVX_API == FALSE)
extern vx_kernel_description_t * vx_kernel_RESIZE_list[];

static void _set_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < _INPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)inputs[i]->t;
    }

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)outputs[i]->t;
    }
} /* _set_inputs_outputs() */

static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_resize_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &node->nn_param.resize;
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, factor );
#undef _SET_PARAM
set_param_error:

    return status;
} /* _create_params */

static void _release_params
    (
    vx_reference * params,
    uint32_t num
    )
{
    uint32_t i;
    vx_scalar scalar;
    for( i = 0; i < num; i ++ )
    {
        scalar = (vx_scalar)params[i];
        vxReleaseScalar( &scalar );
    }
} /* _release_params() */

static vsi_status op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e inputFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputFormat = outputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e enableFormat;
    float scale_factor = self->nn_param.resize.factor;

    enableFormat = ((inputFormat == VSI_NN_TYPE_FLOAT16 && outputFormat == VSI_NN_TYPE_FLOAT16) ||
        (inputFormat == VSI_NN_TYPE_INT16 && outputFormat == VSI_NN_TYPE_INT16) ||
        (inputFormat == VSI_NN_TYPE_INT8 && outputFormat == VSI_NN_TYPE_INT8) ||
        (inputFormat == VSI_NN_TYPE_UINT8 && outputFormat == VSI_NN_TYPE_UINT8));

    if(scale_factor == 0.5f && enableFormat && inputs[0]->attr.size[1] % 2 == 0
        && inputs[0]->attr.size[1] * inputs[0]->attr.size[2] < 65536)
    {
        kernel_info->type = VX_KERNEL_TYPE_VX;
        kernel_info->init_index = 1;
        if (inputFormat == VX_TYPE_FLOAT16  || inputFormat == VX_TYPE_INT16 )
        {
            kernel_info->kernel_index = 1;
        }
        else
        {
            kernel_info->kernel_index = 2;
        }
    }
    else
    {
        kernel_info->type = VX_KERNEL_TYPE_CPU;
        kernel_info->kernel_index = 0;
        kernel_info->init_index = 0;
    }

    return VSI_SUCCESS;
}

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_reference * args;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( params, inputs, outputs );

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}

static vsi_status vx_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_IO_NUM];
    vx_border_t border;
    int32_t sizes[4] = {0};
    uint32_t dims   = 2;
    uint32_t input_size[4] = {1, 1, 1, 1};
    uint32_t output_size[4] = {1, 1, 1, 1};
    uint32_t i;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    for(i = 0; i < inputs[0]->attr.dim_num; ++i)
    {
        input_size[i] = inputs[0]->attr.size[i];
    }
    for(i = 0; i < outputs[0]->attr.dim_num; ++i)
    {
        output_size[i] = outputs[0]->attr.size[i];
    }


    sizes[0] = input_size[0];
    sizes[1] = input_size[1] * input_size[2] * input_size[3];
    self->nn_param.resize.local.local_tensor[0] = vxReshapeTensor(inputs[0]->t, sizes, dims);

    sizes[0] = output_size[0];
    sizes[1] = output_size[1] * output_size[2] * output_size[3];
    self->nn_param.resize.local.local_tensor[1] = vxReshapeTensor(outputs[0]->t, sizes, dims);

    params[0] = (vx_reference)self->nn_param.resize.local.local_tensor[0];
    params[1] = (vx_reference)self->nn_param.resize.local.local_tensor[1];

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _IO_NUM );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute,
    NULL
};
#endif

static vsi_bool _is_same_shape
    (
    vsi_nn_tensor_t * inputs,
    uint32_t *sizes,
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
#if (USE_OVX_API == TRUE)
    if ( ((self->nn_param.resize.align_corners || self->nn_param.resize.half_pixel_centers)
       && (VSI_NN_INTERPOLATION_BILINEAR == self->nn_param.resize.type
          || VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR == self->nn_param.resize.type))
       || _is_same_shape(inputs[0], outputs[0]->attr.size, outputs[0]->attr.dim_num) )
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
#else

    vsi_nn_kernel_info_t kernel_info;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name = "vsi_nn_kernel_resize";
    kernel_info.kernel = vx_kernel_RESIZE_list;

    op_pre_compute(self, inputs, outputs, &kernel_info);

    self->n = vsi_nn_RegisterClientKernelAndNewNode(
        self->graph, &kernel_info);
    if (kernel_info.resource_name) free(kernel_info.resource_name);
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }
    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, inputs, outputs);
    }
#endif
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
    if ( ((self->nn_param.resize.align_corners || self->nn_param.resize.half_pixel_centers)
       && (VSI_NN_INTERPOLATION_BILINEAR == self->nn_param.resize.type
          || VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR == self->nn_param.resize.type) )
        || _is_same_shape(inputs[0], outputs[0]->attr.size, outputs[0]->attr.dim_num) )
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
    vsi_nn_internal_node_t* curr = NULL;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        if (factor != 0)
        {
            outputs[0]->attr.size[0] = (uint32_t)(inputs[0]->attr.size[0] * factor);
            outputs[0]->attr.size[1] = (uint32_t)(inputs[0]->attr.size[1] * factor);
        }
        else
        {
            outputs[0]->attr.size[0] = self->nn_param.resize.size[0];
            outputs[0]->attr.size[1] = self->nn_param.resize.size[1];
        }
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    }

    if ((self->nn_param.resize.align_corners || self->nn_param.resize.half_pixel_centers)
       && (VSI_NN_INTERPOLATION_BILINEAR == self->nn_param.resize.type))
    {
        vsi_nn_internal_init_node_wksp( self );
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESIZE_INTERNAL, 0, 0 );
        curr->node->nn_param.resize_internal.align_corners = self->nn_param.resize.align_corners;
        curr->node->nn_param.resize_internal.factor = self->nn_param.resize.factor;
        curr->node->nn_param.resize_internal.half_pixel_centers = self->nn_param.resize.half_pixel_centers;
        curr->inputs[0]  = inputs[0];
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node(self, curr);
    }
    else if ((self->nn_param.resize.align_corners || self->nn_param.resize.half_pixel_centers)
            && (VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR == self->nn_param.resize.type))
    {
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
#if (USE_OVX_API == FALSE)
    uint32_t i;
    for (i = 0; i < _VSI_NN_RESIZE_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.resize.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.resize.local.local_tensor[i]));
            self->nn_param.resize.local.local_tensor[i] = NULL;
        }
    }
#endif
    if ((self->nn_param.resize.align_corners || self->nn_param.resize.half_pixel_centers)
       && (VSI_NN_INTERPOLATION_BILINEAR == self->nn_param.resize.type
          || VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR == self->nn_param.resize.type))
    {
        vsi_nn_internal_deinit_node_wksp(self);
    }
    else
    {
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

    if (vsi_nn_compareVersion(self->graph, 1, 1, 14) == -1)
    {
        self->nn_param.resize.align_corners      = FALSE;
        self->nn_param.resize.half_pixel_centers = FALSE;
    }

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

