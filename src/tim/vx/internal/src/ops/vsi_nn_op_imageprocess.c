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
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "client/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _ARG_NUM            (14)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_IMAGEPROCESS_list[];

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
    vsi_nn_imageprocess_param * p;
    int32_t i;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.imageprocess);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, crop.enable );
    _SET_PARAM( 1, VX_TYPE_INT32, crop.dim_num );
    for (i = 0; i < p->crop.dim_num; i++)
    {
        _SET_PARAM( 2 + i, VX_TYPE_INT32, crop.start[i] );
    }
    _SET_PARAM( 6, VX_TYPE_BOOL, reverse_channel );
    _SET_PARAM( 7, VX_TYPE_INT32, mean.type );
    _SET_PARAM( 8, VX_TYPE_FLOAT32, mean.scale );
    _SET_PARAM( 9, VX_TYPE_INT32, mean.mean_value_size );
    for (i = 0; i < p->mean.mean_value_size; i++)
    {
        _SET_PARAM( 10 + i, VX_TYPE_FLOAT32, mean.mean_value[i] );
    }
#undef _SET_PARAM
set_param_error:

    return status;
} /* _create_params */

struct _scaletotensor_kernel_params
{
    int32_t ratio[2];
    int32_t offset[2];
    float mean[3];
    float scale;
};

typedef struct _scaletotensor_kernel_params scaletotensor_kernel_params_t;

static vsi_status prepare_params_scaletotensor
    (
    vsi_nn_imageprocess_param *p,
    scaletotensor_kernel_params_t *params,
    vsi_nn_tensor_attr_t *attr_in,
    vsi_nn_tensor_attr_t *attr_out
    )
{
    int32_t i;
    if (p->crop.enable == TRUE)
    {
        params->offset[0] = p->crop.start[0];
        params->offset[1] = p->crop.start[1];
    }
    else
    {
        params->offset[0] = 0;
        params->offset[1] = 0;
    }

    if (p->crop.enable == TRUE)
    {
        params->ratio[0] = (p->crop.length[0] << 15) / attr_out->size[0];
        params->ratio[1] = (p->crop.length[1] << 15) / attr_out->size[1];
    }
    else
    {
        params->ratio[0] = (attr_in->size[0] << 15) / attr_out->size[0];
        params->ratio[1] = (attr_in->size[1] << 15) / attr_out->size[1];
    }

    if (p->mean.type == VSI_NN_IMAGEPROCESS_MEAN_NONE)
    {
        for (i = 0; i < 3; i++)
        {
            params->mean[i] = 0;
        }
    }
    else if (p->mean.type == VSI_NN_IMAGEPROCESS_MEAN_CHANNEL)
    {
        for (i = 0; i < 3; i++)
        {
            params->mean[i] = p->mean.mean_value[i];
        }
    }
    else if (p->mean.type == VSI_NN_IMAGEPROCESS_MEAN_PIXEL)
    {
        for (i = 0; i < 3; i++)
        {
            params->mean[i] = p->mean.mean_value[0];
        }
    }
    params->scale = p->mean.scale;
    return VSI_SUCCESS;
}

static vsi_status prepare_params_grayscaletotensor
    (
    vsi_nn_imageprocess_param *p,
    scaletotensor_kernel_params_t *params,
    vsi_nn_tensor_attr_t *attr_in,
    vsi_nn_tensor_attr_t *attr_out
    )
{
    if (p->crop.enable == TRUE)
    {
        params->offset[0] = p->crop.start[0];
        params->offset[1] = p->crop.start[1];
    }
    else
    {
        params->offset[0] = 0;
        params->offset[1] = 0;
    }

    if (p->crop.enable == TRUE)
    {
        params->ratio[0] = (p->crop.length[0] << 15) / attr_out->size[0];
        params->ratio[1] = (p->crop.length[1] << 15) / attr_out->size[1];
    }
    else
    {
        params->ratio[0] = (attr_in->size[0] << 15) / attr_out->size[0];
        params->ratio[1] = (attr_in->size[1] << 15) / attr_out->size[1];
    }

    if (p->mean.type == VSI_NN_IMAGEPROCESS_MEAN_NONE)
    {
        params->mean[0] = 0;
    }
    else
    {
        params->mean[0] = p->mean.mean_value[0];
    }
    params->scale = p->mean.scale;
    return VSI_SUCCESS;
}

static vsi_status _create_params_vx_scaletotensor
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num,
    vsi_nn_tensor_attr_t *attr_in,
    vsi_nn_tensor_attr_t *attr_out
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_imageprocess_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.imageprocess);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    {
        scaletotensor_kernel_params_t scaletotensor_kernel_params;
        prepare_params_scaletotensor(p, &scaletotensor_kernel_params, attr_in, attr_out);
        _SET_PARAM( 0, VX_TYPE_INT32, scaletotensor_kernel_params.ratio[0]);
        _SET_PARAM( 1, VX_TYPE_INT32, scaletotensor_kernel_params.ratio[1]);
        _SET_PARAM( 2, VX_TYPE_INT32, scaletotensor_kernel_params.offset[0]);
        _SET_PARAM( 3, VX_TYPE_INT32, scaletotensor_kernel_params.offset[1]);
        _SET_PARAM( 4, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[0]);
        _SET_PARAM( 5, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[1]);
        _SET_PARAM( 6, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[2]);
        _SET_PARAM( 7, VX_TYPE_FLOAT32, scaletotensor_kernel_params.scale);
    }
#undef _SET_PARAM
set_param_error:

    return status;
} /* _create_params_vx_scaletotensor */

static vsi_status _create_params_vx_grayscaletotensor
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num,
    vsi_nn_tensor_attr_t *attr_in,
    vsi_nn_tensor_attr_t *attr_out
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_imageprocess_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.imageprocess);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    {
        scaletotensor_kernel_params_t scaletotensor_kernel_params;
        prepare_params_grayscaletotensor(p, &scaletotensor_kernel_params, attr_in, attr_out);
        _SET_PARAM( 0, VX_TYPE_INT32, scaletotensor_kernel_params.ratio[0]);
        _SET_PARAM( 1, VX_TYPE_INT32, scaletotensor_kernel_params.ratio[1]);
        _SET_PARAM( 2, VX_TYPE_INT32, scaletotensor_kernel_params.offset[0]);
        _SET_PARAM( 3, VX_TYPE_INT32, scaletotensor_kernel_params.offset[1]);
        _SET_PARAM( 4, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[0]);
        _SET_PARAM( 5, VX_TYPE_FLOAT32, scaletotensor_kernel_params.scale);
    }
#undef _SET_PARAM
set_param_error:

    return status;
} /* _create_params_vx_scaletotensor */

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

static vsi_status select_kernel_index
    (
    vsi_nn_kernel_info_t * kernel_info,
    vsi_nn_type_e outDataType,
    vx_bool is_copy
    )
{
    if (!is_copy)
    {
        if (outDataType == VSI_NN_TYPE_FLOAT16)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess";
            kernel_info->kernel_index = 1;
        }
        else if (outDataType == VSI_NN_TYPE_INT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess";
            kernel_info->kernel_index = 2;
        }
        else if (outDataType == VSI_NN_TYPE_INT16)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_2";
            kernel_info->kernel_index = 3;
        }
        else if (outDataType == VSI_NN_TYPE_UINT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_2";
            kernel_info->kernel_index = 4;
        }
        else
        {
            VSILOGE("Unsupported data type(imageprocess).\n");
            return VSI_FAILURE;
        }
    }
    else
    {
        if (outDataType == VSI_NN_TYPE_FLOAT16)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_3";
            kernel_info->kernel_index = 5;
        }
        else if (outDataType == VSI_NN_TYPE_INT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_3";
            kernel_info->kernel_index = 6;
        }
        else if (outDataType == VSI_NN_TYPE_INT16)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_3";
            kernel_info->kernel_index = 7;
        }
        else if (outDataType == VSI_NN_TYPE_UINT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_3";
            kernel_info->kernel_index = 8;
        }
        else
        {
            VSILOGE("Unsupported data type(imageprocess).\n");
            return VSI_FAILURE;
        }
    }

    return VSI_SUCCESS;
}

static vsi_status select_kernel_index_gray
    (
    vsi_nn_kernel_info_t * kernel_info,
    vsi_nn_type_e outDataType,
    vx_bool is_copy
    )
{
    if (!is_copy)
    {
        if (outDataType == VSI_NN_TYPE_FLOAT16)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_4";
            kernel_info->kernel_index = 9;
        }
        else if (outDataType == VSI_NN_TYPE_INT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_4";
            kernel_info->kernel_index = 10;
        }
        else if (outDataType == VSI_NN_TYPE_INT16)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_5";
            kernel_info->kernel_index = 11;
        }
        else if (outDataType == VSI_NN_TYPE_UINT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_5";
            kernel_info->kernel_index = 12;
        }
        else
        {
            VSILOGE("Unsupported data type(imageprocess).\n");
            return VSI_FAILURE;
        }
    }
    else
    {
        if (outDataType == VSI_NN_TYPE_FLOAT16)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_5";
            kernel_info->kernel_index = 13;
        }
        else if (outDataType == VSI_NN_TYPE_INT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_5";
            kernel_info->kernel_index = 14;
        }
        else if (outDataType == VSI_NN_TYPE_INT16)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_5";
            kernel_info->kernel_index = 15;
        }
        else if (outDataType == VSI_NN_TYPE_UINT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_imageprocess_5";
            kernel_info->kernel_index = 16;
        }
        else
        {
            VSILOGE("Unsupported data type(imageprocess).\n");
            return VSI_FAILURE;
        }
    }

    return VSI_SUCCESS;
}

static vsi_status vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e outDataType = outputs[0]->attr.dtype.vx_type;
    vx_bool is_copy = (vx_bool)((inputs[0]->attr.size[0] == outputs[0]->attr.size[0])
        && (inputs[0]->attr.size[1] == outputs[0]->attr.size[1]));

    if (inputs[0]->attr.size[2] == 1)
    {
        kernel_info->init_index = 2;
        return select_kernel_index_gray(kernel_info, outDataType, is_copy);
    }
    else
    {
        kernel_info->init_index = 1;
        return select_kernel_index(kernel_info, outDataType, is_copy);
    }
}

#define _ARG_NUM_SCALETOTENSOR            (8)
#define _PARAM_NUM_SCALETOTENSOR          (_ARG_NUM_SCALETOTENSOR + _IO_NUM)

static vsi_status vx_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM_SCALETOTENSOR];
    vx_border_t border;
    vx_reference * args;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( params, inputs, outputs );

    /* Init parameters. */
    _create_params_vx_scaletotensor( self, args, _ARG_NUM_SCALETOTENSOR,
        &(inputs[0]->attr), &(outputs[0]->attr));

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM_SCALETOTENSOR );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    _release_params( args, _ARG_NUM_SCALETOTENSOR );

    return status;
} /* vx_op_compute() */

#define _ARG_NUM_GRAYSCALETOTENSOR            (6)
#define _PARAM_NUM_GRAYSCALETOTENSOR          (_ARG_NUM_GRAYSCALETOTENSOR + _IO_NUM)

static vsi_status vx_gray_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM_GRAYSCALETOTENSOR];
    vx_border_t border;
    vx_reference * args;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( params, inputs, outputs );

    /* Init parameters. */
    _create_params_vx_grayscaletotensor( self, args, _ARG_NUM_GRAYSCALETOTENSOR,
        &(inputs[0]->attr), &(outputs[0]->attr));

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM_GRAYSCALETOTENSOR );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    _release_params( args, _ARG_NUM_GRAYSCALETOTENSOR );

    return status;
} /* vx_gray_op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(IMAGEPROCESS, 1, 1)
        IO_TYPE(D_U8,  D_F16)
        IO_TYPE(D_U8,  D_U8|Q_ASYM)
        IO_TYPE(D_U8,  D_I16|Q_DFP)
        IO_TYPE(D_U8,  D_I8|Q_DFP)
    END_IO_TYPE_DECL(IMAGEPROCESS)
    if (!VALIDATE_OP_IO_TYPES(IMAGEPROCESS, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute,
    vx_gray_op_compute,
    NULL
};

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_kernel_info_t kernel_info;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_imageprocess";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_IMAGEPROCESS_list;

    if (vsi_nn_is_do_vx_op_pre_init(kernel_info.type))
    {
        vx_op_pre_compute(self, inputs, outputs, &kernel_info);
    }

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
    return status;
} /* op_compute() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_imageprocess_param * p;
    uint32_t i;
    p = (vsi_nn_imageprocess_param *)&(self->nn_param.imageprocess);
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        /* TODO */
        if (inputs[0]->attr.dim_num != 4)
        {
            VSILOGE("Only support 4D tensor for image process!(IMAGEPROCESS)\n");
            return FALSE;
        }
        if (p->reverse_channel == TRUE && inputs[0]->attr.size[2] != 3)
        {
            VSILOGE("Only support 3 channels for reverse channel!(IMAGEPROCESS)\n");
            return FALSE;
        }

        if (p->resize.type != VSI_NN_IMAGEPROCESS_RESIZE_NONE)
        {
            outputs[0]->attr.dim_num = p->resize.dim_num;
            for(i = 0; i < (uint32_t)p->resize.dim_num; ++i)
            {
                outputs[0]->attr.size[i] = p->resize.length[i];
            }
        }
        else if (p->crop.enable == TRUE)
        {
            outputs[0]->attr.dim_num = p->crop.dim_num;
            for(i = 0; i < (uint32_t)p->crop.dim_num; ++i)
            {
                outputs[0]->attr.size[i] = p->crop.length[i];
            }
        }
        else
        {
            // CWHN -> WHCN
            outputs[0]->attr.size[0] = inputs[0]->attr.size[1];
            outputs[0]->attr.size[1] = inputs[0]->attr.size[2];
            outputs[0]->attr.size[2] = inputs[0]->attr.size[0];
            outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
        }
    }
    return TRUE;
} /* op_setup() */

typedef struct _vsi_nn_image_data_t
{
    int32_t id;
    vx_image handle;
}vsi_nn_image_data_t;

typedef struct _vsi_nn_image_list_t
{
    vsi_nn_link_list_t link_list;
    vsi_nn_image_data_t image;
} vsi_nn_image_list_t;

static void _init_image_list(vsi_nn_link_list_t *node)
{
    vsi_nn_image_list_t *image_list = (vsi_nn_image_list_t *)node;
    image_list->link_list.next = NULL;
    image_list->link_list.prev = NULL;
    memset(&image_list->image, 0, sizeof(vsi_nn_image_data_t));
}

static vsi_nn_image_list_t* get_image_by_id
(
    vsi_nn_image_list_t* head,
    int32_t id
)
{
    vsi_nn_image_list_t *iter;
    iter = head;
    while(iter)
    {
        if (iter->image.id == id)
        {
            return iter;
        }
        iter = (vsi_nn_image_list_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter);
    }
    iter = (vsi_nn_image_list_t *)vsi_nn_LinkListNewNode(
                sizeof(vsi_nn_image_list_t), _init_image_list);
    iter->image.id = id;
    return iter;
}

vsi_nn_image_list_t* images_head = NULL;
// pipeline:
// 1.crop
// 2.resize
// 3.(val-mean)*scale
// 4.RGBRGBRGB ---> BBBGGGRRR
// 5.revert channel: BBBGGGRRR ---> RRRGGGBBB
vsi_status vsi_nn_InsertImageprocessSingleNode
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_attr_t *attr,
    vsi_nn_imageprocess_param *p,
    uint8_t *data,
    vsi_nn_tensor_t *tensor_out,
    int32_t id
    )
{
    vsi_nn_image_list_t* p_image;
    vx_image image_global;
    if(images_head == NULL)
    {
        images_head = (vsi_nn_image_list_t *)vsi_nn_LinkListNewNode(
            sizeof(vsi_nn_image_list_t), _init_image_list);
    }
    p_image = get_image_by_id(images_head, id);
    image_global = p_image->image.handle;
    if(image_global == NULL)
    {
        vsi_status status;
        vsi_nn_kernel_info_t kernel_info;
        vx_node node = NULL;
        vx_reference params[_PARAM_NUM_SCALETOTENSOR];
        vx_border_t border;
        vx_reference * args;
        vx_image image = NULL;
        vx_context ctx = vxGetContext( (vx_reference)graph->g );
        vx_imagepatch_addressing_t imgInfo;
        vx_bool is_copy = (vx_bool)((attr->size[0] == tensor_out->attr.size[0])
            && (attr->size[1] == tensor_out->attr.size[1]));
        vsi_nn_tensor_t *tensor_temp = NULL;
        vsi_nn_tensor_t *output_scaletotensor = NULL;
        vsi_nn_tensor_t *output_reversetensor = NULL;
        vx_nn_tensor_reverse_params_t para;
        int32_t reverse1_axis[4] = {2};
        uint32_t perm[] = {2, 0, 1, 3};
        vsi_nn_tensor_t out0;
        uint32_t arg_num;
        vx_bool is_gray = FALSE;

        memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
        memset(&out0, 0, sizeof(vsi_nn_tensor_t));
        para.axis = reverse1_axis;
        para.numberOfAxis = 1;

        if (p->platform_type == VSI_NN_PLATFORM_TENSORFLOW)
        {
            vsi_nn_tensor_attr_t attr0;
            memcpy(&attr0, &tensor_out->attr, sizeof(vsi_nn_tensor_attr_t));
            attr0.size[0] = tensor_out->attr.size[1];
            attr0.size[1] = tensor_out->attr.size[2];
            attr0.size[2] = tensor_out->attr.size[0];

            if (attr0.size[2] == 1)
            {
                is_gray= TRUE;
                p->reverse_channel = FALSE;
            }
            is_copy = (vx_bool)((attr->size[0] == attr0.size[0])
                    && (attr->size[1] == attr0.size[1]));
            if (!is_gray)
            {
                output_scaletotensor = vsi_nn_CreateTensor(graph, &attr0);
                if (p->reverse_channel == TRUE)
                {
                    output_reversetensor = vsi_nn_CreateTensor(graph, &attr0);
                }
                tensor_temp = output_scaletotensor;
            }
            else
            {
                out0.t = vxReshapeTensor(tensor_out->t, (int32_t *)attr0.size, attr0.dim_num);
                memcpy(&out0.attr, &attr0, sizeof(vsi_nn_tensor_attr_t));
                tensor_temp = &out0;
            }
        }
        else /* VSI_NN_PLATFORM_CAFFE */
        {
            if (tensor_out->attr.size[2] == 1)
            {
                is_gray= TRUE;
                p->reverse_channel = FALSE;
            }

            if (p->reverse_channel == TRUE)
            {
                output_scaletotensor = vsi_nn_CreateTensor(graph, &(tensor_out->attr));
                tensor_temp = output_scaletotensor;
            }
            else
            {
                tensor_temp = tensor_out;
            }
        }

        args = &params[_IO_NUM];

        status = VSI_FAILURE;
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
        kernel_info.kernel = vx_kernel_IMAGEPROCESS_list;
        if (!is_gray)
        {
            kernel_info.init_index = 1;
            status = select_kernel_index(&kernel_info, tensor_out->attr.dtype.vx_type, is_copy);
        }
        else
        {
            kernel_info.init_index = 2;
            status = select_kernel_index_gray(&kernel_info, tensor_out->attr.dtype.vx_type, is_copy);
        }

        node = vsi_nn_RegisterClientKernelAndNewNode(
            graph, &kernel_info);
        if (kernel_info.resource_name) free(kernel_info.resource_name);
        if( NULL == node )
        {
            VSILOGE("Create scaletotensor node fails");
            status = VSI_FAILURE;
            goto OnError;
        }
        //imgInfo = {width * num_of_channels, height, 1, width * num_of_channels, VX_SCALE_UNITY, VX_SCALE_UNITY, 1, 1};
        imgInfo.dim_x = attr->size[0] * attr->size[2];
        imgInfo.dim_y = attr->size[1];
        imgInfo.stride_x = 1;
        imgInfo.stride_y = imgInfo.dim_x;
        imgInfo.scale_x = VX_SCALE_UNITY;
        imgInfo.scale_y = VX_SCALE_UNITY;
        imgInfo.step_x = 1;
        imgInfo.step_y = 1;

#if defined(__linux__)
        image = vxCreateImageFromHandle(ctx, VX_DF_IMAGE_U8, &imgInfo, (void **)&data, VX_MEMORY_TYPE_HOST);
#else
        image = vxCreateImage(ctx, imgInfo.dim_x, imgInfo.dim_y, VX_DF_IMAGE_U8);
        {
            vx_rectangle_t rect = {0, 0, 0, 0};
            vx_map_id map_id = 0;
            void* imgBaseAddr = NULL;

            rect.end_x = imgInfo.dim_x;
            rect.end_y = imgInfo.dim_y;
            vxMapImagePatch(image, &rect, 0,&map_id, &imgInfo, &imgBaseAddr, VX_WRITE_ONLY,VX_MEMORY_TYPE_HOST, 0);// get data pointer of image in GPU side
            memcpy((vx_uint8*)imgBaseAddr, data, imgInfo.dim_x * imgInfo.dim_y);
            vxUnmapImagePatch(image, map_id);
            imgBaseAddr = NULL;
        }
#endif
        image_global = image;
        p_image->image.handle = image;

        /* Set inputs and outputs */
        params[0] = (vx_reference)image;
        params[1] = (vx_reference)tensor_temp->t;

        /* Init parameters. */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i + _IO_NUM] = (vx_reference)vxCreateScalar( ctx, type, &arg ); \
    status = vxGetStatus( params[i + _IO_NUM] ); \
    if( VSI_SUCCESS != status ) { \
    status = VSI_FAILURE;\
    goto OnError;\
    } \
        } while(0)
        if (!is_gray)
        {
            {
                scaletotensor_kernel_params_t scaletotensor_kernel_params;
                prepare_params_scaletotensor(p, &scaletotensor_kernel_params, attr, &(tensor_temp->attr));
                _SET_PARAM( 0, VX_TYPE_INT32, scaletotensor_kernel_params.ratio[0]);
                _SET_PARAM( 1, VX_TYPE_INT32, scaletotensor_kernel_params.ratio[1]);
                _SET_PARAM( 2, VX_TYPE_INT32, scaletotensor_kernel_params.offset[0]);
                _SET_PARAM( 3, VX_TYPE_INT32, scaletotensor_kernel_params.offset[1]);
                if (p->reverse_channel == TRUE)
                {
                    _SET_PARAM( 4, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[0]);
                    _SET_PARAM( 5, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[1]);
                    _SET_PARAM( 6, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[2]);
                }
                else
                {
                    _SET_PARAM( 4, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[2]);
                    _SET_PARAM( 5, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[1]);
                    _SET_PARAM( 6, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[0]);
                }
                _SET_PARAM( 7, VX_TYPE_FLOAT32, scaletotensor_kernel_params.scale);
            }
            arg_num = _ARG_NUM_SCALETOTENSOR;
        }
        else
        {
            {
                scaletotensor_kernel_params_t scaletotensor_kernel_params;
                prepare_params_scaletotensor(p, &scaletotensor_kernel_params, attr, &(tensor_temp->attr));
                _SET_PARAM( 0, VX_TYPE_INT32, scaletotensor_kernel_params.ratio[0]);
                _SET_PARAM( 1, VX_TYPE_INT32, scaletotensor_kernel_params.ratio[1]);
                _SET_PARAM( 2, VX_TYPE_INT32, scaletotensor_kernel_params.offset[0]);
                _SET_PARAM( 3, VX_TYPE_INT32, scaletotensor_kernel_params.offset[1]);
                _SET_PARAM( 4, VX_TYPE_FLOAT32, scaletotensor_kernel_params.mean[0]);
                _SET_PARAM( 5, VX_TYPE_FLOAT32, scaletotensor_kernel_params.scale);
            }
            arg_num = _ARG_NUM_GRAYSCALETOTENSOR;
        }
#undef _SET_PARAM

        /* Pass parameters to node. */
        status = vsi_nn_ClientNodePassParameters(node, params, _IO_NUM + arg_num);

        border.mode = VX_BORDER_REPLICATE;
        border.constant_value.U32 = 0;
        status |= vxSetNodeAttribute(node, VX_NODE_BORDER, &border, sizeof(border));

        _release_params( args, arg_num);

        if (p->platform_type == VSI_NN_PLATFORM_TENSORFLOW)
        {
            if (p->reverse_channel == TRUE)
            {
                node = vxTensorReverse( graph->g, output_scaletotensor->t, &para,
                    sizeof(vx_nn_tensor_reverse_params_t), output_reversetensor->t );
                if( NULL == node )
                {
                    VSILOGE("Create vxTensorReverse node fails");
                    status = VSI_FAILURE;
                    goto OnError;
                }

                node = vxTensorPermuteNode( graph->g, output_reversetensor->t,
                    tensor_out->t, perm, 4);
                if( NULL == node )
                {
                    VSILOGE("Create vxTensorPermuteNode node fails");
                    status = VSI_FAILURE;
                    goto OnError;
                }
            }
            else
            {
                if (!is_gray)
                {
                    node = vxTensorPermuteNode( graph->g, output_scaletotensor->t,
                        tensor_out->t, perm, 4);
                    if( NULL == node )
                    {
                        VSILOGE("Create vxTensorPermuteNode node fails");
                        status = VSI_FAILURE;
                        goto OnError;
                    }
                }
                else
                {
                    if (out0.t) vxReleaseTensor(&out0.t);
                }
            }
        }
        else /* VSI_NN_PLATFORM_CAFFE */
        {
            if (p->reverse_channel == TRUE)
            {
                node = vxTensorReverse( graph->g, output_scaletotensor->t, &para,
                    sizeof(vx_nn_tensor_reverse_params_t), tensor_out->t );
                if( NULL == node )
                {
                    VSILOGE("Create vxTensorReverse node fails");
                    status = VSI_FAILURE;
                    goto OnError;
                }
            }
        }

    //set graph inputs outputs again, because pre_process changed graph inputs
    {
        uint32_t num_of_graph_inputs;
        vx_reference *graph_inputs = NULL;
        uint32_t num_of_graph_outputs;
        vx_reference *graph_outputs = NULL;
        uint32_t i = 0;

        /* Explicitly set graph inputs and outputs */
        num_of_graph_inputs = 1;
        graph_inputs = (vx_reference *)malloc( num_of_graph_inputs * sizeof( vx_reference ) );

        graph_inputs[0] = (vx_reference)image;

        num_of_graph_outputs = graph->output.num;
        graph_outputs = (vx_reference *)malloc( num_of_graph_outputs * sizeof( vx_reference ) );
        for( i = 0; i < num_of_graph_outputs; i++ )
        {
            graph_outputs[i] = (vx_reference)( ( vsi_nn_GetTensor( graph, graph->output.tensors[i] ) )->t );
        }
        status = vxIdentifyGraphInputsAndOutputs( graph->g,
            num_of_graph_inputs,
            graph_inputs,
            num_of_graph_outputs,
            graph_outputs );

        if ( NULL != graph_inputs)
        {
            free( graph_inputs );
        }
        if ( NULL != graph_outputs)
        {
            free( graph_outputs );
        }
    }
OnError:
        //if(tensor_temp) vsi_nn_ReleaseTensor(&tensor_temp);
        if(output_scaletotensor) vsi_nn_ReleaseTensor(&output_scaletotensor);
        if(output_reversetensor) vsi_nn_ReleaseTensor(&output_reversetensor);
        return status;
    }
    else
    {
#if !defined(__linux__)
        vx_imagepatch_addressing_t imgInfo;
        vx_rectangle_t rect = {0, 0, 0, 0};
        vx_map_id map_id = 0;
        void* imgBaseAddr = NULL;

        //imgInfo = {width * num_of_channels, height, 1, width * num_of_channels, VX_SCALE_UNITY, VX_SCALE_UNITY, 1, 1};
        imgInfo.dim_x = attr->size[0] * attr->size[2];
        imgInfo.dim_y = attr->size[1];
        imgInfo.stride_x = 1;
        imgInfo.stride_y = imgInfo.dim_x;
        imgInfo.scale_x = VX_SCALE_UNITY;
        imgInfo.scale_y = VX_SCALE_UNITY;
        imgInfo.step_x = 1;
        imgInfo.step_y = 1;

        rect.end_x = imgInfo.dim_x;
        rect.end_y = imgInfo.dim_y;
        vxMapImagePatch(image_global, &rect, 0,&map_id, &imgInfo, &imgBaseAddr, VX_WRITE_ONLY,VX_MEMORY_TYPE_HOST, 0);// get data pointer of image in GPU side
        memcpy((vx_uint8*)imgBaseAddr, data, imgInfo.dim_x * imgInfo.dim_y);
        vxUnmapImagePatch(image_global, map_id);
        imgBaseAddr = NULL;
#endif
        return VSI_SUCCESS;
    }
}

vsi_status vsi_nn_op_imageprocess_single_node
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_attr_t *attr,
    vsi_nn_imageprocess_param *p,
    uint8_t *data,
    vsi_nn_tensor_t *tensor_out
    )
{
    return vsi_nn_InsertImageprocessSingleNode(
        graph, attr, p, data, tensor_out, 0);
}

static void _release_image_list(vsi_nn_link_list_t *node)
{
    vsi_nn_image_list_t *image_list = (vsi_nn_image_list_t *)node;
    vxReleaseImage(&(image_list->image.handle));
}

vsi_status vsi_nn_ReleaseImageprocessSingleNode()
{
    vsi_nn_LinkListDeinit((vsi_nn_link_list_t *)images_head, _release_image_list);
    return VSI_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ IMAGEPROCESS,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
