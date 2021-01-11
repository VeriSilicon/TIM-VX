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
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _ARG_NUM            (3)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define USE_OVX_API TRUE

#if (USE_OVX_API == FALSE)
extern vx_kernel_description_t * vx_kernel_CROP_list[];

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
    vsi_nn_crop_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.crop);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, offset[0] );
    _SET_PARAM( 1, VX_TYPE_INT32, offset[1] );
    _SET_PARAM( 2, VX_TYPE_INT32, offset[2] );

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

static vsi_status vx_op_pre_init
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e dataFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e dstFormat = outputs[0]->attr.dtype.vx_type;

    if (dataFormat == VSI_NN_TYPE_FLOAT16
        || (dataFormat == VSI_NN_TYPE_INT16 && dstFormat == VSI_NN_TYPE_INT16))
    {
        kernel_info->kernel_index = 1;
    }
    else if(dataFormat == VSI_NN_TYPE_INT16 && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->kernel_index = 3;
    }
    else
    {
        kernel_info->kernel_index = 2;
    }

    return VSI_SUCCESS;
}

static vsi_status vx_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
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
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

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

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
#if (USE_OVX_API == TRUE)
    vx_nn_stride_slice_params_t param;
    vsi_nn_tensor_t *begin_dims_tensor = NULL;
    vsi_nn_tensor_t *end_dims_tensor = NULL;
    vsi_nn_tensor_t *stride_dims_tensor = NULL;
    vsi_nn_tensor_attr_t attr;
    int32_t start[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t end[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t stride[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t i;

    memset(&param, 0, sizeof(vx_nn_stride_slice_params_t));

    for (i = 0; i < self->nn_param.crop.dims; i++)
    {
        start[i] = self->nn_param.crop.offset[i];
        end[i] = self->nn_param.crop.offset[i] + outputs[0]->attr.size[i];
        stride[i] = 1;
    }

    for (i = self->nn_param.crop.dims; i < inputs[0]->attr.dim_num; i++)
    {
        start[i] = 0;
        end[i] = outputs[0]->attr.size[i];
        stride[i] = 1;
    }

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = inputs[0]->attr.dim_num;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    begin_dims_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)start,
        &attr);
    if( NULL == begin_dims_tensor )
    {
        VSILOGE("Create begin_dims_tensor fail.(crop)");
        return VSI_FAILURE;
    }

    end_dims_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)end,
        &attr);
    if( NULL == end_dims_tensor )
    {
        VSILOGE("Create end_dims_tensor fail.(crop)");
        status = VSI_FAILURE;
        goto OnError;
    }

    stride_dims_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)stride,
        &attr);
    if( NULL == stride_dims_tensor )
    {
        VSILOGE("Create stride_dims_tensor fail.(crop)");
        status = VSI_FAILURE;
        goto OnError;
    }

    param.begin_dims = REQUIRED_IO(begin_dims_tensor);
    param.end_dims = REQUIRED_IO(end_dims_tensor);
    param.stride_dims = REQUIRED_IO(stride_dims_tensor);

    self->n = vxTensorStrideSliceNode(
        self->graph->g,
        inputs[0]->t,
        &param,
        sizeof(vx_nn_stride_slice_params_t),
        outputs[0]->t
        );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
#else
    vsi_nn_kernel_info_t kernel_info;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_crop";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_CROP_list;
    kernel_info.init_index = 1;

    if (vsi_nn_is_do_vx_op_pre_init(kernel_info.type))
    {
        vx_op_pre_init(self, inputs, outputs, &kernel_info);
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
#endif
OnError:
    if (begin_dims_tensor) vsi_nn_ReleaseTensor(&begin_dims_tensor);
    if (end_dims_tensor) vsi_nn_ReleaseTensor(&end_dims_tensor);
    if (stride_dims_tensor) vsi_nn_ReleaseTensor(&stride_dims_tensor);
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(CROP, 1, 1)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_I32,  D_I32)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
    END_IO_TYPE_DECL(CROP)
    if (!VALIDATE_OP_IO_TYPES(CROP, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_crop_param * p;
    int32_t i;
    p = (vsi_nn_crop_param *)&(self->nn_param.crop);
    if (p->axis >= (int32_t)inputs[0]->attr.dim_num)
    {
        VSILOGE("Invalid parameter: axis!\n");
        return FALSE;
    }

    if( VSI_NN_DIM_AUTO != outputs[0]->attr.dim_num )
    {
        return TRUE;
    }

    if (p->dims + p->axis == inputs[0]->attr.dim_num)
    {
        for(i = 0; i < p->axis; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }
        for(i = p->axis; i < (int32_t)inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[1]->attr.size[i];
        }
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    else
    {
        if (p->dims == 1)
        {
            for(i = 0; i <= p->axis; i++)
            {
                outputs[0]->attr.size[i] = inputs[1]->attr.size[i];
                p->offset[i] = p->offset[0];
            }
            for(i = p->axis + 1; i < (int32_t)inputs[0]->attr.dim_num; i++)
            {
                outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
            }
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        }
        else
        {
            VSILOGE("Invalid parameter: offset dims!\n");
            return FALSE;
        }
    }

    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CROP,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

