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
#include "vsi_nn_test.h"
#include "client/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define USE_OVXLIB (0)

#define _ARG_NUM            (2)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#if (USE_OVXLIB)

extern vx_kernel_description_t * vx_kernel_SHUFFLECHANNEL_list[];

static vsi_bool _reshape_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i = 0;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t axis = 0;
    vsi_nn_shufflechannel_param * p = NULL;
    uint32_t before_size = 1;
    uint32_t after_size = 1;
    uint32_t * input_sizes = inputs[0]->attr.size;
    uint32_t dims = inputs[0]->attr.dim_num;

    p = &(self->nn_param.shufflechannel);
    axis = p->axis;

    for ( i = 0; i < (uint32_t)axis; i++)
    {
        before_size *= input_sizes[i];
    }
    for ( i = axis + 1; i < dims; i++)
    {
        after_size *= input_sizes[i];
    }

    if (axis == 2 && after_size == 1)
    {
        sizes[0] = input_sizes[0];
        sizes[1] = input_sizes[1];
        sizes[2] = input_sizes[2];
    }
    else
    {
        sizes[0] = before_size;
        sizes[1] = input_sizes[axis];
        sizes[2] = after_size;
        p->axis = 1;
    }
    dims = 3;

    p->local->input_tensor = vxReshapeTensor(inputs[0]->t, (int32_t *)sizes, dims);
    p->local->output_tensor = vxReshapeTensor(outputs[0]->t, (int32_t *)sizes, dims);

    return TRUE;
}

static void _set_inputs_outputs
    (
    vsi_nn_node_t * self,
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_shufflechannel_param * p = NULL;

    p = &(self->nn_param.shufflechannel);

    params[0] = (vx_reference)p->local->input_tensor;
    params[1] = (vx_reference)p->local->output_tensor;
} /* _set_inputs_outputs() */

static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_context ctx;
    vsi_nn_shufflechannel_param * p = NULL;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.shufflechannel);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, group_number );
    _SET_PARAM( 1, VX_TYPE_INT32, axis );
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
    uint32_t i = 0;
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
    vx_reference * args = NULL;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( self, params, inputs, outputs );

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}

static vsi_status vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputDataFormat    = outputs[0]->attr.dtype.vx_type;
    int8_t        inputFixedPointPos  = inputs[0]->attr.dtype.fl;
    int8_t        outputFixedPointPos = outputs[0]->attr.dtype.fl;
    int32_t       inputZeroPoint      = inputs[0]->attr.dtype.zero_point;
    int32_t       outputZeroPoint     = outputs[0]->attr.dtype.zero_point;
    vx_float32    inputScale          = inputs[0]->attr.dtype.scale;
    vx_float32    outputScale         = outputs[0]->attr.dtype.scale;
    int32_t       axis                = self->nn_param.shufflechannel.axis;
    uint32_t      *sizes              = inputs[0]->attr.size;
    vsi_bool      is16Bits            = FALSE;
    vsi_bool      is8Bits             = FALSE;

    is16Bits = ((inputDataFormat == VSI_NN_TYPE_FLOAT16 && outputDataFormat == VSI_NN_TYPE_FLOAT16)
            || (inputDataFormat == VSI_NN_TYPE_INT16 && outputDataFormat == VSI_NN_TYPE_INT16
            && inputFixedPointPos == outputFixedPointPos)) ? TRUE : FALSE;
    is8Bits = ((inputDataFormat == VSI_NN_TYPE_INT8 && outputDataFormat == VSI_NN_TYPE_INT8
            && inputFixedPointPos == outputFixedPointPos)
            || (inputDataFormat == VSI_NN_TYPE_UINT8 && outputDataFormat == VSI_NN_TYPE_UINT8
            && inputZeroPoint == outputZeroPoint && inputScale == outputScale)) ? TRUE : FALSE;
#define VSI_NN_TENSOR_WIDTH_MAX (65536)
    kernel_info->kernel_index = 0;
    if (sizes[0] < VSI_NN_TENSOR_WIDTH_MAX && sizes[1] < VSI_NN_TENSOR_WIDTH_MAX)
    {
        if ( is16Bits && axis == 2 )
        {
            kernel_info->kernel_index = 1;
        }
        else if ( is8Bits && axis == 2)
        {
            kernel_info->kernel_index = 2;
        }
        else if ( is16Bits && axis == 1)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_shufflechannel_axis1";
            kernel_info->kernel_index = 3;
        }
        else if ( is8Bits && axis == 1)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_shufflechannel_axis1";
            kernel_info->kernel_index = 4;
        }
    }
#undef VSI_NN_TENSOR_WIDTH_MAX

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
    vx_reference * args = NULL;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( self, params, inputs, outputs );

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
#if(USE_OVXLIB)
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_info_t kernel_info;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));

    /* setup input/output shape */
    _reshape_tensor( self, inputs, outputs);

    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_shufflechannel";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_SHUFFLECHANNEL_list;
    kernel_info.init_index = 1;

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
#else
    vsi_status status = VSI_FAILURE;
    vx_nn_reorg_params_ext2_t param;
    vsi_nn_tensor_t *block_size_tensor = NULL;
    vsi_nn_tensor_attr_t attr;
    uint8_t data = 1;

    memset(&param, 0, sizeof(vx_nn_reorg_params_ext2_t));
    memset(&attr, 0, sizeof(attr));
    attr.size[0] = 2;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    block_size_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        &data,
        &attr);
    if( NULL == block_size_tensor )
    {
        VSILOGE("Create block_size_tensor fail.(shufflechannel)");
        return VSI_FAILURE;
    }

    self->nn_param.shufflechannel.local->block_size_tensor = block_size_tensor;
    param.base.block_size = REQUIRED_IO(block_size_tensor);

    param.base.type = VX_REORG_SHUFFLE_CHANNEL;
    param.axis = &self->nn_param.shufflechannel.axis;
    param.num_group = &self->nn_param.shufflechannel.group_number;

    self->n = vxReorgLayer2( self->graph->g,
        inputs[0]->t,
        (vx_nn_reorg_params_t *)&param,
        sizeof(vx_nn_reorg_params_ext2_t),
        outputs[0]->t);

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

    return status;
#endif
} /* op_compute() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;
    vsi_nn_shufflechannel_param *p = NULL;
    int32_t axis = 0;

    if( NULL == self )
    {
        return ret;
    }

    p = &(self->nn_param.shufflechannel);
    axis = p->axis;

    if (axis < 0)
    {
        axis = axis + inputs[0]->attr.dim_num;
        p->axis = axis;
    }

    if (p->axis < 0)
    {
        VSILOGD("shufflechannel Invalid Axis: %d", p->axis);
        return FALSE;
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        memcpy( outputs[0]->attr.size, inputs[0]->attr.size,
            sizeof(uint32_t) * inputs[0]->attr.dim_num );
    }

    return TRUE;
} /* op_setup() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_shufflechannel_param *p = NULL;
    int32_t axis = 0;
    int32_t dims = (int32_t)inputs[0]->attr.dim_num;
    int32_t num_group = 0;
    uint32_t *shape = inputs[0]->attr.size;

    p = &(self->nn_param.shufflechannel);
    axis = p->axis;
    num_group = p->group_number;

    if (axis > (dims - 1))
    {
        VSILOGE("Invalid Axis: %d, (SHUFFLECHANNEL) at [%s : %d]\n", axis, __FILE__, __LINE__);
        return FALSE;
    }
    if (shape[axis] % num_group)
    {
        VSILOGE("Invalid group_number: %d, (SHUFFLECHANNEL) at [%s : %d]\n", num_group, __FILE__, __LINE__);
        return FALSE;
    }

    {
        BEGIN_IO_TYPE_DECL(SHUFFLECHANNEL, 1, 1)
            IO_TYPE(D_F16,  D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
            IO_TYPE(D_F32,  D_F32)
            IO_TYPE(D_BF16, D_F32)
            IO_TYPE(D_F32,  D_BF16)
        END_IO_TYPE_DECL(SHUFFLECHANNEL)
        if(!VALIDATE_OP_IO_TYPES(SHUFFLECHANNEL, self, inputs, self->input.num, outputs, self->output.num)) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
    }

    return TRUE;
} /* op_check() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_shufflechannel_lcl_data_t *local = NULL;
    vsi_nn_shufflechannel_param *p = NULL;

    p = &(self->nn_param.shufflechannel);
    self->nn_param.shufflechannel.axis = 2;
    local = (vsi_nn_shufflechannel_lcl_data_t *)malloc(sizeof(vsi_nn_shufflechannel_lcl_data_t));
    if (NULL == local)
    {
        VSILOGE("Malloc fail, (SHUFFLECHANNEL) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    memset(local, 0, sizeof(vsi_nn_shufflechannel_lcl_data_t));
    p->local = local;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_shufflechannel_param *p = &(self->nn_param.shufflechannel);
    if (p->local)
    {
        if (p->local->input_tensor)
        {
            vxReleaseTensor(&(p->local->input_tensor));
            p->local->input_tensor = NULL;
        }
        if (p->local->output_tensor)
        {
            vxReleaseTensor(&(p->local->output_tensor));
            p->local->output_tensor = NULL;
        }
        if (p->local->block_size_tensor != NULL)
        {
            vsi_nn_ReleaseTensor(&(p->local->block_size_tensor));
        }

        vsi_nn_safe_free(p->local);
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
    /* op_name    */ SHUFFLECHANNEL,
    /* init       */ op_init,
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

