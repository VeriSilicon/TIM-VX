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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"
#include "utils/vsi_nn_constraint_check.h"

#define _ARG_NUM            (2)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_SPACE2DEPTH_list[];

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
    vsi_nn_space2depth_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.space2depth);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, block_size[0] );
    _SET_PARAM( 1, VX_TYPE_INT32, block_size[1] );
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

static vsi_status vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_status status = VX_SUCCESS;
    vsi_nn_type_e dataFormat = inputs[0]->attr.dtype.vx_type;
    int8_t input_fixPointPos = 0;
    int8_t output_fixPointPos = 0;
    vx_bool dataTypeFlg = FALSE;
    vsi_nn_tensor_attr_t attr[2];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(inputs[0]->t, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(outputs[0]->t, &attr[1]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    input_fixPointPos  = attr[0].dtype.fl;
    output_fixPointPos = attr[1].dtype.fl;

    if(input_fixPointPos == output_fixPointPos)
        dataTypeFlg = TRUE;

    if ((dataFormat == VSI_NN_TYPE_INT16 && dataTypeFlg) || dataFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->kernel_index = 2;
    }
    else
    {
        VSILOGE("Not support input or output data format!(PRELU)\n");
        return VSI_FAILURE;
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

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    int32_t size_x = self->nn_param.space2depth.block_size[0];
    int32_t size_y = self->nn_param.space2depth.block_size[1];
    if (size_x == size_y)
    {
        vx_nn_reorg_params_t param;
        vsi_nn_tensor_t *block_size_tensor = NULL;
        memset(&param, 0, sizeof(vx_nn_reorg_params_t));

        block_size_tensor = vsi_nn_VariableToTensor(self,
            (uint8_t *)&self->nn_param.space2depth.block_size[0],
            VSI_NN_TYPE_INT32);
        if( NULL == block_size_tensor )
        {
            VSILOGE("Create block_size_tensor fail.(space2depth)");
            return VSI_FAILURE;
        }
        self->nn_param.space2depth.local.block_size_tensor = block_size_tensor;
        param.block_size = REQUIRED_IO(block_size_tensor);
        param.type = VX_REORG_SPACE_TO_DEPTH;

        self->n = vxReorgLayer2( self->graph->g,
            inputs[0]->t,
            &param,
            sizeof(vx_nn_reorg_params_t),
            outputs[0]->t);

        if( NULL != self->n )
        {
            status = VSI_SUCCESS;
        }
    }
    else
    {
        vsi_nn_kernel_info_t kernel_info;
        memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_space2depth";
        //kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
        kernel_info.kernel = vx_kernel_SPACE2DEPTH_list;
        kernel_info.kernel_index = 1;
        //kernel_info.init_index = 0;
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
    }

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if(self->nn_param.space2depth.block_size[0] < 0
        || self->nn_param.space2depth.block_size[1] < 0)
    {
        VSILOGE("Block size can't be less than zero in space to depth");
        return FALSE;
    }

    {
        BEGIN_IO_TYPE_DECL(SPACE2DEPTH, 1, 1)
            IO_TYPE(D_F16,  D_F16)
            IO_TYPE(D_I8|Q_DFP,  D_I8|Q_DFP)
            IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
            IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
            IO_TYPE(D_F32,  D_F32)
            IO_TYPE(D_F32,  D_BF16)
            IO_TYPE(D_BF16, D_F32)

            /* HW 9.0 */
            IO_TYPE(D_BF16, D_BF16)
        END_IO_TYPE_DECL(SPACE2DEPTH)
        if(!VALIDATE_OP_IO_TYPES(SPACE2DEPTH, self, inputs, self->input.num, outputs, self->output.num)) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
    }

    return TRUE;
} /* op_check() */


static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t size_x = node->nn_param.space2depth.block_size[0];
    uint32_t size_y = node->nn_param.space2depth.block_size[1];
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = inputs[0]->attr.size[0] / size_x;
        outputs[0]->attr.size[1] = inputs[0]->attr.size[1] / size_y;
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2] * size_x * size_y;
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.space2depth.local.block_size_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.space2depth.local.block_size_tensor));
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
    /* op_name    */ SPACE2DEPTH,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
