
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
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _ARG_NUM            (1)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define USE_OVX_API TRUE

#if (USE_OVX_API == FALSE)
extern vx_kernel_description_t * vx_kernel_REVERSE_list[];

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
    vsi_nn_reverse_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &node->nn_param.reverse;
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, axis[0] );
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

static vsi_status op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e in_dataType = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e out_dataType = outputs[0]->attr.dtype.vx_type;
    uint32_t i;
    uint32_t changed_num = 1;

    for( i = self->nn_param.reverse.axis[0] + 1; i < inputs[0]->attr.dim_num; i++ )
    {
        changed_num *= inputs[0]->attr.size[inputs[0]->attr.dim_num - 1 - i];
    }

    if ((in_dataType != VSI_NN_TYPE_INT16 || out_dataType != VSI_NN_TYPE_INT16)
        && self->nn_param.reverse.axis[0] != 0)
    {
        VSILOGE("tensorReverse shader unsupport format or axis:%d!\n",
            self->nn_param.reverse.axis[0]);
        return VSI_FAILURE;
    }
    else if (changed_num >= 65536)
    {
        VSILOGE("tensorReverse unsupport change num:%d!\n", changed_num);
        return VSI_FAILURE;
    }

    kernel_info->kernel_index = 1;

    return VSI_SUCCESS;
}

static void reshape_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vx_reference * params,
    uint32_t index
    )
{
    uint32_t i;
    int32_t size[4] = {0};
    int32_t size0[4] = {1, 1, 1, 1};
    uint32_t dims = 2;

    for( i = 0; i < input->attr.dim_num; i++ )
    {
        size0[i] = input->attr.size[i];
    }

    size[0] = size0[0] * size0[1] * size0[2];
    size[1] = size0[3];

    self->nn_param.reverse.local.local_tensor[index] =
        vxReshapeTensor(input->t, size, dims);
    params[index] = (vx_reference)self->nn_param.reverse.local.local_tensor[index];
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

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    reshape_tensor_shape(self, inputs[0], params, 0);
    reshape_tensor_shape(self, outputs[0], params, 1);

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

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
    vx_nn_tensor_reverse_params_t para;
    vsi_nn_reverse_param * p;
    int32_t axes[VSI_NN_MAX_DIM_NUM] = {0};
    p = &self->nn_param.reverse;
    memcpy(axes, p->axis, sizeof(int32_t) * p->axis_num);
    para.axis = axes;
    para.numberOfAxis = p->axis_num;
    self->n = vxTensorReverse( self->graph->g, inputs[0]->t, &para,
        sizeof(vx_nn_tensor_reverse_params_t), outputs[0]->t );
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
#else
    vsi_nn_kernel_info_t kernel_info;
    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_reverse";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_REVERSE_list;
    kernel_info.init_index = 1;

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


static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(REVERSE, 1, 1)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_U8|Q_DFP,   D_U8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I32|Q_DFP,  D_I32|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM,  D_I8|Q_ASYM)
        IO_TYPE(D_I16|Q_ASYM, D_I16|Q_ASYM)
        IO_TYPE(D_I32|Q_ASYM, D_I32|Q_ASYM)
        IO_TYPE(D_U8|Q_SYM_PC,   D_U8|Q_SYM_PC)
        IO_TYPE(D_I8|Q_SYM_PC,   D_I8|Q_SYM_PC)
        IO_TYPE(D_I16|Q_SYM_PC,  D_I16|Q_SYM_PC)
        IO_TYPE(D_I32|Q_SYM_PC,  D_I32|Q_SYM_PC)
        IO_TYPE(D_U8,   D_U8)
        IO_TYPE(D_I8,   D_I8)
        IO_TYPE(D_I16,  D_I16)
        IO_TYPE(D_I32,  D_I32)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_F32,  D_BF16)
        IO_TYPE(D_BF16, D_F32)

        /* HW 9.0 */
        IO_TYPE(D_BF16, D_BF16)
    END_IO_TYPE_DECL(REVERSE)
    if(!VALIDATE_OP_IO_TYPES(REVERSE, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }
    return TRUE;
} /* op_check() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
#if (USE_OVX_API == FALSE)
    uint32_t i;
    for (i = 0; i < _VSI_NN_REVERSE_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.reverse.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.reverse.local.local_tensor[i]));
            self->nn_param.reverse.local.local_tensor[i] = NULL;
        }
    }
#endif
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ REVERSE,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
