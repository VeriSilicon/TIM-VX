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
#include "vsi_nn_log.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"
#include "vsi_nn_tensor_util_prv.h"
#include "vsi_nn_error.h"

static vsi_status _try_set_high_presision_tensor
    (
    vsi_nn_tensor_t **inputs
    )
{
    vsi_status status;
    vsi_nn_vxtensor_attr_t attr;

    status = VSI_SUCCESS;
    attr = VSI_NN_TENSOR_ATTR_HIGH_PRECISION;

    if(VSI_NN_TYPE_FLOAT32 == inputs[1]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[1], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }
    if(VSI_NN_TYPE_FLOAT32 == inputs[2]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[2], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }
    if(VSI_NN_TYPE_FLOAT32 == inputs[3]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[3], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }
    if(VSI_NN_TYPE_FLOAT32 == inputs[4]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[4], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }

    return status;
}

static vsi_bool _require_reshape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs
    )
{
    /*
        We support 3d batchnorm at version 1.1.12
    */
    if (vsi_nn_compareVersion(self->graph, 1, 1, 12) == -1)
    {
        return FALSE;
    }
    return (3 == inputs[0]->attr.dim_num)||(5 == inputs[0]->attr.dim_num);
}

static vsi_bool _is_dynamic_batchnorm
    (
    vsi_nn_tensor_t ** inputs
    )
{
    uint32_t i = 0;
    for (i = 1; i < 5 ; i++) {
        if (FALSE == inputs[i]->attr.is_const) {
            return TRUE;
        }
    }
    return FALSE;
}

static vsi_status _static_batchnorm
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
#define _TENSOR_LEN 64
    vsi_status         status;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_tensor_t* reshape_tensors[6] = { NULL };
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM];
    uint32_t new_rank = 4;
    vsi_nn_tensor_t* input0 = NULL;
    vsi_nn_tensor_t* output = NULL;
    char reshape0_tensor_name[_TENSOR_LEN];
    char reshape1_tensor_name[_TENSOR_LEN];
    char batch_norm_tensor_name[_TENSOR_LEN];

    memset(reshape0_tensor_name, 0, sizeof(reshape0_tensor_name));
    memset(reshape1_tensor_name, 0, sizeof(reshape1_tensor_name));
    memset(batch_norm_tensor_name, 0, sizeof(batch_norm_tensor_name));

    status = VSI_FAILURE;

    status = _try_set_high_presision_tensor(inputs);
    if(VSI_SUCCESS != status)
    {
        VSILOGE("Set tensor attr of high presision fail");
        return status;
    }
    if (_require_reshape(self, inputs))
    {
        if (3 == inputs[0]->attr.dim_num)
        {
            shape[0] = inputs[0]->attr.size[0];
            shape[1] = 1;
            shape[2] = inputs[0]->attr.size[1];
            shape[3] = inputs[0]->attr.size[2];
        }
        else if (5 == inputs[0]->attr.dim_num)
        {
            shape[0] = inputs[0]->attr.size[0] * inputs[0]->attr.size[1];
            shape[1] = inputs[0]->attr.size[2];
            shape[2] = inputs[0]->attr.size[3];
            shape[3] = inputs[0]->attr.size[4];
        }

        input0 = vsi_nn_kernel_insert_reshape_node(self->graph,
            inputs[0], shape, (uint32_t)new_rank, VSI_NN_OPTIMIZE_BACKWARD);
        CHECK_PTR_FAIL_GOTO(input0, "Create tensor fail.", final);
        reshape_tensors[0] = input0;
        snprintf(reshape0_tensor_name, sizeof(reshape0_tensor_name), "uid_%u_sub_uid_%u_out_0", self->uid, 0);
        if (vxSetReferenceName((vx_reference)reshape_tensors[0]->t, reshape0_tensor_name) == VSI_FAILURE)
        {
            VSILOGW("Set uid %u reshape 0 node output name fail", self->uid);
            goto final;
        }
        output = vsi_nn_kernel_insert_reshape_node(self->graph,
            outputs[0], shape, (uint32_t)new_rank, VSI_NN_OPTIMIZE_FORWARD);
        CHECK_PTR_FAIL_GOTO(output, "Create tensor fail.", final);
        reshape_tensors[5] = output;
        snprintf(reshape1_tensor_name, sizeof(reshape1_tensor_name), "uid_%u_sub_uid_%u_out_0", self->uid, 1);
        if (vxSetReferenceName((vx_reference)outputs[0]->t, reshape1_tensor_name) == VSI_FAILURE)
        {
            VSILOGW("Set uid %u reshap 1 node output name fail", self->uid);
            goto final;
        }
    }
    else
    {
        reshape_tensors[0] = inputs[0];
        reshape_tensors[5] = outputs[0];
    }

    reshape_tensors[1] = inputs[1];
    reshape_tensors[2] = inputs[2];
    reshape_tensors[3] = inputs[3];
    reshape_tensors[4] = inputs[4];

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_float32( param, "eps", self->nn_param.batch_norm.eps );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "batch_norm",
        reshape_tensors, 5,
        &reshape_tensors[5], 1, param );

    if ( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release(&param);

    if (output)
    {
        snprintf(batch_norm_tensor_name, sizeof(batch_norm_tensor_name), "uid_%u_sub_uid_%u_out_0", self->uid, 2);
        if (vxSetReferenceName((vx_reference)output->t, batch_norm_tensor_name) == VSI_FAILURE)
        {
            VSILOGW("Set uid %u instance_norm node output name fail", self->uid);
            goto final;
        }
    }

final:
    vsi_safe_release_tensor(input0);
    vsi_safe_release_tensor(output);

    return status;
}

static void _expand_dims_for_param
    (
    vsi_nn_tensor_attr_t* attr,
    uint32_t dim_num,
    vsi_size_t* shapes_expand
    )
{
    uint32_t i = 0;

    if (attr->dim_num == 1 && dim_num > 2)
    {
        /* [C] reshape to [1, 1, C, 1] */
        for (i = 0; i < dim_num; i++)
        {
            if (i == dim_num - 2)
            {
                shapes_expand[i] = attr->size[0];
            }
            else
            {
                shapes_expand[i] = 1;
            }
        }
    }
    else
    {
        for (i = 0; i < attr->dim_num; i++)
        {
            shapes_expand[i] = attr->size[i];
        }
    }
}

static vsi_status _dynamic_batchnorm
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_size_t  shapes[4][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    vsi_size_t  shapes_expand[2][VSI_NN_MAX_DIM_NUM] = {{ 0 }};
    vsi_size_t  *shapes_ptr[4] = {NULL};
    vsi_size_t  *shapes_in[3] = {NULL};
    vsi_size_t rank_in[3] = {0};
    uint32_t new_rank = 0;
    vsi_nn_tensor_t* reshape_tensors[6] = { NULL };
    vsi_bool ret = TRUE;
    uint32_t i = 0;

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_float32( param, "eps", self->nn_param.batch_norm.eps );

    rank_in[0] = (vsi_size_t)inputs[0]->attr.dim_num;
    rank_in[1] = (vsi_size_t)inputs[0]->attr.dim_num;
    rank_in[2] = (vsi_size_t)inputs[0]->attr.dim_num;

    /* [C] reshape to [1, 1, C, 1] if need */
    _expand_dims_for_param(&(inputs[1]->attr), inputs[0]->attr.dim_num, shapes_expand[0]);
    _expand_dims_for_param(&(inputs[3]->attr), inputs[0]->attr.dim_num, shapes_expand[1]);

    shapes_in[0] = inputs[0]->attr.size;
    shapes_in[1] = shapes_expand[0];
    shapes_in[2] = shapes_expand[1];
    for (i = 0; i < 4; i++)
    {
        shapes_ptr[i] = shapes[i];
    }

    ret = vsi_nn_kernel_optimize_broadcast_shape(
            (const vsi_size_t**)shapes_in, rank_in, 3,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes_ptr, shapes[3], &new_rank);

    if( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
            inputs[0], shapes[0], new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
            inputs[1], shapes[1], new_rank );
        reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
            inputs[2], shapes[1], new_rank );
        reshape_tensors[3] = vsi_nn_reshape_tensor( self->graph,
            inputs[3], shapes[2], new_rank );
        reshape_tensors[4] = vsi_nn_reshape_tensor( self->graph,
            inputs[4], shapes[2], new_rank );

        reshape_tensors[5] = vsi_nn_reshape_tensor( self->graph,
            outputs[0], shapes[3], new_rank );
    }
    else
    {
        reshape_tensors[0] = inputs[0];
        reshape_tensors[1] = inputs[1];
        reshape_tensors[2] = inputs[2];
        reshape_tensors[3] = inputs[3];
        reshape_tensors[4] = inputs[4];

        reshape_tensors[5] = outputs[0];
    }

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "batchnorm_single",
        reshape_tensors, 5,
        &reshape_tensors[5], 1, param );

    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    if (ret)
    {
        for ( i = 0; i < 6; i++)
        {
            if (reshape_tensors[i])
            {
                vsi_nn_ReleaseTensor( &reshape_tensors[i] );
            }
        }
    }

    vsi_nn_kernel_param_release( &param );

    return status;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    if (_is_dynamic_batchnorm(inputs))
    {
        status = _dynamic_batchnorm(self, inputs, outputs);
    }
    else
    {
        status = _static_batchnorm(self, inputs, outputs);
    }
    return status;
} /* op_compute() */

static vsi_bool _dynamic_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*TODO: Check tensor shapes. */
    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t rank = inputs[0]->attr.dim_num;

    /* check inputs outputs data type */
    BEGIN_IO_TYPE_DECL(BATCHNORM_SINGLE, 5, 1)
        IO_TYPE(D_F16,          D_F16, D_F16, D_F16, D_F32, D_U8|Q_ASYM)
        IO_TYPE(D_F16,          D_F16, D_F16, D_F16, D_F32, D_I16|Q_DFP)
        IO_TYPE(D_F16,          D_F16, D_F16, D_F16, D_F32, D_I8|Q_DFP)
        IO_TYPE(D_F16,          D_F16, D_F16, D_F16, D_F32, D_F16)
        IO_TYPE(D_U8|Q_ASYM,    D_F16, D_F16, D_F16, D_F32, D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,    D_F16, D_F16, D_F16, D_F32, D_F16)
        IO_TYPE(D_I8|Q_DFP,     D_F16, D_F16, D_F16, D_F32, D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,     D_F16, D_F16, D_F16, D_F32, D_F16)
        IO_TYPE(D_I16|Q_DFP,    D_F16, D_F16, D_F16, D_F32, D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,    D_F16, D_F16, D_F16, D_F32, D_F16)
        IO_TYPE(D_F32,          D_F32, D_F32, D_F32, D_F32, D_F32)
        IO_TYPE(D_F16,          D_F32, D_F32, D_F32, D_F32, D_F16)
        IO_TYPE(D_U8|Q_ASYM,    D_F32, D_F32, D_F32, D_F32, D_U8|Q_ASYM)
    END_IO_TYPE_DECL(BATCHNORM_SINGLE)
    if(!VALIDATE_OP_IO_TYPES(BATCHNORM_SINGLE, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    for(i = 0; i < rank; i++)
    {
        vx_int32 shape0 = (int32_t)(inputs[0]->attr.size[i]);

        for ( j = 1; j < self->input.num; j++)
        {
            uint32_t rank1 = inputs[j]->attr.dim_num;
            vx_int32 shape1 = (int32_t)(rank1 > i ? inputs[j]->attr.size[i] : 1);

            if(shape0 != shape1 && shape1 != 1)
            {
                VSILOGE("Invalid broadcast for inputs[%d] size[%u]", j, shape1);
                return FALSE;
            }
        }
    }
    return TRUE;
}

static vsi_bool _static_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(BATCH_NORM, 5, 1)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_F32, D_F32,  D_F32)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_F32, D_F32,  D_F16)
        IO_TYPE(D_F32,        D_F32,  D_F32,  D_F32, D_F32,  D_F32)
        IO_TYPE(D_F32,        D_F32,  D_F32,  D_F32, D_F32,  D_F16)
        IO_TYPE(D_F32,        D_F32,  D_F32,  D_F32, D_F32,  D_BF16)
        IO_TYPE(D_BF16,       D_F32,  D_F32,  D_F32, D_F32,  D_BF16)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_F32, D_F32,  D_U8|Q_ASYM)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_F32, D_F32,  D_I8|Q_DFP)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_F32, D_F32,  D_I8|Q_ASYM)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_F32, D_F32,  D_I8|Q_SYM)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_F32, D_F32,  D_I16|Q_DFP)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_F32, D_F32,  D_I16|Q_ASYM)
        IO_TYPE(D_F16,        D_F32,  D_F32,  D_F32, D_F32,  D_I16|Q_SYM)
        IO_TYPE(D_I8|Q_DFP,   D_F32,  D_F32,  D_F32, D_F32,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,  D_F32,  D_F32,  D_F32, D_F32,  D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,   D_F32,  D_F32,  D_F32, D_F32,  D_I8|Q_SYM)
        IO_TYPE(D_I8|Q_DFP,   D_F32,  D_F32,  D_F32, D_F32,  D_F16)
        IO_TYPE(D_I8|Q_ASYM,  D_F32,  D_F32,  D_F32, D_F32,  D_F16)
        IO_TYPE(D_I8|Q_SYM,   D_F32,  D_F32,  D_F32, D_F32,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F32,  D_F32, D_F32,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM, D_F32,  D_F32,  D_F32, D_F32,  D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,  D_F32,  D_F32,  D_F32, D_F32,  D_I16|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F32,  D_F32, D_F32,  D_F16)
        IO_TYPE(D_I16|Q_ASYM, D_F32,  D_F32,  D_F32, D_F32,  D_F16)
        IO_TYPE(D_I16|Q_SYM,  D_F32,  D_F32,  D_F32, D_F32,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F32,  D_F32, D_F32,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_F32,  D_F32, D_F32,  D_F16)
    END_IO_TYPE_DECL(BATCH_NORM)
    if (!VALIDATE_OP_IO_TYPES(BATCH_NORM, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }
    return TRUE;
}

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if (_is_dynamic_batchnorm(inputs))
    {
        return _dynamic_check(self, inputs, outputs);
    }
    else
    {
        return _static_check(self, inputs, outputs);
    }
} /* op_check() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ BATCH_NORM,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ 5,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
