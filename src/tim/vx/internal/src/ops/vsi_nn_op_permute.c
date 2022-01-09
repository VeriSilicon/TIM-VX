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
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_bool _is_same_memory_shape
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t input_dims[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t perm_dims[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t i = 0;
    uint32_t idx = 0;
    uint32_t dim_num0 = inputs[0]->attr.dim_num;
    uint32_t dim_num1 = self->nn_param.permute.dim_num;

    if (dim_num0 != dim_num1)
        return FALSE;

    /********squeeze tensor shape*******/
    for (i = 0; i < inputs[0]->attr.dim_num; i++)
    {
        if (inputs[0]->attr.size[i] == 1)
        {
            dim_num0 --;
        }
        else
        {
            input_dims[idx++] = i;
        }
    }

    for (i = 0, idx = 0; i < self->nn_param.permute.dim_num; i++)
    {
        uint32_t d = self->nn_param.permute.perm[i];

        if (inputs[0]->attr.size[d] == 1)
        {
            dim_num1 --;
        }
        else
        {
            perm_dims[idx++] = d;
        }
    }

    if (dim_num0 != dim_num1)
        return FALSE;

    for (i = 0; i < dim_num0; i++)
    {
        if (input_dims[i] != perm_dims[i])
            return FALSE;
    }

    return TRUE;
} /* _is_same_memory_shape */

static vsi_bool _is_same_quant
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_dtype_t *dtype,*_dtype;

    dtype = &inputs[0]->attr.dtype;
    _dtype = &outputs[0]->attr.dtype;

    if(vsi_nn_DtypeCompare(dtype, _dtype) == FALSE)
    {
        return FALSE;
    }

    return TRUE;
} /* _is_same_quant */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    uint32_t perm[VSI_NN_MAX_DIM_NUM] = {0};
    status = VSI_SUCCESS;

    if (self->nn_param.permute.local.initialized == FALSE)
    {
        memcpy(perm, self->nn_param.permute.perm,
            sizeof(uint32_t) * self->nn_param.permute.dim_num);
        self->n = vxTensorPermuteNode(
            self->graph->g,
            inputs[0]->t,
            outputs[0]->t,
            perm,
            self->nn_param.permute.dim_num
            );

        if( NULL != self->n )
        {
            status = VSI_SUCCESS;
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
    BEGIN_IO_TYPE_DECL(PERMUTE, 1, 1)
        IO_TYPE(D_F16,  D_F16)
        IO_TYPE(D_F16,  D_F32)
        IO_TYPE(D_I16,  D_I16)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_F32)
        IO_TYPE(D_I8|Q_SYM_PC,   D_I8|Q_SYM_PC)
        IO_TYPE(D_BOOL8,  D_BOOL8)
        IO_TYPE(D_BOOL8,  D_I8|Q_DFP)
        IO_TYPE(D_F32,  D_F32)
        IO_TYPE(D_F32,  D_BF16)
        IO_TYPE(D_F32,  D_F16)
        IO_TYPE(D_BF16, D_F32)
        IO_TYPE(D_BF16, D_BF16)
        IO_TYPE(D_I32,  D_I32)
    END_IO_TYPE_DECL(PERMUTE)
    if (!VALIDATE_OP_IO_TYPES(PERMUTE, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret;
    uint32_t i;
    uint32_t axis;

    if( self->nn_param.permute.dim_num != inputs[0]->attr.dim_num )
    {
        VSILOGE( "Error permute dims '%u' vs '%u' ",
            self->nn_param.permute.dim_num, inputs[0]->attr.dim_num );
        return FALSE;
    }

    ret = TRUE;
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        for( i = 0; i < self->nn_param.permute.dim_num; i ++ )
        {
            axis = self->nn_param.permute.perm[i];
            if( axis >= inputs[0]->attr.dim_num )
            {
                VSILOGE( "Error permute axis '%u', the dim is '%u' ",
                    axis, inputs[0]->attr.dim_num );
                ret = FALSE;
                break;
            }
            outputs[0]->attr.size[i] = inputs[0]->attr.size[axis];
        }
    }

    return ret;
} /* op_setup() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status     status;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM];
    uint32_t i = 0;

    status = VSI_SUCCESS;

    if (_is_same_memory_shape(self, inputs, outputs) == FALSE ||
        _is_same_quant(self, inputs, outputs) == FALSE ||
        (inputs[0]->t != NULL && outputs[0]->t != NULL))
    {
        return status;
    }

    VSILOGD("Optimize %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);

    for (i = 0; i < self->nn_param.permute.dim_num; i++)
    {
        shape[i] = inputs[0]->attr.size[self->nn_param.permute.perm[i]];
    }

    if( direction == VSI_NN_OPTIMIZE_BACKWARD )
    {
        if(NULL == inputs[0]->t && NULL != outputs[0]->t)
        {
            inputs[0]->t = vsi_nn_safe_reshape_tensor( outputs[0]->t,
                (void*)inputs[0]->attr.size, (vsi_size_t)inputs[0]->attr.dim_num, sizeof(inputs[0]->attr.size[0]) );
            if( inputs[0]->t == NULL )
            {
                status = VSI_FAILURE;
            }
            self->nn_param.permute.local.initialized = TRUE;
        }
    }
    else
    {
        if(NULL == outputs[0]->t)
        {
            vsi_bool ret;
            ret = vsi_nn_ReshapeTensor( self->graph, inputs[0], outputs[0],
                shape, (vsi_size_t)self->nn_param.permute.dim_num );
            if( ret == FALSE )
            {
                status = VSI_FAILURE;
            }
            self->nn_param.permute.local.initialized = TRUE;
        }
    }

    //vsi_nn_ReshapeTensor(self->graph, inputs[0], outputs[0], shape, self->nn_param.permute.dim_num);

    return status;
} /* op_optimize() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ PERMUTE,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

