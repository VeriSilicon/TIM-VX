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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

static vsi_bool _is_dataconvert_op
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_dtype_t *dtype,*_dtype;

    dtype = &inputs[0]->attr.dtype;
    _dtype = &outputs[0]->attr.dtype;

    if ( dtype->qnt_type == VSI_NN_QNT_TYPE_NONE &&
         dtype->vx_type != VSI_NN_TYPE_FLOAT32 &&
         dtype->vx_type != VSI_NN_TYPE_FLOAT16 &&
         _dtype->qnt_type == VSI_NN_QNT_TYPE_NONE &&
         vsi_nn_OpCheck(VSI_NN_OP_DATACONVERT, self, inputs, outputs) )
    {
        return TRUE;
    }

    if (vsi_nn_DtypeCompare(dtype, _dtype) == FALSE)
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
    vsi_status status = VSI_FAILURE;

    if ( _is_dataconvert_op(self, inputs, outputs) )
    {
        vsi_nn_internal_compute_node( self );
        status = VSI_SUCCESS;
    }
    else
    {
        vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
        vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
        vsi_size_t new_rank = 0;
        vsi_bool ret;

        if ( NULL == self )
        {
            return status;
        }

        ret = vsi_nn_kernel_optimize_element_shape(
                inputs[0]->attr.size, inputs[0]->attr.dim_num,
                shape, &new_rank );
        if ( ret )
        {
            reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                    inputs[0], shape, new_rank );
            reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                    outputs[0], shape, new_rank );

            self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                    "cast",
                    &reshape_tensors[0], 1,
                    &reshape_tensors[1], 1, NULL );

            vsi_nn_ReleaseTensor( &reshape_tensors[0] );
            vsi_nn_ReleaseTensor( &reshape_tensors[1] );
        }

        if ( self->n )
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
    BEGIN_IO_TYPE_DECL(CAST, 1, 1)
        IO_TYPE(D_F32,        D_F32)
        IO_TYPE(D_F32,        D_I32)
        IO_TYPE(D_F32,        D_U32)
        IO_TYPE(D_F32,        D_F16)
        IO_TYPE(D_F32,        D_BOOL8)
        IO_TYPE(D_I32,        D_F32)
        IO_TYPE(D_I32,        D_I32)
        IO_TYPE(D_I32,        D_U32)
        IO_TYPE(D_I32,        D_F16)
        IO_TYPE(D_I32,        D_BOOL8)
        IO_TYPE(D_BOOL8,      D_F32)
        IO_TYPE(D_BOOL8,      D_I32)
        IO_TYPE(D_BOOL8,      D_U32)
        IO_TYPE(D_U32,        D_F32)
        IO_TYPE(D_U32,        D_I32)
        IO_TYPE(D_U32,        D_U32)
        IO_TYPE(D_U32,        D_BOOL8)
        IO_TYPE(D_F16,        D_I16|Q_DFP)
        IO_TYPE(D_F16,        D_I8|Q_DFP)
        IO_TYPE(D_F16,        D_U8|Q_ASYM)
        IO_TYPE(D_F16,        D_BOOL8)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,  D_BOOL8)
        IO_TYPE(D_I16,        D_F16)
        IO_TYPE(D_I16,        D_I8|Q_DFP)
        IO_TYPE(D_I16,        D_U8|Q_ASYM)
        IO_TYPE(D_I16,        D_BOOL8)
        IO_TYPE(D_I16,        D_I32)
        IO_TYPE(D_I16,        D_U32)
        IO_TYPE(D_I16,        D_F32)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,   D_BOOL8)
        IO_TYPE(D_I8,         D_F16)
        IO_TYPE(D_I8,         D_I16|Q_DFP)
        IO_TYPE(D_I8,         D_U8|Q_ASYM)
        IO_TYPE(D_I8,         D_BOOL8)
        IO_TYPE(D_I8,         D_I32)
        IO_TYPE(D_I8,         D_U32)
        IO_TYPE(D_I8,         D_F32)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_BOOL8)
        IO_TYPE(D_U8,         D_F16)
        IO_TYPE(D_U8,         D_I16|Q_DFP)
        IO_TYPE(D_U8,         D_I8|Q_DFP)
        IO_TYPE(D_U8,         D_BOOL8)
        IO_TYPE(D_U8,         D_I32)
        IO_TYPE(D_U8,         D_U32)
        IO_TYPE(D_U8,         D_F32)
        IO_TYPE(D_F32,        D_I16|Q_DFP)
        IO_TYPE(D_F32,        D_I8|Q_DFP)
        IO_TYPE(D_F32,        D_U8|Q_ASYM)
        IO_TYPE(D_I32,        D_I16|Q_DFP)
        IO_TYPE(D_I32,        D_I8|Q_DFP)
        IO_TYPE(D_I32,        D_U8|Q_ASYM)
        IO_TYPE(D_F16,        D_F32)
        IO_TYPE(D_F16,        D_I32)
        IO_TYPE(D_F16,        D_I16)
        IO_TYPE(D_F16,        D_U8)
        IO_TYPE(D_F16,        D_I8)
        IO_TYPE(D_F16,        D_F16)
        IO_TYPE(D_BOOL8,      D_F16)
        IO_TYPE(D_BOOL8,      D_I16|Q_DFP)
        IO_TYPE(D_BOOL8,      D_I8|Q_DFP)
        IO_TYPE(D_BOOL8,      D_U8|Q_ASYM)
        IO_TYPE(D_BOOL8,      D_BOOL8)
        IO_TYPE(D_BOOL8,      D_I16)
        IO_TYPE(D_BOOL8,      D_I8)
        IO_TYPE(D_BOOL8,      D_U8)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_F32)
        IO_TYPE(D_U8|Q_ASYM,  D_I32)
        IO_TYPE(D_BF16,       D_BF16)
    END_IO_TYPE_DECL(CAST)
    if(!VALIDATE_OP_IO_TYPES(CAST, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */


static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status     status;

    status = VSI_SUCCESS;

    if ( _is_dataconvert_op(self, inputs, outputs) )
    {
        vsi_nn_internal_optimize_node( self, direction );
    }

    return status;
} /* op_optimize() */


static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;

    if ( NULL == self )
    {
        return FALSE;
    }
    ret = vsi_nn_op_common_setup(self, inputs, outputs);

    if (  _is_dataconvert_op(self, inputs, outputs) )
    {
        vsi_nn_internal_node_t* curr = NULL;
        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_DATACONVERT, 1, 1);
        if (NULL == curr)
        {
            return FALSE;
        }
        curr->inputs[0]     = inputs[0];
        curr->outputs[0]    = outputs[0];

        vsi_nn_internal_setup_node(self, curr);
    }

    return ret;
}

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_internal_deinit_node_wksp(self);
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_internal_init_node_wksp(self);
    return status;
} /* op_init() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CAST,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS
