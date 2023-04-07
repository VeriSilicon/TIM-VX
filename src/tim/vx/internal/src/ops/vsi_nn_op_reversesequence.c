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
#include "utils/vsi_nn_constraint_check.h"

typedef struct _reversesequence_local_data_t {
    int32_t placeholder;
} reversesequence_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t* param = NULL;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    uint32_t i = 0;
    uint32_t new_rank = 3;
    vsi_size_t shapes[VSI_NN_MAX_DIM_NUM] = { 1 };
    int32_t batch_axis = (int32_t)self->nn_param.reversesequence.batch_axis;
    int32_t time_axis  = (int32_t)self->nn_param.reversesequence.time_axis;

    if (inputs[0]->attr.dim_num == 2)
    {
        shapes[0] = 1;
        shapes[1] = inputs[0]->attr.size[0];
        shapes[2] = inputs[0]->attr.size[1];
    }
    if (inputs[0]->attr.dim_num > 2)
    {
        shapes[2] = inputs[0]->attr.size[inputs[0]->attr.dim_num - 1];
        shapes[1] = inputs[0]->attr.size[inputs[0]->attr.dim_num - 2];
        for (i = 0;i < inputs[0]->attr.dim_num - 2; i++)
        {
            shapes[0] = shapes[0] * inputs[0]->attr.size[i];
        }
    }

    reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
            inputs[0], shapes, new_rank );
    reshape_tensors[1] = inputs[1];
    reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
            outputs[0], shapes, new_rank );

    param = vsi_nn_kernel_param_create();
    if (batch_axis == (int32_t)inputs[0]->attr.dim_num - 1)
    {
        batch_axis = 2;
    }
    else
    {
        batch_axis = 1;
    }
    if (time_axis == (int32_t)inputs[0]->attr.dim_num - 1)
    {
        time_axis = 2;
    }
    else
    {
        time_axis = 1;
    }

    vsi_nn_kernel_param_add_int32(param, "batch_axis", batch_axis);
    vsi_nn_kernel_param_add_int32(param, "time_axis", time_axis);

    self->n = (vx_node)vsi_nn_kernel_selector(self->graph,"reversesequence",
        &reshape_tensors[0],_INPUT_NUM,&reshape_tensors[2],_OUTPUT_NUM,param);
    if ( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release(&param);
    return status;
} /* op_compute()  */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    int32_t batch_axis = (int32_t)self->nn_param.reversesequence.batch_axis;
    int32_t time_axis = (int32_t)self->nn_param.reversesequence.time_axis;
    BEGIN_IO_TYPE_DECL(REVERSESEQUENCE, 2, 1)
        IO_TYPE(D_F32,        D_I32,   D_F32)
        IO_TYPE(D_F16,        D_I32,   D_F16)
        IO_TYPE(D_BF16,       D_I32,   D_BF16)
        IO_TYPE(D_I16|Q_SYM,  D_I32,   D_I16|Q_SYM)
        IO_TYPE(D_U8|Q_ASYM,  D_I32,   D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,   D_I32,   D_I8|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,  D_I32,   D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_I32,   D_I8|Q_DFP)
        IO_TYPE(D_F32,        D_I32,   D_I16|Q_SYM)
        IO_TYPE(D_F16,        D_I32,   D_I16|Q_SYM)
        IO_TYPE(D_F32,        D_I32,   D_I16|Q_DFP)
        IO_TYPE(D_F16,        D_I32,   D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_SYM,  D_I32,   D_F32)
        IO_TYPE(D_U8|Q_ASYM,  D_I32,   D_F32)
        IO_TYPE(D_I8|Q_SYM,   D_I32,   D_F32)
        IO_TYPE(D_I16|Q_DFP,  D_I32,   D_F32)
        IO_TYPE(D_I8|Q_DFP,   D_I32,   D_F32)
        IO_TYPE(D_F32,        D_I32,   D_U8|Q_ASYM)
        IO_TYPE(D_F16,        D_I32,   D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,  D_I32,   D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_I32,   D_F16)
        IO_TYPE(D_I8|Q_SYM,   D_I32,   D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_I32,   D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_I32,   D_F16)
        IO_TYPE(D_F32,        D_I32,   D_I8|Q_SYM)
        IO_TYPE(D_F16,        D_I32,   D_I8|Q_SYM)
        IO_TYPE(D_F32,        D_I32,   D_I8|Q_DFP)
        IO_TYPE(D_F16,        D_I32,   D_I8|Q_DFP)
    END_IO_TYPE_DECL(REVERSESEQUENCE)

    if (!VALIDATE_OP_IO_TYPES(
            REVERSESEQUENCE, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }
    if (inputs[0]->attr.dim_num < 2)
    {
        VSILOGE("inputs[0] dim should be greater than 2");
        return FALSE;
    }
    if ((batch_axis != (int32_t)inputs[0]->attr.dim_num - 1 &&
        batch_axis != (int32_t)inputs[0]->attr.dim_num - 2) ||
        (time_axis != (int32_t)inputs[0]->attr.dim_num - 1 &&
        time_axis != (int32_t)inputs[0]->attr.dim_num - 2))
    {
        VSILOGE("batch_axis must be inputs[0]->attr.dim_num - 1 \
            of inputs[0]->attr.dim_num - 1, so do time_axis");
        return FALSE;
    }
    if (inputs[1]->attr.size[0] != inputs[0]->attr.size[batch_axis])
    {
        VSILOGE("inputs[1] should have shape `[batch_size]`");
        return FALSE;
    }

    return TRUE;
} /* op_check() */


__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ REVERSESEQUENCE,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

