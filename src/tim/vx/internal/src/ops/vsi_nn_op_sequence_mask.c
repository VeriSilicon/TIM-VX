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
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _ARG_NUM            (1)
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
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_kernel_node_t    n = NULL;
    int32_t max_len = self->nn_param.sequence_mask.maxlen;

    param =vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "max_len", max_len );
    n = vsi_nn_kernel_selector( self->graph, "sequence_mask", inputs, 2, outputs, 1, param );
    if ( n != NULL )
    {
        self->n = (vx_node)n;
        status = VSI_SUCCESS;
    }

    if (param != NULL)
    {
        vsi_nn_kernel_param_release( &param );
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
    vsi_nn_sequence_mask_param * p = NULL;

    BEGIN_IO_TYPE_DECL(SEQUENCE_MASK, 2, 1)
        IO_TYPE(D_U8|Q_ASYM,  D_I32, D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,   D_I32, D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I32, D_I16|Q_DFP)
        IO_TYPE(D_U8,   D_I32, D_U8)
        IO_TYPE(D_I8,   D_I32, D_I8)
        IO_TYPE(D_I16,  D_I32, D_I16)
        IO_TYPE(D_F16,  D_I32, D_F16)
        IO_TYPE(D_I32,  D_I32, D_U8|Q_ASYM)
        IO_TYPE(D_I32,  D_I32, D_BOOL8)
        IO_TYPE(D_I16,  D_I32, D_BOOL8)
        IO_TYPE(D_I16,  D_I16, D_BOOL8)
        IO_TYPE(D_I8,   D_I32, D_BOOL8)
        IO_TYPE(D_I8,   D_I16, D_BOOL8)
        IO_TYPE(D_U8,   D_I32, D_BOOL8)
        IO_TYPE(D_U8,   D_I16, D_BOOL8)
        IO_TYPE(D_F16,  D_I32, D_BOOL8)
        IO_TYPE(D_F16,  D_I16, D_BOOL8)
        IO_TYPE(D_I32,  D_I32, D_I32)
        IO_TYPE(D_I32,  D_I32, D_I32|Q_DFP)
        IO_TYPE(D_I32,  D_I32, D_F32)

        IO_TYPE(D_U8|Q_ASYM,  D_NONE, D_U8|Q_ASYM)
        IO_TYPE(D_I32,  D_NONE, D_U8|Q_ASYM)
        IO_TYPE(D_I32,  D_NONE, D_BOOL8)
    END_IO_TYPE_DECL(SEQUENCE_MASK)
    if (!VALIDATE_OP_IO_TYPES(SEQUENCE_MASK, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    p = &(self->nn_param.sequence_mask);
    if (p->maxlen < 0)
    {
        VSILOGE("Max length must bigger than 1");
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
    /* TODO: Add code to comput outputs' shape. */
    uint32_t i = 0;
    vsi_nn_sequence_mask_param * p = NULL;

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        p = &(self->nn_param.sequence_mask);
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num + 1;
        outputs[0]->attr.size[0] = p->maxlen;
        for (i = 0; i < (uint32_t)inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i+1] = inputs[0]->attr.size[i];
        }
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SEQUENCE_MASK,
    /* init       */ NULL,
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
