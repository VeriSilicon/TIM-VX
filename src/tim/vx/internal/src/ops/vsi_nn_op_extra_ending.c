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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t *tmp_tensor = NULL;
    vsi_nn_tensor_t *input_tensor[2] = {NULL};
    vsi_nn_extra_ending_param *p = &self->nn_param.extra_ending;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.dim_num = 2;
    attr.size[0] = p->length;
    attr.size[1] = 1;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    attr.vtl = FALSE;
    tmp_tensor = vsi_nn_CreateTensorFromData(self->graph,
        (uint8_t*)&p->value, &attr);

    input_tensor[0] = inputs[0];
    input_tensor[1] = tmp_tensor;

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "signal_frame",
        input_tensor, 2,
        outputs, 1, NULL );

    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_safe_release_tensor(tmp_tensor);

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(EXTRA_ENDING, 1, 1)
        IO_TYPE(D_F16,        D_F16)
        IO_TYPE(D_F16,        D_U8|Q_ASYM)
        IO_TYPE(D_F16,        D_I16|Q_DFP)
        IO_TYPE(D_F16,        D_I8|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_I16|Q_DFP)
        IO_TYPE(D_U8|Q_ASYM,  D_I8|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_U8|Q_ASYM)
        IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,   D_I16|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
    END_IO_TYPE_DECL(EXTRA_ENDING)
    if (!VALIDATE_OP_IO_TYPES(EXTRA_ENDING, self, inputs, self->input.num, outputs, self->output.num))
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
    /* TODO: Add code to comput outputs' shape. */
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        VSILOGE("output size cannot be zero!(EXTRA_ENDING)\n");
        return FALSE;
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
    /* op_name    */ EXTRA_ENDING,
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
