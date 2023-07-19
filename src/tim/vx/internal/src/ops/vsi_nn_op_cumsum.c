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
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_kernel_node_t    n = NULL;

    int32_t axis = self->nn_param.cumsum.axis;
    int32_t exclusive = (int32_t)self->nn_param.cumsum.exclusive;
    int32_t reverse = (int32_t)self->nn_param.cumsum.reverse;

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "axis", (int32_t)axis );
    vsi_nn_kernel_param_add_int32( param, "exclusive", (int32_t)exclusive );
    vsi_nn_kernel_param_add_int32( param, "reverse", (int32_t)reverse );
    n = vsi_nn_kernel_selector( self->graph, "cumsum", inputs, 1, outputs, 1, param );
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
    BEGIN_IO_TYPE_DECL(CUMSUM, 1, 1)
        IO_TYPE(D_U32,         D_U32)
        IO_TYPE(D_F32,         D_F32)
        IO_TYPE(D_F16,         D_F16)
        IO_TYPE(D_BF16,        D_BF16)
        IO_TYPE(D_U8|Q_ASYM,   D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_DFP,    D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,   D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,    D_I8|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,   D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM,  D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,   D_I16|Q_SYM)
        IO_TYPE(D_F16,         D_U8|Q_ASYM)
        IO_TYPE(D_F16,         D_I8|Q_DFP)
        IO_TYPE(D_F16,         D_I8|Q_ASYM)
        IO_TYPE(D_F16,         D_I8|Q_SYM)
        IO_TYPE(D_F16,         D_I16|Q_DFP)
        IO_TYPE(D_F16,         D_I16|Q_ASYM)
        IO_TYPE(D_F16,         D_I16|Q_SYM)
    END_IO_TYPE_DECL(CUMSUM)
    if (!VALIDATE_OP_IO_TYPES(CUMSUM, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.cumsum.axis = 0;
    self->nn_param.cumsum.exclusive = FALSE;
    self->nn_param.cumsum.reverse = FALSE;

    return status;
} /* op_init() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* TODO: Add code to comput outputs' shape. */
    uint32_t i = 0;

    VSI_UNREFERENCED(self);

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        for (i = 0; i < outputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
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
    /* op_name    */ CUMSUM,
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
