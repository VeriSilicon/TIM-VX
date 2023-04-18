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
#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"
#include "kernel/vsi_nn_kernel.h"
#include "vsi_nn_tensor_util_prv.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VX_FAILURE;
    int32_t axis = self->nn_param.l2_normalize.axis;
    vsi_nn_kernel_param_t * param = NULL;

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "axis", axis );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "l2_norm",
        inputs, 1,
        outputs, 1, param );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );

    return status;
} /* op_compute() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    if (vsi_nn_compareVersion(self->graph, 1, 1, 15) == -1)
    {
        self->nn_param.l2_normalize.axis = 2;
    }

    return status;
} /* op_init() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = vsi_nn_is_stream_process_supported_types(self->graph, inputs, self->input.num);

    if (!ret)
    {
        BEGIN_IO_TYPE_DECL(L2_NORMALIZE, 1, 1)
            IO_TYPE(D_F32,  D_F16)
            IO_TYPE(D_F32,  D_F32)
            IO_TYPE(D_I32,  D_I32)
            IO_TYPE(D_BF16, D_BF16)
            IO_TYPE(D_F16,  D_F32)
            IO_TYPE(D_F16,  D_F16)
            IO_TYPE(D_F16,  D_U8|Q_ASYM)
            IO_TYPE(D_F16,  D_I16|Q_DFP)
            IO_TYPE(D_F16,  D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,   D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,   D_F16)
            IO_TYPE(D_I16|Q_DFP,  D_I16|Q_DFP)
            IO_TYPE(D_I16|Q_DFP,  D_F16)
            IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM,  D_F16)
        END_IO_TYPE_DECL(L2_NORMALIZE)
        if (!VALIDATE_OP_IO_TYPES(L2_NORMALIZE, self, inputs, self->input.num, outputs, self->output.num))
        {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
    }

    return TRUE;
} /* op_check() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ L2_NORMALIZE,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
