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
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util_prv.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"
#include "vsi_nn_error.h"

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
    vsi_nn_kernel_node_t    n = NULL;
    float eps = self->nn_param.rmsnorm.eps;
    int32_t axis = self->nn_param.rmsnorm.axis;

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_float32(param, "eps", eps);
    vsi_nn_kernel_param_add_int32(param, "axis", axis);
    n = vsi_nn_kernel_selector(self->graph, "rms_norm",
        inputs, _INPUT_NUM, outputs, _OUTPUT_NUM, param);
    if (n != NULL)
    {
        self->n = (vx_node)n;
        status = VSI_SUCCESS;
    }

    if (param != NULL)
    {
        vsi_nn_kernel_param_release(&param);
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
    vsi_bool ret = vsi_nn_is_stream_process_supported_types(self->graph, inputs, self->input.num);

    if (!ret)
    {
        BEGIN_IO_TYPE_DECL(RMS_NORM, 2, 1)
            IO_TYPE(D_F32, D_F32, D_F32)
            IO_TYPE(D_F16, D_F32, D_F16)
            IO_TYPE(D_F16, D_F32, D_F16)
            IO_TYPE(D_F16, D_F32, D_U8 | Q_ASYM)
            IO_TYPE(D_F16, D_F32, D_U8 | Q_ASYM)
            IO_TYPE(D_F16, D_F32, D_I8 | Q_DFP)
            IO_TYPE(D_F16, D_F32, D_I8 | Q_DFP)
            IO_TYPE(D_F16, D_F32, D_I8 | Q_ASYM)
            IO_TYPE(D_F16, D_F32, D_I8 | Q_ASYM)
            IO_TYPE(D_F16, D_F32, D_I8 | Q_SYM)
            IO_TYPE(D_F16, D_F32, D_I8 | Q_SYM)
            IO_TYPE(D_F16, D_F32, D_I16 | Q_DFP)
            IO_TYPE(D_F16, D_F32, D_I16 | Q_DFP)
            IO_TYPE(D_F16, D_F32, D_I16 | Q_ASYM)
            IO_TYPE(D_F16, D_F32, D_I16 | Q_ASYM)
            IO_TYPE(D_F16, D_F32, D_I16 | Q_SYM)
            IO_TYPE(D_F16, D_F32, D_I16 | Q_SYM)
            IO_TYPE(D_BF16, D_F32, D_BF16)
            IO_TYPE(D_U8 | Q_ASYM, D_F32, D_F16)
            IO_TYPE(D_U8 | Q_ASYM, D_F32, D_U8 | Q_ASYM)
            IO_TYPE(D_I16 | Q_DFP, D_F32, D_I16 | Q_DFP)
            IO_TYPE(D_I16 | Q_ASYM, D_F32, D_I16 | Q_ASYM)
            IO_TYPE(D_I16 | Q_SYM, D_F32, D_I16 | Q_SYM)
            IO_TYPE(D_I16 | Q_DFP, D_F32, D_F16)
            IO_TYPE(D_I16 | Q_ASYM, D_F32, D_F16)
            IO_TYPE(D_I16 | Q_SYM, D_F32, D_F16)
            IO_TYPE(D_I8 | Q_DFP, D_F32, D_I8 | Q_DFP)
            IO_TYPE(D_I8 | Q_ASYM, D_F32, D_I8 | Q_ASYM)
            IO_TYPE(D_I8 | Q_SYM, D_F32, D_I8 | Q_SYM)
            IO_TYPE(D_I8 | Q_DFP, D_F32, D_F16)
            IO_TYPE(D_I8 | Q_ASYM, D_F32, D_F16)
            IO_TYPE(D_I8 | Q_SYM, D_F32, D_F16)
            IO_TYPE(D_U8 | Q_ASYM, D_F32, D_U8 | Q_ASYM)
            IO_TYPE(D_U8 | Q_ASYM, D_F32, D_F16)
            IO_TYPE(D_I16 | Q_DFP, D_F32, D_I16 | Q_DFP)
            IO_TYPE(D_I16 | Q_ASYM, D_F32, D_I16 | Q_ASYM)
            IO_TYPE(D_I16 | Q_SYM, D_F32, D_I16 | Q_SYM)
            IO_TYPE(D_I16 | Q_DFP, D_F32, D_F16)
            IO_TYPE(D_I16 | Q_ASYM, D_F32, D_F16)
            IO_TYPE(D_I16 | Q_SYM, D_F32, D_F16)
            IO_TYPE(D_I8 | Q_DFP, D_F32, D_I8 | Q_DFP)
            IO_TYPE(D_I8 | Q_ASYM, D_F32, D_I8 | Q_ASYM)
            IO_TYPE(D_I8 | Q_SYM, D_F32, D_I8 | Q_SYM)
            IO_TYPE(D_I8 | Q_DFP, D_F32, D_F16)
            IO_TYPE(D_I8 | Q_ASYM, D_F32, D_F16)
            IO_TYPE(D_I8 | Q_SYM, D_F32, D_F16)
            END_IO_TYPE_DECL(RMS_NORM)
            if (!VALIDATE_OP_IO_TYPES(RMS_NORM, self, inputs, self->input.num, outputs, self->output.num))
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

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;

    if (NULL == self)
    {
        return FALSE;
    }

    ret = vsi_nn_op_common_setup(self, inputs, outputs);

    return ret;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    self->nn_param.rmsnorm.axis = 0;
    self->nn_param.rmsnorm.eps = 1e-8f;
    return VSI_SUCCESS;
} /* op_init() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RMSNORM,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

