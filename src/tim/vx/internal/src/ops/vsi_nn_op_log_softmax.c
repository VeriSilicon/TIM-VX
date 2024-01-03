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
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status _log_softmax_op_compute
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    int32_t axis = 0;
    float betaValue = 0;

    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_log_softmax_param * p = NULL;

    if( NULL == self )
    {
        return VSI_FAILURE;
    }
    status = VSI_FAILURE;

    p = &(self->nn_param.log_softmax);
    axis = p->axis;
    betaValue = p->betaValue;

    // TODO: This optimzie is a hack for gpu path,
    // it should be moved to gpu kernel setup.

    param =vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_int32( param, "axis", axis );
    vsi_nn_kernel_param_add_float32( param, "beta", betaValue );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
            kernel_name,
            inputs, 1,
            outputs, 1, param );

    vsi_nn_kernel_param_release( &param );

    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    return status;
} /* _log_softmax_op_compute() */

static vsi_bool _log_softmax_op_setup
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    VSI_UNREFERENCED(kernel_name);

    /* TODO: Add code to comput outputs' shape. */
    if( NULL == self )
    {
        return FALSE;
    }

    if (self->nn_param.log_softmax.axis < 0)
        self->nn_param.log_softmax.axis += (int32_t)inputs[0]->attr.dim_num;

    if (self->nn_param.log_softmax.axis < 0)
    {
        VSILOGD("LogSoftMax Invalid Axis: %d", self->nn_param.log_softmax.axis);
        return FALSE;
    }

    vsi_nn_op_common_setup(self, inputs, outputs);

    return TRUE;
} /* _log_softmax_op_setup() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(LOG_SOFTMAX, 1, 1)
        IO_TYPE(D_F16,          D_U8|Q_ASYM)
        IO_TYPE(D_F16,          D_I16|Q_DFP)
        IO_TYPE(D_F16,          D_I16|Q_ASYM)
        IO_TYPE(D_F16,          D_I16|Q_SYM)
        IO_TYPE(D_F16,          D_I8|Q_DFP)
        IO_TYPE(D_F16,          D_I8|Q_ASYM)
        IO_TYPE(D_F16,          D_I8|Q_SYM)
        IO_TYPE(D_F16,          D_F16)
        IO_TYPE(D_F32,          D_F32)
        IO_TYPE(D_BF16,         D_F16)
        IO_TYPE(D_BF16,         D_F32)
        IO_TYPE(D_BF16,         D_BF16)
        IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,    D_F16)
        IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_SYM,     D_I8|Q_SYM)
        IO_TYPE(D_I8|Q_DFP,     D_F16)
        IO_TYPE(D_I8|Q_ASYM,    D_F16)
        IO_TYPE(D_I8|Q_SYM,     D_F16)
        IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_ASYM,   D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_SYM,    D_I16|Q_SYM)
        IO_TYPE(D_I16|Q_DFP,    D_F16)
        IO_TYPE(D_I16|Q_ASYM,   D_F16)
        IO_TYPE(D_I16|Q_SYM,    D_F16)
    END_IO_TYPE_DECL(LOG_SOFTMAX)
    if (!VALIDATE_OP_IO_TYPES(LOG_SOFTMAX, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

#ifdef __cplusplus
extern "C" {
#endif

#define DEF_LOG_SOFTMAX_OP(name, kernel_name) \
            static vsi_status op_compute_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _log_softmax_op_compute( ""#kernel_name, self, inputs, outputs ); \
            } \
            static vsi_bool op_setup_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _log_softmax_op_setup( ""#kernel_name, self, inputs, outputs ); \
            } \
DEF_OP_REG  \
    ( \
    /* op_name    */ name, \
    /* init       */ NULL, \
    /* compute    */ op_compute_##kernel_name, \
    /* deinit     */ vsi_nn_op_common_deinit, \
    /* check      */ op_check, \
    /* setup      */ op_setup_##kernel_name, \
    /* optimize   */ NULL, \
    /* input_num  */ 1, \
    /* output_num */ 1 \
    )
/*            DEF_OP_REG(name, op_init_##kernel_name, op_compute_##kernel_name, \
                    NULL, NULL, op_setup_##kernel_name, NULL, 1, 1)*/

DEF_LOG_SOFTMAX_OP( LOG_SOFTMAX, log_softmax );

#undef DEF_LOG_SOFTMAX_OP

#ifdef __cplusplus
}
#endif
