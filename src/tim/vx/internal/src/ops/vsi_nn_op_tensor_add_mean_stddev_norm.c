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
#include "libnnext/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VX_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    vsi_nn_tensor_add_mean_stddev_norm_param * p = NULL;
    float eps;

    p     = &(self->nn_param.tensor_add_mean_stddev_norm);
    param = vsi_nn_kernel_param_create();
    eps   = p->eps;
    vsi_nn_kernel_param_add_float32( param, "eps",  eps );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                    "add_mean_std_norm",
                    inputs,  _INPUT_NUM,
                    outputs, _OUTPUT_NUM, param );

    vsi_nn_kernel_param_release( &param );

    if( self->n )
    {
        status = VSI_SUCCESS;
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
    BEGIN_IO_TYPE_DECL(TENSOR_ADD_MEAN_STDDEV_NORM, 2, 1)
        IO_TYPE(D_F16, D_F16, D_F16)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_F16)
        IO_TYPE(D_I16|Q_DFP, D_I16|Q_DFP, D_F16)
        IO_TYPE(D_U8|Q_ASYM, D_U8|Q_ASYM, D_F32)
        IO_TYPE(D_F32, D_F32, D_F32)
        IO_TYPE(D_F32, D_F32, D_F16)
        IO_TYPE(D_F32, D_F16, D_F32)
        IO_TYPE(D_F32, D_F16, D_F16)
        IO_TYPE(D_F16, D_F32, D_F32)
        IO_TYPE(D_F16, D_F32, D_F16)
        IO_TYPE(D_F16, D_F16, D_F32)
    END_IO_TYPE_DECL(TENSOR_ADD_MEAN_STDDEV_NORM)
    if(!VALIDATE_OP_IO_TYPES(TENSOR_ADD_MEAN_STDDEV_NORM, self, inputs, self->input.num, outputs, self->output.num)) {
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
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* TODO: Add code to comput outputs' shape. */
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = inputs[0]->attr.size[0];
        outputs[0]->attr.size[1] = inputs[0]->attr.size[1];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    }
    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ TENSOR_ADD_MEAN_STDDEV_NORM,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
