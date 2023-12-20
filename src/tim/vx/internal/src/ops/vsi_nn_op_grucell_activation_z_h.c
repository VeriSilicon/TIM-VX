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

typedef struct _grucell_activation_z_h_local_data_t {
    int32_t placeholder;
} grucell_activation_z_h_local_data_t;

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_grucell_activation_param* p = &self->nn_param.grucell_activation;
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t* param;

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32(param, "activation", p->activation);
    vsi_nn_kernel_param_add_int32(param, "recurrent_activation", p->recurrent_activation);

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "grucell_activation_z_h",
        inputs, GRUCELL_ACT_Z_H_IN_CNT,
        outputs, GRUCELL_ACT_Z_H_OUT_CNT,
        param );

    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    VSI_UNREFERENCED(self);
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    VSI_UNREFERENCED(self);

    if (VSI_NN_DIM_AUTO == outputs[GRUCELL_ACT_Z_H_OUT_OUTPUT]->attr.dim_num)
    {
        outputs[GRUCELL_ACT_Z_H_OUT_OUTPUT]->attr.dim_num = \
            inputs[GRUCELL_ACT_Z_H_HSTATE]->attr.dim_num;

        memcpy( outputs[GRUCELL_ACT_Z_H_OUT_OUTPUT]->attr.size,
            inputs[GRUCELL_ACT_Z_H_HSTATE]->attr.size,
            inputs[GRUCELL_ACT_Z_H_HSTATE]->attr.dim_num * sizeof(vsi_size_t) );
    }

    if (VSI_NN_DIM_AUTO == outputs[GRUCELL_ACT_Z_H_OUT_HSTATE]->attr.dim_num)
    {
        outputs[GRUCELL_ACT_Z_H_OUT_HSTATE]->attr.dim_num = \
            inputs[GRUCELL_ACT_Z_H_HSTATE]->attr.dim_num;

        memcpy( outputs[GRUCELL_ACT_Z_H_OUT_HSTATE]->attr.size,
            inputs[GRUCELL_ACT_Z_H_HSTATE]->attr.size,
            inputs[GRUCELL_ACT_Z_H_HSTATE]->attr.dim_num * sizeof(vsi_size_t) );
    }

    return TRUE;
} /* op_setup() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GRUCELL_ACTIVATION_Z_H,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ GRUCELL_ACT_Z_H_IN_CNT,
    /* output_num */ GRUCELL_ACT_Z_H_OUT_CNT
    );

__END_DECLS
