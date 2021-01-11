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

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_t n;
    vsi_nn_kernel_param_t* param;

    param = vsi_nn_kernel_param_create();
    n = vsi_nn_kernel_selector( self->graph, "grucell_activation_sma",
        inputs, GRUCELL_ACTIVATION_SMA_INPUT_COUNT,
        outputs, GRUCELL_ACTIVATION_SMA_OUTPUT_COUNT,
        param );

    self->n = (vx_node)n;
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
    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_ACTIVATION_SMA_OUTPUT_OUTPUT]->attr.dim_num)
    {
        outputs[GRUCELL_ACTIVATION_SMA_OUTPUT_OUTPUT]->attr.dim_num = \
            inputs[GRUCELL_ACTIVATION_SMA_INPUT_H_STATE]->attr.dim_num;
        memcpy( outputs[GRUCELL_ACTIVATION_SMA_OUTPUT_OUTPUT]->attr.size,
            inputs[GRUCELL_ACTIVATION_SMA_INPUT_H_STATE]->attr.size,
            inputs[GRUCELL_ACTIVATION_SMA_INPUT_H_STATE]->attr.dim_num * sizeof( uint32_t ) );
    }

    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_ACTIVATION_SMA_OUTPUT_H_STATE]->attr.dim_num)
    {
        outputs[GRUCELL_ACTIVATION_SMA_OUTPUT_H_STATE]->attr.dim_num = \
            inputs[GRUCELL_ACTIVATION_SMA_OUTPUT_OUTPUT]->attr.dim_num;
        memcpy( outputs[GRUCELL_ACTIVATION_SMA_OUTPUT_H_STATE]->attr.size,
            inputs[GRUCELL_ACTIVATION_SMA_OUTPUT_OUTPUT]->attr.size,
            inputs[GRUCELL_ACTIVATION_SMA_OUTPUT_OUTPUT]->attr.dim_num * sizeof( uint32_t ) );
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    return vsi_nn_op_common_deinit(self);
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GRUCELL_ACTIVATION_INTERNAL_SMA,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ GRUCELL_ACTIVATION_SMA_INPUT_COUNT,
    /* output_num */ GRUCELL_ACTIVATION_SMA_OUTPUT_COUNT
    );
__END_DECLS
