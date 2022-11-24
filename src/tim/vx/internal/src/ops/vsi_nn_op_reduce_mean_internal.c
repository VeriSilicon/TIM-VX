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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

typedef struct _reduce_mean_internal_local_data_t {
    int32_t placeholder;
} reduce_mean_internal_local_data_t;

/*
 Declare number of input and output.
 */
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
    int32_t * axis = self->nn_param.reduce_mean_internal.axis;
    int32_t axis_num = self->nn_param.reduce_mean_internal.axis_num;
    float scale = self->nn_param.reduce_mean_internal.scale;
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { {0} };
    int32_t new_axis[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    uint32_t axis_size = 0;
    uint32_t rank_in = 0;
    uint32_t rank_out = 0;
    vsi_bool ret = FALSE;
    vsi_nn_kernel_param_t * param = NULL;

    ret = vsi_nn_kernel_optimize_reduce_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            axis, axis_num,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], &rank_in, shapes[1], &rank_out,
            new_axis, &axis_size);

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "axis_num", axis_size );
    vsi_nn_kernel_param_add_float32( param, "scale", scale );

    if (ret)
    {
        uint32_t i = 0;
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
            inputs[0], shapes[0], rank_in );
        for (i = 0; i < axis_size; i++)
        {
            shapes[0][i] = 1;
        }
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
            outputs[0], shapes[0], rank_in );

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                "reduce_mean",
                &reshape_tensors[0], 1,
                &reshape_tensors[1], 1, param );

        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
    }

    if ( self->n )
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
    vsi_nn_reduce_mean_internal_param * p = &(self->nn_param.reduce_mean_internal);

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        uint32_t i = 0;
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;

        memcpy(outputs[0]->attr.size, inputs[0]->attr.size, inputs[0]->attr.dim_num * sizeof(vsi_size_t));

        for (i = 0; i < p->axis_num; i++)
        {
            outputs[0]->attr.size[p->axis[i]] = 1;
        }
    }

    return TRUE;
} /* op_setup() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ REDUCE_MEAN_INTERNAL,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS
