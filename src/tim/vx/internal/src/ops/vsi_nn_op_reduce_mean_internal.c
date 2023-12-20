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
    int32_t axis_num = self->nn_param.reduce_mean_internal.axis_num;
    float scale = self->nn_param.reduce_mean_internal.scale;
    vsi_enum type = self->nn_param.reduce_mean_internal.type;
    int32_t *axis = self->nn_param.reduce_mean_internal.axis;
    vsi_nn_kernel_param_t * param = NULL;

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "axis_num", axis_num );
    vsi_nn_kernel_param_add_float32( param, "scale", scale );
    vsi_nn_kernel_param_add_str( param, "axis", (const char*)axis );

    if (type == VSI_NN_REDUCE_MAX)
    {
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                "reduce_max",
                inputs, 1,
                outputs, 1, param );
    }
    else
    {
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                "reduce_mean",
                inputs, 1,
                outputs, 1, param );
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
