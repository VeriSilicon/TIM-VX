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
#include "vsi_nn_error.h"

typedef struct _reducel2_local_data_t {
    int32_t placeholder;
} reducel2_local_data_t;

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
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return vsi_nn_internal_compute_node( self );

} /* op_compute() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return vsi_nn_internal_optimize_node( self, direction );
}


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
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* square_tensor = NULL;
    vsi_nn_internal_tensor_t* reducesum_tensor = NULL;
    vsi_nn_internal_node_t* square_node = NULL;
    vsi_nn_internal_node_t* reducesum_node = NULL;
    vsi_nn_internal_node_t* sqrt_node = NULL;
    vsi_nn_kernel_dtype_e in_dtype;

    vsi_nn_reducel2_param * p0 = &self->nn_param.reducel2;

    vsi_nn_internal_init_node_wksp(self);

    in_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    if (in_dtype == U8 || in_dtype == I8)
    {
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    else
    {
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    }

    square_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
    CHECK_PTR_FAIL_GOTO(square_tensor, "Create internal tensor failed", final);
    square_node = vsi_nn_internal_new_node( self, VSI_NN_OP_SQUARE, 0, 0);
    CHECK_PTR_FAIL_GOTO(square_node, "Create internal node failed", final);

    square_node->inputs[0] = inputs[0];
    square_node->outputs[0] = square_tensor->t;
    vsi_nn_internal_setup_node( self, square_node );

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;

    reducesum_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
    CHECK_PTR_FAIL_GOTO(square_tensor, "Create internal tensor failed", final);
    reducesum_node = vsi_nn_internal_new_node( self, VSI_NN_OP_REDUCE, 0, 0);
    CHECK_PTR_FAIL_GOTO(reducesum_node, "Create internal node failed", final);

    reducesum_node->node->nn_param.reduce.type = VSI_NN_REDUCE_SUM;
    reducesum_node->node->nn_param.reduce.axis = p0->axis;
    reducesum_node->node->nn_param.reduce.axis_num = p0->axis_num;
    reducesum_node->node->nn_param.reduce.keep_dim = p0->keep_dim;

    reducesum_node->inputs[0] = square_tensor->t;
    reducesum_node->outputs[0] = reducesum_tensor->t;
    vsi_nn_internal_setup_node( self, reducesum_node );

    sqrt_node = vsi_nn_internal_new_node( self, VSI_NN_OP_SQRT, 0, 0);
    CHECK_PTR_FAIL_GOTO(sqrt_node, "Create internal node failed", final);

    sqrt_node->inputs[0] = reducesum_tensor->t;
    sqrt_node->outputs[0] = outputs[0];
    vsi_nn_internal_setup_node( self, sqrt_node );

    /* TODO: Add code to comput outputs' shape. */
    return TRUE;
final:
    return FALSE;
} /* op_setup() */


__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ REDUCEL2,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

