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
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)

static vsi_bool _get_stackconcat_shape
    (
    const vsi_ssize_t* shape_x, const int32_t rank_x,
    const vsi_ssize_t* shape_output, const int32_t rank_output,
    const int32_t axis,
    vsi_ssize_t* out_shape_0, uint32_t* out_rank_0,
    vsi_ssize_t* out_shape_1, uint32_t* out_rank_1,
    vsi_ssize_t* out_shape_output, uint32_t* out_rank_output
    )
{
    int32_t i = 0;
    vsi_size_t innerSize = 1;
    vsi_size_t outerSize = 1;

    for ( i = 0; i < rank_x; i++)
    {
        innerSize *= shape_x[i];
    }
    for ( i = axis + 1; i < rank_x; i++)
    {
        outerSize *= shape_x[i];
    }
    out_shape_0[0] = innerSize;
    out_shape_0[1] = shape_x[axis];
    out_shape_0[2] = outerSize;
    *out_rank_0 = 3;

    out_shape_1[0] = 1;
    out_shape_1[1] = 1;
    *out_rank_1 = 2;

    innerSize = 1;
    outerSize = 1;
    for ( i = 0; i < axis; i++)
    {
        innerSize *= shape_output[i];
    }
    for ( i = axis + 1; i < rank_output; i++)
    {
        outerSize *= shape_output[i];
    }
    out_shape_output[0] = innerSize;
    out_shape_output[1] = shape_output[axis];
    out_shape_output[2] = outerSize;
    *out_rank_output = 3;

    return TRUE;
}
static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_size_t shape[3][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t rank[3] = {0};

    _get_stackconcat_shape(
        (vsi_ssize_t*)inputs[0]->attr.size, inputs[0]->attr.dim_num,
        (vsi_ssize_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num,
        self->nn_param.tensorstackconcat.axis,
        (vsi_ssize_t*)shape[0], &rank[0], (vsi_ssize_t*)shape[1], &rank[1], (vsi_ssize_t*)shape[2], &rank[2] );

    reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
        inputs[0], shape[0], rank[0] );
    reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
        inputs[1], shape[1], rank[1] );
    reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
        outputs[0], shape[2], rank[2] );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "tensorstackconcat",
        &reshape_tensors[0], 2,
        &reshape_tensors[2], 1, NULL );

    vsi_nn_ReleaseTensor( &reshape_tensors[0] );
    vsi_nn_ReleaseTensor( &reshape_tensors[1] );
    vsi_nn_ReleaseTensor( &reshape_tensors[2] );

    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    return status;
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;
    vsi_nn_tensorstackconcat_param *p = NULL;
    int32_t axis = 0;

    VSI_UNREFERENCED(outputs);

    if ( NULL == self )
    {
        return ret;
    }

    p = &(self->nn_param.tensorstackconcat);
    axis = p->axis;

    if (axis < 0)
    {
        axis = axis + inputs[0]->attr.dim_num;
        p->axis = axis;
    }

    return TRUE;
} /* op_setup() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_tensorstackconcat_param *p = NULL;
    int32_t axis = 0;
    int32_t dims = (int32_t)inputs[0]->attr.dim_num;
    int32_t out_dims = (int32_t)outputs[0]->attr.dim_num;

    p = &(self->nn_param.tensorstackconcat);
    axis = p->axis;

    if (axis < 0)
    {
        axis = axis + dims;
    }

    if (axis > (dims - 1))
    {
        VSILOGE("Invalid Axis: %d, (TENSORSTACKCONCAT) at [%s : %d]\n", axis, __FILE__, __LINE__);
        return FALSE;
    }
    if ( VSI_NN_DIM_AUTO == out_dims )
    {
        VSILOGE("Invalid output, (TENSORSTACKCONCAT) at [%s : %d]\n", __FILE__, __LINE__);
        return FALSE;
    }
    if ( dims != out_dims )
    {
        VSILOGE("Input and output's dims not matched, (TENSORSTACKCONCAT) at [%s : %d]\n", __FILE__, __LINE__);
        return FALSE;
    }

    {
        BEGIN_IO_TYPE_DECL(TENSORSTACKCONCAT, 2, 1)
            IO_TYPE(D_F16,       D_I32, D_F16)
            IO_TYPE(D_BF16,      D_I32, D_BF16)
            IO_TYPE(D_U8|Q_ASYM, D_I32, D_U8|Q_ASYM)
            IO_TYPE(D_I16|Q_DFP, D_I32, D_I16|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,  D_I32, D_I8|Q_DFP)
        END_IO_TYPE_DECL(TENSORSTACKCONCAT)
        if (!VALIDATE_OP_IO_TYPES(TENSORSTACKCONCAT, self, inputs, self->input.num, outputs, self->output.num))
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

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.tensorstackconcat.axis = 1;

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ TENSORSTACKCONCAT,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
