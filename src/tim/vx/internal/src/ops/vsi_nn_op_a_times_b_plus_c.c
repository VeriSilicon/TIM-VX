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
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
        "a_times_b_plus_c",
        inputs, 3,
        outputs, 1, NULL );
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
    vsi_nn_tensor_t * a_times_b[2] = {NULL};
    vsi_nn_tensor_attr_t attr;
    vsi_bool ret = FALSE;

    memset(&attr, 0, sizeof(attr));
    memcpy(attr.size, outputs[0]->attr.size,  VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ));
    attr.dim_num = outputs[0]->attr.dim_num;
    attr.vtl = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    a_times_b[0] = vsi_nn_CreateTensor(self->graph, &attr);
    ret = vsi_nn_OpCheck(VSI_NN_OP_MULTIPLY, self, inputs, a_times_b);
    if (!ret)
    {
        goto final;
    }

    a_times_b[1] = inputs[2];
    ret = vsi_nn_OpCheck(VSI_NN_OP_ADD, self, a_times_b, outputs);
final:
    if (a_times_b[0]) vsi_nn_ReleaseTensor(&a_times_b[0]);
    return ret;

} /* op_check() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ A_TIMES_B_PLUS_C,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_eltwise_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
