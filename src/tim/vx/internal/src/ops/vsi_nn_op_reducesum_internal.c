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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t rank_in = 0;
    uint32_t rank_out = 0;
    int32_t new_axis[VSI_NN_MAX_DIM_NUM];
    uint32_t axis_size = 0;
    vsi_bool ret;

    ret = vsi_nn_kernel_optimize_reduce_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            (int32_t *)(self->nn_param.reducesum_internal.axis),
            self->nn_param.reducesum_internal.axis_num,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], &rank_in, shapes[1], &rank_out,
            new_axis, &axis_size);

    if( ret )
    {
        self->nn_param.reducesum_internal.local->reshaped_input =
                vsi_nn_reshape_tensor( self->graph,
                inputs[0], shapes[0], rank_in );
        self->nn_param.reducesum_internal.local->reshaped_output =
                vsi_nn_reshape_tensor( self->graph,
                outputs[0], shapes[1], rank_out );

        self->n = vxTensorReduceSumNode( self->graph->g,
                   self->nn_param.reducesum_internal.local->reshaped_input->t,
                   self->nn_param.reducesum_internal.local->reshaped_output->t,
                   (uint32_t *)new_axis, axis_size, FALSE);

    }

    if( NULL != self->n )
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
    /* TODO: Add code to comput outputs' shape. */
    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.reducesum_internal.local   =
    (vsi_nn_reducesum_lcl_data_t *)malloc(sizeof(vsi_nn_reducesum_lcl_data_t));

    if (NULL == self->nn_param.reducesum_internal.local)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(self->nn_param.reducesum_internal.local, 0, sizeof(vsi_nn_reducesum_lcl_data_t));
    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.reducesum_internal.local != NULL)
    {
        if (self->nn_param.reducesum_internal.local->reshaped_input != NULL)
        {
            vsi_nn_ReleaseTensor(&(self->nn_param.reducesum_internal.local->reshaped_input));
        }
        if (self->nn_param.reducesum_internal.local->reshaped_output != NULL)
        {
            vsi_nn_ReleaseTensor(&(self->nn_param.reducesum_internal.local->reshaped_output));
        }

        free(self->nn_param.reducesum_internal.local);
        self->nn_param.reducesum_internal.local = NULL;
    }

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ REDUCESUM_INTERNAL,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
