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
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_util.h"


static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_internal_compute_node( self );
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

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_internal_deinit_node_wksp( self );
    return status;
} /* op_deinit() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_relu_keras_param * p;
    vsi_nn_internal_node_t* curr = NULL;
    float alpha = 0;
    float max_value = 0;
    float threshold = 0;
    uint32_t max_raw = 0;
    if( NULL == self )
    {
        return FALSE;
    }

    p = &(self->nn_param.relu_keras);
    alpha = p->alpha;
    max_value = p->max_value;
    threshold = p->threshold;

    max_raw = *(uint32_t*)&max_value;

    vsi_nn_internal_init_node_wksp(self);

    if (alpha == 0 && max_raw == VSI_NN_FLOAT32_INF && threshold == 0)
    {
        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RELU, 0, 0);
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];
    }
    else if (alpha == 1.0f && max_value == 1.0f && threshold == -1.0f)
    {
        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RELU1, 0, 0);
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];
    }
    else if (alpha == 0 && max_value == 6.0f && threshold == 0)
    {
        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RELU6, 0, 0);
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];
    }
    else if (alpha == 0.1 && max_value == VSI_NN_FLOAT32_INF && threshold == 0)
    {
        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_LEAKY_RELU, 0, 0);
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];
    }
    else
    {
        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RELU_KERAS_INTERNAL, 0, 0);
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];
        curr->node->nn_param.relu_keras_internal.max_value = max_value;
        curr->node->nn_param.relu_keras_internal.alpha = alpha;
        curr->node->nn_param.relu_keras_internal.threshold = threshold;
    }

    vsi_nn_internal_setup_node(self, curr);

    return TRUE;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RELU_KERAS,
    /* init       */ NULL,
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
