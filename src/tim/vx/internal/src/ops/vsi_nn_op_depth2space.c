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
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_math.h"

static vsi_status vsi_nn_depth2space_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_tensor_t *block_size_tensor = NULL;
    vx_nn_reorg_params_t param;

    status = VSI_FAILURE;
    memset(&param, 0, sizeof(vx_nn_reorg_params_t));

    block_size_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.depth2space.block_size,
        VSI_NN_TYPE_INT32);
    if( NULL == block_size_tensor )
    {
        VSILOGE("Create block_size_tensor fail.(depth2space)");
        return VSI_FAILURE;
    }
    self->nn_param.depth2space.local.block_size_tensor = block_size_tensor;
    param.block_size = REQUIRED_IO(block_size_tensor);
    param.type = VX_REORG_DEPTH_TO_SPACE;

    self->n = vxReorgLayer2( self->graph->g,
        inputs[0]->t,
        &param,
        sizeof(vx_nn_reorg_params_t),
        outputs[0]->t);
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
    return status;
} /* vsi_nn_depth2space_compute() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;

    if (self->nn_param.depth2space.mode == VSI_NN_DEPTH2SPACE_DCR)
    {
        status = vsi_nn_depth2space_compute(self, inputs, outputs);
    }
    else if (self->nn_param.depth2space.mode == VSI_NN_DEPTH2SPACE_CRD)
    {
        status = vsi_nn_internal_compute_node( self );
    }
    else
    {
        VSILOGE("Unknown depth2space mode.(depth2space)");
        return status;
    }

    return status;
} /* op_compute() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    if (self->nn_param.depth2space.mode == VSI_NN_DEPTH2SPACE_CRD)
    {
        return vsi_nn_internal_optimize_node(self, direction );
    }
    else
    {
        return VSI_SUCCESS;
    }
} /* op_optimize() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;

    if(self->nn_param.depth2space.block_size < 0)
    {
        VSILOGE("Block size can't be less than zero in depth to space");
        return FALSE;
    }

    ret = vsi_nn_OpCheck(VSI_NN_OP_STRIDED_SLICE, self, inputs, outputs);

    return ret;
} /* op_check() */

static void op_set_depth2space_param_value(vsi_nn_nn_param_t *nn_param,
                                    vsi_nn_op_t  type_name,
                                    vsi_nn_depth2space_mode_e   mode,
                                    vx_uint32   block_size
                                    )
{
    if (type_name == VSI_NN_OP_DEPTH2SPACE_INTERNAL)
    {
        nn_param->depth2space_internal.block_size = block_size;
        nn_param->depth2space_internal.mode = mode;
    }
}

static vsi_bool op_set_depth2space_internal
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_op_t  type_name
    )
{
    vsi_bool retn = TRUE;
    vsi_nn_internal_node_t* curr = NULL;

    vsi_nn_internal_init_node_wksp( self );

    curr = vsi_nn_internal_new_node( self, type_name, 0, 0 );
    op_set_depth2space_param_value(&(curr->node->nn_param), type_name,
        self->nn_param.depth2space.mode, self->nn_param.depth2space.block_size);
    curr->inputs[0]  = inputs[0];
    curr->outputs[0] = outputs[0];
    retn = vsi_nn_internal_setup_node(self, curr);

    return retn;
}

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    if (vsi_nn_compareVersion(self->graph, 1, 1, 22) == -1)
    {
        self->nn_param.depth2space.mode = VSI_NN_DEPTH2SPACE_DCR;
    }

    return status;
} /* op_init() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    uint32_t size = node->nn_param.depth2space.block_size;
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = inputs[0]->attr.size[0] * size;
        outputs[0]->attr.size[1] = inputs[0]->attr.size[1] * size;
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2] / (size * size);
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    }

    if (node->nn_param.depth2space.mode == VSI_NN_DEPTH2SPACE_CRD)
    {
        ret = op_set_depth2space_internal(node, inputs, outputs, VSI_NN_OP_DEPTH2SPACE_INTERNAL);
    }
    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.depth2space.local.block_size_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.depth2space.local.block_size_tensor));
    }

    if (self->nn_param.depth2space.mode == VSI_NN_DEPTH2SPACE_CRD)
    {
        vsi_nn_internal_deinit_node_wksp(self);
    }
    else
    {
        vsi_nn_op_common_deinit(self);
    }

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ DEPTH2SPACE,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
