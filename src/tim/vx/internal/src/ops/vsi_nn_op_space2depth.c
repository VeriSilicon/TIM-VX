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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_error.h"
#include "vsi_nn_test.h"
#include "utils/vsi_nn_constraint_check.h"

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
    if (self->nn_param.space2depth.block_size[0] == self->nn_param.space2depth.block_size[1])
    {
        vx_nn_reorg_params_t param;
        vsi_nn_tensor_t *block_size_tensor = NULL;
        vsi_nn_tensor_attr_t attr;
        memset(&param, 0, sizeof(vx_nn_reorg_params_t));

        memset(&attr, 0, sizeof(attr));
        attr.size[0] = 2;
        attr.size[1] = 1;
        attr.dim_num = 2;
        attr.is_const = TRUE;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        block_size_tensor = vsi_nn_CreateTensorFromData(
            self->graph,
            (uint8_t *)self->nn_param.space2depth.block_size,
            &attr);
        TEST_CHECK_PTR(block_size_tensor, final);

        param.block_size = REQUIRED_IO(block_size_tensor);
        param.type = VX_REORG_SPACE_TO_DEPTH;

        self->n = vxReorgLayer2( self->graph->g,
            inputs[0]->t,
            &param,
            sizeof(vx_nn_reorg_params_t),
            outputs[0]->t);

        if ( NULL != self->n )
        {
            status = VSI_SUCCESS;
        }
final:
        if (block_size_tensor) vsi_nn_ReleaseTensor(&block_size_tensor);
    }
    else
    {
        status = vsi_nn_internal_compute_node( self );
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
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    if (self->nn_param.space2depth.block_size[0] != self->nn_param.space2depth.block_size[1])
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

    if (self->nn_param.space2depth.block_size[0] < 0
        || self->nn_param.space2depth.block_size[1] < 0)
    {
        VSILOGE("Block size can't be less than zero in space to depth");
        return FALSE;
    }

    ret = vsi_nn_OpCheck(VSI_NN_OP_STRIDED_SLICE, self, inputs, outputs);

    return ret;
} /* op_check() */

static vsi_bool op_set_space2depth_internal
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_op_t  type_name
    )
{
    vsi_bool retn = FALSE;
    vsi_nn_internal_node_t* curr = NULL;

    vsi_nn_internal_init_node_wksp( self );

    curr = vsi_nn_internal_new_node( self, type_name, 0, 0 );
    CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
    curr->node->nn_param.space2depth_internal.block_size_x =
                        self->nn_param.space2depth.block_size[0];
    curr->node->nn_param.space2depth_internal.block_size_y =
                        self->nn_param.space2depth.block_size[1];
    curr->inputs[0]  = inputs[0];
    curr->outputs[0] = outputs[0];
    retn = vsi_nn_internal_setup_node(self, curr);

final:
    return retn;
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    uint32_t size_x = node->nn_param.space2depth.block_size[0];
    uint32_t size_y = node->nn_param.space2depth.block_size[1];
    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = inputs[0]->attr.size[0] / size_x;
        outputs[0]->attr.size[1] = inputs[0]->attr.size[1] / size_y;
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2] * size_x * size_y;
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    }

    if (size_x != size_y)
    {
        ret = op_set_space2depth_internal(node, inputs, outputs, VSI_NN_OP_SPACE2DEPTH_INTERNAL);
    }

    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.space2depth.block_size[0] != self->nn_param.space2depth.block_size[1])
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
    /* op_name    */ SPACE2DEPTH,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
