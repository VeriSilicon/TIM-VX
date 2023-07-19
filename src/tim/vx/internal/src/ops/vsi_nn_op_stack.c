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
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_link_list.h"
#include "utils/vsi_nn_dtype_util.h"
#include "vsi_nn_error.h"

#define _ARG_NUM            (1)
#define _INPUT_NUM          VSI_NN_STACK_MAX_INPUTS
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (2)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)


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
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_stack_param * p;
    uint32_t i, j;
    vsi_size_t block_size = 1;
    vsi_size_t block_num = 1;
    uint32_t axis;
    vsi_size_t input_shape[2] = {1, 1};
    vsi_size_t output_shape[2] = {1, 1};
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_t *output_rs = NULL;
    vsi_nn_stack_lcl_data * data = NULL;
    vsi_bool ret = FALSE;
    vx_int8 is_scalar = vsi_nn_GetTensorIsScalar(inputs[0]);

    vsi_nn_internal_init_node_wksp( node );

    p = (vsi_nn_stack_param *)&(node->nn_param.stack);
    axis = p->axis;

    for (i = 0; i < axis; i++)
    {
        block_size *= inputs[0]->attr.size[i];
    }

    for (i = axis; i < inputs[0]->attr.dim_num; i++)
    {
        block_num *= inputs[0]->attr.size[i];
    }

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = is_scalar ? 1 : inputs[0]->attr.dim_num + 1;

        for (i = 0, j = 0; j < outputs[0]->attr.dim_num; j++)
        {
            if (j == p->axis)
            {
                outputs[0]->attr.size[j] = node->input.num;
            }
            else
            {
                outputs[0]->attr.size[j] = inputs[0]->attr.size[i ++];
            }
        }
    }

    if (1 == node->input.num)
    {
        curr = vsi_nn_internal_new_node( node, VSI_NN_OP_RESHAPE2, 1, 1);
        CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];
        curr->node->nn_param.reshape2.dim_num = outputs[0]->attr.dim_num;
        curr->node->nn_param.reshape2.size = outputs[0]->attr.size;
        ret = vsi_nn_internal_setup_node(node, curr);
        goto final;
    }

    input_shape[0] = block_size;
    input_shape[1] = block_num;

    curr = vsi_nn_internal_new_node( node, VSI_NN_OP_CONCAT, node->input.num, node->output.num );
    CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
    for (i = 0; i < node->input.num; i++)
    {
        vsi_nn_tensor_t *input_rs = NULL;
        /* Malloc ptr */
        data = (vsi_nn_stack_lcl_data *)malloc( sizeof(vsi_nn_stack_lcl_data) );
        CHECK_PTR_FAIL_GOTO_RLS_INTERNAL_NODE(data, curr, "Create buffer failed", final);
        memset( data, 0, sizeof(vsi_nn_stack_lcl_data) );

        input_rs = vsi_nn_reshape_tensor(node->graph, inputs[i], input_shape, 2);
        data->src_in     = input_rs;
    /* Store node, ptr */
        vsi_nn_LinkListPushStart(
            (vsi_nn_link_list_t **)&node->nn_param.stack.lcl_data,
            (vsi_nn_link_list_t *)data );

        curr->inputs[i] = input_rs;
    }

    if (block_num == 1)
    {
        output_shape[0] = block_size;
        output_shape[1] = node->input.num;
        axis = 1;
    }
    else
    {
        output_shape[0] = block_size * node->input.num;
        output_shape[1] = block_num;
        axis = 0;
    }

    /* Malloc ptr */
    data = (vsi_nn_stack_lcl_data *)malloc( sizeof(vsi_nn_stack_lcl_data) );
    CHECK_PTR_FAIL_GOTO_RLS_INTERNAL_NODE(data, curr, "Create buffer failed", final);
    memset( data, 0, sizeof(vsi_nn_stack_lcl_data) );

    output_rs = vsi_nn_reshape_tensor(node->graph, outputs[0], output_shape, 2);
    if (output_rs == NULL)
    {
        vsi_nn_internal_release_node(&curr);
        VSILOGD("Create reshape tensor failed\n");
        vsi_nn_safe_free(data);
        goto final;
    }
    data->src_in = output_rs;
    /* Store node, ptr */
    vsi_nn_LinkListPushStart(
        (vsi_nn_link_list_t **)&node->nn_param.stack.lcl_data,
        (vsi_nn_link_list_t *)data );

    curr->outputs[0] = output_rs;
    curr->node->nn_param.concat.axis = axis;
    ret = vsi_nn_internal_setup_node(node, curr);

final:
    return ret;
} /* op_setup() */

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
} /* op_optimize() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_stack_lcl_data * data;
    vsi_nn_stack_lcl_data * tmp;

    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    data = self->nn_param.stack.lcl_data;
    while( NULL != data )
    {
        tmp = (vsi_nn_stack_lcl_data *)vsi_nn_LinkListPopStart(
            (vsi_nn_link_list_t **)&data );
        vsi_nn_ReleaseTensor(&tmp->src_in);
        free( tmp );
    }

    vsi_nn_internal_deinit_node_wksp( self );

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
}
#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ STACK,
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
