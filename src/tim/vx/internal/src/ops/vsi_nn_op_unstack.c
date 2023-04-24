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
#include "libnnext/vsi_nn_vxkernel.h"
#include "vsi_nn_internal_node.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (VSI_NN_UNSTACK_MAX_OUTPUTS)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
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
    return vsi_nn_internal_optimize_node( self, direction );
} /* op_optimize() */

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
    vsi_nn_unstack_param * p;
    vsi_nn_tensor_attr_t attr;
    int32_t use_virtual_tensor = 1;
    uint32_t tensor_num = self->output.num;
    vsi_nn_internal_tensor_t* input_tensor = NULL;
    vsi_nn_internal_tensor_t** output_tensors = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_size_t* reshape_input_size = NULL;
    uint32_t *slices = NULL;
    vsi_size_t block_size = 1;
    vsi_size_t block_num = 1;
    uint32_t axis = 0;
    uint32_t i = 0, j = 0;
    uint32_t rank = inputs[0]->attr.dim_num;
    int8_t is_scalar = (rank - 1) == 0 ? TRUE : FALSE;

    vsi_nn_internal_init_node_wksp( self );

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        p = (vsi_nn_unstack_param *)&(self->nn_param.unstack);

        if (p->axis == 0)
        {
            for (j = 0; j < self->output.num; j++)
            {
                for (i = 0; i < inputs[0]->attr.dim_num - 1; i++)
                {
                    outputs[j]->attr.size[i] = inputs[0]->attr.size[i + 1];
                }
                outputs[j]->attr.size[0] = is_scalar ? 1 : outputs[j]->attr.size[0];
            }

            for (j = 0; j < self->output.num; j++)
            {
                outputs[j]->attr.dim_num = is_scalar ? 1 : (rank - 1);
                vsi_nn_SetTensorIsScalar(outputs[j], is_scalar);
            }
        }
        else if (p->axis == 1)
        {
            for (j = 0; j < self->output.num; j++)
            {
                outputs[j]->attr.size[0] = inputs[0]->attr.size[0];

                for (i = 1; i < inputs[0]->attr.dim_num-1; i++)
                {
                    outputs[j]->attr.size[i] = inputs[0]->attr.size[i + 1];
                }
                outputs[j]->attr.dim_num = inputs[0]->attr.dim_num - 1;
            }
        }
        else if (p->axis == 2)
        {
            for (j = 0; j < self->output.num; j++)
            {
                outputs[j]->attr.size[0] = inputs[0]->attr.size[0];
                outputs[j]->attr.size[1] = inputs[0]->attr.size[1];

                for (i = 2; i < inputs[0]->attr.dim_num - 1; i++)
                {
                    outputs[j]->attr.size[i] = inputs[0]->attr.size[i + 1];
                }
                outputs[j]->attr.dim_num = inputs[0]->attr.dim_num - 1;
            }
        }
        else if (p->axis == 3)
        {
            for (j = 0; j < self->output.num; j++)
            {
                outputs[j]->attr.size[0] = inputs[0]->attr.size[0];
                outputs[j]->attr.size[1] = inputs[0]->attr.size[1];
                outputs[j]->attr.size[2] = inputs[0]->attr.size[2];
                outputs[j]->attr.dim_num = inputs[0]->attr.dim_num - 1;
            }
        }
    }

    axis = self->nn_param.unstack.axis;

    for (i = 0; i < axis; i++)
    {
        block_size *= inputs[0]->attr.size[i];
    }

    for (i = axis + 1; i < inputs[0]->attr.dim_num; i++)
    {
        block_num *= inputs[0]->attr.size[i];
    }

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_init_tensor_attr(&attr, &inputs[0]->attr.dtype, use_virtual_tensor);
    input_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE2, 0, 0 );
    reshape_input_size = (vsi_size_t*)vsi_nn_internal_new_node_param(curr,
        VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    reshape_input_size[0] = block_size;
    reshape_input_size[1] = tensor_num;
    reshape_input_size[2] = block_num;

    curr->node->nn_param.reshape2.size = reshape_input_size;
    curr->node->nn_param.reshape2.dim_num = 3;
    curr->inputs[0] = inputs[0];
    curr->outputs[0] = input_tensor->t;
    vsi_nn_internal_setup_node( self, curr );

    slices = (uint32_t *)vsi_nn_internal_new_node_param(curr,
        tensor_num * sizeof(uint32_t));
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_SPLIT, 1, tensor_num );
    curr->node->nn_param.split.axis = 1;
    curr->node->nn_param.split.slices = slices;
    curr->node->nn_param.split.slices_num = tensor_num;
    curr->inputs[0] = input_tensor->t;
    output_tensors = (vsi_nn_internal_tensor_t**)malloc(tensor_num * sizeof(vsi_nn_internal_tensor_t*));
    for (i = 0; i < tensor_num; i++)
    {
        slices[i] = 1;
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        vsi_nn_internal_init_tensor_attr(&attr, &outputs[i]->attr.dtype, use_virtual_tensor);
        output_tensors[i] = vsi_nn_internal_new_tensor( self, &attr, 0.0f );
        curr->outputs[i] = output_tensors[i]->t;
    }
    vsi_nn_internal_setup_node( self, curr );

    for (i = 0; i < tensor_num; i++)
    {
        vsi_size_t* output_size = NULL;

        output_size = (vsi_size_t *)vsi_nn_internal_new_node_param(curr,
            VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));

        memcpy(output_size, outputs[i]->attr.size, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));

        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE2, 0, 0 );
        curr->node->nn_param.reshape2.size = output_size;
        curr->node->nn_param.reshape2.dim_num = outputs[i]->attr.dim_num;
        curr->inputs[0] = output_tensors[i]->t;
        curr->outputs[0] = outputs[i];
        vsi_nn_internal_setup_node( self, curr );
    }

    vsi_nn_safe_free(output_tensors);

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
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
    /* op_name    */ UNSTACK,
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
