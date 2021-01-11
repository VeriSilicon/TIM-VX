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
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_internal_node.h"

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
    return vsi_nn_internal_compute_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    uint32_t i = 0;

    if ( self->nn_param.squeeze.axis_num == 0 )
    {
        for ( i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            if (inputs[0]->attr.size[i] != 1)
            {
                VSILOGE("the size of rank %d must be reported if squeezing a dimension that is not 1",
                        i);
                ret = FALSE;
            }
        }
    }
    else
    {
        for ( i = 0; i < self->nn_param.squeeze.axis_num; i++)
        {
            int32_t rank = self->nn_param.squeeze.axis[i];
            if (inputs[0]->attr.size[rank] != 1)
            {
                VSILOGE("the size of rank %d must be reported if squeezing a dimension that is not 1",
                        rank);
                ret = FALSE;
            }
        }
    }

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* TODO: Add code to comput outputs' shape. */
    uint32_t i = 0;
    uint32_t outIdx = 0;
    vsi_bool shouldSqueeze[VSI_NN_MAX_DIM_NUM] = {FALSE};
    uint32_t numDimsSqueezed = 0;
    vsi_nn_internal_node_t* curr = NULL;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        memset(shouldSqueeze, 0, sizeof(vsi_bool) * VSI_NN_MAX_DIM_NUM);

        if ( self->nn_param.squeeze.axis_num == 0 )
        {
            outputs[0]->attr.size[0] = 1;
            outputs[0]->attr.dim_num = 1;
        }
        else
        {
            for ( i = 0; i < self->nn_param.squeeze.axis_num; i++)
            {
                int32_t rank = self->nn_param.squeeze.axis[i];

                rank = rank < 0 ? rank + inputs[0]->attr.dim_num : rank;

                if ( !shouldSqueeze[rank] )
                {
                    ++numDimsSqueezed;
                }
                shouldSqueeze[rank] = TRUE;
            }

            for ( i = 0; i < inputs[0]->attr.dim_num; i++)
            {
                if (!shouldSqueeze[i])
                {
                    outputs[0]->attr.size[outIdx ++] = inputs[0]->attr.size[i];
                }
            }

            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num - numDimsSqueezed;
        }
    }

    vsi_nn_internal_init_node_wksp( self );
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE, 0, 0 );
    curr->node->nn_param.reshape.size = outputs[0]->attr.size;
    curr->node->nn_param.reshape.dim_num = outputs[0]->attr.dim_num;
    curr->inputs[0] = inputs[0];
    curr->outputs[0] = outputs[0];
    vsi_nn_internal_setup_node( self, curr );

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_internal_deinit_node_wksp( self );

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_internal_optimize_node( self, direction );
}

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SQUEEZE,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

