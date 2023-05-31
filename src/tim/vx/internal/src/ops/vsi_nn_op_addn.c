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
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_constraint_check.h"
#include "vsi_nn_kernel_prv.h"

static int32_t _get_input_num
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** inputs
    )
{
    int32_t num;
    num = (int32_t)(self->input.num - 1);
    while( num >= 0 && NULL == inputs[num] )
    {
        num --;
    }
    if( 0 > num )
    {
        return -1;
    }

    num++;
    return num;
}

vsi_bool _is_float32_data_format
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t input_num = 0;
    uint32_t i = 0;

    input_num = _get_input_num(self, inputs);

    if (outputs[0]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT32)
    {
        return FALSE;
    }

    for ( i = 0; i < input_num; i++)
    {
        if (inputs[i]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT32)
        {
            return FALSE;
        }
    }

    return TRUE;
}

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
    return TRUE;
} /* op_check() */

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

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    uint32_t i;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* temp_output_tensor = NULL;
    vsi_bool is_sp_supported = vx_false_e;
    uint32_t input_num = 0;

    vsi_nn_internal_init_node_wksp( self );

    input_num = _get_input_num(self, inputs);

    is_sp_supported = vsi_nn_is_sp_supported_broadcast(self->graph, inputs, input_num, outputs[0]);

    for(i = 0; i < input_num -1; i++)
    {
        /* loop call add for input_num -1 times */

        /* setup input for each add */
        curr = vsi_nn_internal_new_node( self, VSI_NN_OP_ADD, 0, 0 );
        if(i == 0)
        {
            curr->inputs[0] = inputs[i];
        }
        else
        {
            curr->inputs[0] = temp_output_tensor->t;
        }
        curr->inputs[1] = inputs[i+1];

        /* setup output for each add */
        if (i < input_num - 2)
        {
            memset(&attr, 0, sizeof(attr));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = TRUE;
            attr.is_const = FALSE;
            if (VSI_NN_TYPE_INT32 == outputs[0]->attr.dtype.vx_type)
            {
                attr.dtype.vx_type = VSI_NN_TYPE_INT32;
            }
            else if ( _is_float32_data_format( self, inputs, outputs ) ||
                      is_sp_supported )
            {
                attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
            }
            else
            {
                attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
            }

            temp_output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

            curr->outputs[0] = temp_output_tensor->t;
        }
        else
        {
            curr->outputs[0] = outputs[0];
        }

        vsi_nn_internal_setup_node( self, curr );
    }
    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_internal_deinit_node_wksp( self );

    return status;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ ADDN,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
