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

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    int i;
    for( i = 0; i < 10; i ++ )
    {
        if( NULL == outputs[i] )
        {
            break;
        }
        if( NULL != outputs[i]->t )
        {
            continue;
        }
        outputs[i]->t = inputs[0]->t;
    }
    return VSI_SUCCESS;
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

static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    int i;
    for( i = 0; i < 10; i ++ )
    {
        if( NULL == outputs[i] )
        {
            break;
        }
        if( outputs[i]->attr.vtl != inputs[0]->attr.vtl )
        {
            VSILOGW( "The tensor virtual attr changed in %#x op.", node->op );
        }
        if( outputs[i]->attr.is_const != inputs[0]->attr.is_const )
        {
            VSILOGW( "The tensor const attr changed in %#x op.", node->op );
        }
        if( VSI_NN_DIM_AUTO == outputs[i]->attr.dim_num )
        {
            if( NULL != outputs[i]->t )
            {
                if( NULL == inputs[0]->t )
                {
                    memcpy( inputs[0], outputs[i], sizeof( vsi_nn_tensor_t ) );
                }
                else
                {
                    VSILOGE( "Invalid NOOP tensors." );
                    vxReleaseTensor( &outputs[i]->t );
                    memcpy( outputs[i], inputs[0], sizeof( vsi_nn_tensor_t ) );
                }
            }
        }
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    self->n = NULL;
    vsi_nn_InitTensorsId( self->input.tensors, self->input.num );
    vsi_nn_InitTensorsId( self->output.tensors, self->output.num );
    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ NOOP,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 10
    );
#ifdef __cplusplus
}
#endif

