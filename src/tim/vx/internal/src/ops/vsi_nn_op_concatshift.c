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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_constraint_check.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vx_node cp_node = NULL;

    /* call CONCAT's op_compute */
    status = vsi_nn_OpCompute( VSI_NN_OP_CONCAT, self, inputs, outputs );

    if( VSI_SUCCESS == status )
    {
        cp_node = vxTensorCopyNode(self->graph->g,
                self->nn_param.concatshift.lcl_data->src_tensor,
                outputs[1]->t );

        if( NULL != cp_node )
        {
            self->nn_param.concatshift.lcl_data->cp_node = cp_node;
        }
        else
        {
            VSILOGE( "Create vxTensorCopyNode fail." );
            status = VSI_FAILURE;
        }
    }

    return status;
} /* op_compute() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_FAILURE;

    /* call CONCAT's op_deinit */
    status = vsi_nn_OpDeinit( VSI_NN_OP_CONCAT, self );

    if( NULL != self->nn_param.concatshift.lcl_data )
    {
        vxReleaseNode( &self->nn_param.concatshift.lcl_data->cp_node );
        vxReleaseTensor( &self->nn_param.concatshift.lcl_data->src_tensor );
        free( self->nn_param.concatshift.lcl_data );
        self->nn_param.concatshift.lcl_data = NULL;
    }

    return status;
} /* op_deinit() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;

    /* call CONCAT's op_check */
    ret = vsi_nn_OpCheck( VSI_NN_OP_CONCAT, self, inputs, outputs );

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;
    vsi_nn_concatshift_lcl_data * data = NULL;

    /* call CONCAT's op_setup */
    ret = vsi_nn_OpSetup( VSI_NN_OP_CONCAT, self, inputs, outputs );

    if( VSI_NN_DIM_AUTO == outputs[1]->attr.dim_num )
    {
        outputs[1]->attr.dim_num = outputs[0]->attr.dim_num;
        memcpy( &outputs[1]->attr.size, &outputs[0]->attr.size, sizeof(outputs[0]->attr.size) );

        outputs[1]->attr.size[self->nn_param.concatshift.axis] = self->nn_param.concatshift.keep_size;
    }

    data = ( vsi_nn_concatshift_lcl_data *)malloc(sizeof(vsi_nn_concatshift_lcl_data) );
    if( NULL != data )
    {
        memset( data, 0x00, sizeof(vsi_nn_concatshift_lcl_data) );
        self->nn_param.concatshift.lcl_data = data;
    }
    else
    {
        ret = VSI_FAILURE;
    }

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
    vsi_status      status = VSI_SUCCESS;
    uint32_t        axis;
    vx_tensor       out_view_tensor;
    vsi_size_t        start[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t        end[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t        i = 0;
    uint32_t        keep_size = 0;

    VSILOGD("Optimize %s", vsi_nn_OpGetName(self->op));
    vsi_nn_OpOptimize(VSI_NN_OP_CONCAT, self, inputs, outputs, direction);

    if(direction == VSI_NN_OPTIMIZE_BACKWARD)
    {
        return VSI_SUCCESS;
    }

    if( NULL == outputs[0]->t )
    {
        vsi_nn_TensorReinit( self->graph, outputs[0] );
    }
    if( NULL == outputs[1]->t )
    {
        vsi_nn_TensorReinit( self->graph, outputs[1] );
    }

    axis = self->nn_param.concatshift.axis;
    keep_size = self->nn_param.concatshift.keep_size;
    for( i = 0; i < outputs[0]->attr.dim_num; i++ )
    {
        if( i == axis )
        {
            start[i] = outputs[0]->attr.size[i] - keep_size;
        }
        else
        {
            start[i] = 0;
        }

        end[i] = outputs[0]->attr.size[i];
    }

    out_view_tensor = vsi_nn_CreateViewTensor(self->graph, start, end, outputs[0]);
    if( out_view_tensor != NULL )
    {
        self->nn_param.concatshift.lcl_data->src_tensor = out_view_tensor;
    }
    else
    {
        VSILOGE( "Create tensor %d from view fail.", i );
        status = VSI_FAILURE;
    }

    return status;
} /* op_optimize() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CONCATSHIFT,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 16,
    /* output_num */ 2
    );
#ifdef __cplusplus
}
#endif

