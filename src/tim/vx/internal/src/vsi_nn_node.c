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
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_types.h"
#include "vsi_nn_types_prv.h"
#include "utils/vsi_nn_util.h"

vsi_nn_node_t * vsi_nn_NewNode
    (
    vsi_nn_graph_t * graph,
    vsi_nn_op_t      op,
    vsi_size_t         input_num,
    vsi_size_t         output_num
    )
{
    vsi_nn_node_prv_t* node;

    node = NULL;
    if(NULL == graph || FALSE == vsi_nn_OpIsValid(op))
    {
        VSILOGE("Create node %s. fail", vsi_nn_OpGetName(op));
        goto final;
    }

    node = (vsi_nn_node_prv_t *)malloc( sizeof( vsi_nn_node_prv_t ) );
    if( NULL != node )
    {
        memset( node, 0, sizeof( vsi_nn_node_prv_t ) );
        node->pon.graph = graph;
        node->pon.op = op;
        node->pon.vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
        node->pon.vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
        node->pon.vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

        /* init op */
        vsi_nn_OpInit( node->pon.op, &node->pon );

        if( 0 == input_num && 0 == output_num )
            {
            vsi_nn_OpGetIoNum( op, &node->pon, &input_num, &output_num );
            }

        /* init output struct */
        node->pon.output.num = (uint32_t)output_num;
        node->pon.output.tensors = (vsi_nn_tensor_id_t *) malloc(
            output_num * sizeof( vsi_nn_tensor_id_t ) );
        if (NULL == node->pon.output.tensors)
        {
            goto final;
        }
        vsi_nn_InitTensorsId( node->pon.output.tensors, (uint32_t)output_num );

        /* init input struct */
        node->pon.input.num = (uint32_t)input_num;
        node->pon.input.tensors = (vsi_nn_tensor_id_t *) malloc(
            input_num * sizeof( vsi_nn_tensor_id_t ) );
        if (NULL == node->pon.input.tensors)
        {
            goto final;
        }
        vsi_nn_InitTensorsId( node->pon.input.tensors, (uint32_t)input_num );
        node->pon.attr.const_tensor_preload_type = VSI_NN_NODE_PRELOAD_NONE;
        node->pon.attr.enable_op_constraint_check = TRUE;
    }
    else
    {
        goto final;
    }

    node->pon.uid = VSI_NN_NODE_UID_NA;

    return (vsi_nn_node_t*)node;
final:
    if (node)
    {
        vsi_nn_safe_free(node->pon.output.tensors);
        vsi_nn_safe_free(node->pon.input.tensors);
    }
    vsi_nn_safe_free(node);

    return NULL;
} /* vsi_nn_NewNode() */

/*
* Deprecated: Use vsi_nn_NewNode() instead
*/
vsi_nn_node_t * vsi_nn_CreateNode
    (
    vsi_nn_graph_t * graph,
    vsi_nn_op_t      op
    )
{
    return vsi_nn_NewNode( graph, op, 0, 0 );
} /* vsi_nn_CreateNode() */

void vsi_nn_ReleaseNode
    (
    vsi_nn_node_t ** node
    )
{
    vsi_nn_node_prv_t* ptr;
    ptr = (NULL != node) ? (vsi_nn_node_prv_t*)*node : NULL;
    if( NULL != ptr)
    {
        vsi_nn_OpDeinit( ptr->pon.op, &ptr->pon );
        if( NULL != ptr->pon.input.tensors )
        {
            free( ptr->pon.input.tensors );
        }
        if( NULL != ptr->pon.output.tensors )
        {
            free( ptr->pon.output.tensors );
        }
        free( ptr );
        *node = NULL;
    }
} /* vsi_nn_ReleaseNode() */

vsi_status vsi_nn_SetNodeInputsAndOutputs
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t * const inputs[],
    int input_num,
    vsi_nn_tensor_t * const outputs[],
    int output_num
    )
{
    vsi_status status = VSI_SUCCESS;
    int i;
    vsi_nn_tensor_id_t id;
    if( !node )
    {
        return VSI_FAILURE;
    }
    if( inputs && input_num > 0 )
    {
        assert(input_num <= (int)node->input.num);
        for(i = 0; i < input_num; i ++)
        {
            id = vsi_nn_get_tensor_id(node->graph, inputs[i]);
            node->input.tensors[i] = id;
        }
    }
    if( outputs && output_num > 0 )
    {
        assert(output_num <= (int)node->output.num);
        for(i = 0; i < output_num; i ++)
        {
            id = vsi_nn_get_tensor_id(node->graph, outputs[i]);
            node->output.tensors[i] = id;
        }
    }
    return status;
} /* vsi_nn_SetNodeInputsAndOutputs() */

void vsi_nn_PrintNode
    (
    vsi_nn_node_t * node,
    vsi_nn_node_id_t id
    )
{
#define _MAX_PRINT_BUF_SZ   (1024)
    uint32_t i;
    int count;
    char buf[_MAX_PRINT_BUF_SZ];
    vsi_bool is_out_of_bound = FALSE;
    int temp = 0;

    if( NULL == node )
    {
        return;
    }
    count = snprintf( &buf[0], _MAX_PRINT_BUF_SZ, "%s", "[in:" );
    for( i = 0; i < node->input.num; i ++ )
    {
        temp = snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
            " %d,", node->input.tensors[i] );
        if ( temp >= _MAX_PRINT_BUF_SZ - count || temp == -1 )
        {
            is_out_of_bound = TRUE;
            goto final;
        }
        count += temp;
    }
    count --;
    temp = snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
        "%s", " ], [out:" );
    if ( temp >= _MAX_PRINT_BUF_SZ - count || temp == -1 )
    {
            is_out_of_bound = TRUE;
            goto final;
    }
    count += temp;
    for( i = 0; i < node->output.num; i ++ )
    {
        /* -3 means reserve memory for ending symbols --" ]" */
        temp = snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count - 3,
            " %d,", node->output.tensors[i] );
        if ( temp >= _MAX_PRINT_BUF_SZ - count - 3 || temp == -1 )
        {
            is_out_of_bound = TRUE;
            goto final;
        }
        count += temp;
    }
    count --;
    count += snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
        "%s", " ]" );
final:
    if ( is_out_of_bound )
    {
        VSILOGW("Buffer is already full, cannot print all messages for (%16s)node[%u] [%08x]",
            vsi_nn_OpGetName(node->op), id, node->n );
    }
    VSILOGI( "(%16s)node[%u] %s [%08x]", vsi_nn_OpGetName(node->op), id, buf, node->n );
} /* vsi_nn_PrintNode() */

#if VX_GRAPH_BATCH_OPT_SUPPORT
vsi_status vsi_nn_SetNodeBatchSplitNum
(
    vsi_nn_node_t* node,
    int8_t split_num
)
{
    vsi_status status = VSI_SUCCESS;
    if (node == NULL || split_num < 1)
    {
        status = VSI_FAILURE;
        goto final;
    }
    ((vsi_nn_node_prv_t*)node)->split_num = split_num;

    final:
    return status;
}
#endif

vsi_status vsi_nn_update_node_attr
    (
    vsi_nn_node_t *node
    )
{
    vsi_status status = VSI_FAILURE;

    VSI_UNREFERENCED(node);

#if(defined(VX_PRELOAD_CONST_TENSOR_SUPPORT) && VX_PRELOAD_CONST_TENSOR_SUPPORT)
    if(node)
    {
        /* some node don't have a `n`, skip it */
        status = VSI_SUCCESS;
        if(node->n)
        {
            vx_enum preload_type = VX_PRELOAD_NULL;
            switch(node->attr.const_tensor_preload_type)
            {
                default:
                case VSI_NN_NODE_PRELOAD_NONE:
                    preload_type = VX_PRELOAD_NULL;
                    break;

                case VSI_NN_NODE_PRELOAD_VIPSRAM:
                    preload_type = VX_PRELOAD_CONST_TENSOR_VIPSRAM;
                    break;

                case VSI_NN_NODE_PRELOAD_AXISRAM:
                    preload_type = VX_PRELOAD_CONST_TENSOR_AXISRAM;
                    break;
            }
            status = vxSetNodeAttribute(node->n, VX_NODE_ATTRIBUTE_CONST_TENSOR_CACHE,
                &preload_type, sizeof(preload_type));
        }
    }
#else
    status = VSI_SUCCESS;
#endif

    return status;
} /* vsi_nn_update_node_attr() */
