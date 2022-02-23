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
#include "utils/vsi_nn_util.h"

vsi_nn_node_t * vsi_nn_NewNode
    (
    vsi_nn_graph_t * graph,
    vsi_nn_op_t      op,
    vsi_size_t         input_num,
    vsi_size_t         output_num
    )
{
    vsi_nn_node_t * node;

    node = NULL;
    if(NULL == graph || FALSE == vsi_nn_OpIsValid(op))
    {
        VSILOGE("Create node %s. fail", vsi_nn_OpGetName(op));
        return NULL;
    }

    node = (vsi_nn_node_t *)malloc( sizeof( vsi_nn_node_t ) );
    if( NULL != node )
    {
        memset( node, 0, sizeof( vsi_nn_node_t ) );
        node->graph = graph;
        node->op = op;
        node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
        node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
        node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

        /* init op */
        vsi_nn_OpInit( node->op, node );

        if( 0 == input_num && 0 == output_num )
            {
            vsi_nn_OpGetIoNum( op, node, &input_num, &output_num );
            }

        /* init output struct */
        node->output.num = (uint32_t)output_num;
        node->output.tensors = (vsi_nn_tensor_id_t *) malloc(
            output_num * sizeof( vsi_nn_tensor_id_t ) );
        vsi_nn_InitTensorsId( node->output.tensors, (uint32_t)output_num );

        /* init input struct */
        node->input.num = (uint32_t)input_num;
        node->input.tensors = (vsi_nn_tensor_id_t *) malloc(
            input_num * sizeof( vsi_nn_tensor_id_t ) );
        vsi_nn_InitTensorsId( node->input.tensors, (uint32_t)input_num );
        node->attr.const_tensor_preload_type = VSI_NN_NODE_PRELOAD_NONE;
        node->attr.enable_op_constraint_check = TRUE;
    }

    node->uid = VSI_NN_NODE_UID_NA;
    return node;
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
    vsi_nn_node_t * ptr;
    ptr = (NULL != node) ? *node : NULL;
    if( NULL != ptr)
    {
        vsi_nn_OpDeinit( ptr->op, ptr );
        if( NULL != ptr->input.tensors )
        {
            free( ptr->input.tensors );
        }
        if( NULL != ptr->output.tensors )
        {
            free( ptr->output.tensors );
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

    if( NULL == node )
    {
        return;
    }
    count = snprintf( &buf[0], _MAX_PRINT_BUF_SZ, "%s", "[in:" );
    for( i = 0; i < node->input.num; i ++ )
    {
        if( count >= _MAX_PRINT_BUF_SZ )
        {
            break;
        }
        count += snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
            " %d,", node->input.tensors[i] );
    }
    count --;
    count += snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
        "%s", " ], [out:" );
    for( i = 0; i < node->output.num; i ++ )
    {
        if( count >= _MAX_PRINT_BUF_SZ )
        {
            break;
        }
        count += snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
            " %d,", node->output.tensors[i] );
    }
    count --;
    count += snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
        "%s", " ]" );
    VSILOGI( "(%16s)node[%u] %s [%08x]", vsi_nn_OpGetName(node->op), id, buf, node->n );
} /* vsi_nn_PrintNode() */

vsi_status vsi_nn_update_node_attr
    (
    vsi_nn_node_t *node
    )
{
    vsi_status status = VSI_FAILURE;

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
