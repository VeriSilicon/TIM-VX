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
#include <stdlib.h>
#include <string.h>

#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_types.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_rnn.h"
#include "vsi_nn_test.h"
#include "vsi_nn_internal_node.h"
#include "vsi_nn_version.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_vdata.h"
#include "utils/vsi_nn_map.h"
#include "vsi_nn_graph_optimization.h"

static vsi_status _set_reference_name
    (
    vsi_nn_graph_t *graph,
    vsi_nn_node_t *node
    )
{
#define _NODE_ID_LEN 64
    vsi_status status;
    vsi_nn_tensor_t *tensor;
    uint32_t i;
    char name[_NODE_ID_LEN];

    if(NULL == node || NULL == graph)
    {
        return VSI_FAILURE;
    }

    status = VSI_SUCCESS;
    memset(name, 0, sizeof(char) * _NODE_ID_LEN);
    snprintf(name, sizeof(char) * _NODE_ID_LEN, "uid_%u", node->uid);
    if(node && node->n)
    {
        status = vxSetReferenceName((vx_reference)node->n, name);
    }
    TEST_CHECK_STATUS(status, final);
    for(i = 0; i < node->output.num; i++)
    {
        memset(name, 0, sizeof(char) * _NODE_ID_LEN);
        snprintf(name, sizeof(char) * _NODE_ID_LEN, "uid_%u_out_%u", node->uid, i);
        tensor = vsi_nn_GetTensor(graph, node->output.tensors[i]);
        if(tensor && tensor->t)
        {
            status = vxSetReferenceName((vx_reference)tensor->t, name);
            TEST_CHECK_STATUS(status, final);
        }
    }

final:
    return status;
} /* _set_reference_name() */

static vsi_status _check_swapped_tensors
    (
    const vsi_nn_graph_t* graph
    )
{
    uint32_t i = 0;
    vsi_status status = VSI_SUCCESS;

    VSILOGD("Check swapped tensors");
    for( i = 0; i < graph->node_num; i++ )
    {
        vsi_nn_node_t* node = vsi_nn_GetNode( graph, (vsi_nn_node_id_t)i );

        /* For NBG node, all inputs/outputs need to be set if tensors are swapped */
        if( node && VSI_NN_OP_NBG == node->op )
        {
            uint32_t idx, j;
            vsi_nn_tensor_t* tensor = NULL;

            idx = 0;
            for( j = 0; j < node->input.num; j++ )
            {
                tensor = vsi_nn_GetTensor( graph, node->input.tensors[j] );
                if( tensor && tensor->is_swapped )
                {
                    status = vxSetParameterByIndex( node->n, idx, (vx_reference)tensor->t );
                    if( VSI_SUCCESS != status )
                    {
                        VSILOGE( "Set input parameter %d for node[%08x] fail!", idx, node->n );
                        goto final;
                    }
                    tensor->is_swapped = FALSE;
                }
                idx++;
            }

            for( j = 0; j < node->output.num; j++ )
            {
                tensor = vsi_nn_GetTensor( graph, node->output.tensors[j] );
                if( tensor && tensor->is_swapped )
                {
                    status = vxSetParameterByIndex( node->n, idx, (vx_reference)tensor->t );
                    if( VSI_SUCCESS != status )
                    {
                        VSILOGE( "Set output parameter %d for node[%08x] fail!", idx, node->n );
                        goto final;
                    }
                    tensor->is_swapped = FALSE;
                }
                idx++;
            }
        }
    }

final:
    return status;
} /* _check_swapped_tensors() */

static void free_io_buffer
    (
    vsi_nn_tensor_t **buffer
    )
{
    if(buffer)
    {
        free(buffer);
        buffer = NULL;
    }
} /* free_io_buffer() */

static vsi_nn_tensor_t **allocate_io_buffer
    (
    vsi_nn_graph_t * graph
    )
{
    vsi_nn_tensor_t **buffer;

    buffer = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *) * graph->max_node_io);
    if(NULL == buffer)
    {
        return NULL;
    }

    return buffer;
} /* allocate_io_buffer() */

static vsi_status update_max_node_io
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    uint32_t i,max_io;
    vsi_status status;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node;

    status = VSI_SUCCESS;
    max_io = VSI_NN_MAX_IO_NUM; /* default max node io */
    for(i = 0; i < graph->node_num; i++)
    {
        node_id = node_list[i];
        node = vsi_nn_GetNode( graph, node_id );
        if(node->input.num > max_io)
        {
            max_io = node->input.num;
        }
        if(node->output.num > max_io)
        {
            max_io = node->output.num;
        }
    }

    graph->max_node_io = max_io;
    return status;
} /* update_max_node_io() */

static vsi_status optimize_node_backward
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    int32_t i;
    vsi_status status;
    vsi_nn_tensor_t **inputs;
    vsi_nn_tensor_t **outputs;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node;

    status = VSI_SUCCESS;
    inputs = allocate_io_buffer(graph);
    outputs = allocate_io_buffer(graph);
    if(NULL == inputs || NULL == outputs)
    {
        VSILOGE("allocate io buffer fail");
        status =  VSI_FAILURE;
        goto final;
    }

    for( i = graph->node_num - 1; i >= 0; i-- )
    {
        node_id = node_list[i];
        memset( inputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );
        memset( outputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );

        /* Get inputs, outputs. */
        node = vsi_nn_GetNode( graph, node_id );
        vsi_nn_GetTensors( graph, node->input.tensors,
            node->input.num, inputs );
        vsi_nn_GetTensors( graph, node->output.tensors,
            node->output.num, outputs );

        status = vsi_nn_OpOptimize(node->op, node, inputs, outputs, VSI_NN_OPTIMIZE_BACKWARD);
        if( status != VSI_SUCCESS )
        {
            VSILOGE( "Backward optimize node[%u] %s fail",
                node_id, vsi_nn_OpGetName(node->op));
            break;
        }
    }

final:
    free_io_buffer(inputs);
    free_io_buffer(outputs);
    return status;
} /* optimize_node_backward() */

static vsi_status optimize_node_forward
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    uint32_t i;
    vsi_status status;
    vsi_nn_tensor_t **inputs;
    vsi_nn_tensor_t **outputs;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node;

    status = VSI_SUCCESS;
    inputs = allocate_io_buffer(graph);
    outputs = allocate_io_buffer(graph);
    if(NULL == inputs || NULL == outputs)
    {
        VSILOGE("allocate io buffer fail");
        status =  VSI_FAILURE;
        goto final;
    }

    for( i = 0; i < graph->node_num; i++ )
    {
        node_id = node_list[i];
        memset( inputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );
        memset( outputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );

        /* Get inputs, outputs. */
        node = vsi_nn_GetNode( graph, node_id );
        vsi_nn_GetTensors( graph, node->input.tensors,
            node->input.num, inputs );
        vsi_nn_GetTensors( graph, node->output.tensors,
            node->output.num, outputs );

        status = vsi_nn_OpOptimize(node->op, node, inputs, outputs, VSI_NN_OPTIMIZE_FORWARD);
        if( status != VSI_SUCCESS )
        {
            VSILOGE( "Forward optimize node[%u] %s fail",
                node_id, vsi_nn_OpGetName(node->op));
            break;
        }
    }

final:
    free_io_buffer(inputs);
    free_io_buffer(outputs);
    return status;
} /* optimize_node_forward() */

static vsi_status compute_node
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    uint32_t i,j;
    vsi_status status;
    vsi_nn_tensor_t **inputs;
    vsi_nn_tensor_t **outputs;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node;

    status = VSI_SUCCESS;
    inputs = allocate_io_buffer(graph);
    outputs = allocate_io_buffer(graph);
    if(NULL == inputs || NULL == outputs)
    {
        VSILOGE("allocate io buffer fail");
        status =  VSI_FAILURE;
        goto final;
    }

    VSILOGI("Create vx node");
    for( i = 0; i < graph->node_num; i++ )
    {
        node_id = node_list[i];
        memset( inputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );
        memset( outputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );

        /* Get inputs, outputs. */
        node = vsi_nn_GetNode( graph, node_id );
        vsi_nn_GetTensors( graph, node->input.tensors,
            node->input.num, inputs );
        vsi_nn_GetTensors( graph, node->output.tensors,
            node->output.num, outputs );

        /* Create vx output tensor */
        for ( j = 0; j < node->output.num; j++ )
        {
            if( NULL == outputs[j] || NULL != outputs[j]->t )
                continue;
            vsi_nn_TensorReinit( graph, outputs[j] );
        }

        /* Create vx node */
        VSILOGD("Instance node[%d] \"%s\" ...", node_id, vsi_nn_OpGetName(node->op));
        status = vsi_nn_OpCompute( node->op, node, inputs, outputs );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Create node[%d] %s fail", node_id, vsi_nn_OpGetName(node->op));
            break;
        }
        status = _set_reference_name(graph, node);
        if( VSI_SUCCESS != status )
        {
            VSILOGW("Set reference name fail");
        }

        status = vsi_nn_update_node_attr(node);
        if( VSI_SUCCESS != status )
        {
            VSILOGW("Update node attribute fail");
        }
    }

final:
    free_io_buffer(inputs);
    free_io_buffer(outputs);
    return status;
} /* compute_node */

static vsi_status optimize_node
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    vsi_status status;

    status = VSI_FAILURE;
    VSILOGD("Backward optimize neural network");
    status = optimize_node_backward(graph, node_list);
    if(status != VSI_SUCCESS)
    {
        return VSI_FAILURE;
    }

    VSILOGD("Forward optimize neural network");
    status = optimize_node_forward(graph, node_list);
    if(status != VSI_SUCCESS)
    {
        return VSI_FAILURE;
    }

    return status;
} /* optimize_node() */

static vsi_status setup_node
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    uint32_t i;
    vsi_status status;
    vsi_bool ret;
    vsi_nn_tensor_t **inputs;
    vsi_nn_tensor_t **outputs;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node;

    status = VSI_SUCCESS;
    ret = TRUE;
    inputs = allocate_io_buffer(graph);
    outputs = allocate_io_buffer(graph);
    if(NULL == inputs || NULL == outputs)
    {
        VSILOGE("allocate io buffer fail");
        status =  VSI_FAILURE;
        goto final;
    }

    for( i = 0; i < graph->node_num; i++ )
    {
        node_id = node_list[i];
        memset( inputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );
        memset( outputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );

        /* Get inputs, outputs. */
        node = vsi_nn_GetNode( graph, node_id );
        vsi_nn_GetTensors( graph, node->input.tensors,
            node->input.num, inputs );
        vsi_nn_GetTensors( graph, node->output.tensors,
            node->output.num, outputs );

        VSILOGD("Setup node id[%u] uid[%u] op[%s]",
            node_id, node->uid, vsi_nn_OpGetName(node->op));
        if( vsi_nn_OpCheck( node->op, node, inputs, outputs ) )
        {
            vsi_nn_print_node_io(graph, node, 0x01);
            ret = vsi_nn_OpGenerateTensor( node, inputs, outputs );
            if(ret != TRUE)
            {
                VSILOGE( "Setup node[%u] %s fail", node_id, vsi_nn_OpGetName(node->op));
                status = VSI_FAILURE;
                break;
            }
            vsi_nn_print_node_io(graph, node, 0x02);
        }
        else
        {
            VSILOGE( "Check node[%u] %s fail", node_id, vsi_nn_OpGetName(node->op));
            status = VSI_FAILURE;
            break;
        }
    }

final:
    free_io_buffer(inputs);
    free_io_buffer(outputs);
    return status;
} /* setup_node() */

vsi_nn_graph_t * vsi_nn_CreateGraph
    (
    vsi_nn_context_t ctx,
    uint32_t        max_tensor_num,
    uint32_t        max_node_num
    )
{
    vsi_nn_graph_t * graph;
    graph = NULL;

    VSILOGI( "%s", vsi_nn_GetVersion() );

    if( NULL == ctx )
    {
        return graph;
    }

    graph = (vsi_nn_graph_t *)malloc( sizeof( vsi_nn_graph_t ) );
    if( NULL != graph )
    {
        memset( graph, 0, sizeof( vsi_nn_graph_t ) );
        graph->g = vxCreateGraph( ctx->c );
        if( NULL != graph->g )
        {
            /* Configure driver mem aligned size,
             * driver requests address and tensor size are aligend to 64 bytes. */
            const uint32_t ADDRESS_ALIGN_BYTES = 64;
            graph->handle_manager.align_start_size = ADDRESS_ALIGN_BYTES;
            #ifdef VX_WRAP_USER_MEMORY_SIZE_ALIGNMENT
                graph->handle_manager.align_block_size = (VX_WRAP_USER_MEMORY_SIZE_ALIGNMENT);
            #else
            {
                const uint32_t MEMORY_BLOCK_ALIGN_BYTES = 4096;
                graph->handle_manager.align_block_size = MEMORY_BLOCK_ALIGN_BYTES;
            }
            #endif
            graph->tensor_num = 0;
            graph->node_num = 0;
            graph->ctx = ctx;
            graph->rnn_wksp = NULL;
            graph->node_table = (vsi_nn_map_t *)malloc( sizeof( vsi_nn_map_t ) );
            graph->tensor_table = (vsi_nn_map_t *)malloc( sizeof( vsi_nn_map_t ) );
            vsi_nn_MapInit( graph->node_table );
            vsi_nn_MapInit( graph->tensor_table );
        }
        else
        {
            VSILOGE( "Create vx graph fail." );
            free( graph );
            graph = NULL;
        }
    }

    return graph;
} /* vsi_nn_CreateGraph() */

void vsi_nn_ReleaseGraph
    (
    vsi_nn_graph_t ** graph
    )
{
    uint32_t i;
    vsi_nn_graph_t  * ptr;

    ptr = *graph;
    if( NULL != graph && NULL != * graph )
    {
        if( NULL != ptr->tensors )
        {
            for( i = 0; i < ptr->tensor_num; i++ )
            {
                vsi_nn_RemoveTensor( *graph, (vsi_nn_tensor_id_t)i );
            }
            free( (*graph)->tensor_table );
        }
        if( ptr->complete_signal.exists
            && NULL != ptr->complete_signal.tensor )
        {
            vsi_nn_ReleaseTensor( &ptr->complete_signal.tensor );
        }
        if( NULL != ptr->nodes )
        {
            for( i = 0; i < ptr->node_num; i++ )
            {
                vsi_nn_RemoveNode( *graph, (vsi_nn_node_id_t)i );
            }
            free( (*graph)->node_table );
        }
        if( NULL != ptr->input.tensors )
        {
            free( ptr->input.tensors );
        }
        if( NULL != ptr->output.tensors )
        {
            free( ptr->output.tensors );
        }
        if( NULL != ptr->rnn_wksp )
        {
            vsi_nn_rnn_DeinitWksp( ptr );
        }
        if( NULL != ptr->g )
        {
            vxReleaseGraph( &ptr->g );
        }
        free( ptr );
        *graph = NULL;
    }

} /* vsi_nn_ReleaseGraph() */

/*
* Create vx tensor and nodes.
* */
vsi_status vsi_nn_SetupGraph
    (
    vsi_nn_graph_t * graph,
    vsi_bool          sort
    )
{
    uint32_t i;
    vsi_status status;
    vsi_nn_node_id_t *sorted_nodes;
    vsi_nn_node_id_t *nodes_list;
    vsi_bool dirty = FALSE;

    status = VSI_FAILURE;
    sorted_nodes = NULL;
    nodes_list = NULL;
    if( NULL == graph )
    {
        return status;
    }

    /* Optimize graph */
    status = vsi_nn_OptimizeGraph(graph, &dirty);
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    /* Prepare node list */
    nodes_list = (vsi_nn_node_id_t *)malloc(
        graph->node_num * sizeof( vsi_nn_node_id_t ) );
    if( !nodes_list )
    {
        goto final;
    }
    if( TRUE == sort || dirty)
    {
        VSILOGD( "Sort graph nodes.");
        sorted_nodes = vsi_nn_SortGraphNode( graph );
        if (NULL == sorted_nodes)
        {
            VSILOGW("Sort graph nodes failure.");
            free(nodes_list);
            nodes_list = NULL;
            return status;
        }
        memcpy(nodes_list, sorted_nodes,
            graph->node_num * sizeof( vsi_nn_node_id_t ));
    }
    else
    {
        for ( i = 0; i < graph->node_num; i++ )
        {
            nodes_list[i] = i;
        }
    }

    status = update_max_node_io( graph, nodes_list );
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    /* Preprocess node and tensor */
    status = setup_node( graph, nodes_list );
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    /* Optimize graph */
    status = optimize_node( graph, nodes_list );
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    /* Create vx node and vx virtual tensor */
    status = compute_node( graph, nodes_list );
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    /* Try setup graph complete signal node. */
    status = vsi_nn_TrySetupCompleteSignalNode( graph );
    TEST_CHECK_STATUS( status, final );

    /* Setup binary graph inputs and outputs. */
    status = vsi_nn_setup_binary_graph_inputs_outputs( graph );
    TEST_CHECK_STATUS( status, final );

final:
    if( NULL != sorted_nodes )
    {
        free( sorted_nodes );
    }
    if ( NULL != nodes_list )
    {
        free( nodes_list );
    }
    return status;
} /* vsi_nn_SetupGraph() */

/*
* Call vx verify graph.
* */
vsi_status vsi_nn_VerifyGraph
    (
    vsi_nn_graph_t * graph
    )
{
    vsi_status status;
    status = VSI_FAILURE;
    if( NULL != graph->g )
    {
        status = vxVerifyGraph( graph->g );
    }
    return status;
} /* vsi_nn_VerifyGraph() */

vsi_status vsi_nn_RunGraph
    (
    const vsi_nn_graph_t * graph
    )
{
    vsi_status status;
    status = VSI_FAILURE;
    if( NULL != graph->g )
    {
        if( vsi_nn_HasRNN( graph ) )
        {
            status = vsi_nn_rnn_feed_internal_state( graph );
        }
        else
        {
            status = VSI_SUCCESS;
        }

        if( VSI_SUCCESS == status )
        {
            status = _check_swapped_tensors( graph );
        }

        if( VSI_SUCCESS == status )
        {
            status = vxProcessGraph( graph->g );
        }

        if( VSI_SUCCESS == status && vsi_nn_HasRNN( graph ) )
        {
            status = vsi_nn_rnn_save_internal_state( graph );
        }
    }
    return status;
} /* vsi_nn_RunGraph() */

vsi_status vsi_nn_GenerateNBG(
    vsi_nn_graph_t * graph,
    void * nbg_buffer,
    size_t * size
    )
{
    return (VX_SUCCESS == vxGenerateNBG( graph->g, nbg_buffer, size ))? VSI_SUCCESS : VSI_FAILURE;
} /* vsi_nn_GenerateNBG() */

vsi_status vsi_nn_AsyncRunGraph
    (
    vsi_nn_graph_t * graph
    )
{
    vsi_status status;
    status = VSI_FAILURE;
    if( NULL != graph->g )
    {
        if( vsi_nn_HasRNN( graph ) )
        {
            status = vsi_nn_rnn_feed_internal_state( graph );
        }
        else
        {
            status = VSI_SUCCESS;
        }

        if( VSI_SUCCESS == status )
        {
            status = _check_swapped_tensors( graph );
        }

        if( VSI_SUCCESS == status )
        {
            status = vxScheduleGraph(graph->g);
        }
    }
    return status;
} /* vsi_nn_AsynRunGraph() */


vsi_status vsi_nn_AsyncRunWait
    (
        vsi_nn_graph_t * graph
    )
{
    vsi_status status;
    status = VSI_FAILURE;
    if( NULL != graph->g )
    {
        status = vxWaitGraph(graph->g);
        if( VSI_SUCCESS == status && vsi_nn_HasRNN( graph ) )
        {
            status = vsi_nn_rnn_save_internal_state( graph );
        }
    }
    return status;
}


vsi_status vsi_nn_SetGraphVersion
    (
    vsi_nn_graph_t * graph,
    uint32_t major,
    uint32_t minor,
    uint32_t patch
    )
{
    graph->version.major = major;
    graph->version.minor = minor;
    graph->version.patch = patch;
    return VSI_SUCCESS;
} /* vsi_nn_SetGraphVersion() */

vsi_status vsi_nn_GetGraphVersion
    (
    vsi_nn_graph_t * graph,
    uint32_t * major,
    uint32_t * minor,
    uint32_t * patch
    )
{
    *major = graph->version.major;
    *minor = graph->version.minor;
    *patch = graph->version.patch;
    return VSI_SUCCESS;
} /* vsi_nn_GetGraphVersion() */

static vsi_nn_tensor_id_t _add_tensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id,
    vsi_nn_tensor_attr_t * attr,
    uint8_t             * data
    )
{
    vsi_nn_tensor_t * tensor;
    tensor = NULL;
    if( NULL == graph || NULL == attr )
    {
        return VSI_NN_TENSOR_ID_NA;
    }
    if( VSI_NN_TENSOR_ID_AUTO == id )
    {
        id = graph->cur_tid;
        graph->tensor_num = graph->cur_tid;
    }

    if (TRUE == attr->is_created_from_handle)
    {
        tensor = vsi_nn_CreateTensorFromHandle( graph, data, attr );
    }
    else if( VSI_NN_TYPE_VDATA == attr->dtype.vx_type )
    {
        if( NULL == data )
        {
            id = VSI_NN_TENSOR_ID_NA;
        }
        else
        {
            tensor = vsi_nn_CreateVDataTensor( graph, data, attr );
        }
    }
    else if( NULL != data )
    {
        tensor = vsi_nn_CreateTensorFromData( graph, data, attr );
    }
    else
    {
        tensor = vsi_nn_CreateTensor( graph, attr );
    }

    if( NULL != tensor )
    {
        vsi_nn_MapAdd( graph->tensor_table, (vsi_nn_map_key_t)id, (void *)tensor );
        graph->cur_tid ++;
    }
    else
    {
        id = VSI_NN_TENSOR_ID_NA;
    }
    return id;
}

vsi_nn_tensor_id_t vsi_nn_AddTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id,
    vsi_nn_tensor_attr_t * attr,
    uint8_t             * data
    )
{
    attr->is_created_from_handle = FALSE;
    return _add_tensor(graph, id, attr, data);
} /* vsi_nn_AddTensor() */

vsi_nn_tensor_id_t vsi_nn_AddTensorFromHandle
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id,
    vsi_nn_tensor_attr_t * attr,
    uint8_t             * data
    )
{
    attr->is_created_from_handle = TRUE;
    return _add_tensor(graph, id, attr, data);
}

vsi_nn_tensor_id_t vsi_nn_AttachTensorToGraph
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id,
    vsi_nn_tensor_t      * tensor
    )
{
    if( NULL == graph || NULL == tensor )
    {
        return VSI_NN_TENSOR_ID_NA;
    }
    if( VSI_NN_TENSOR_ID_AUTO == id )
    {
        id = graph->cur_tid;
        graph->tensor_num = graph->cur_tid;
    }
    graph->cur_tid ++;
    vsi_nn_MapAdd( graph->tensor_table, (vsi_nn_map_key_t)id, (void *)tensor );
    return id;
} /* vsi_nn_AttachTensorToGraph() */

/*
 * Deprecated, Use vsi_nn_RemoveTensor() instead
 */
void vsi_nn_DeleteTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id
    )
{
    vsi_nn_RemoveTensor( graph, id );
} /* vsi_nn_DeleteTensor() */

void vsi_nn_RemoveTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id
    )
{
    vsi_nn_tensor_t * tensor;
    if( NULL != graph )
    {
        tensor = vsi_nn_GetTensor( graph, id );
        if( NULL != tensor )
        {
            vsi_nn_ReleaseTensor( &tensor );
            vsi_nn_MapRemove( graph->tensor_table,
                    (vsi_nn_map_key_t)id );
        }
    }
} /* vsi_nn_RemoveTensor() */

vsi_nn_tensor_t * vsi_nn_GetTensor
    (
    const vsi_nn_graph_t   * graph,
    vsi_nn_tensor_id_t id
    )
{
    vsi_nn_tensor_t * tensor;
    tensor = NULL;
    if( NULL != graph )
    {
        tensor = vsi_nn_MapGet( graph->tensor_table, (vsi_nn_map_key_t)id );
    }
    return tensor;
} /* vsi_nn_GetTensor() */

vsi_nn_node_t * vsi_nn_GetNode
    (
    const vsi_nn_graph_t   * graph,
    vsi_nn_node_id_t   id
    )
{
    vsi_nn_node_t * node;
    node = NULL;
    if( NULL != graph )
    {
        node = vsi_nn_MapGet( graph->node_table, (vsi_nn_map_key_t)id );
    }
    return node;
} /* vsi_nn_GetTensor() */

void vsi_nn_GetTensors
    (
    vsi_nn_graph_t     * graph,
    vsi_nn_tensor_id_t * tensors_id,
    uint32_t            num,
    vsi_nn_tensor_t   ** tensors
    )
{
    uint32_t i;

    if( NULL == graph || NULL == graph->tensors
        || NULL == tensors_id || NULL == tensors)
    {
        return;
    }
    memset( &tensors[0], 0, sizeof( vsi_nn_tensor_t * ) * num  );
    if( num > graph->max_node_io )
    {
        VSILOGW( "Tensor num(%d) is greater than the MAX(%d), \
                 set to max num.", num, graph->max_node_io );
        num = graph->max_node_io;
    }
    for( i = 0; i < num; i++ )
    {
        if( VSI_NN_TENSOR_ID_NA == tensors_id[i] )
        {
            continue;
        }
        if( tensors_id[i] >= graph->tensor_num )
        {
            VSILOGE( "Tensor id %d/%d", tensors_id[i], graph->tensor_num );
            continue;
        }
        tensors[i] = vsi_nn_GetTensor( graph, tensors_id[i] );
    }
} /* vsi_nn_GetTensors() */

vsi_nn_node_t * vsi_nn_AddNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_op_t           op,
    uint32_t              input_num,
    uint32_t              output_num,
    vsi_nn_node_id_t    * node_id
    )
{
    vsi_nn_node_t * node;
    vsi_nn_node_id_t id;

    if( NULL == graph )
    {
        return NULL;
    }

    id = graph->cur_nid;
    node = vsi_nn_NewNode(graph, op, input_num, output_num);
    if( NULL != node )
    {
        vsi_nn_MapAdd( graph->node_table, (vsi_nn_map_key_t)id, (void *)node );
        graph->cur_nid ++;
        graph->node_num = graph->cur_nid;
    }
    else
    {
        id = VSI_NN_NODE_ID_NA;
    }

    if( NULL != node_id )
    {
        *node_id = id;
    }
    return node;
} /* vsi_nn_AddNode() */

/*
 * Deprecated, Use vsi_nn_AddNode instead
 */
vsi_nn_node_t * vsi_nn_AppendNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_op_t           op,
    vsi_nn_node_id_t    * node_id
    )
{
    return vsi_nn_AddNode( graph, op, 0, 0, node_id );
} /* vsi_nn_AppendNode() */

void vsi_nn_RemoveNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_node_id_t      id
    )
{
    vsi_nn_node_t * node;
    if( NULL != graph )
    {
        node = vsi_nn_GetNode( graph, id );
        if( NULL != node )
        {
            vsi_nn_ReleaseNode( &node );
            vsi_nn_MapRemove( graph->node_table,
                    (vsi_nn_map_key_t)id );
        }
    }
} /* vsi_nn_RemoveNode() */

vsi_bool vsi_nn_SetGraphInputs
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_tensor_id_t  * tensors_id,
    uint32_t             tensor_num
    )
{
    vsi_bool ret;
    ret = FALSE;

    if( NULL == graph || tensor_num == 0 )
    {
        return ret;
    }

    graph->input.tensors = (vsi_nn_tensor_id_t *)malloc(
        tensor_num * sizeof( vsi_nn_tensor_id_t ) );

    if( NULL != graph->input.tensors )
    {
        graph->input.num = tensor_num;
        ret = TRUE;
        if( NULL != tensors_id )
        {
            memcpy( graph->input.tensors, tensors_id,
                tensor_num * sizeof( vsi_nn_tensor_id_t ) );
        }
    }

    return ret;
} /* vsi_nn_SetGreaphInputs() */

vsi_bool vsi_nn_SetGraphOutputs
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_tensor_id_t  * tensors_id,
    uint32_t             tensor_num
    )
{
    vsi_bool ret;
    ret = FALSE;

    if( NULL == graph || tensor_num == 0 )
    {
        return ret;
    }

    graph->output.tensors = (vsi_nn_tensor_id_t *)malloc(
        tensor_num * sizeof( vsi_nn_tensor_id_t ) );
    if( NULL != graph->output.tensors )
    {
        graph->output.num = tensor_num;
        ret = TRUE;
        if( NULL != tensors_id )
        {
            memcpy( graph->output.tensors, tensors_id,
                tensor_num * sizeof( vsi_nn_tensor_id_t ) );
        }
    }

    return ret;

} /* vsi_nn_SetGraphOutputs() */

vsi_nn_node_id_t * vsi_nn_SortGraphNode
    (
    vsi_nn_graph_t * graph
    )
{
    uint32_t i,j;
    uint32_t             count;
    vsi_bool             dirty;
    vsi_bool             all_tensor_processed;
    vsi_bool           * tensors;
    vsi_nn_node_id_t   * nodes;
    vsi_nn_node_id_t   * sorted_nodes;
    vsi_nn_node_t      * node;
    vsi_nn_node_id_t     node_id;
    vsi_nn_tensor_id_t   tensor_id;
    vsi_nn_tensor_t    * tensor;

    if( NULL == graph || NULL == graph->nodes
        || NULL == graph->tensors )
    {
        return NULL;
    }

    tensors      = NULL;
    sorted_nodes = NULL;
    nodes        = NULL;
    node         = NULL;
    node_id      = VSI_NN_NODE_ID_NA;

    /* Init variables. */
    tensors = (vsi_bool *)malloc(
        graph->tensor_num * sizeof( vsi_bool ) );

    if( NULL == tensors )
    {
        goto _SortGraphNodeFinally;
    }

    sorted_nodes = (vsi_nn_node_id_t *)malloc(
        graph->node_num * sizeof( vsi_nn_node_id_t ) );
    nodes = (vsi_nn_node_id_t *)malloc(
        graph->node_num * sizeof( vsi_nn_node_id_t ) );

    if( NULL == sorted_nodes || NULL == nodes)
    {
        goto _SortGraphNodeFinally;
    }

    for( i = 0; i < graph->tensor_num; i++ )
    {
        tensor = vsi_nn_GetTensor( graph, (vsi_nn_tensor_id_t)i );
        if( NULL == tensor
        || TRUE == tensor->attr.is_const )
        {
            tensors[i] = TRUE;
        }
        else
        {
            tensors[i] = FALSE;
        }
    }

    for( i = 0; i < graph->input.num; i++ )
    {
        tensor_id = graph->input.tensors[i];
        if( tensor_id != VSI_NN_TENSOR_ID_NA )
        {
            tensors[tensor_id] = TRUE;
        }
    }

    for( i = 0; i < graph->node_num; i++ )
    {
        nodes[i] = i;
    }
    count = graph->node_num;
    do
    {
        dirty = FALSE;
        all_tensor_processed = FALSE;
        for( i = 0; i < count; i ++ )
        {
            node_id = nodes[i];
            node = vsi_nn_GetNode( graph, node_id );
            all_tensor_processed = TRUE;
            for( j = 0; j < node->input.num; j ++ )
            {
                tensor_id = node->input.tensors[j];
                if( VSI_NN_TENSOR_ID_NA == tensor_id )
                {
                    continue;
                }
                if( FALSE == tensors[tensor_id] )
                {
                    all_tensor_processed = FALSE;
                    break;
                }
            }
            if( TRUE == all_tensor_processed )
            {
                sorted_nodes[graph->node_num - count] = nodes[i];
                nodes[i] = nodes[count - 1];
                count --;
                i --;
                dirty = TRUE;
                for( j = 0; j < node->output.num; j ++ )
                {
                    tensor_id = node->output.tensors[j];
                    if( VSI_NN_TENSOR_ID_NA == tensor_id )
                    {
                        continue;
                    }
                    tensors[tensor_id] = TRUE;
                }
            }
        }
        if( FALSE == dirty )
        {
            if( FALSE == all_tensor_processed )
            {
                // TODO: Log all unprocessed tensors
                VSILOGW("Unprocessed node %u", node_id);
            }
            break;
        }
    } while( count > 0 );

    if( count != 0 )
    {
        free( sorted_nodes );
        sorted_nodes = NULL;
    }

_SortGraphNodeFinally:

    /* Release memory. */
    free( tensors );
    free( nodes );
    return sorted_nodes;
} /* vsi_nn_SortGraphNode() */

uint32_t vsi_nn_GetNodesByUids
    (
    vsi_nn_graph_t   * graph,
    uint32_t        * node_uids,
    uint32_t          node_uids_size,
    vsi_nn_node_id_t * nodes,
    uint32_t          nodes_num
    )
{
    uint32_t sz;
    uint32_t i;
    uint32_t j;
    vsi_nn_node_t * node;

    sz = 0;
    if( NULL == nodes || 0 >= nodes_num )
    {
        return sz;
    }
    if( NULL != node_uids )
    {
        for( i = 0; i < node_uids_size; i++ )
        {
            for( j = 0; j < graph->node_num; j++ )
            {
                node = vsi_nn_GetNode( graph, (vsi_nn_node_id_t)j );
                if( node_uids[i] == node->uid )
                {
                    nodes[sz] = (vsi_nn_node_id_t)j;
                    sz ++;
                    break;
                }
            }
        }
    }
    else
    {
        for( j = 0; j < graph->node_num; j++ )
        {
            nodes[j] = (vsi_nn_node_id_t)j;
        }
        sz = graph->node_num;
    }
    return sz;
} /* vsi_nn_GetNodesByUids() */

void vsi_nn_DumpGraphNodeOutputs
    (
    vsi_nn_graph_t * graph,
    const char     * path,
    uint32_t       * node_uids,
    uint32_t         node_uids_size,
    vsi_bool         force_fp32,
    vsi_nn_dim_fmt_e data_fmt
    )
{
    vsi_nn_DumpGraphNodeOutputsEx(graph, path, NULL, node_uids, node_uids_size, force_fp32, data_fmt );
} /* vsi_nn_DumpGraphNodeOutputs() */

void vsi_nn_DumpGraphNodeOutputsEx
    (
    vsi_nn_graph_t * graph,
    const char     * path,
    const char     * prefix,
    uint32_t       * node_uids,
    uint32_t         node_uids_size,
    vsi_bool         force_fp32,
    vsi_nn_dim_fmt_e data_fmt
    )
{
#define _MAX_TENSOR_NAME_SZ (1024)
#define _SHAPE_BUF_SZ   (64)
    char shape[_SHAPE_BUF_SZ] = { 0 };
    char filename[_MAX_TENSOR_NAME_SZ] = { 0 };
    char filename_prefix[_SHAPE_BUF_SZ] = { 0 };
    const char * op_name;
    uint32_t i;
    uint32_t o;
    uint32_t node_num;
    vsi_nn_node_id_t * nodes;
    vsi_nn_node_t    * node;
    vsi_nn_tensor_t  * tensor;

    if(vsi_nn_CheckFilePath(path) == FALSE)
    {
        return ;
    }

    if( NULL == node_uids )
    {
        node_num = graph->node_num;
    }
    else
    {
        if( node_uids_size <= 0 )
        {
            VSILOGE("Error node_uids_size: %d.", node_uids_size);
            return;
        }
        node_num = node_uids_size;
    }
    nodes = (vsi_nn_node_id_t *)malloc( node_num * sizeof( vsi_nn_node_id_t ) );
    if( NULL == nodes )
    {
        VSILOGE("Malloc nodes memory fail.");
        return;
    }
    node_num = vsi_nn_GetNodesByUids( graph, node_uids, node_uids_size,
        nodes, node_num );

    if( NULL != prefix )
    {
        strncpy(filename_prefix, prefix, _SHAPE_BUF_SZ);
        filename_prefix[_SHAPE_BUF_SZ - 1] = '\0';

        strncat(filename_prefix, "_", _SHAPE_BUF_SZ - 1);
        filename_prefix[_SHAPE_BUF_SZ - 1] = '\0';
    }

    VSILOGD("Dump %u nodes.", node_num);
    for( i = 0; i < node_num; i++ )
    {
        node = vsi_nn_GetNode( graph, (vsi_nn_node_id_t)i );

        if( node->internal_node_wksp ) /* dump internal nodes if any */
        {
            vsi_nn_internal_dump_node_output(graph, path, filename_prefix,
                force_fp32, node);
        }

        for( o = 0; o < node->output.num; o++ )
        {
            tensor = vsi_nn_GetTensor( graph, node->output.tensors[o] );
            if( NULL != tensor )
            {
                if( TRUE == tensor->attr.vtl )
                {
                    VSILOGW("Uid %u node's tensor %d is virtual",
                        node->uid, o);
                    continue;
                }
                // TODO: Support different tensor format
                vsi_nn_ShapeToString( tensor->attr.size, tensor->attr.dim_num,
                    shape, _SHAPE_BUF_SZ, FALSE );
                op_name = vsi_nn_OpGetName( node->op );
                snprintf( filename, _MAX_TENSOR_NAME_SZ,
                    "%s/%s%s_uid_%u_t_%u_s_%s.txt", path, filename_prefix, op_name, node->uid, o, shape);
                if( FALSE == force_fp32 )
                {
                    vsi_nn_SaveTensorToText( graph, tensor, filename, NULL );
                }
                else
                {
                    vsi_nn_SaveTensorToTextByFp32( graph, tensor, filename, NULL );
                }
            }
        }
    }
    free( nodes );
} /* vsi_nn_DumpGraphNodeOutputsEx */

void vsi_nn_PrintGraph
    (
    vsi_nn_graph_t * graph
    )
{
    vsi_nn_tensor_t * tensor;
    vsi_nn_node_t * node;
    uint32_t i;

    if( NULL == graph )
    {
        return;
    }

    VSILOGI( "Graph:" );
    VSILOGI( "***************** Tensors ******************" );
    for( i = 0; i < graph->tensor_num; i ++ )
    {
        tensor = vsi_nn_GetTensor( graph, (vsi_nn_tensor_id_t)i );
        if( NULL != tensor )
        {
            vsi_nn_PrintTensor( tensor, (vsi_nn_tensor_id_t)i );
        }
    }
    VSILOGI( "***************** Nodes ******************" );
    for( i = 0; i < graph->node_num; i ++ )
    {
        node = vsi_nn_GetNode( graph, (vsi_nn_node_id_t)i );
        if( NULL != node )
        {
            vsi_nn_PrintNode( node, (vsi_nn_node_id_t)i );
        }
    }
    VSILOGI("******************************************" );
} /* vsi_nn_PrintGraph() */

void vsi_nn_DumpGraphToJson
    (
    vsi_nn_graph_t *graph
    )
{
#define _SHAPE_BUF_SIZE 64
    uint32_t i,j;
    FILE *fp;
    vsi_nn_tensor_rel_t *tensor_ref, *tio;
    vsi_nn_tensor_rel_table_t *table;
    vsi_nn_node_t *node,*in_node;
    vsi_nn_tensor_t *tensor;
    char shape[_SHAPE_BUF_SIZE] = { 0 };

    if(NULL == graph)
    {
        return ;
    }

    fp = fopen("graph.json", "w+");
    if(NULL == fp)
    {
        VSILOGE("Create dump file fail");
        return ;
    }

    tensor_ref = vsi_nn_CreateTensorRelevance(graph);
    if(NULL == tensor_ref)
    {
        VSILOGE("build tensor io fail");
        fclose(fp);
        return ;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "\t\"Layers\":{\n");
    for(i = 0; i < graph->node_num; i++)
    {
        node = vsi_nn_GetNode(graph, i);
        if(node)
        {
            fprintf(fp, "\t\t\"uid_%u\":{\n\t\t\t\"op\": \"%s\",\n",
                node->uid, vsi_nn_OpGetName(node->op));

            /* dump inputs */
            fprintf(fp, "\t\t\t\"inputs\": [ ");
            for(j = 0; j < node->input.num; j++)
            {
                tio = &tensor_ref[node->input.tensors[j]];
                if(NULL == vsi_nn_GetTensor(graph, node->input.tensors[j]))
                {
                    if(j == node->input.num - 1)
                    {
                        fprintf(fp, "\"not used\" ");
                    }
                    else
                    {
                        fprintf(fp, "\"not used\", ");
                    }
                }
                else
                {
                    if(tio->input.num > 0)
                    {
                        table = tio->input.table;

                        /* tensor only 1 input node */
                        in_node = vsi_nn_GetNode(graph, table[0].node);
                        if(j == node->input.num - 1)
                        {
                            fprintf(fp, "\"@uid_%u:out%u\" ", in_node->uid, table[0].index);
                        }
                        else
                        {
                            fprintf(fp, "\"@uid_%u:out%u\", ", in_node->uid, table[0].index);
                        }
                    }
                    else
                    {
                        if(j == node->input.num - 1)
                        {
                            fprintf(fp, "\"datainput_%u:out0\" ", j);
                        }
                        else
                        {
                            fprintf(fp, "\"datainput_%u:out0\", ", j);
                        }
                    }
                }
            }

            /* dump input shape */
            fprintf(fp, "],\n\t\t\t\"inut_shape\": [ ");
            for(j = 0; j < node->input.num; j++)
            {
                tensor = vsi_nn_GetTensor(graph, node->input.tensors[j]);
                if(NULL != tensor && vsi_nn_ShapeToString( tensor->attr.size, tensor->attr.dim_num,
                    shape, _SHAPE_BUF_SIZE, TRUE ) > 0)
                {
                    fprintf(fp, "[%s ]", shape);
                }
                else
                {
                    fprintf(fp, "[]");
                }
                if(j < node->input.num - 1)
                {
                    fprintf(fp, ",");
                }
            }

            /* dump output */
            fprintf(fp, " ],\n\t\t\t\"outputs\": [ ");
            for(j = 0; j < node->output.num; j++)
            {
                if(j == node->output.num - 1)
                {
                    fprintf(fp, "\"out%u\" ", j);
                }
                else
                {
                    fprintf(fp, "\"out%u\", ", j);
                }
            }

            //output shape
            fprintf(fp, "],\n\t\t\t\"output_shape\": [ ");
            for(j = 0; j < node->output.num; j++)
            {
                tensor = vsi_nn_GetTensor(graph, node->output.tensors[j]);
                if(NULL != tensor && vsi_nn_ShapeToString( tensor->attr.size, tensor->attr.dim_num,
                    shape, _SHAPE_BUF_SIZE, TRUE ) > 0)
                {
                    fprintf(fp, "[%s ]", shape);
                }
                else
                {
                    fprintf(fp, "[]");
                }
                if(j < node->output.num - 1)
                {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, " ]\n\t\t}");

            if(i != graph->node_num - 1)
            {
                fprintf(fp, ",");
            }
            fprintf(fp, "\n");
        }
    }
    fprintf(fp, "\t}\n}\n");

    vsi_nn_ReleaseTensorRelevance(graph, tensor_ref);
    fclose(fp);
} /* vsi_nn_DumpGraphToJson() */

/*
 * Documented in vsi_nn_graph.h
 */
vsi_status vsi_nn_TrySetupCompleteSignalNode
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_tensor_t* signal_tensor = NULL;
    vsi_nn_node_t* signal_node = NULL;
    vsi_nn_tensor_attr_t signal_tensor_attr;
    vsi_status status = VSI_FAILURE;
    if( graph->complete_signal.exists )
    {
        if( !graph->complete_signal.write_address )
        {
            VSILOGW("COMPLETE signal is set with null write addres.");
            return VSI_FAILURE;
        }
        VSILOGD("Setup COMPLETE signal, value \"%d\", write address \"%p\"",
                graph->complete_signal.value, graph->complete_signal.write_address);
        /* Setup signal tensor attr */
        memset( &signal_tensor_attr, 0, sizeof(vsi_nn_tensor_attr_t) );
        signal_tensor_attr.size[0] = 8;
        signal_tensor_attr.size[1] = 1;
        signal_tensor_attr.dim_num = 2;
        signal_tensor_attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
        signal_tensor_attr.vtl = FALSE;
        /* Setup signal node */
        signal_node = vsi_nn_CreateNode( graph, VSI_NN_OP_EXTRA_ENDING );
        TEST_CHECK_PTR( signal_node, final );

        signal_node->nn_param.extra_ending.length = sizeof(int64_t);
        memcpy( &signal_node->nn_param.extra_ending.value,
                &graph->complete_signal.value, sizeof(int64_t));

        if( graph->output.num > 1 )
        {
            VSILOGE("Not support COMPLETE signal with multi graph outputs.");
        }
        else
        {
            tensor = vsi_nn_GetTensor( graph, graph->output.tensors[0] );
            signal_tensor = vsi_nn_CreateTensorFromHandle( graph,
                    (uint8_t*)graph->complete_signal.write_address,
                    &signal_tensor_attr);
            status = vsi_nn_OpCompute( signal_node->op, signal_node,
                    &tensor, &signal_tensor );
            TEST_CHECK_STATUS( status, final );
        }
        graph->complete_signal.tensor = signal_tensor;
        status = VSI_SUCCESS;
    }
    else
    {
        status = VSI_SUCCESS;
    }
final:
    if( signal_node )
    {
        vsi_nn_ReleaseNode( &signal_node );
    }
    return status;
} /* vsi_nn_TrySetupCompleteSignalNode() */


/*
 * Documented in vsi_nn_graph.h
 */
vsi_status vsi_nn_setup_binary_graph_inputs_outputs
    (
    vsi_nn_graph_t* graph
    )
{
    uint32_t i,j;
    vsi_status status;
    uint32_t num_of_graph_inputs;
    uint32_t num_of_graph_real_inputs;
    vx_reference *graph_inputs = NULL;
    uint32_t num_of_graph_outputs;
    uint32_t num_of_graph_real_outputs;
    vx_reference *graph_outputs = NULL;
    vsi_nn_tensor_t *tensor;

    num_of_graph_real_inputs = 0;
    num_of_graph_real_outputs = 0;

    /* Explicitly set graph inputs and outputs */
    num_of_graph_inputs = graph->input.num;
    for( i = 0; i < num_of_graph_inputs; i++ )
    {
        tensor = vsi_nn_GetTensor( graph, graph->input.tensors[i] );
        if (tensor)
        {
            num_of_graph_real_inputs += 1;
        }
        else
        {
            ;//do nothing
        }
    }
    graph_inputs = (vx_reference *)malloc( num_of_graph_real_inputs * sizeof( vx_reference ) );
    for( i = 0, j = 0; i < num_of_graph_inputs; i++ )
    {
        tensor = vsi_nn_GetTensor( graph, graph->input.tensors[i] );
        if (tensor)
        {
            if(j > num_of_graph_real_inputs -1)
            {
                status = VSI_FAILURE;
                goto final;
            }
            graph_inputs[j++] = (vx_reference)( tensor->t );
        }
        else
        {
            ;//do nothing
        }
    }
    num_of_graph_outputs = graph->output.num;
    if( graph->complete_signal.exists )
    {
        num_of_graph_outputs += 1;
    }
    for( i = 0; i < num_of_graph_outputs; i++ )
    {
        tensor = vsi_nn_GetTensor( graph, graph->output.tensors[i] );
        if (tensor)
        {
            num_of_graph_real_outputs += 1;
        }
        else
        {
            ;//do nothing
        }
    }
    graph_outputs = (vx_reference *)malloc( num_of_graph_real_outputs * sizeof( vx_reference ) );
    for( i = 0, j = 0; i < num_of_graph_outputs; i++ )
    {
        tensor = vsi_nn_GetTensor( graph, graph->output.tensors[i] );
        if (tensor)
        {
            if(j > num_of_graph_real_outputs -1)
            {
                status = VSI_FAILURE;
                goto final;
            }
            graph_outputs[j++] = (vx_reference)( tensor->t );
        }
        else
        {
            ;//do nothing
        }
    }
    if( graph->complete_signal.exists )
    {
        graph_outputs[num_of_graph_real_outputs - 1] = \
                (vx_reference)graph->complete_signal.tensor->t;
    }

    status = vxIdentifyGraphInputsAndOutputs( graph->g,
        num_of_graph_real_inputs,
        graph_inputs,
        num_of_graph_real_outputs,
        graph_outputs );

    if( VSI_SUCCESS != status )
    {
        goto final;
    }

final:
    if ( NULL != graph_inputs)
    {
        free( graph_inputs );
    }
    if ( NULL != graph_outputs)
    {
        free( graph_outputs );
    }
    return status;
} /* vsi_nn_setup_binary_graph_inputs_outputs() */

vsi_status vsi_nn_SetupRNNConnections
    (
    vsi_nn_graph_t* graph,
    const vsi_nn_rnn_external_connection_t* connections,
    uint32_t connections_count
    )
{
    return vsi_nn_rnn_InitWksp( graph, connections, connections_count, NULL );
} /* vsi_nn_SetupRNNConnections() */

vsi_status vsi_nn_ResetRNNBuffers
    (
    vsi_nn_graph_t* graph
    )
{
    return vsi_nn_rnn_ResetBuffers( graph );
} /* vsi_nn_ResetRNNBuffers() */

vsi_bool vsi_nn_HasRNN
    (
    const vsi_nn_graph_t* graph
    )
{
    return NULL != graph && NULL != graph->rnn_wksp;
} /* vsi_nn_HasRNN() */

void  vsi_nn_get_tensor_consumers
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_id_t tensor_id,
    vsi_nn_node_t** nodes,
    uint32_t* count
    )
{
    vsi_nn_node_t* node = NULL;
    uint32_t i, j = 0;
    uint32_t nodes_count = 0;
    for(i = 0; i < graph->node_num; i++)
    {
        node = vsi_nn_GetNode(graph, i);
        for(j = 0; j < node->input.num; j++)
        {
            if(node->input.tensors[j] == tensor_id)
            {
                if(nodes != NULL)
                {
                    nodes[nodes_count] = node;
                }
                nodes_count += 1;
                break;
            }
        }
    }
    if(count != NULL)
    {
        *count = nodes_count;
    }
} /* vsi_nn_get_tensor_consumers() */

void vsi_nn_get_tensor_provider
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_id_t tensor_id,
    vsi_nn_node_t** node
    )
{
    vsi_nn_node_t* cur_node = NULL;
    uint32_t i, j = 0;
    for(i = 0; i < graph->node_num; i++)
    {
        cur_node = vsi_nn_GetNode(graph, i);
        for(j = 0; j < cur_node->output.num; j++)
        {
            if(cur_node->output.tensors[j] == tensor_id)
            {
                *node = cur_node;
                return;
            }
        }
    }
} /* vsi_nn_get_tensor_provider() */

vsi_status vsi_nn_SetGraphPreloadSize
    (
    vsi_nn_graph_t* graph,
    vsi_nn_graph_attr_preload_type_e attr,
    uint32_t size
    )
{
    vsi_status status;
    status = VSI_FAILURE;

#if(defined(VX_PRELOAD_CONST_TENSOR_SUPPORT) && VX_PRELOAD_CONST_TENSOR_SUPPORT)
    if(graph && graph->g)
    {
        switch(attr)
        {
            case VSI_NN_GRAPH_PRELOAD_VIPSRAM:
            {
                status = vxSetGraphAttribute(graph->g, VX_GRAPH_VIP_SRAM_PRE_LOAD, &size, sizeof(size));
                break;
            }

            case VSI_NN_GRAPH_PRELOAD_AXISRAM:
            {
                status = vxSetGraphAttribute(graph->g, VX_GRAPH_AXI_SRAM_PRE_LOAD, &size, sizeof(size));
                break;
            }

            default:
            {
                VSILOGE("Unsupported graph attribute: %d", attr);
                break;
            }
        }
    }
#else
    status = VSI_SUCCESS;
#endif

    return status;
}

vsi_nn_tensor_id_t vsi_nn_get_tensor_id
    (
    vsi_nn_graph_t* graph,
    const vsi_nn_tensor_t * tensor
    )
{
    uint32_t i;
    vsi_nn_tensor_t * iter;
    if( !graph || !tensor )
    {
        return VSI_NN_TENSOR_ID_NA;
    }
    for(i = 0; i < graph->tensor_num; i++)
    {
        iter = vsi_nn_GetTensor( graph, i );
        if(iter && iter == tensor)
        {
            return i;
        }
    }
    return VSI_NN_TENSOR_ID_NA;
} /* vsi_nn_get_tensor_id() */

vsi_status vsi_nn_SetGraphPriority
    (
    vsi_nn_graph_t* graph,
    uint32_t priority
    )
{
    vsi_status status = VSI_FAILURE;
#ifdef VX_GRAPH_PREEMPTION_SUPPORT
    if(graph && graph->g)
    {
        status = vxSetGraphAttribute(graph->g, VX_GRAPH_PRIORITY_VALUE_VIV, &priority, sizeof(priority));
    }
#else
    VSILOGE("Current driver not support graph priority.");
#endif
    return status;
}
