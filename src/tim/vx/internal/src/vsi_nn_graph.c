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
#include "utils/vsi_nn_map.h"
#include "utils/vsi_nn_dtype_util.h"
#include "vsi_nn_graph_optimization.h"
#include "vsi_nn_error.h"
#include "vsi_nn_types_prv.h"

static vsi_status _set_reference_node_name
    (
    vsi_nn_graph_t *graph,
    vsi_nn_node_t *node
    )
{
#define _NODE_ID_LEN 64
    vsi_status status;
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

final:
    return status;
} /* _set_reference_node_name() */

static vsi_status _set_reference_tensor_name
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
} /* _set_reference_tensor_name() */

static vsi_status _set_parameter_for_swap_handle
    (
    vsi_nn_graph_t* graph,
    vsi_nn_node_t* node,
    vsi_nn_tensor_t* tensor,
    uint32_t idx
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_swap_handle_cache_item_t* item = NULL;
    status = vxSetParameterByIndex( node->n, idx, (vx_reference)tensor->t );
    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Set parameter %d for node[%08x] fail!", idx, node->n );
        goto final;
    }
    tensor->is_swapped = FALSE;

    if (!((vsi_nn_graph_prv_t*)graph)->swap_handle_cache.is_feature_on) {
       goto final;
    }

    item = (vsi_nn_swap_handle_cache_item_t *)
        malloc( sizeof(vsi_nn_swap_handle_cache_item_t) );
    if( NULL == item )
    {
        VSILOGE( "Create swap handle cache item fail." );
        goto final;
    }

    memset( item, 0, sizeof(vsi_nn_swap_handle_cache_item_t) );
    item->node = node;
    item->idx = idx;
    item->tensor = tensor;
    vsi_nn_LinkListPushStart(
        (vsi_nn_link_list_t **)&(((vsi_nn_graph_prv_t*)graph)->swap_handle_cache.cache_list),
        (vsi_nn_link_list_t *)item );

final:
    return status;
} /* _set_parameter_for_swap_handle() */

static vsi_status _check_swapped_tensors
    (
    vsi_nn_graph_t* graph
    )
{
    uint32_t i = 0;
    vsi_status status = VSI_SUCCESS;

    VSILOGD("Check swapped tensors");
    if (((vsi_nn_graph_prv_t*)graph)->swap_handle_cache.is_feature_on
        && ((vsi_nn_graph_prv_t*)graph)->swap_handle_cache.is_cached)
    {
        vsi_nn_swap_handle_cache_item_t* cur_item =
            ((vsi_nn_graph_prv_t*)graph)->swap_handle_cache.cache_list;
        while( NULL != cur_item && VSI_SUCCESS == status )
        {
            status = vxSetParameterByIndex( cur_item->node->n, cur_item->idx,
                (vx_reference)(cur_item->tensor->t) );
            cur_item = (vsi_nn_swap_handle_cache_item_t *)
                vsi_nn_LinkListNext( (vsi_nn_link_list_t *)cur_item );
        }
        goto final;
    }
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
                    status = _set_parameter_for_swap_handle(graph, node, tensor, idx);
                    if( VSI_SUCCESS != status )
                    {
                        VSILOGE( "_set_parameter_for_swap_handle for input fail!");
                        goto final;
                    }
                }
                idx++;
            }

            for( j = 0; j < node->output.num; j++ )
            {
                tensor = vsi_nn_GetTensor( graph, node->output.tensors[j] );
                if( tensor && tensor->is_swapped )
                {
                    status = _set_parameter_for_swap_handle(graph, node, tensor, idx);
                    if( VSI_SUCCESS != status )
                    {
                        VSILOGE( "_set_parameter_for_swap_handle for output fail!");
                        goto final;
                    }
                }
                idx++;
            }
        }
    }

    if (NULL != ((vsi_nn_graph_prv_t*)graph)->swap_handle_cache.cache_list)
    {
        ((vsi_nn_graph_prv_t*)graph)->swap_handle_cache.is_cached = TRUE;
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
    uint32_t i = 0,max_io = 0;
    vsi_status status = VSI_FAILURE;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node = NULL;

    status = VSI_SUCCESS;
    max_io = VSI_NN_MAX_IO_NUM; /* default max node io */
    for(i = 0; i < graph->node_num; i++)
    {
        node_id = node_list[i];
        node = vsi_nn_GetNode( graph, node_id );

        if (node && node->input.num > max_io)
        {
            max_io = node->input.num;
        }
        if (node && node->output.num > max_io)
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
        CHECK_PTR_FAIL_GOTO( node, "Get node fail.", final );

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
        CHECK_PTR_FAIL_GOTO( node, "Get node fail.", final );

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
        CHECK_PTR_FAIL_GOTO( node, "Get node fail.", final );

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
        status = _set_reference_tensor_name(graph, node);
        if( VSI_SUCCESS != status )
        {
            VSILOGW("Set reference node[%d] %s output tensor name fail",
                node_id, vsi_nn_OpGetName(node->op));
        }

        /* Create vx node */
        VSILOGD("Instance node[%d] \"%s\" ...", node_id, vsi_nn_OpGetName(node->op));
        status = vsi_nn_OpCompute( node->op, node, inputs, outputs );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Create node[%d] %s fail", node_id, vsi_nn_OpGetName(node->op));
            break;
        }
        status = _set_reference_node_name(graph, node);
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
        CHECK_PTR_FAIL_GOTO( node, "Get node fail.", final );

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

#if VX_GRAPH_BATCH_OPT_SUPPORT
static vsi_bool canBatchSplit
(
    vsi_nn_node_t* node,
    uint32_t inputBtachNum
)
{
    vsi_bool ret;
    uint32_t i;
    ret = TRUE;

    switch(node->op)
    {
        case VSI_NN_OP_SOFTMAX:
            if (node->nn_param.softmax.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_LOG_SOFTMAX:
            if (node->nn_param.log_softmax.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_LAYER_NORM:
            if (node->nn_param.layernorm.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_REDUCE:
            for (i = 0; i < node->nn_param.reduce.axis_num; i++)
            {
                int index = node->nn_param.reduce.axis[i];
                if (index == (int32_t)inputBtachNum - 1)
                {
                    ret = FALSE;
                    break;
                }
            }
            break;
        case VSI_NN_OP_CONCAT:
            if (node->nn_param.concat.axis == inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_TENSORSTACKCONCAT:
            if (node->nn_param.tensorstackconcat.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_STACK:
            if (node->nn_param.stack.axis == inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_UNSTACK:
            if (node->nn_param.unstack.axis == inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_CONCATSHIFT:
            if (node->nn_param.concatshift.axis == inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_SPLIT:
            if (node->nn_param.split.axis == inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_BATCH2SPACE:
        case VSI_NN_OP_SPACE2BATCH:
        case VSI_NN_OP_BATCH_NORM:
            ret = FALSE;
            break;
        case VSI_NN_OP_CROP:
            if (node->nn_param.crop.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_CUMSUM:
            if (node->nn_param.cumsum.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_INSTANCE_NORM:
            for (i = 0; i < (uint32_t)node->nn_param.instancenorm.axis_num; i++)
            {
                int index = node->nn_param.instancenorm.axis[i];
                if (index == (int32_t)inputBtachNum - 1)
                {
                    ret = FALSE;
                    break;
                }
            }
            break;
        case VSI_NN_OP_L2NORMALIZESCALE:
            if (node->nn_param.l2normalizescale.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_L2_NORMALIZE:
            if (node->nn_param.l2_normalize.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_LPNORM:
            if (node->nn_param.lpnorm.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_LRN:
            if (node->nn_param.lrn.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_MOMENTS:
            for (i = 0; i < (uint32_t)node->nn_param.moments.axis_num; i++)
            {
                int index = node->nn_param.moments.axis[i];
                if (index == (int32_t)inputBtachNum - 1)
                {
                    ret = FALSE;
                    break;
                }
            }
            break;
        case VSI_NN_OP_REPEAT:
            if (node->nn_param.repeat.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_GATHER:
            if (node->nn_param.gather.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_GATHER_ELEMENTS:
            if (node->nn_param.gather_elements.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_SCATTER_ELEMENTS:
            if (node->nn_param.scatter_elements.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_SHUFFLECHANNEL:
            if (node->nn_param.shufflechannel.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        case VSI_NN_OP_TOPK:
            if (node->nn_param.topk.axis == (int32_t)inputBtachNum - 1)
            {
                ret = FALSE;
            }
            break;
        default:
            break;
    }

    return ret;
}

static vsi_status batchInference_graph
(
    vsi_nn_graph_t* graph,
    vsi_nn_node_id_t* nodes_list
)
{
    vsi_size_t i, j, k;
    vsi_status status;
    vsi_bool ret;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_tensor_t** inputs = NULL;
    vsi_nn_tensor_t** outputs = NULL;
    vsi_nn_tensor_attr_t* original_inputs_attr = NULL;
    vsi_nn_tensor_attr_t* original_outputs_attr = NULL;
    vsi_nn_tensor_id_t* approximateConstTensor = NULL;
    vsi_size_t approximateConstTensor_count = 0;
    vsi_bool has_inputTensor = FALSE;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t* node;
    vsi_size_t num_of_node_inputs = 0;
    vsi_size_t batchCount = 0;
    vsi_size_t batchNum = 1;

    vx_hardware_caps_params_t   hw_param;
    vx_context  ctx = vxGetContext((vx_reference)graph->g);

    for (i = 0; i < graph->node_num; i++)
    {
        node_id = nodes_list[i];
        node = vsi_nn_GetNode(graph, node_id);
        /* For NBG node, donot infer shape*/
        if (node && node->op == VSI_NN_OP_NBG)
        {
            status = VSI_SUCCESS;
            goto final;
        }
    }

    memset(&hw_param, 0, sizeof(vx_hardware_caps_params_t));
    status = vxQueryHardwareCaps(ctx, &hw_param, sizeof(vx_hardware_caps_params_t));

    /*initial tensor shape*/
    status = setup_node(graph, nodes_list);
    if (VSI_SUCCESS != status)
    {
        goto final;
    }

    status = VSI_SUCCESS;
    ret = TRUE;
    inputs = allocate_io_buffer(graph);
    outputs = allocate_io_buffer(graph);
    original_inputs_attr = (vsi_nn_tensor_attr_t*)malloc(sizeof(vsi_nn_tensor_attr_t) * graph->max_node_io);
    original_outputs_attr = (vsi_nn_tensor_attr_t*)malloc(sizeof(vsi_nn_tensor_attr_t) * graph->max_node_io);
    approximateConstTensor = (vsi_nn_tensor_id_t*)malloc(sizeof(vsi_nn_tensor_id_t) * graph->tensor_num);
    CHECK_PTR_FAIL_GOTO(approximateConstTensor, "Malloc fail.", final);
    memset(approximateConstTensor, -1, sizeof(vsi_nn_tensor_id_t) * graph->tensor_num);

    if (NULL == inputs || NULL == outputs || NULL == original_inputs_attr || NULL == original_outputs_attr)
    {
        VSILOGE("allocate buffer fail");
        status = VSI_FAILURE;
        goto final;
    }

    for (i = 0; i < graph->node_num; i++)
    {
        node_id = nodes_list[i];
        memset(inputs, 0, graph->max_node_io * sizeof(vsi_nn_tensor_t*));
        memset(outputs, 0, graph->max_node_io * sizeof(vsi_nn_tensor_t*));
        memset(original_inputs_attr, 0, graph->max_node_io * sizeof(vsi_nn_tensor_attr_t));
        memset(original_outputs_attr, 0, graph->max_node_io * sizeof(vsi_nn_tensor_attr_t));

        /* Get inputs, outputs. */
        node = vsi_nn_GetNode(graph, node_id);
        CHECK_PTR_FAIL_GOTO(node, "Get node fail.", final);

        vsi_nn_GetTensors(graph, node->input.tensors,
            node->input.num, inputs);
        vsi_nn_GetTensors(graph, node->output.tensors,
            node->output.num, outputs);
        batchNum = 1;
        /*get input batch number*/
        has_inputTensor = FALSE;
        for (j = 0; j < node->input.num; j++)
        {
            vx_bool is_const = FALSE;
            if (inputs[j] == NULL)
            {
                continue;
            }
            memcpy(&original_inputs_attr[j], &inputs[j]->attr, sizeof(vsi_nn_tensor_attr_t));
            for (k = 0; k < approximateConstTensor_count; k++)
            {
                if (node->input.tensors[j] == approximateConstTensor[k])
                {
                    is_const = TRUE;
                }
            }
            if (inputs[j]->attr.is_const != TRUE && is_const != TRUE)
            {
                has_inputTensor = TRUE;
                if (batchNum < inputs[j]->attr.size[inputs[j]->attr.dim_num - 1])
                {
                    batchNum = inputs[j]->attr.size[inputs[j]->attr.dim_num - 1];
                }
            }
        }

        for (j = 0; j < node->output.num; j++)
        {
            if (outputs[j] == NULL)
            {
                continue;
            }
            memcpy(&original_outputs_attr[j], &outputs[j]->attr, sizeof(vsi_nn_tensor_attr_t));
            if (!has_inputTensor)
            {
                approximateConstTensor[approximateConstTensor_count++] = node->output.tensors[j];
            }
            if (original_outputs_attr[j].dim_num < 1)
            {
                break;
            }
        }
        if (j != node->output.num)
        {
            continue;
        }

        if (batchNum > 1 && canBatchSplit(node, original_inputs_attr[0].dim_num))
        {
            vsi_size_t iterator_list_index = 0;
            vsi_size_t list_index = 0;
            vsi_size_t* iterator_list = (vsi_size_t*)malloc(sizeof(vsi_size_t) * (batchNum + 1));
            CHECK_PTR_FAIL_GOTO(iterator_list, "Malloc fail.", final);
            memset(iterator_list, 0, sizeof(uint32_t) * (batchNum + 1));

            if (((vsi_nn_node_prv_t*)node)->split_num > 0)
            {/*user defined batch count*/
                iterator_list[iterator_list_index++] = ((vsi_nn_node_prv_t*)node)->split_num;
                if (((vsi_nn_node_prv_t*)node)->split_num == 1)
                {/*if user set split_num = 1, there is no need to batch split.*/
                    vsi_nn_safe_free(iterator_list);
                    continue;
                }
            }
            /*iterate through each vaild batch count*/
            for (batchCount = batchNum; batchCount > 1; batchCount--)
            {

                /*for some node with big batch num, should limit to max core count.*/
                if (batchCount > (hw_param.coreCount == 0?24 : hw_param.coreCount))
                {
                    continue;
                }
                if (batchNum % batchCount != 0)
                {
                    continue;
                }
                iterator_list[iterator_list_index++] = batchCount;
            }

            /*iterate through each vaild batch count*/
            for (list_index = 0; list_index < iterator_list_index; list_index++)
            {
                batchCount = iterator_list[list_index];

                /*set node input batch*/
                num_of_node_inputs = node->input.num;
                for (k = 0; k < num_of_node_inputs; k++)
                {
                    tensor = inputs[k];
                    if (tensor)
                    {
                        vx_bool is_const = FALSE;
                        uint32_t index = 0;
                        for (index = 0; index < approximateConstTensor_count; index++)
                        {
                            if (node->input.tensors[k] == approximateConstTensor[index])
                            {
                                is_const = TRUE;
                            }
                        }
                        if (is_const != TRUE && tensor->attr.is_const != TRUE)
                        {
                            if (original_inputs_attr[k].size[tensor->attr.dim_num - 1] / batchCount < 1
                                || original_inputs_attr[k].size[tensor->attr.dim_num - 1] % batchCount != 0)
                            {
                                break;
                            }
                            else
                            {
                                tensor->attr.size[tensor->attr.dim_num - 1] =
                                    original_inputs_attr[k].size[tensor->attr.dim_num - 1] / batchCount;
                            }
                        }
                    }
                }
                if (k != num_of_node_inputs)
                {
                    continue;
                }

                /*reset output tensor size, dim_num and other parameter,
                    if not, it will affect vsi_nn_OpGenerateTensor*/
                for (j = 0; j < node->output.num; j++)
                {
                    if (outputs[j] == NULL)
                    {
                        continue;
                    }
                    outputs[j]->attr.dim_num = VSI_NN_DIM_AUTO;
                    for (k = 0; k < VSI_NN_MAX_DIM_NUM; k++)
                    {
                        outputs[j]->attr.size[k] = 0;
                    }
                }
                if (node->internal_node_wksp != NULL)
                {
                    vsi_nn_internal_init_node_wksp(node);
                }

                /*node shape inference: */
                if (vsi_nn_OpCheck(node->op, node, inputs, outputs))
                {
                    vsi_nn_print_node_io(graph, node, 0x01);
                    ret = vsi_nn_OpGenerateTensor(node, inputs, outputs);
                    if (ret != TRUE)
                    {
                        VSILOGD("Cannot split node[%u] %s on input_batch_count=%u",
                            node_id, vsi_nn_OpGetName(node->op), batchCount);
                        continue;
                    }
                    vsi_nn_print_node_io(graph, node, 0x02);

                    /*check if the node can be splited on batch*/
                    for (j = 0; j < node->output.num; j++)
                    {
                        if (outputs[j] == NULL)
                        {
                            continue;
                        }

                        tensor = outputs[j];
                        /*can be splited if the batch dim size of the output shape is changed.*/
                        if (tensor->attr.size[tensor->attr.dim_num - 1] ==
                            original_outputs_attr[j].size[original_outputs_attr[j].dim_num - 1])
                        {
                            VSILOGD("Cannot split node[%u] %s on input_batch_count=%u",
                                    node_id,
                                    vsi_nn_OpGetName(node->op),
                                    batchCount);
                            break;
                        }
                    }

                    if (j == node->output.num )
                    {
                        /*save the verified batch count*/
                        ((vsi_nn_node_prv_t*)node)->split_num = batchCount;
                        break;
                    }
                }
                else
                {
                    VSILOGD("Cannot split node[%u] %s on input_batch_count=%u",
                    node_id,
                    vsi_nn_OpGetName(node->op),
                    batchCount);
                    continue;
                }
            }

            vsi_nn_safe_free(iterator_list);
            /*restore node input batch number*/
            num_of_node_inputs = node->input.num;
            for (k = 0; k < num_of_node_inputs; k++)
            {
                tensor = inputs[k];
                if (tensor)
                {
                    tensor->attr.size[tensor->attr.dim_num - 1] =
                        original_inputs_attr[k].size[tensor->attr.dim_num - 1] ;
                }
            }

            /*reset the output tensors*/
            for (j = 0; j < node->output.num; j++)
            {
                if (outputs[j] == NULL)
                {
                    continue;
                }
                outputs[j]->attr.dim_num = VSI_NN_DIM_AUTO;
                for (k = 0; k < VSI_NN_MAX_DIM_NUM; k++)
                {
                    outputs[j]->attr.size[k] = 0;
                }
            }
            if (node->internal_node_wksp != NULL)
            {
                vsi_nn_internal_init_node_wksp(node);
            }

            /*restore node output shape*/
            if (vsi_nn_OpCheck(node->op, node, inputs, outputs))
            {
                ret = vsi_nn_OpGenerateTensor(node, inputs, outputs);
            }
        }
    }

final:
    for (i = 0; i < graph->node_num; i++)
    {
        node_id = nodes_list[i];
        node = vsi_nn_GetNode(graph, node_id);
        if (node == NULL || node->op == VSI_NN_OP_NBG)
        {
            break;
        }

        vsi_nn_GetTensors(graph, node->input.tensors,
            node->input.num, inputs);
        vsi_nn_GetTensors(graph, node->output.tensors,
            node->output.num, outputs);
        for (j = 0; outputs && j < node->output.num; j++)
        {
            if (outputs[j] == NULL)
            {
                continue;
            }
            /*reset attr->size*/
            outputs[j]->attr.dim_num = VSI_NN_DIM_AUTO;
            for (k = 0; k < VSI_NN_MAX_DIM_NUM; k++)
            {
                outputs[j]->attr.size[k] = 0;
            }
        }
        if (node->internal_node_wksp != NULL)
        {
            vsi_nn_internal_init_node_wksp(node);
        }
    }

    free_io_buffer(inputs);
    free_io_buffer(outputs);

    if (original_inputs_attr != NULL)
    {
        free(original_inputs_attr);
    }
    if (original_outputs_attr != NULL)
    {
        free(original_outputs_attr);
    }
    if (approximateConstTensor != NULL)
    {
        free(approximateConstTensor);
    }

    return status;
} /* batchInference_graph() */

static vsi_status update_vxnode_batchNum
(
    vsi_nn_graph_t* graph,
    vsi_nn_node_id_t* node_list
)
{
    uint32_t i, j;
    vsi_status status;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t* node;
    vsi_nn_internal_node_t* inode;

    status = VSI_SUCCESS;
    for (i = 0; i < graph->node_num; i++)
    {
        node_id = node_list[i];
        node = vsi_nn_GetNode(graph, node_id);
        CHECK_PTR_FAIL_GOTO(node, "Get node fail.", final);
        if (node->n != NULL)
        {
            vxSetNodeBatch(node->n, (uint32_t)((vsi_nn_node_prv_t*)node)->split_num);
            if (((vsi_nn_node_prv_t*)node)->split_num > 1)
            {
                VSILOGD("split node[%u] %s to %ds on batch dim",
                    node_id,
                    vsi_nn_OpGetName(node->op),
                    ((vsi_nn_node_prv_t*)node)->split_num);
            }
        }

        for (j = 1; j < 100; j++)
        {
            inode = vsi_nn_internal_get_node_by_uid(node, j);
            if (inode == NULL)
            {
                break;
            }
            else
            {
                if (inode->node->n != NULL)
                {
                    vxSetNodeBatch(inode->node->n, (uint32_t)((vsi_nn_node_prv_t*)node)->split_num);
                }
            }
        }

    }

    final:
    return status;
} /* update_vxnode_batchNum() */
#endif

vsi_status vsi_nn_InferShape
(
    vsi_nn_graph_t* graph
)
{
    uint32_t i, j, k;
    vsi_status status;
    vsi_nn_tensor_t** outputs = NULL;
    vsi_nn_node_t* node;
    vsi_nn_node_id_t* nodes_list = NULL;
    status = VSI_SUCCESS;

    for (i = 0; i < graph->node_num; i++)
    {
        node = vsi_nn_GetNode(graph, i);
        /* For NBG node, donot infer shape*/
        if (node && node->op == VSI_NN_OP_NBG)
        {
            status = VSI_FAILURE;
            goto final;
        }
    }

    outputs = allocate_io_buffer(graph);
    if (NULL == outputs)
    {
        VSILOGE("allocate buffer fail");
        status = VSI_FAILURE;
        goto final;
    }

    /*reset all nodes' output shape*/
    for (i = 0; i < graph->node_num; i++)
    {
        memset(outputs, 0, graph->max_node_io * sizeof(vsi_nn_tensor_t*));
        node = vsi_nn_GetNode(graph, i);
        CHECK_PTR_FAIL_GOTO(node, "Get node fail.", final);

        vsi_nn_GetTensors(graph, node->output.tensors,
            node->output.num, outputs);
        CHECK_PTR_FAIL_GOTO(outputs, "Get node's output fail.", final);
        for (j = 0; j < node->output.num; j++)
        {
            if (outputs[j] == NULL)
            {
                continue;
            }
            /*reset attr->size*/
            outputs[j]->attr.dim_num = VSI_NN_DIM_AUTO;
            for (k = 0; k < VSI_NN_MAX_DIM_NUM; k++)
            {
                outputs[j]->attr.size[k] = 0;
            }
        }
        if (node->internal_node_wksp != NULL)
        {
            vsi_nn_internal_init_node_wksp(node);
        }
    }

    /*setup nodes.*/
    nodes_list = (vsi_nn_node_id_t*)malloc(
        graph->node_num * sizeof(vsi_nn_node_id_t));
    if (!nodes_list)
    {
        goto final;
    }
    for (i = 0; i < graph->node_num; i++)
    {
        nodes_list[i] = i;
    }

    status = setup_node(graph, nodes_list);
    if (VSI_SUCCESS != status)
    {
        goto final;
    }

    final:
    free_io_buffer(outputs);
    if (NULL != nodes_list)
    {
        free(nodes_list);
    }

    return status;
}

static vsi_status set_graph_precision
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    uint32_t i, j;
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

    if(vsi_nn_IsGraphFastMode(graph))
    {
        goto final;
    }
    for( i = 0; i < graph->node_num; i++ )
    {
        node_id = node_list[i];
        memset( inputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );
        memset( outputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );
        /* Get inputs, outputs. */
        node = vsi_nn_GetNode( graph, node_id );
        CHECK_PTR_FAIL_GOTO( node, "Get node fail.", final );

        vsi_nn_GetTensors( graph, node->input.tensors,
            node->input.num, inputs );
        vsi_nn_GetTensors( graph, node->output.tensors,
            node->output.num, outputs );

        for(j = 0; j < node->input.num; j++)
        {
            if(inputs[j] != NULL && inputs[j]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32)
            {
                vsi_nn_SetTensorAttr(inputs[j], VSI_NN_TENSOR_ATTR_HIGH_PRECISION);
            }
        }
        for(j = 0; j < node->output.num; j++)
        {
            if(outputs[j] != NULL && outputs[j]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32)
            {
                vsi_nn_SetTensorAttr(outputs[j], VSI_NN_TENSOR_ATTR_HIGH_PRECISION);
            }
        }
    }
final:
    free_io_buffer(inputs);
    free_io_buffer(outputs);
    return status;
}
vsi_nn_graph_t * vsi_nn_CreateGraph
    (
    vsi_nn_context_t ctx,
    uint32_t        max_tensor_num,
    uint32_t        max_node_num
    )
{
    vsi_nn_graph_t * graph;
    graph = NULL;

    VSI_UNREFERENCED(max_tensor_num);
    VSI_UNREFERENCED(max_node_num);

    VSILOGI( "%s", vsi_nn_GetVersion() );

    if( NULL == ctx )
    {
        return graph;
    }

    graph = (vsi_nn_graph_t *)malloc( sizeof( vsi_nn_graph_prv_t ) );
    if( NULL != graph )
    {
        memset( graph, 0, sizeof( vsi_nn_graph_prv_t ) );
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
            ((vsi_nn_graph_prv_t*) graph)->options =
                (vsi_nn_runtime_option_t *)malloc( sizeof( vsi_nn_runtime_option_t ));
            CHECK_PTR_FAIL_GOTO(((vsi_nn_graph_prv_t*) graph)->options, "Create graph options fail.", error);
            graph->node_table = (vsi_nn_map_t *)malloc( sizeof( vsi_nn_map_t ) );
            graph->tensor_table = (vsi_nn_map_t *)malloc( sizeof( vsi_nn_map_t ) );
            graph->isAllowFastMode = TRUE;
            vsi_nn_MapInit( graph->node_table );
            vsi_nn_MapInit( graph->tensor_table );
            vsi_nn_initOptions_runtime( ((vsi_nn_graph_prv_t*) graph)->options, ctx );
        }
        else
        {
            VSILOGE( "Create vx graph fail." );
            free(graph);
            graph = NULL;
        }
    }

    return graph;
error:
    return graph;
} /* vsi_nn_CreateGraph() */

void vsi_nn_ReleaseGraph
    (
    vsi_nn_graph_t ** graph
    )
{
    uint32_t i;
    vsi_nn_graph_t  * ptr;

    ptr = (NULL != graph) ? *graph : NULL;
    if( NULL != ptr)
    {
        if( NULL != ptr->nodes )
        {
            for( i = 0; i < ptr->node_num; i++ )
            {
                vsi_nn_RemoveNode( *graph, (vsi_nn_node_id_t)i );
            }
            free( (*graph)->node_table );
        }
        if( NULL != ptr->g )
        {
            vxReleaseGraph( &ptr->g );
        }
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
        if( NULL != ((vsi_nn_graph_prv_t*)ptr)->swap_handle_cache.cache_list )
        {
            vsi_nn_swap_handle_cache_item_t* item = ((vsi_nn_graph_prv_t*)ptr)->swap_handle_cache.cache_list;
            while( NULL != item )
            {
                vsi_nn_swap_handle_cache_item_t* tmp = (vsi_nn_swap_handle_cache_item_t *)
                    vsi_nn_LinkListPopStart( (vsi_nn_link_list_t **)&item );
                free( tmp );
            }
        }
        if (NULL != ((vsi_nn_graph_prv_t*)ptr)->options)
        {
            free(((vsi_nn_graph_prv_t*)ptr)->options);
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

#if VX_GRAPH_BATCH_OPT_SUPPORT
    if (((vsi_nn_graph_prv_t*)graph)->options->enable_batch_opt)
    {
        /*processing batch splitting*/
        status = batchInference_graph(graph, nodes_list);
        if (VSI_SUCCESS != status)
        {
            goto final;
        }
    }
#endif

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

    /* set tensor's precision before compute_node
    so that internal tensor can know the precision information*/
    status = set_graph_precision(graph, nodes_list);
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

#if VX_GRAPH_BATCH_OPT_SUPPORT
    /* update vxnode's batch_count */
    status = update_vxnode_batchNum(graph, nodes_list);
    if (VSI_SUCCESS != status)
    {
        goto final;
    }
#endif
    /* set precision again to make sure any tensor created by compute_node have correct precesion infor*/
    status = set_graph_precision(graph, nodes_list);
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
    uint8_t             *  data,
    int8_t                 is_from_axisram
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
            VSILOGE("VDATA mode is no longer be supported!");
        }
    }
    else if( NULL != data )
    {
        if (TRUE == is_from_axisram)
        {
            VSILOGE("Can't create a tensor from AXI-SRAM with data.");
        }
        else
        {
            tensor = vsi_nn_CreateTensorFromData( graph, data, attr );
        }
    }
    else
    {
        if (TRUE == is_from_axisram)
        {
            tensor = vsi_nn_CreateTensorFromAXISRAM(graph, attr);
        }
        else
        {
            tensor = vsi_nn_CreateTensor(graph, attr);
        }

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
    return _add_tensor(graph, id, attr, data, FALSE);
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
    return _add_tensor(graph, id, attr, data, FALSE);
}

vsi_nn_tensor_id_t vsi_nn_AddTensorFromView
(
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_id_t parent_id,
    vsi_size_t* start,
    vsi_size_t* end
)
{
    uint32_t i = 0;
    vx_tensor view_vxt = NULL;
    vsi_nn_tensor_t* parent_tensor = NULL;
    vsi_nn_tensor_t* new_tensor =NULL;
    vsi_nn_tensor_id_t id = VSI_NN_TENSOR_ID_NA;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0x0, sizeof(vsi_nn_tensor_attr_t));
    parent_tensor = vsi_nn_GetTensor(graph, parent_id);
    if (NULL == parent_tensor)
    {
        VSILOGE("Create view tensor failed, parent tensor is invalid.");
        id = VSI_NN_TENSOR_ID_NA;
        goto final;
    }

    /* new tensor's all attribuites are inherited from parent tensor except 'size' */
    attr = parent_tensor->attr;
    for (i = 0; i < attr.dim_num; i++)
    {
        attr.size[i] = end[i] - start[i];
    }
    id = _add_tensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL, FALSE);
    if (VSI_NN_TENSOR_ID_NA == id)
    {
        VSILOGE("Create view tensor failed, new tensor could not be created.");
        goto final;
    }

    new_tensor = vsi_nn_GetTensor(graph, id);
    if (new_tensor && new_tensor->t)
    {
        vxReleaseTensor(&(new_tensor->t));
    }
    else
    {
        VSILOGE("Create view tensor failed, new tensor or vxTensor is NULL.");
        id = VSI_NN_TENSOR_ID_NA;
        goto final;
    }

    view_vxt = vsi_nn_CreateViewTensor(graph, start, end, parent_tensor);
    if ( NULL != view_vxt)
    {
        new_tensor->t = view_vxt;
    }
    else
    {
        VSILOGE("Create view tensor failed, view vxTensor could not be created.");
        id = VSI_NN_TENSOR_ID_NA;
        goto final;
    }
final:
    return id;
}

vsi_nn_tensor_id_t vsi_nn_AddTensorFromAXISRAM
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id,
    vsi_nn_tensor_attr_t * attr
    )
{
    return _add_tensor(graph, id, attr, NULL, TRUE);
} /* vsi_nn_AddTensorFromAXISRAM() */

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
} /* vsi_nn_GetNode() */

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

vsi_nn_node_t * vsi_nn_AddExternalNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_op_t           op,
    const void          * proc,
    vsi_nn_node_id_t    * node_id,
    const char          * kernel_name
    )
{
    vsi_nn_node_prv_t* node;
    vsi_nn_node_id_t id;
    vsi_nn_op_proc_t * node_proc;

    VSI_UNREFERENCED(node_id);

    node_proc = (vsi_nn_op_proc_t*)proc;

    if( NULL == graph )
    {
        return NULL;
    }
    node = (vsi_nn_node_prv_t*)malloc(sizeof(vsi_nn_node_prv_t));

    if( NULL != node )
    {
        memset(node, 0, sizeof(vsi_nn_node_prv_t));
        node->pon.graph = graph;
        node->pon.op = op;
        node->pon.vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
        node->pon.vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
        node->pon.vx_param.down_scale_size_rounding =
            VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

        /* init op */
        if(node_proc->init != NULL){
            //TODO
        }

        /* init output struct */
        node->pon.output.num = node_proc->output_num;
        node->pon.output.tensors = (vsi_nn_tensor_id_t*)malloc(
            node_proc->output_num * sizeof( vsi_nn_tensor_id_t ) );
        if (NULL == node->pon.output.tensors)
        {
            VSILOGE("Create output tensor id %s. fail", vsi_nn_OpGetName(op));
            vsi_nn_safe_free(node);
            return NULL;
        }
        vsi_nn_InitTensorsId(node->pon.output.tensors, node_proc->output_num);

        /* init input struct */
        node->pon.input.num = node_proc->input_num;
        node->pon.input.tensors = (vsi_nn_tensor_id_t*)malloc(
            node_proc->input_num * sizeof( vsi_nn_tensor_id_t ) );
        if (NULL == node->pon.input.tensors)
        {
            VSILOGE("Create input tensor id %s. fail", vsi_nn_OpGetName(op));
            vsi_nn_safe_free(node->pon.output.tensors);
            vsi_nn_safe_free(node);
            return NULL;
        }
        vsi_nn_InitTensorsId(node->pon.input.tensors, node_proc->input_num);
        node->pon.attr.const_tensor_preload_type = VSI_NN_NODE_PRELOAD_NONE;
        node->pon.attr.enable_op_constraint_check = TRUE;
    }
    id = graph->cur_nid;
    if(NULL != node){
        vsi_nn_MapAdd( graph->node_table, (vsi_nn_map_key_t)id, (void *)node );
        graph->node_num = graph->cur_nid;
        graph->cur_nid ++;
    }
    vsi_nn_OpRegisterExternalOvxInit(op, kernel_name, node_proc);
    return (vsi_nn_node_t*)node;
} /* vsi_nn_AddExternalNode() */

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

    if( NULL == graph )
    {
        return ret;
    }

    if ( tensor_num == 0 )
    {
        return TRUE;
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
    uint32_t i = 0,j = 0;
    uint32_t             count = 1;
    vsi_bool             dirty = TRUE;
    vsi_bool             all_tensor_processed = FALSE;
    vsi_bool           * tensors = NULL;
    vsi_nn_node_id_t   * nodes = NULL;
    vsi_nn_node_id_t   * sorted_nodes = NULL;
    vsi_nn_node_t      * node = NULL;
    vsi_nn_node_id_t     node_id;
    vsi_nn_tensor_id_t   tensor_id;
    vsi_nn_tensor_t    * tensor = NULL;

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
    CHECK_PTR_FAIL_GOTO( tensors, "Create buffer fail.", final );
    memset(tensors, 0, graph->tensor_num * sizeof( vsi_bool ));

    sorted_nodes = (vsi_nn_node_id_t *)malloc(
        graph->node_num * sizeof( vsi_nn_node_id_t ) );
    CHECK_PTR_FAIL_GOTO( sorted_nodes, "Create buffer fail.", final );
    memset(sorted_nodes, 0, graph->node_num * sizeof( vsi_nn_node_id_t ));

    nodes = (vsi_nn_node_id_t *)malloc(
        graph->node_num * sizeof( vsi_nn_node_id_t ) );
    CHECK_PTR_FAIL_GOTO( nodes, "Create buffer fail.", final );
    memset(sorted_nodes, 0, graph->node_num * sizeof( vsi_nn_node_id_t ));

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
            CHECK_PTR_FAIL_GOTO( node, "Get node fail.", final );

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

final:

    /* Release memory. */
    vsi_nn_safe_free( tensors );
    vsi_nn_safe_free( nodes );

    if ( count != 0 )
    {
        vsi_nn_safe_free( sorted_nodes );
    }

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

                if ( node && node_uids[i] == node->uid )
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
    char filename_prefix[_SHAPE_BUF_SZ + 1] = { 0 };
    const char * op_name;
    uint32_t i;
    uint32_t o;
    uint32_t node_num;
    vsi_nn_node_id_t * nodes;
    vsi_nn_node_t    * node;
    vsi_nn_tensor_t  * tensor;

    VSI_UNREFERENCED(data_fmt);

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
        vsi_nn_strncpy(filename_prefix, prefix, _SHAPE_BUF_SZ);
        filename_prefix[_SHAPE_BUF_SZ - 1] = '\0';

        vsi_nn_strncat(filename_prefix, "_", _SHAPE_BUF_SZ - 1);
        filename_prefix[_SHAPE_BUF_SZ - 1] = '\0';
    }

    VSILOGD("Dump %u nodes.", node_num);
    for( i = 0; i < node_num; i++ )
    {
        node = vsi_nn_GetNode( graph, (vsi_nn_node_id_t)i );
        CHECK_PTR_FAIL_GOTO( node, "Get node fail.", final );

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

final:
    vsi_nn_safe_free( nodes );
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
    uint32_t i, j, data_input_count = 0;
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

    fp = vsi_nn_fopen("graph.json", "w+");
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

    /* dump meta data */
    fprintf(fp, "\t\"MetaData\":{\n");
    fprintf(fp, "\t\t\"Name\": \"Ovxlib_Debug_Graph\",\n");
    fprintf(fp, "\t\t\"AcuityVersion\": \"UNKNOWN\",\n");
    fprintf(fp, "\t\t\"Platform\": \"UNKNOWN\",\n");
    fprintf(fp, "\t\t\"Org_Platform\": \"UNKNOWN\"\n");
    fprintf(fp, "\t},\n");

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
                    /* this path may cause netron display abnormally */
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
                        CHECK_PTR_FAIL_GOTO( in_node, "Get node fail.", final );
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
                            fprintf(fp, "\"@data_input_uid_%u:out0\" ", graph->node_num + data_input_count + 1);
                        }
                        else
                        {
                            fprintf(fp, "\"@data_input_uid_%u:out0\", ", graph->node_num + data_input_count + 1);
                        }
                        data_input_count += 1;
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

            if(i != graph->node_num - 1 || data_input_count > 0)
            {
                fprintf(fp, ",");
            }
            fprintf(fp, "\n");
        }
    }

    /* dump all norm_tensor and const tensor into json as input layer */
    for (i = 0; i < data_input_count; i++)
    {
        fprintf(fp, "\t\t\"data_input_uid_%u\":{\n\t\t\t\"op\": \"%s\",\n",
            graph->node_num + i + 1, "DATA_INPUT");

        /* dump inputs */
        fprintf(fp, "\t\t\t\"inputs\": [ ");

        /* dump input shape */
        fprintf(fp, "],\n\t\t\t\"inut_shape\": [ ");
        fprintf(fp, "[%s ]", "");

        /* dump output */
        fprintf(fp, " ],\n\t\t\t\"outputs\": [ ");
        fprintf(fp, "\"out%u\" ", 0);

        //output shape
        fprintf(fp, "],\n\t\t\t\"output_shape\": [ ");
        fprintf(fp, "[%s ]", "");

        fprintf(fp, " ]\n\t\t}");

        if (i != data_input_count - 1)
        {
            fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\t}\n}\n");

final:
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
        signal_tensor_attr.is_created_from_handle = TRUE;
        signal_tensor_attr.is_handle_malloc_by_ovxlib = FALSE;
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
    uint32_t i,j,k,p;
    vsi_status status = VSI_FAILURE;
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
    /*update inputs for nbg node  who has crop scalar parameter as inputs*/
    for (i = 0; i < graph->node_num; i++)
    {
        vsi_nn_node_t* node = vsi_nn_GetNode(graph, i);
        uint32_t numParams = 0;

        if (node && node->op == VSI_NN_OP_NBG)
        {
            status = vxQueryNode(
                node->n, VX_NODE_PARAMETERS, &numParams, sizeof(numParams));
            for (j = 0; j < numParams; j++)
            {
                vx_parameter param = 0;
                vx_enum type = 0;
                param = vxGetParameterByIndex(node->n, j);
                if (param != NULL)
                {
                    status = vxQueryParameter(param, VX_PARAMETER_TYPE, &type, sizeof(vx_enum));
                    if (type == VX_TYPE_SCALAR)
                    {
                        num_of_graph_real_inputs++;
                    }

                    vxReleaseParameter(&param);
                    param = NULL;
                }
            }
        }
    }
    graph_inputs = (vx_reference *)malloc( num_of_graph_real_inputs * sizeof( vx_reference ) );
    CHECK_PTR_FAIL_GOTO( graph_inputs, "Create buffer fail.", final );
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
            for (k = 0; k < graph->node_num; k++)
            {
                vsi_nn_node_t* node = vsi_nn_GetNode(graph, k);

                if (node && node->op == VSI_NN_OP_NBG)
                {
                    vx_parameter param = 0;
                    vx_reference ref = 0;
                    vx_enum type = 0;
                    uint32_t scalar_index = j;
                    param = vxGetParameterByIndex(node->n, scalar_index);

                    if (param != NULL)
                    {
                        status = vxQueryParameter(param,
                                                    VX_PARAMETER_TYPE,
                                                    &type,
                                                    sizeof(vx_enum));
                        vxReleaseParameter(&param);
                        param = NULL;

                        if (type != VX_TYPE_SCALAR)
                        {
                            break;
                        }
                    }

                    for (p = scalar_index; p < scalar_index+4; p++)
                    {
                        param = vxGetParameterByIndex(node->n, p);

                        if (param != NULL)
                        {
                            status = vxQueryParameter(param,
                                                        VX_PARAMETER_TYPE,
                                                        &type,
                                                        sizeof(vx_enum));
                            if (type == VX_TYPE_SCALAR)
                            {
                                vxQueryParameter(param,
                                                    VX_PARAMETER_REF,
                                                    &ref,
                                                    sizeof(vx_reference));
                                graph_inputs[j++] = ref;
                                vxReleaseReference(&ref);
                            }

                            vxReleaseParameter(&param);
                        }
                    }
                }
            }
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
    CHECK_PTR_FAIL_GOTO( graph_outputs, "Create buffer fail.", final );
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
    vsi_nn_safe_free(graph_inputs);
    vsi_nn_safe_free(graph_outputs);

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
        CHECK_PTR_FAIL_GOTO( node, "Get node fail.", final );

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

final:
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
        CHECK_PTR_FAIL_GOTO( cur_node, "Get node fail.", final );

        for(j = 0; j < cur_node->output.num; j++)
        {
            if(cur_node->output.tensors[j] == tensor_id)
            {
                *node = cur_node;
                return;
            }
        }
    }

final:
    return;
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

    VSI_UNREFERENCED(graph);
    VSI_UNREFERENCED(attr);
    VSI_UNREFERENCED(size);

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
    VSI_UNREFERENCED(graph);
    VSI_UNREFERENCED(priority);
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

vsi_status vsi_nn_SetGraphFastMode
    (
    vsi_nn_graph_t* graph,
    vsi_bool fastmode
    )
{
    vsi_status status = VSI_SUCCESS;
    if(graph)
    {
        graph->isAllowFastMode = fastmode;
    }
    else
    {
        status = VSI_FAILURE;
    }
    return status;
}

vsi_bool vsi_nn_IsGraphFastMode
    (
    const vsi_nn_graph_t* graph
    )
{
    return NULL == graph ? FALSE : graph->isAllowFastMode;
}

vsi_status vsi_nn_CopyTensorViaGraphs
    (
    vsi_nn_graph_t *src_graph,
    vsi_nn_tensor_id_t src_tensor_id,
    vsi_nn_graph_t *dst_graph,
    vsi_nn_tensor_id_t dst_tensor_id
    )
{
    vsi_status status = VSI_FAILURE;
    uint8_t *data = NULL;
    vsi_nn_tensor_t *src_tensor = NULL;
    vsi_nn_tensor_t *dst_tensor = NULL;
    vsi_size_t i;

    src_tensor = vsi_nn_GetTensor(src_graph, src_tensor_id);
    TEST_CHECK_PTR(src_tensor, final);
    dst_tensor = vsi_nn_GetTensor(dst_graph, dst_tensor_id);
    TEST_CHECK_PTR(dst_tensor, final);

    /* Check shape and dtype */
    if(src_tensor->attr.dim_num != dst_tensor->attr.dim_num)
    {
        VSILOGE("The dim_num of src_tensor and dst_tensor don't match.");
        return status;
    }
    for(i=0; i<src_tensor->attr.dim_num; i++)
    {
        if(src_tensor->attr.size[i] != dst_tensor->attr.size[i])
        {
            VSILOGE("The shape of src_tensor and dst_tensor don't match.");
            return status;
        }
    }
    if(vsi_nn_DtypeCompare(&src_tensor->attr.dtype, &dst_tensor->attr.dtype) == FALSE)
    {
        VSILOGE("The dtype of src_tensor and dst_tensor don't match.");
        return status;
    }

    data = vsi_nn_ConvertTensorToData(src_graph, src_tensor);
    TEST_CHECK_PTR(data, final);

    status = vsi_nn_CopyDataToTensor(dst_graph, dst_tensor, data);
    TEST_CHECK_STATUS(status, final);

final:
    vsi_nn_safe_free(data);
    return status;
} /* vsi_nn_CopyTensorViaGraphs() */

vsi_status vsi_nn_ExecuteGraphLoop
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *max_iteration_tensor
    )
{
    int32_t i,j,loop_var_num,max_iteration;
    vsi_status status = VSI_FAILURE;
    vsi_nn_tensor_t *iteration_index = NULL;
    vsi_nn_tensor_t *iteration_cond_out = NULL;
    uint8_t *data = NULL;
    int8_t cond = 0;
    vsi_size_t sz = 0;

    sz = vsi_nn_ShapeProduct(max_iteration_tensor->attr.size, max_iteration_tensor->attr.dim_num);
    if(1 != sz) // it's shape should be 1.
    {
        VSILOGE("Invalid max_iteration_tensor.");
        return status;
    }

    loop_var_num = graph->input.num - 2;
    iteration_index = vsi_nn_GetTensor(graph, graph->input.tensors[0]);
    iteration_cond_out = vsi_nn_GetTensor(graph, graph->output.tensors[0]);

    data = vsi_nn_ConvertTensorToData(NULL, max_iteration_tensor);
    TEST_CHECK_PTR(data, final);
    max_iteration = ((int32_t *)data)[0];
    vsi_nn_safe_free(data);

    for(i=0; i<max_iteration; i++)
    {
        status = vsi_nn_CopyDataToTensor(graph, iteration_index, &i);
        TEST_CHECK_STATUS(status, final);

        status = vsi_nn_RunGraph(graph);
        TEST_CHECK_STATUS(status, final);

        /*
            Loop Graph inputs: iteration_index, iteration_cond_in, loop_vars...
            Loop Graph outputs: iteration_cond_out, loop_vars...
        */
        data = vsi_nn_ConvertTensorToData(graph, iteration_cond_out);
        TEST_CHECK_PTR(data, final);
        cond = ((int8_t *)data)[0];
        vsi_nn_safe_free(data);
        if(cond == FALSE)
        {
            break;
        }

        // Update condition
        status = vsi_nn_CopyTensorViaGraphs(
            graph, graph->output.tensors[0],
            graph, graph->input.tensors[1]
        );
        TEST_CHECK_STATUS(status, final);
        for(j=0; j<loop_var_num; j++)
        {
            // Update loop_vars
            status = vsi_nn_CopyTensorViaGraphs(
                graph, graph->output.tensors[j + 1],
                graph, graph->input.tensors[j + 2]
            );
            TEST_CHECK_STATUS(status, final);
        }
    }

final:
    vsi_nn_safe_free(data);
    return status;
} /* vsi_nn_ExecuteGraphLoop() */

typedef enum {
    VSI_NN_ENABLE_I8TOU8 = 0,
    VSI_NN_ENABLE_OPCHECK,
    VSI_SAVE_FILE_TYPE,
    VSI_USE_IMAGE_PROCESS,
    VSI_NN_LOG_LEVEL,
    VSI_NN_ENABLE_CONCAT_OPTIMIZE,
    VSI_NN_ENABLE_DATACONVERT_OPTIMIZE,
    VSI_VX_ENABLE_STREAM_PROCESSOR,
    VSI_NN_FORCE_RGB888_OUT_NHWC,
    VSI_NN_ENABLE_SLICE_OPTIMIZE,
    VSI_VX_ENABLE_BATCH_OPT,
    VIV_VX_ENABLE_SHADER,
    VSI_USE_FROM_HANDLE,
    VIV_VX_ENABLE_GRAPH_TRANSFORM
} VSI_PUBLIC_TYPE vsi_nn_runtime_variable;

typedef struct {
    const char* key;
    int32_t value;
} VSI_PUBLIC_TYPE keyValuePair;

char* vsi_nn_GetRunTimeVariable
    (
    const vsi_nn_graph_t* graph,
    const char* key
    )
{
    int32_t isVaid = 1;
    int32_t value = -1;
#define varSize 256
    char* value_str = (char*)malloc(sizeof(char) * varSize);
    CHECK_PTR_FAIL_GOTO(value_str, "Create value_str fail.", final);
    CHECK_PTR_FAIL_GOTO(graph, "Graph is NULL!", final);
    memset(value_str, 0, varSize);
    char tmp_value[varSize] = {0};
    VSI_UNREFERENCED(tmp_value);
    vsi_nn_runtime_option_t* options = ((vsi_nn_graph_prv_t*)graph)->options;
    switch (vsi_nn_GetVariable(key))
    {
        case VIV_VX_ENABLE_SHADER:
            value =options->enable_shader;
            break;
        case VSI_NN_ENABLE_OPCHECK:
            value = options->enable_opcheck;
            break;
        case VSI_NN_ENABLE_I8TOU8:
            value = options->enable_i8_to_u8;
            break;
        case VSI_VX_ENABLE_STREAM_PROCESSOR:
            value = options->enable_stream_processor;
            break;
        case VSI_VX_ENABLE_BATCH_OPT:
            value = options->enable_batch_opt;
            break;
        case VSI_NN_FORCE_RGB888_OUT_NHWC:
            value = options->enable_rgb88_planar_nhwc;
            break;
        case VSI_SAVE_FILE_TYPE:
            value = options->enable_save_file_type;
            break;
        case VSI_NN_ENABLE_CONCAT_OPTIMIZE:
            value = options->enable_concat_optimize;
            break;
        case VSI_NN_ENABLE_SLICE_OPTIMIZE:
            value = options->enable_slice_optimize;
            break;
        case VSI_USE_IMAGE_PROCESS:
            if (options->enable_use_image_process != -1)
            {
                value = options->enable_use_image_process;
            }
            else
            {
                isVaid = 0;
            }
            break;
        case VSI_USE_FROM_HANDLE:
            if (options->enable_use_from_handle != -1)
            {
                value = options->enable_use_from_handle;
            }
            else
            {
                isVaid = 0;
            }
            break;
        default:
            isVaid = 0;
            VSILOGE("Not support this key: %s.", key);
    }
    if (isVaid == 1)
    {
        snprintf(tmp_value, varSize, "%d", value);
        memcpy(value_str, tmp_value, varSize);
    } else
    {
        goto final;
    }
#undef varSize
    return value_str;
final:
#undef varSize
    vsi_nn_safe_free(value_str);
    return value_str;
}

vsi_status vsi_nn_SetRunTimeVariable
    (
    vsi_nn_graph_t* graph,
    const char* key,
    const char* value
     )
{
    vsi_status status = VSI_SUCCESS;
    size_t size = 1;  // placeholder, not used in vxSetGraphAttribute.
    if (graph == NULL)
    {
        status = VSI_FAILURE;
        return status;
    }
    vsi_nn_runtime_option_t* options = ((vsi_nn_graph_prv_t*)graph)->options;
    VSI_UNREFERENCED(size);
    if (vsi_nn_getenv(key) == NULL)
    {
        switch (vsi_nn_GetVariable(key) )
        {
            case VIV_VX_ENABLE_SHADER:
                options->enable_shader = atoi(value);
                break;
            case VSI_NN_ENABLE_OPCHECK:
                options->enable_opcheck = atoi(value);
                break;
            case VSI_NN_ENABLE_I8TOU8:
                options->enable_i8_to_u8 = atoi(value);
                break;
            case VSI_VX_ENABLE_STREAM_PROCESSOR:
                options->enable_stream_processor = atoi(value);
                options->config.support_stream_processor = atoi(value);
                status = query_hardware_caps_runtime(graph->ctx, options);
                break;
            case VSI_VX_ENABLE_BATCH_OPT:
                options->enable_batch_opt = atoi(value);
                break;
            case VSI_NN_FORCE_RGB888_OUT_NHWC:
                options->enable_rgb88_planar_nhwc = atoi(value);
                break;
            case VSI_NN_ENABLE_CONCAT_OPTIMIZE:
                options->enable_concat_optimize = atoi(value);
                break;
            case VSI_NN_ENABLE_DATACONVERT_OPTIMIZE:
                options->enable_dataconvert_optimize = atoi(value);
                break;
            case VSI_NN_ENABLE_SLICE_OPTIMIZE:
                options->enable_slice_optimize = atoi(value);
                break;
            case VSI_SAVE_FILE_TYPE:
                options->enable_save_file_type = atoi(value);
                break;
            case VSI_USE_IMAGE_PROCESS:
                options->enable_use_image_process = atoi(value);
                break;
            case VSI_USE_FROM_HANDLE:
                options->enable_use_from_handle = atoi(value);
                break;
            case VIV_VX_ENABLE_GRAPH_TRANSFORM:
#ifdef VX_GRAPH_TRANSFORM_OPTION_SUPPORT
                if (graph && graph->g) {
                    status = vxSetGraphAttribute(
                        graph->g, VX_GRAPH_VSI_TRANSFORM_OPTIONS, value, size);
                }
#else
                status = VSI_FAILURE;
                VSILOGE("VX_GRAPH_TRANSFORM_OPTION_SUPPORT is not defined, please check driver version.");
#endif
                break;
            default:
#ifdef VX_GRAPH_ENV_SUPPORT
                status = vxSetGraphEnv(graph->g, key, value);
#else
                status = VSI_FAILURE;
                VSILOGE("VX_GRAPH_ENV_SUPPORT is not defined, please check driver version.");
#endif
                break;
        }
    }
    return status;
}

int32_t vsi_nn_GetVariable(const char* variableKey) {
    keyValuePair dict[] = {
        {"VSI_NN_ENABLE_I8TOU8", VSI_NN_ENABLE_I8TOU8},
        {"VSI_NN_ENABLE_OPCHECK", VSI_NN_ENABLE_OPCHECK},
        {"VSI_SAVE_FILE_TYPE", VSI_SAVE_FILE_TYPE},
        {"VSI_USE_IMAGE_PROCESS", VSI_USE_IMAGE_PROCESS},
        {"VSI_NN_ENABLE_CONCAT_OPTIMIZE", VSI_NN_ENABLE_CONCAT_OPTIMIZE},
        {"VSI_NN_ENABLE_DATACONVERT_OPTIMIZE", VSI_NN_ENABLE_DATACONVERT_OPTIMIZE},
        {"VSI_VX_ENABLE_STREAM_PROCESSOR", VSI_VX_ENABLE_STREAM_PROCESSOR},
        {"VSI_NN_FORCE_RGB888_OUT_NHWC", VSI_NN_FORCE_RGB888_OUT_NHWC},
        {"VSI_NN_ENABLE_SLICE_OPTIMIZE", VSI_NN_ENABLE_SLICE_OPTIMIZE},
        {"VSI_VX_ENABLE_BATCH_OPT", VSI_VX_ENABLE_BATCH_OPT},
        {"VIV_VX_ENABLE_SHADER", VIV_VX_ENABLE_SHADER},
        {"VSI_USE_FROM_HANDLE", VSI_USE_FROM_HANDLE},
        {"VIV_VX_ENABLE_GRAPH_TRANSFORM", VIV_VX_ENABLE_GRAPH_TRANSFORM},
        {NULL, -1}
    };
    for (int32_t i = 0; dict[i].key != NULL; i++) {
        if (strcmp(dict[i].key, variableKey) == 0) {
            return dict[i].value;
        }
    }
    return -1;
}

OVXLIB_API char* vsi_nn_GenerateGraphJson
    (
    vsi_nn_graph_t* graph
    )
{
    char* json = NULL;
    VSI_UNREFERENCED(graph);
#ifdef VX_GENERATE_GRAPH_JSON_API_SUPPORT
    if (graph && graph->g)
    {
        json = vxGenerateGraphJson(graph->g);
    }
#endif
    return json;
}

OVXLIB_API vsi_status vsi_nn_ReleaseGraphJson
    (
    char* json
    )
{
    vsi_status status = VSI_FAILURE;
    VSI_UNREFERENCED(json);
#ifdef VX_GENERATE_GRAPH_JSON_API_SUPPORT
    if (json) {
        status = vxReleaseGraphJson(json);
    }
#endif

    return status;
}