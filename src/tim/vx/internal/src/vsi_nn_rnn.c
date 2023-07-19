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
#include <stdio.h>
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_rnn_prv.h"
#include "vsi_nn_internal_node.h"
#include "vsi_nn_error.h"

/**********************************************************
* MACROS
**********************************************************/
#define RNN_WKSP(_GRAPH) ( (vsi_nn_rnn_wksp_t *)((_GRAPH)->rnn_wksp) )

/**********************************************************
* LOCAL FUNCTIONS
**********************************************************/
static vsi_status internal_buffer_init
    (
    vsi_nn_rnn_internal_buffer_t* buffer,
    vsi_nn_tensor_t* tensor,
    float default_value
    )
{
    vsi_status  status      = VSI_FAILURE;
    vsi_size_t    element_num = 0;
    vsi_size_t    i           = 0;
    uint32_t    stride      = 0;
    vsi_size_t    data_size   = 0;
    uint8_t*    data        = NULL;

    if( NULL == tensor )
    {
        VSILOGE("input tensor is NULL.");
        return status;
    }

    if( TRUE == tensor->attr.vtl )
    {
        VSILOGE("Internal tensors cannot be dumpped.");
        return status;
    }

    if( NULL == buffer )
    {
        VSILOGE("Internal buffer is NULL.");
        return status;
    }

    memcpy(&buffer->attr, &tensor->attr, sizeof(tensor->attr));
    data_size = vsi_nn_GetTensorSize( buffer->attr.size, buffer->attr.dim_num, buffer->attr.dtype.vx_type );
    element_num = vsi_nn_GetElementNum(tensor);
    stride = vsi_nn_TypeGetBytes( tensor->attr.dtype.vx_type );

    data = (uint8_t *)malloc(data_size);
    if ( NULL == data )
    {
        VSILOGE("Out of memoery.");
        goto error;
    }

    /* init data with zero */
    for( i = 0; i < element_num; i++ )
    {
        status = vsi_nn_Float32ToDtype(default_value, data + i * stride, &buffer->attr.dtype);
        if( VSI_SUCCESS != status )
        {
            VSILOGE("Convert default value to dtype fail");
            goto error;
        }
    }

    buffer->data = data;
    buffer->data_size = data_size;

error:
    if( VSI_SUCCESS != status )
    {
        vsi_nn_safe_free(data);
    }
    return status;
} /* internal_buffer_init() */

static vsi_status internal_buffer_deinit
    (
    vsi_nn_rnn_internal_buffer_t* buffer
    )
{
    vsi_status status = VSI_FAILURE;

    if( NULL == buffer )
    {
        VSILOGE("Internal buffer is NULL.");
        return status;
    }

    vsi_nn_safe_free( buffer->data );

    return VSI_SUCCESS;
} /* internal_buffer_deinit() */

static vsi_status internal_buffer_copy_to_tensor
    (
    const vsi_nn_graph_t* graph,
    vsi_nn_rnn_internal_buffer_t* buffer,
    vsi_nn_tensor_id_t tensorid
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_size_t request_data_size = 0;
    vsi_nn_tensor_t* tensor = NULL;

    if( NULL == buffer )
    {
        VSILOGE("Internal buffer is NULL.\n");
        return status;
    }

    tensor = vsi_nn_GetTensor( graph, tensorid );
    if ( NULL == tensor )
    {
        VSILOGE("tensor is NULL.");
        return status;
    }
    request_data_size = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num, tensor->attr.dtype.vx_type );
    if( request_data_size != buffer->data_size )
    {
        VSILOGE("Internal buffer size error.\n");
        return status;
    }

    status = vsi_nn_CopyDataToTensor( graph, tensor, buffer->data );

    return status;
} /* internal_buffer_copy_to_tensor() */

static vsi_status internal_buffer_copy_from_tensor
    (
    const vsi_nn_graph_t* graph,
    vsi_nn_rnn_internal_buffer_t* buffer,
    vsi_nn_tensor_id_t tensorid
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_size_t request_data_size = 0;
    uint8_t* data = NULL;
    vsi_nn_tensor_t* tensor = NULL;

    if( NULL == buffer )
    {
        VSILOGE("Internal buffer is NULL.\n");
        return status;
    }

    tensor = vsi_nn_GetTensor( graph, tensorid );
    CHECK_PTR_FAIL_GOTO( tensor, "Get tensor fail.", final );
    request_data_size = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num, tensor->attr.dtype.vx_type );
    if( request_data_size != buffer->data_size )
    {
        VSILOGE("Internal buffer size error.\n");
        return status;
    }

    data = vsi_nn_ConvertTensorToData( graph, tensor );
    if( buffer->data && data )
    {
        memcpy( buffer->data, data, request_data_size );
        status = VSI_SUCCESS;
    }

final:
    vsi_nn_safe_free( data );

    return status;
} /* internal_buffer_copy_from_tensor() */

static vsi_status _swap_rnn_tensor_handle
    (
    const vsi_nn_graph_t* graph,
    vsi_nn_tensor_id_t output_id,
    vsi_nn_tensor_id_t input_id
    )
{
    vsi_nn_tensor_t* tensor_out = NULL;
    vsi_nn_tensor_t* tensor_in = NULL;

    tensor_out = vsi_nn_GetTensor( graph, output_id );
    tensor_in = vsi_nn_GetTensor( graph, input_id );

    return vsi_nn_SwapTensorHandle( tensor_out, tensor_in );
} /* _swap_rnn_tensor_handle() */

/**********************************************************
* PUBLIC FUNCTIONS
**********************************************************/
vsi_status vsi_nn_rnn_feed_internal_state
    (
    const vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_rnn_connection_t * cur_conn = NULL;
    uint32_t i = 0;

    /* copy previous data from internal buffer to related input tensors */
    if( NULL != graph && NULL != graph->rnn_wksp )
    {
        /* don't copy/swap for first inference */
        if( RNN_WKSP(graph)->is_first_run )
        {
            RNN_WKSP(graph)->is_first_run = FALSE;
        }
        else
        {
            cur_conn = RNN_WKSP(graph)->external_connection_list;
            while( NULL != cur_conn && VSI_SUCCESS == status )
            {
                if( cur_conn->tensor_swappable )
                {
                    status = _swap_rnn_tensor_handle( graph, cur_conn->connection.output,
                                cur_conn->connection.inputs[0] );
                    if( VSI_SUCCESS != status )
                    {
                        VSILOGE("Swap handle of RNN input/output fail.");
                        break;
                    }
                }
                else
                {
                    for( i = 0; i < cur_conn->connection_inputs_count; i++ )
                    {
                        vsi_nn_tensor_id_t input = cur_conn->connection.inputs[i];

                        status = internal_buffer_copy_to_tensor( graph, &cur_conn->buffer, input );
                        if( VSI_SUCCESS != status )
                        {
                            break;
                        }
                    }
                }
                cur_conn = (vsi_nn_rnn_connection_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)cur_conn );
            }
        }
    }

    return status;
} /* vsi_nn_rnn_feed_internal_state() */

vsi_status vsi_nn_rnn_save_internal_state
    (
    const vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_rnn_connection_t * cur_conn = NULL;

    if( VSI_SUCCESS == status )
    {
        /* copy tensors' data to internal buffer */
        if( NULL != graph->rnn_wksp )
        {
            cur_conn = RNN_WKSP(graph)->external_connection_list;
            while( NULL != cur_conn && VSI_SUCCESS == status )
            {
                if( !cur_conn->tensor_swappable )
                {
                    status = internal_buffer_copy_from_tensor( graph,
                                &cur_conn->buffer, cur_conn->connection.output );
                }
                cur_conn = (vsi_nn_rnn_connection_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)cur_conn );
            }
        }
    }

    return status;
} /* vsi_nn_rnn_save_internal_state() */

vsi_status vsi_nn_rnn_DeinitWksp
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_rnn_connection_t * cur_conn = NULL;

    if( NULL == graph )
    {
        status = VSI_FAILURE;
        return status;
    }

    if( NULL == graph->rnn_wksp )
    {
        return status;
    }

    while( NULL != RNN_WKSP(graph)->external_connection_list )
    {
        cur_conn = (vsi_nn_rnn_connection_t *)vsi_nn_LinkListPopStart(
            (vsi_nn_link_list_t **)&RNN_WKSP(graph)->external_connection_list );
        internal_buffer_deinit( &cur_conn->buffer );
        vsi_nn_safe_free( cur_conn );
    }

    vsi_nn_safe_free( graph->rnn_wksp );

    return status;
} /* vsi_nn_rnn_DeinitWksp() */

vsi_status vsi_nn_rnn_InitWksp
    (
    vsi_nn_graph_t* graph,
    const vsi_nn_rnn_external_connection_t* connections,
    uint32_t connections_count,
    void* user_data
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t i = 0;
    uint32_t j = 0;
    vsi_nn_rnn_connection_t* cur_conn = NULL;
    vsi_nn_tensor_t* output_tensor = NULL;
    vsi_nn_tensor_t* input_tensor = NULL;

    if( NULL == graph )
    {
        status = VSI_FAILURE;
        return status;
    }

    vsi_nn_rnn_DeinitWksp( graph );

    graph->rnn_wksp = malloc( sizeof( vsi_nn_rnn_wksp_t ) );
    if( NULL == graph->rnn_wksp )
    {
        VSILOGE("Malloc memory for rnn_wksp fail, Out of memory.");
        status = VSI_FAILURE;
        return status;
    }

    memset( graph->rnn_wksp, 0x00, sizeof( vsi_nn_rnn_wksp_t ) );
    RNN_WKSP(graph)->user_data = user_data;
    RNN_WKSP(graph)->is_first_run = TRUE;
    for( i = 0; i < connections_count; i++ )
    {
        cur_conn = (vsi_nn_rnn_connection_t *)malloc( sizeof( vsi_nn_rnn_connection_t ) );
        if( NULL == cur_conn )
        {
            VSILOGE("Malloc memory for connection fail, Out of memory.");
            status = VSI_FAILURE;
            break;
        }
        memset( cur_conn, 0x00, sizeof( vsi_nn_rnn_connection_t ) );
        memcpy( &cur_conn->connection, &connections[i], sizeof( connections[i] ) );

        output_tensor = vsi_nn_GetTensor( graph, cur_conn->connection.output );
        CHECK_PTR_FAIL_GOTO( output_tensor, "Get tensor fail.", OnError );

        for( j = 0; j < VSI_NN_MAX_RNN_CONNECTION_INPUTS; j++ )
        {
            if( VSI_NN_TENSOR_ID_NA == cur_conn->connection.inputs[j] )
            {
                break;
            }
            /* make sure input tensors have the same size and dtype with output tensor */
            input_tensor = vsi_nn_GetTensor( graph, cur_conn->connection.inputs[j] );
            CHECK_PTR_FAIL_GOTO( input_tensor, "Get tensor fail.", OnError );

            if( output_tensor->attr.dim_num != input_tensor->attr.dim_num
                || output_tensor->attr.dtype.vx_type != input_tensor->attr.dtype.vx_type
                || 0 != memcmp(output_tensor->attr.size, input_tensor->attr.size,
                    output_tensor->attr.dim_num * sizeof(output_tensor->attr.size[0])) )
            {
                VSILOGE("The tensors in connections must have the same size and dtype.");
                status = VSI_FAILURE;
                goto OnError;
            }
        }

        if( j == VSI_NN_MAX_RNN_CONNECTION_INPUTS )
        {
            VSILOGE("The count of inputs is greater than maximum value: %d.", VSI_NN_MAX_RNN_CONNECTION_INPUTS);
            status = VSI_FAILURE;
            goto OnError;
        }
        else
        {
            cur_conn->connection_inputs_count = j;
        }

        if( cur_conn->connection_inputs_count == 1 )
        {
            input_tensor = vsi_nn_GetTensor( graph, cur_conn->connection.inputs[0] );
            CHECK_PTR_FAIL_GOTO( input_tensor, "Get tensor fail.", OnError );

            if( output_tensor && output_tensor->attr.is_created_from_handle
                && input_tensor && input_tensor->attr.is_created_from_handle )
            {
                cur_conn->tensor_swappable = TRUE;
            }
        }

        if( !cur_conn->tensor_swappable )
        {
            internal_buffer_init( &cur_conn->buffer,
                vsi_nn_GetTensor( graph, cur_conn->connection.output ), 0.0f );
        }

        vsi_nn_LinkListPushEnd(
            (vsi_nn_link_list_t **)&RNN_WKSP(graph)->external_connection_list,
            (vsi_nn_link_list_t *)cur_conn );
    }

    return status;

OnError:
    vsi_nn_safe_free( cur_conn );
    return VSI_FAILURE;
} /* vsi_nn_rnn_InitWksp() */

vsi_status vsi_nn_rnn_ResetBuffers
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_rnn_connection_t * cur_conn = NULL;

    if( NULL == graph )
    {
        status = VSI_FAILURE;
        return status;
    }

    if( NULL == graph->rnn_wksp )
    {
        return status;
    }

    if( NULL != graph->rnn_wksp )
    {
        RNN_WKSP(graph)->is_first_run = TRUE;
        cur_conn = RNN_WKSP(graph)->external_connection_list;
        while( NULL != cur_conn && VSI_SUCCESS == status )
        {
            if( !cur_conn->tensor_swappable )
            {
                status = internal_buffer_deinit( &cur_conn->buffer );
                status = internal_buffer_init( &cur_conn->buffer,
                    vsi_nn_GetTensor( graph, cur_conn->connection.output ), 0.0f );
            }

            cur_conn = (vsi_nn_rnn_connection_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)cur_conn );
        }
    }

    return status;
} /* vsi_nn_rnn_ResetBuffers() */

vsi_status vsi_nn_rnn_RunGraph
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_rnn_feed_internal_state( graph );

    if( VSI_SUCCESS == status )
    {
        status = vsi_nn_RunGraph( graph );
    }

    if( VSI_SUCCESS == status )
    {
        status = vsi_nn_rnn_save_internal_state( graph );
    }

    return status;
} /* vsi_nn_rnn_RunGraph() */
