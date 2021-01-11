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

#include "vsi_nn_graph_optimization.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_test.h"


static vsi_bool _is_asymm_int8_norm_tensor
    (
        vsi_nn_tensor_t * tensor
    )
{
    vsi_bool ret = FALSE;

    ret = ( tensor != NULL
   && tensor->attr.vtl == FALSE && tensor->attr.is_const == FALSE
   && tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT8
   && tensor->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC);

    return ret;
}/* _is_asymm_int8_norm_tensor() */

static vsi_bool _is_asymm_int8_const_tensor
    (
        vsi_nn_tensor_t * tensor
    )
{
    vsi_bool ret = FALSE;

    ret = ( tensor != NULL
   && tensor->attr.is_const == TRUE
   && tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT8
   && tensor->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC);

    return ret;
}/* _is_asymm_int8_const_tensor() */

static vsi_bool _is_asymm_int8_virtual_tensor
    (
        vsi_nn_tensor_t * tensor
    )
{
    vsi_bool ret = FALSE;

    ret = ( tensor != NULL
   && tensor->attr.vtl == TRUE
   && tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT8
   && tensor->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC);

    return ret;
}/* _is_asymm_int8_virtual_tensor() */

static vsi_status _add_forward_node
    (
    vsi_nn_graph_t* graph,
    vsi_nn_node_t** first_node,
    uint32_t nodes_count,
    vsi_nn_node_t* node,
    vsi_nn_tensor_id_t input,
    vsi_nn_tensor_id_t output
    )
{
    uint32_t i = 0;
    uint32_t j = 0;

    /* Reconnect node tensors */
    for(i = 0; i < nodes_count; i++)
    {
        for(j = 0; j < first_node[i]->input.num; j++)
        {
            if(first_node[i]->input.tensors[j] == input)
            {
                first_node[i]->input.tensors[j] = output;
            }
        }
    }

    node->input.tensors[0] = input;
    node->output.tensors[0] = output;

    return VSI_SUCCESS;
}/* _add_forward_node() */

static vsi_status _add_backward_node
    (
    vsi_nn_graph_t* graph,
    vsi_nn_node_t* last_node,
    vsi_nn_node_t* node,
    vsi_nn_tensor_id_t input,
    vsi_nn_tensor_id_t output
    )
{
    uint32_t i = 0;

    /* Reconnect node output tensors */
    for(i = 0; i < (int32_t)last_node->output.num; i++)
    {
        if(last_node->output.tensors[i] == output)
        {
            last_node->output.tensors[i] = input;
            break;
        }
    }

    node->input.tensors[0] = input;
    node->output.tensors[0] = output;

    return VSI_SUCCESS;
}/* _add_backward_node() */

static vsi_status _add_dataconvert_node
    (
    vsi_nn_graph_t* graph,
    uint32_t idx,
    vsi_nn_opt_direction_e direction,
    vsi_nn_node_t** nodes,
    uint32_t nodes_count,
    vsi_nn_tensor_id_t input,
    vsi_nn_tensor_id_t output
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_node_t* node = NULL;

    /* Add dataconvert node */
    node = vsi_nn_AddNode(graph, VSI_NN_OP_DATACONVERT, 1, 1, NULL);
    node->uid = (uint32_t)(VSI_NN_DATACONVERT_NODE_UID_BASE) + idx;
    if( NULL == node ) {
        status = VSI_FAILURE;
        goto final;
    }

    if( direction == VSI_NN_OPTIMIZE_FORWARD )
    {
        /* Reconnect node input tensors */
        VSILOGD("add a dataconvert op to input norm tensor[%d] ", input);
        status = _add_forward_node(graph, nodes, nodes_count, node, input, output);
    }
    else
    {
        /* Reconnect node output tensors */
        VSILOGD("add a dataconvert op to output norm tensor[%d] ", output);
        status = _add_backward_node(graph, nodes[0], node, input, output);
    }

final:
    return status;
} /* _add_dataconvert_node() */

static void _get_graph_input_asymm_int8_norm_tensor
    (
    vsi_nn_graph_t* graph,
    uint32_t* count,
    vsi_nn_tensor_id_t *tensor_ids,
    uint32_t* valid_count
    )
{
    vsi_nn_node_t* node = NULL;
    uint32_t i = 0, j = 0, k = 0;
    uint32_t tensor_count = 0;
    uint32_t id_count = 0;

    for(i = 0; i < graph->node_num; i++)
    {
        node = vsi_nn_GetNode(graph, i);
        for(j = 0; j < node->input.num; j++)
        {
            vsi_nn_tensor_id_t id = node->input.tensors[j];
            vsi_nn_tensor_t * tensor = vsi_nn_GetTensor(graph, id);
            if (_is_asymm_int8_norm_tensor(tensor))
            {
                if(tensor_ids != NULL)
                {
                    for ( k = 0; k < id_count; k++)
                    {
                        if (tensor_ids[k] == id)
                            break;
                    }
                    if (k == id_count)
                    {
                        tensor_ids[id_count ++] = id;
                    }

                }
                tensor_count += 1;
            }
        }
    }

    if(count != NULL)
    {
        *count = tensor_count;
    }

    if(valid_count != NULL)
    {
        *valid_count = id_count;
    }
} /* _get_graph_input_asymm_int8_norm_tensor() */

static void _get_graph_output_asymm_int8_norm_tensor
    (
    vsi_nn_graph_t* graph,
    uint32_t* count,
    vsi_nn_tensor_id_t *tensor_ids
    )
{
    vsi_nn_node_t* node = NULL;
    uint32_t i = 0, j = 0;
    uint32_t tensor_count = 0;

    for(i = 0; i < graph->node_num; i++)
    {
        node = vsi_nn_GetNode(graph, i);
        for(j = 0; j < node->output.num; j++)
        {
            vsi_nn_tensor_id_t id = node->output.tensors[j];
            vsi_nn_tensor_t * tensor = vsi_nn_GetTensor(graph, id);
            if (_is_asymm_int8_norm_tensor(tensor))
            {
                if(tensor_ids != NULL)
                {
                   tensor_ids[tensor_count] = id;
                }
                tensor_count += 1;
            }
        }
    }

    if(count != NULL)
    {
        *count = tensor_count;
    }
} /* _get_graph_output_asymm_int8_norm_tensor() */

static vsi_status _add_graph_dataconvert_for_int8
    (
    vsi_nn_graph_t* graph,
    vsi_bool *dirty
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_tensor_attr_t attr;
    uint32_t input_count;
    uint32_t input_valid_count = 0;
    vsi_nn_tensor_id_t *input_ids = NULL;
    vsi_nn_node_t*** input_nodes = NULL;
    uint32_t i = 0;
    uint32_t output_count;
    vsi_nn_tensor_id_t *output_ids = NULL;
    vsi_nn_node_t** output_nodes = NULL;
    uint32_t dataconvert_idx = 0;

    _get_graph_input_asymm_int8_norm_tensor(graph, &input_count, NULL, NULL);

    if(input_count != 0)
    {
        input_ids = (vsi_nn_tensor_id_t *)malloc(sizeof(vsi_nn_tensor_id_t) * input_count);
        _get_graph_input_asymm_int8_norm_tensor(graph, NULL, input_ids, &input_valid_count);

        if ( input_valid_count > 0 )
        {
            input_nodes = (vsi_nn_node_t***)malloc(sizeof(vsi_nn_node_t**) * input_valid_count);
        }

        for ( i = 0; i < input_valid_count; i++)
        {
            uint32_t nodes_count = 0;
            vsi_nn_get_tensor_consumers(graph, input_ids[i], NULL, &nodes_count);

            if(nodes_count > 0)
            {
                input_nodes[i] = (vsi_nn_node_t**)malloc(sizeof(vsi_nn_node_t*)*nodes_count);
                vsi_nn_get_tensor_consumers(graph, input_ids[i], input_nodes[i], NULL);

                *dirty = TRUE;
            }
        }
    }

    _get_graph_output_asymm_int8_norm_tensor(graph, &output_count, NULL);

    if(output_count > 0)
    {
        output_ids = (vsi_nn_tensor_id_t*)malloc(sizeof(vsi_nn_tensor_id_t) * output_count);
        _get_graph_output_asymm_int8_norm_tensor(graph, NULL, output_ids);

        output_nodes = (vsi_nn_node_t**)malloc(sizeof(vsi_nn_node_t*) * output_count);

        for ( i = 0; i < output_count; i++)
        {
            vsi_nn_get_tensor_provider(graph, output_ids[i], &output_nodes[i]);
            *dirty = TRUE;
        }
    }

    if ( input_valid_count > 0 )
    {
        for ( i = 0; i < input_valid_count; i++)
        {
            uint32_t nodes_count = 0;
            vsi_nn_get_tensor_consumers(graph, input_ids[i], NULL, &nodes_count);

            if(nodes_count != 0)
            {
                vsi_nn_tensor_id_t id = input_ids[i];
                vsi_nn_tensor_t * tensor = vsi_nn_GetTensor(graph, id);
                vsi_nn_tensor_id_t output;

               memcpy(&attr, &tensor->attr, sizeof(vsi_nn_tensor_attr_t));
               attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
               attr.dtype.zero_point += 128;
               attr.vtl = TRUE;
               output = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL );

               _add_dataconvert_node(graph, dataconvert_idx ++, VSI_NN_OPTIMIZE_FORWARD,
                   input_nodes[i], nodes_count, id, output);
            }
        }

        if(input_nodes)
        {
            free(input_nodes);
            input_nodes = NULL;
        }
    }

    if ( output_count > 0 )
    {
        for ( i = 0; i < output_count; i++)
        {
            vsi_nn_tensor_id_t id = output_ids[i];
            vsi_nn_tensor_t * tensor = vsi_nn_GetTensor(graph, id);
            vsi_nn_tensor_id_t input;

            memcpy(&attr, &tensor->attr, sizeof(vsi_nn_tensor_attr_t));
            attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
            attr.dtype.zero_point += 128;
            attr.vtl = TRUE;
            input = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL );

            _add_dataconvert_node(graph, dataconvert_idx ++, VSI_NN_OPTIMIZE_BACKWARD,
                &output_nodes[i], 1, input, id);
        }

        if(output_nodes)
        {
            free(output_nodes);
            output_nodes = NULL;
        }
    }

    if (input_ids)
    {
        free(input_ids);
        input_ids = NULL;
    }
    if (output_ids)
    {
        free(output_ids);
        output_ids = NULL;
    }

    return status;
} /* _add_graph_dataconvert_for_int8() */

static vsi_status _add_graph_data_convert
    (
    vsi_nn_graph_t* graph,
    vsi_bool *dirty
    )
{
    vsi_status status = VSI_FAILURE;

    status = _add_graph_dataconvert_for_int8(graph, dirty);
    TEST_CHECK_STATUS(status, final);

final:
    return status;
}/* _add_graph_data_convert() */

static vsi_status _set_raw_tensor_attr
    (
    vx_tensor               tensor,
    vsi_nn_tensor_attr_t    attr,
    const vsi_nn_vxtensor_attr_t vx_attr
    )
{
    vsi_status status;

    status = VSI_SUCCESS;
    if( NULL == tensor )
    {
        return VSI_FAILURE;
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( vx_attr, VSI_NN_TENSOR_ATTR_CONST ) )
    {
        vx_enum data_lifetime;
        if(attr.is_const == TRUE)
        {
            data_lifetime = VX_TENSOR_LIFE_TIME_STATIC;
        }
        else
        {
            data_lifetime = VX_TENSOR_LIFE_TIME_DYNAMIC;
        }
        status = vxSetTensorAttribute(tensor,
                                      VX_TENSOR_LIFETIME,
                                      &data_lifetime,
                                      sizeof(vx_enum));
    }
    if( VSI_SUCCESS == status && vsi_nn_hasattr( vx_attr, VSI_NN_TENSOR_ATTR_HIGH_PRECISION ) )
    {
        vx_enum precision = VX_TENSOR_PRECISION_HIGH;
        status = vxSetTensorAttribute(tensor,
                                      VX_TENSOR_PRECISION,
                                      &precision,
                                      sizeof(vx_enum));
    }

    return status;
}/* _set_raw_tensor_attr() */

static vsi_bool _try_set_const_raw_tensor
    (
    vx_tensor               tensor,
    vsi_nn_tensor_attr_t    attr
    )
{
    vsi_status status;
    vsi_bool ret;
    vsi_nn_vxtensor_attr_t vx_attr;

    ret = TRUE;
    status = VSI_SUCCESS;
    if( TRUE == attr.is_const )
    {
        vx_attr = VSI_NN_TENSOR_ATTR_CONST;
        status = _set_raw_tensor_attr(tensor, attr, vx_attr);
    }
    if( VSI_FAILURE == status )
    {
        ret = FALSE;
    }

    return ret;
} /* _try_set_const_raw_tensor() */

vsi_status vsi_nn_CopyDataToRawTensor
    (
    vsi_nn_graph_t       * graph,
    vx_tensor              tensor,
    uint8_t              * data,
    vsi_nn_tensor_attr_t   attr
    )
{
    vsi_status         status = VSI_FAILURE;
    if( NULL == graph || NULL == data || NULL == tensor )
    {
        return status;
    }

    if( attr.is_created_from_handle )
    {
        uint8_t* ptr = NULL;
        vxSwapTensorHandle( tensor, NULL, (void **)&ptr);
        if ( ptr == NULL )
        {
            VSILOGE("vxSwapTensorHandle fail.");
            return VSI_FAILURE;
        }
        memcpy( ptr, data, vsi_nn_GetTensorSize(attr.size, attr.dim_num,
                    attr.dtype.vx_type));
        status = vxSwapTensorHandle( tensor, ptr, NULL );
        status |= vxFlushHandle( (vx_reference)tensor );
    }
    else
    {
        status = vsi_nn_copy_tensor_patch(tensor, &attr, data, VX_WRITE_ONLY);
    }

    _try_set_const_raw_tensor(tensor, attr);

    return status;
} /* vsi_nn_CopyDataToRawTensor() */

static vx_tensor _create_const_raw_tensor
    (
    vsi_nn_graph_t  * graph,
    uint8_t         * data,
    vsi_nn_tensor_attr_t attr
    )
{
    vx_tensor tensor = NULL;
    vx_tensor_create_params_t params;
    float * scales = NULL;

    memset( &params, 0, sizeof( vx_tensor_create_params_t ) );
    params.num_of_dims = attr.dim_num;
    params.sizes = attr.size;
    params.data_format = (vsi_enum)attr.dtype.vx_type;
    params.quant_format = (vsi_enum)attr.dtype.qnt_type;
    switch( attr.dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        params.quant_data.dfp.fixed_point_pos = (uint8_t)attr.dtype.fl;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        params.quant_data.affine.scale = attr.dtype.scale;
        params.quant_data.affine.zeroPoint = (int32_t)attr.dtype.zero_point;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC:
#ifdef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
        // This is a hack that driver doesn't support const scale
        scales = (float *)malloc(sizeof(float) * attr.dtype.scale_dim);
        memcpy(scales, attr.dtype.scales, attr.dtype.scale_dim * sizeof(float));
        params.quant_data.affinePerChannel.channelDim = attr.dtype.channel_dim;
        params.quant_data.affinePerChannel.scaleCount = attr.dtype.scale_dim;
        params.quant_data.affinePerChannel.scales = scales;
        params.quant_data.affinePerChannel.zeroPoint = NULL;
        params.quant_data.affinePerChannel.zeroPointCount = 0;
        break;
#else
    VSILOGE( "can't support qnt_type VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC." );
#endif
    default:
        break;
    }

    if( TRUE == attr.is_created_from_handle )
    {
        vx_tensor_addressing addr;
        uint32_t stride_size[VSI_NN_MAX_DIM_NUM];
        uint32_t buf_sz;

        buf_sz = vsi_nn_GetStrideSize( &attr, stride_size );
        if( buf_sz > 0 )
        {
            uint32_t align_start_size = graph->handle_manager.align_start_size;
            uint32_t align_block_size = graph->handle_manager.align_block_size;
            if (data == NULL)
            {
                data = vsi_nn_MallocAlignedBuffer(buf_sz, align_start_size,
                    align_block_size);
                attr.is_handle_malloc_by_ovxlib = TRUE;
            }
            else
            {
                attr.is_handle_malloc_by_ovxlib = FALSE;
                if (!vsi_nn_IsBufferAligned(data, align_start_size))
                {
                    VSILOGE( "vsi_nn_IsBufferAligned is FALSE." );
                    if( scales )
                    {
                        free( scales );
                    }
                    return NULL;
                }
            }
            if( data )
            {
                addr = vxCreateTensorAddressing(graph->ctx->c,
                    attr.size, stride_size, (vx_uint8)attr.dim_num);
#ifdef VX_13_NN_COMPATIBLITY
                tensor = vxCreateTensorFromHandle2(graph->ctx->c,
                    &params, sizeof(vx_tensor_create_params_t),
                    addr, data, VX_MEMORY_TYPE_HOST);
#else
                tensor = vxCreateTensorFromHandle(graph->ctx->c,
                    &params, sizeof(vx_tensor_create_params_t),
                    addr, data, VX_MEMORY_TYPE_HOST);
#endif
                //memset(data, 0x5A, buf_sz);
                vxReleaseTensorAddressing( &addr );
                vxFlushHandle( (vx_reference)tensor );
            }
        }
    }
    else if( FALSE == attr.vtl )
    {
        tensor = vxCreateTensor2( graph->ctx->c,
            &params, sizeof( vx_tensor_create_params_t ) );
    }
    else
    {
        tensor = vxCreateVirtualTensor2( graph->g,
            &params, sizeof( vx_tensor_create_params_t ) );
    }
    if( NULL == tensor )
    {
        VSILOGE( "Create vx tensor fail." );
    }
    if( scales )
    {
        free( scales );
    }

    return tensor;
} /* _create_const_raw_tensor() */

vx_tensor vsi_nn_CreateRawTensorFromData
    (
    vsi_nn_graph_t       * graph,
    uint8_t             * data,
    vsi_nn_tensor_attr_t * attr
    )
{
    vsi_status status;
    vx_tensor  tensor;

    status = VSI_FAILURE;
    tensor = NULL;

    if( NULL == graph || NULL == data || NULL == attr )
    {
        return NULL;
    }

    tensor = _create_const_raw_tensor( graph, data, *attr );

    status = vsi_nn_CopyDataToRawTensor( graph, tensor, data, *attr );

    if( VSI_SUCCESS != status )
    {
        VSILOGE("Create tensor from data fail.");
        if( NULL != tensor )
        {
            vxReleaseTensor( &tensor );
            tensor = NULL;
        }
    }
    return tensor;
}/* vsi_nn_CreateRawTensorFromData() */

static void _convert_const_I8toU8
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_id_t id
    )
{
    uint8_t    * data = NULL;
    vsi_nn_tensor_t * tensor = vsi_nn_GetTensor(graph, id);
    vsi_nn_tensor_attr_t *attr = &tensor->attr;
    uint32_t sz = 0;
    uint32_t i = 0;

    sz = vsi_nn_GetElementNum( tensor );

    data = vsi_nn_ConvertTensorToData( graph, tensor );
    if( NULL == data )
    {
        VSILOGE( "Convert data fail." );
        return ;
    }

    for( i = 0; i < sz; i++ )
    {
        data[i] = data[i] ^ 0x80;
    }

    attr->dtype.vx_type = VSI_NN_TYPE_UINT8;
    attr->dtype.zero_point += 128;

    if ( tensor->t ) vxReleaseTensor(&tensor->t);
    tensor->t = vsi_nn_CreateRawTensorFromData(graph, data, attr);
}/* _convert_const_I8toU8() */

static vsi_status _convert_graph_const_tensor
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t node_num = graph->node_num;
    vsi_nn_node_t* node = NULL;
    uint32_t i = 0;
    uint32_t j = 0;

    for(i = 0; i < node_num; i++)
    {
        node = vsi_nn_GetNode(graph, i);
        for(j = 0; j < node->input.num; j++)
        {
           vsi_nn_tensor_id_t id = node->input.tensors[j];
           vsi_nn_tensor_t * tensor = vsi_nn_GetTensor(graph, id);

           if (_is_asymm_int8_const_tensor(tensor))
           {
               _convert_const_I8toU8(graph, id);
           }
        }
    }

    return status;
} /* _convert_graph_const_tensor() */

static vsi_status _convert_virtual_tensor_attr
    (
    vsi_nn_tensor_t * tensor
    )
{
    if (_is_asymm_int8_virtual_tensor(tensor))
    {
        tensor->attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
        tensor->attr.dtype.zero_point += 128;
    }

    return VSI_SUCCESS;
}/* _convert_virtual_tensor_attr() */

static vsi_status _convert_graph_virtual_tensor
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t node_num = graph->node_num;
    vsi_nn_node_t* node = NULL;
    uint32_t i = 0;
    uint32_t j = 0;

    for(i = 0; i < node_num; i++)
    {
        node = vsi_nn_GetNode(graph, i);
        for(j = 0; j < node->input.num; j++)
        {
           vsi_nn_tensor_id_t id = node->input.tensors[j];
           vsi_nn_tensor_t * tensor = vsi_nn_GetTensor(graph, id);

           status = _convert_virtual_tensor_attr(tensor);
        }

        for(j = 0; j < node->output.num; j++)
        {
           vsi_nn_tensor_id_t id = node->output.tensors[j];
           vsi_nn_tensor_t * tensor = vsi_nn_GetTensor(graph, id);

           status = _convert_virtual_tensor_attr(tensor);
        }
    }

    return status;
} /* _convert_graph_virtual_tensor() */

static vsi_status _graph_optimization_convert_int8_to_uint8
(
    vsi_nn_graph_t* graph,
    vsi_bool *dirty
)
{
    vsi_status status = VSI_FAILURE;
    status = _convert_graph_virtual_tensor(graph);
    TEST_CHECK_STATUS(status, final);

    status = _convert_graph_const_tensor(graph);
    TEST_CHECK_STATUS(status, final);

    status = _add_graph_data_convert(graph, dirty);
    TEST_CHECK_STATUS(status, final);

final:
    return status;
}/* _graph_optimization_convert_int8_to_uint8() */

vsi_status vsi_nn_OptimizeGraph
    (
    vsi_nn_graph_t* graph,
    vsi_bool *dirty
    )
{
    vsi_status status = VSI_FAILURE;

    status = _graph_optimization_convert_int8_to_uint8(graph, dirty);
    TEST_CHECK_STATUS(status, final);

final:
    return status;
} /* vsi_nn_OptimizeGraph() */

