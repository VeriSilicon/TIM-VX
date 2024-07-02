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

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#include "vsi_nn/vsi_nn.h"
#include "vsi_nn_context.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"

static VSI_NN_error_e _convert_error_code
    (vsi_status status);

static VSI_NN_error_e _fill_quant_param
    (
    vsi_nn_tensor_attr_t* attr,
    VSI_NN_tensor_type_e dtype,
    const VSI_NN_tensor_quant_param* quant_param
    );

static vsi_nn_type_e _convert_internal_type
    (
    VSI_NN_tensor_type_e dtype
    );

static vsi_nn_pad_e _convert_implicit_padding
    (
    VSI_NN_implicit_padding_e padding
    );

static vsi_nn_pad_mode_e _convert_padding_mode
    (
    VSI_NN_padding_mode_e mode
    );

static int32_t _convert_rounding_type
    (
    VSI_NN_rounding_e rounding
    );

static int32_t _convert_rounding_type
    (
    VSI_NN_rounding_e rounding
    )
{
    switch(rounding)
    {
        case VSI_NN_ROUNDING_FLOOR:
            return VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;
        case VSI_NN_ROUNDING_CEIL:
            return VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING;
        default:
            assert(FALSE);
            break;
    }
    return -1;
} /* _convert_rounding_type() */

static vsi_nn_pad_e _convert_implicit_padding
    (
    VSI_NN_implicit_padding_e padding
    )
{
    switch(padding)
    {
        case VSI_NN_IMPLICIT_PADDING_NONE:
            return VSI_NN_PAD_AUTO;
        case VSI_NN_IMPLICIT_PADDING_VALID:
            return VSI_NN_PAD_VALID;
        case VSI_NN_IMPLICIT_PADDING_SAME:
            return VSI_NN_PAD_SAME;
        default:
            return VSI_NN_PAD_AUTO;
    }
} /* _convert_implicit_padding() */

static vsi_nn_pad_mode_e _convert_padding_mode
    (
    VSI_NN_padding_mode_e mode
    )
{
    switch(mode)
    {
        case VSI_NN_PADDING_MODE_CONSTANT:
            return VSI_NN_PAD_MODE_CONSTANT;
        case VSI_NN_PADDING_MODE_REFLECT:
            return VSI_NN_PAD_MODE_REFLECT;
        case VSI_NN_PADDING_MODE_SYMMETRIC:
            return VSI_NN_PAD_MODE_SYMMETRIC;
        case VSI_NN_PADDING_MODE_REPLICATE:
            return VSI_NN_PAD_MODE_REPLICATE;
        default:
            assert(FALSE);
            break;
    }
    return VSI_NN_PAD_MODE_CONSTANT;
} /* _convert_padding_mode() */

static vsi_nn_type_e _convert_internal_type
    (
    VSI_NN_tensor_type_e dtype
    )
{
    switch(dtype)
    {
        case VSI_NN_TENSOR_FLOAT16:
            return VSI_NN_TYPE_FLOAT16;
        case VSI_NN_TENSOR_FLOAT32:
            return VSI_NN_TYPE_FLOAT32;
        case VSI_NN_TENSOR_FLOAT64:
            return VSI_NN_TYPE_FLOAT64;
        case VSI_NN_TENSOR_BOOL8:
            return VSI_NN_TYPE_BOOL8;
        case VSI_NN_TENSOR_INT8:
            return VSI_NN_TYPE_INT8;
        case VSI_NN_TENSOR_INT16:
            return VSI_NN_TYPE_INT16;
        case VSI_NN_TENSOR_INT32:
            return VSI_NN_TYPE_INT32;
        case VSI_NN_TENSOR_INT64:
            return VSI_NN_TYPE_INT64;
        case VSI_NN_TENSOR_UINT8:
            return VSI_NN_TYPE_UINT8;
        case VSI_NN_TENSOR_UINT16:
            return VSI_NN_TYPE_UINT16;
        case VSI_NN_TENSOR_UINT32:
            return VSI_NN_TYPE_UINT32;
        case VSI_NN_TENSOR_UINT64:
            return VSI_NN_TYPE_UINT64;
        case VSI_NN_TENSOR_BFLOAT16:
            return VSI_NN_TYPE_BFLOAT16;
        default:
            assert(FALSE);
            return VSI_NN_TYPE_FLOAT32;
    }
} /* _convert_internal_type() */

static VSI_NN_error_e _convert_error_code
    (vsi_status status)
{
    switch(status)
    {
        case VSI_SUCCESS:
            return VSI_NN_ERROR_OK;
        case VSI_FAILURE:
            return VSI_NN_ERROR_API_FAIL;
        default:
            return VSI_NN_ERROR_API_FAIL;
    }
} /* _convert_error_code() */

static VSI_NN_error_e _fill_quant_param
    (
    vsi_nn_tensor_attr_t* attr,
    VSI_NN_tensor_type_e dtype,
    const VSI_NN_tensor_quant_param* quant_param
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_VALUED_ERROR;
    assert(attr != NULL);
    assert(quant_param != NULL);
    switch(quant_param->type)
    {
        case VSI_NN_TENSOR_QUANT8_DFP:
            if(dtype == VSI_NN_TENSOR_INT8)
            {
                attr->dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
                attr->dtype.fl = (int8_t)quant_param->param.dfp.fraction_length;
                error = VSI_NN_ERROR_OK;
            }
            break;
        case VSI_NN_TENSOR_QUANT16_DFP:
            if(dtype == VSI_NN_TENSOR_INT16)
            {
                attr->dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
                attr->dtype.fl = (int8_t)quant_param->param.dfp.fraction_length;
                error = VSI_NN_ERROR_OK;
            }
            break;
        case VSI_NN_TENSOR_QUANT32_DFP:
            if(dtype == VSI_NN_TENSOR_INT32)
            {
                attr->dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
                attr->dtype.fl = (int8_t)quant_param->param.dfp.fraction_length;
                error = VSI_NN_ERROR_OK;
            }
            break;
        case VSI_NN_TENSOR_QUANT64_DFP:
            if(dtype == VSI_NN_TENSOR_INT64)
            {
                attr->dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
                attr->dtype.fl = (int8_t)quant_param->param.dfp.fraction_length;
                error = VSI_NN_ERROR_OK;
            }
            break;
        case VSI_NN_TENSOR_QUANT8_SYMM:
            if(dtype == VSI_NN_TENSOR_INT8)
            {
                attr->dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC;
                attr->dtype.scale = quant_param->param.symm.scale;
                error = VSI_NN_ERROR_OK;
            }
            break;
        case VSI_NN_TENSOR_QUANT32_SYMM:
            if(dtype == VSI_NN_TENSOR_INT32)
            {
                attr->dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC;
                attr->dtype.scale = quant_param->param.symm.scale;
                error = VSI_NN_ERROR_OK;
            }
            break;
        case VSI_NN_TENSOR_QUANT8_ASYMM:
            if(dtype == VSI_NN_TENSOR_INT8 || dtype == VSI_NN_TENSOR_UINT8)
            {
                attr->dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
                attr->dtype.scale = quant_param->param.asymm.scale;
                attr->dtype.zero_point = quant_param->param.asymm.zero_point;
                error = VSI_NN_ERROR_OK;
            }
            break;
        case VSI_NN_TENSOR_QUANT8_PERCHANNEL_SYMM:
            if(dtype == VSI_NN_TENSOR_INT8)
            {
                attr->dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC;
                attr->dtype.scales = quant_param->param.perchannel_symm.scales;
                attr->dtype.scale_dim = quant_param->param.perchannel_symm.scale_count;
                attr->dtype.channel_dim = quant_param->param.perchannel_symm.channel_dim;
                error = VSI_NN_ERROR_OK;
            }
            break;
        default:
            break;
    }
    return error;
} /* _fill_quant_param() */

VSI_NN_context* VSI_NN_context_create()
{
    return vsi_nn_CreateContext();
} /* VSI_NN_context_create() */


void VSI_NN_context_release
    (
    _IN VSI_NN_context** ctx_ptr
    )
{
    vsi_nn_ReleaseContext(ctx_ptr);
} /* VSI_NN_context_release() */

VSI_NN_graph* VSI_NN_graph_create
    (
    VSI_NN_context* ctx
    )
{
    return vsi_nn_CreateGraph(ctx, 0, 0);
} /* VSI_NN_graph_create() */

void VSI_NN_graph_release
    (
    _IN VSI_NN_graph** graph_ptr
    )
{
    vsi_nn_ReleaseGraph(graph_ptr);
} /* VSI_NN_graph_release() */

VSI_NN_error_e VSI_NN_graph_identify_input_output
    (
    _IN VSI_NN_graph* graph,
    _IN const VSI_NN_tensor** input_tensors,
    _IN const int32_t input_tensors_num,
    _IN const VSI_NN_tensor** output_tensors,
    _IN const int32_t output_tensors_num
    )
{
    int32_t i;
    if(!vsi_nn_SetGraphInputs(graph, NULL, (uint32_t)input_tensors_num))
    {
        return VSI_NN_ERROR_API_FAIL;
    }
    if(!vsi_nn_SetGraphOutputs(graph, NULL, (uint32_t)output_tensors_num))
    {
        return VSI_NN_ERROR_API_FAIL;
    }
    for(i = 0; i < input_tensors_num; i ++)
    {
        graph->input.tensors[i] = vsi_nn_get_tensor_id(graph, input_tensors[i]);
    }
    for(i = 0; i < output_tensors_num; i ++)
    {
        graph->output.tensors[i] = vsi_nn_get_tensor_id(graph, output_tensors[i]);
    }
    return VSI_NN_ERROR_OK;
} /* VSI_NN_graph_identify_input_output() */

VSI_NN_error_e VSI_NN_graph_verify
    (
    _IN VSI_NN_graph* graph
    )
{
    vsi_status status;

    vsi_nn_PrintGraph(graph);
    status = vsi_nn_SetupGraph(graph, TRUE);
    if(status == VSI_SUCCESS)
    {
        status = vsi_nn_VerifyGraph(graph);
    }
    return _convert_error_code(status);
} /* VSI_NN_graph_verify() */

VSI_NN_error_e VSI_NN_graph_compute
    (
    _IN const VSI_NN_graph* graph
    )
{
    vsi_status status = vsi_nn_RunGraph(graph);
    return _convert_error_code(status);
} /* VSI_NN_graph_compute() */

VSI_NN_tensor* VSI_NN_tensor_create
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor_type_e dtype,
    _IN const int32_t* shape,
    _IN int32_t ndim,
    _IN const VSI_NN_tensor_quant_param* quant_param,
    _IN void* memory,
    _IN size_t memory_size,
    _IN int32_t is_constant
    )
{
    vsi_nn_tensor_id_t id = VSI_NN_TENSOR_ID_NA;
    vsi_nn_tensor_attr_t attr;
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    VSI_NN_tensor * tensor = NULL;
    assert(graph);
    assert(shape);
    assert(ndim <= VSI_NN_MAX_DIM_NUM);
    assert(!is_constant || memory);
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.vtl = FALSE;
    attr.dim_num = (uint32_t)ndim;
    memcpy(attr.size, shape, sizeof(int32_t) * ndim);
    attr.dtype.vx_type = _convert_internal_type(dtype);
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if(quant_param)
    {
        error = _fill_quant_param(&attr, dtype, quant_param);
        if( error != VSI_NN_ERROR_OK )
        {
            return NULL;
        }
    }
    if(is_constant)
    {
        attr.is_const = TRUE;
        id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, memory);
    }
    else
    {
        // TODO: Fixme
        id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, memory);
        //id = vsi_nn_AddTensorFromHandle(graph, VSI_NN_TENSOR_ID_AUTO, &attr, memory);
    }

    if(id != VSI_NN_TENSOR_ID_NA)
    {
        tensor = vsi_nn_GetTensor(graph, id);
    }
    return tensor;
} /* VSI_NN_tensor_create() */

VSI_NN_tensor* VSI_NN_tensor_create_virtual
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor_type_e dtype,
    _IN const VSI_NN_tensor_quant_param* quant_param
    )
{
    vsi_nn_tensor_id_t id = VSI_NN_TENSOR_ID_NA;
    vsi_nn_tensor_attr_t attr;
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    VSI_NN_tensor * tensor = NULL;
    assert(graph);
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.vtl = TRUE;
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.dtype.vx_type = _convert_internal_type(dtype);
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if(quant_param)
    {
        error = _fill_quant_param(&attr, dtype, quant_param);
        if( error != VSI_NN_ERROR_OK )
        {
            return NULL;
        }
    }
    id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);
    if(id != VSI_NN_TENSOR_ID_NA)
    {
        tensor = vsi_nn_GetTensor(graph, id);
    }
    return tensor;
} /* VSI_NN_tensor_create_virtual() */

int32_t VSI_NN_tensor_get_size
    (
    _IN const VSI_NN_tensor* tensor
    )
{
    if(NULL == tensor)
    {
        return 0;
    }
    return (int32_t)vsi_nn_GetElementNum(tensor);
} /* VSI_NN_tensor_get_size() */

int32_t VSI_NN_tensor_get_bytes
    (
    _IN const VSI_NN_tensor* tensor
    )
{
    int32_t sz = 0;
    if(NULL == tensor)
    {
        return 0;
    }
    sz = (int32_t)vsi_nn_GetTensorSize(
                tensor->attr.size,
                tensor->attr.dim_num,
                tensor->attr.dtype.vx_type);
    return sz;
} /* VSI_NN_tensor_get_bytes() */

VSI_NN_error_e VSI_NN_tensor_read
    (
    _IN VSI_NN_tensor* tensor,
    _IN void* memory,
    _IN size_t memory_size
    )
{
    if(VSI_NN_tensor_get_bytes(tensor) != memory_size)
    {
        return VSI_NN_ERROR_VALUED_ERROR;
    }
    vsi_nn_CopyTensorToBuffer(NULL, tensor, memory);
    return VSI_NN_ERROR_OK;
} /* VSI_NN_tensor_read() */

VSI_NN_error_e VSI_NN_tensor_write
    (
    _IN VSI_NN_tensor* tensor,
    _IN void* memory,
    _IN size_t memory_size
    )
{
    vsi_status status;
    if(VSI_NN_tensor_get_bytes(tensor) != memory_size)
    {
        return VSI_NN_ERROR_VALUED_ERROR;
    }
    status = vsi_nn_CopyDataToTensor(NULL, tensor, memory);
    return _convert_error_code(status);
} /* VSI_NN_tensor_write() */

VSI_NN_error_e VSI_NN_tensor_swap
    (
    _IN VSI_NN_tensor* tensor1,
    _IN VSI_NN_tensor* tensor2
    )
{
    vsi_status status;
    status = vsi_nn_SwapTensorHandle(tensor1, tensor2);
    return _convert_error_code(status);
} /* VSI_NN_tensor_swap() */

VSI_NN_error_e VSI_NN_tensor_swap_memory
    (
    _IN VSI_NN_tensor* tensor,
    _IN void* new_memory,
    _INOUT void** old_memory
    )
{
    vsi_status status;
    status = vsi_nn_SwapHandle(tensor, new_memory, old_memory);
    return _convert_error_code(status);
} /* VSI_NN_tensor_swap_memory() */

VSI_NN_error_e VSI_NN_tensor_flush_memory
    (
    _IN const VSI_NN_tensor* tensor
    )
{
    vsi_status status;
    status = vsi_nn_FlushHandle(tensor);
    return _convert_error_code(status);
} /* VSI_NN_tensor_swap_memory() */

VSI_NN_error_e VSI_NN_node_conv_1d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t stride,
    _IN int32_t dilation,
    _IN int32_t pad_front, _IN int32_t pad_end,
    _IN VSI_NN_implicit_padding_e implicit_padding
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !kernel || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_CONV1D, 3, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[3] = {NULL};
        node->nn_param.conv1d.stride = stride;
        node->nn_param.conv1d.pad[0] = pad_front;
        node->nn_param.conv1d.pad[1] = pad_end;
        node->nn_param.conv1d.dilation = dilation;
        node->nn_param.conv1d.pad_type = _convert_implicit_padding(implicit_padding);
        node_inputs[0] = input;
        node_inputs[1] = kernel;
        node_inputs[2] = bias;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 3, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_conv_1d() */

VSI_NN_error_e VSI_NN_node_conv_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t dilation_h, _IN int32_t dilation_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !kernel || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_CONV2D, 3, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[3] = {NULL};
        node->nn_param.conv2d.stride[0] = stride_w;
        node->nn_param.conv2d.stride[1] = stride_h;
        node->nn_param.conv2d.pad[0] = pad_w_front;
        node->nn_param.conv2d.pad[1] = pad_w_end;
        node->nn_param.conv2d.pad[2] = pad_h_front;
        node->nn_param.conv2d.pad[3] = pad_h_end;
        node->nn_param.conv2d.dilation[0] = dilation_w;
        node->nn_param.conv2d.dilation[1] = dilation_h;
        node->nn_param.conv2d.pad_type = _convert_implicit_padding(implicit_padding);
        node_inputs[0] = input;
        node_inputs[1] = kernel;
        node_inputs[2] = bias;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 3, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_conv_2d() */

VSI_NN_error_e VSI_NN_node_depthwise_conv_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t multiplier,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t dilation_h, _IN int32_t dilation_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !kernel || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_CONV2D, 3, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[3] = {NULL};
        node->nn_param.conv2d.stride[0] = stride_w;
        node->nn_param.conv2d.stride[1] = stride_h;
        node->nn_param.conv2d.pad[0] = pad_w_front;
        node->nn_param.conv2d.pad[1] = pad_w_end;
        node->nn_param.conv2d.pad[2] = pad_h_front;
        node->nn_param.conv2d.pad[3] = pad_h_end;
        node->nn_param.conv2d.dilation[0] = dilation_w;
        node->nn_param.conv2d.dilation[1] = dilation_h;
        node->nn_param.conv2d.pad_type = _convert_implicit_padding(implicit_padding);
        node->nn_param.conv2d.multiplier = multiplier;
        node->nn_param.conv2d.group = kernel->attr.size[2];
        node_inputs[0] = input;
        node_inputs[1] = kernel;
        node_inputs[2] = bias;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 3, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_depthwise_conv_2d() */

VSI_NN_error_e VSI_NN_node_grouped_conv_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t group_number,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t dilation_h, _IN int32_t dilation_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !kernel || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_CONV2D, 3, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[3] = {NULL};
        node->nn_param.conv2d.stride[0] = stride_w;
        node->nn_param.conv2d.stride[1] = stride_h;
        node->nn_param.conv2d.pad[0] = pad_w_front;
        node->nn_param.conv2d.pad[1] = pad_w_end;
        node->nn_param.conv2d.pad[2] = pad_h_front;
        node->nn_param.conv2d.pad[3] = pad_h_end;
        node->nn_param.conv2d.dilation[0] = dilation_w;
        node->nn_param.conv2d.dilation[1] = dilation_h;
        node->nn_param.conv2d.pad_type = _convert_implicit_padding(implicit_padding);
        node->nn_param.conv2d.group = group_number;
        node_inputs[0] = input;
        node_inputs[1] = kernel;
        node_inputs[2] = bias;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 3, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_grouped_conv_2d() */

VSI_NN_error_e VSI_NN_node_transposed_conv_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t dilation_h, _IN int32_t dilation_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN int32_t output_pad_h, _IN int32_t output_pad_w
    )
{

    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !kernel || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_DECONVOLUTION, 3, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[3] = {NULL};
        node->nn_param.deconv.stride[0] = stride_w;
        node->nn_param.deconv.stride[1] = stride_h;
        node->nn_param.deconv.pad[0] = pad_w_front;
        node->nn_param.deconv.pad[1] = pad_w_end;
        node->nn_param.deconv.pad[2] = pad_h_front;
        node->nn_param.deconv.pad[3] = pad_h_end;
        node->nn_param.deconv.output_padding[0] = output_pad_w;
        node->nn_param.deconv.output_padding[1] = output_pad_h;
        node_inputs[0] = input;
        node_inputs[1] = kernel;
        node_inputs[2] = bias;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 3, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_transposed_conv_2d() */

/** Pooling */
VSI_NN_error_e VSI_NN_node_average_pool_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t ksize_h, _IN int32_t ksize_w,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding,
    _IN VSI_NN_rounding_e size_rounding
    )
{

    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_POOL, 1, 1, NULL);
    if(node)
    {
        node->nn_param.pool.ksize[0] = ksize_w;
        node->nn_param.pool.ksize[1] = ksize_h;
        node->nn_param.pool.stride[0] = stride_w;
        node->nn_param.pool.stride[1] = stride_h;
        node->nn_param.pool.pad[0] = pad_w_front;
        node->nn_param.pool.pad[1] = pad_w_end;
        node->nn_param.pool.pad[2] = pad_h_front;
        node->nn_param.pool.pad[3] = pad_h_end;
        node->nn_param.pool.pad_type = _convert_implicit_padding(implicit_padding);
        node->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_AVG;
        node->nn_param.pool.round_type = _convert_rounding_type(size_rounding);
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_average_pool_2d() */

VSI_NN_error_e VSI_NN_node_max_pool_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t ksize_h, _IN int32_t ksize_w,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding,
    _IN VSI_NN_rounding_e size_rounding
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_POOL, 1, 1, NULL);
    if(node)
    {
        node->nn_param.pool.ksize[0] = ksize_w;
        node->nn_param.pool.ksize[1] = ksize_h;
        node->nn_param.pool.stride[0] = stride_w;
        node->nn_param.pool.stride[1] = stride_h;
        node->nn_param.pool.pad[0] = pad_w_front;
        node->nn_param.pool.pad[1] = pad_w_end;
        node->nn_param.pool.pad[2] = pad_h_front;
        node->nn_param.pool.pad[3] = pad_h_end;
        node->nn_param.pool.pad_type = _convert_implicit_padding(implicit_padding);
        node->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
        node->nn_param.pool.round_type = _convert_rounding_type(size_rounding);
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_max_pool_2d() */

VSI_NN_error_e VSI_NN_node_l2_pool_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t ksize_h, _IN int32_t ksize_w,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding,
    _IN VSI_NN_rounding_e size_rounding
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_POOL, 1, 1, NULL);
    if(node)
    {
        node->nn_param.pool.ksize[0] = ksize_w;
        node->nn_param.pool.ksize[1] = ksize_h;
        node->nn_param.pool.stride[0] = stride_w;
        node->nn_param.pool.stride[1] = stride_h;
        node->nn_param.pool.pad[0] = pad_w_front;
        node->nn_param.pool.pad[1] = pad_w_end;
        node->nn_param.pool.pad[2] = pad_h_front;
        node->nn_param.pool.pad[3] = pad_h_end;
        node->nn_param.pool.pad_type = _convert_implicit_padding(implicit_padding);
        node->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_L2;
        node->nn_param.pool.round_type = _convert_rounding_type(size_rounding);
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_l2_pool_2d() */

VSI_NN_error_e VSI_NN_node_unpool_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{

    assert(FALSE);
    return VSI_NN_ERROR_API_FAIL;
} /* VSI_NN_node_unpool_2d() */

/** Normalization */
VSI_NN_error_e VSI_NN_node_batch_normalization
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* mean,
    _IN VSI_NN_tensor* variance,
    _IN VSI_NN_tensor* offset,
    _IN VSI_NN_tensor* scale,
    _IN VSI_NN_tensor* output,
    _IN float variance_epsilon
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_BATCH_NORM, 5, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[5] = {NULL};
        node->nn_param.batch_norm.eps = variance_epsilon;
        node_inputs[0] = input;
        node_inputs[1] = mean;
        node_inputs[2] = variance;
        node_inputs[3] = scale;
        node_inputs[4] = offset;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 5, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_batch_normalization() */

VSI_NN_error_e VSI_NN_node_l2_normalization
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_L2_NORMALIZE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.l2_normalize.axis = axis;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_l2_normalization() */

VSI_NN_error_e VSI_NN_node_local_response_normalization
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t depth_radius,
    _IN float bias,
    _IN float alpha,
    _IN float beta,
    _IN int32_t axis
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_LRN2, 1, 1, NULL);
    if(node)
    {
        node->nn_param.lrn.type = VX_CONVOLUTIONAL_NETWORK_NORM_ACROSS_MAPS;
        node->nn_param.lrn.size = axis;
        node->nn_param.lrn.alpha = alpha;
        node->nn_param.lrn.beta = beta;
        node->nn_param.lrn.bias = bias;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_local_response_normalization() */

VSI_NN_error_e VSI_NN_node_instance_normalization
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* offset,
    _IN VSI_NN_tensor* scale,
    _IN VSI_NN_tensor* output,
    _IN float variance_epsilon
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_INSTANCE_NORM, 3, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[3] = {NULL};
        node->nn_param.instancenorm.eps = variance_epsilon;
        node_inputs[0] = input;
        node_inputs[1] = scale;
        node_inputs[2] = offset;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 3, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_instance_normalization() */

/** Math */
VSI_NN_error_e VSI_NN_node_add
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_ADD, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_add() */

VSI_NN_error_e VSI_NN_node_mul
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_MULTIPLY, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node->nn_param.multiply.scale = 1.0f;
        node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
        node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_mul() */

VSI_NN_error_e VSI_NN_node_div
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_DIVIDE, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node->nn_param.divide.scale = 1.0f;
        node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
        node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_div() */

VSI_NN_error_e VSI_NN_node_sub
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SUBTRACT, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_sub() */

VSI_NN_error_e VSI_NN_node_floor
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_FLOOR, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_floor() */

VSI_NN_error_e VSI_NN_node_square
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SQUARE, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_square() */

VSI_NN_error_e VSI_NN_node_sqrt
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SQRT, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_sqrt() */

VSI_NN_error_e VSI_NN_node_rsqrt
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RSQRT, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_rsqrt() */

VSI_NN_error_e VSI_NN_node_matmul
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output,
    _IN int transpose_input1,
    _IN int transpose_input2,
    _IN int transpose_output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_MATRIXMUL, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        node->nn_param.matrixmul.transpose[0] = transpose_input1;
        node->nn_param.matrixmul.transpose[1] = transpose_input2;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_matmul() */

VSI_NN_error_e VSI_NN_node_abs
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_ABS, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_abs() */

VSI_NN_error_e VSI_NN_node_pow
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_POW, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_pow() */

VSI_NN_error_e VSI_NN_node_maximum
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_MAXIMUM, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_maximum() */

VSI_NN_error_e VSI_NN_node_minimum
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_MINIMUM, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_minimum() */

VSI_NN_error_e VSI_NN_node_exp
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_EXP, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_exp() */

VSI_NN_error_e VSI_NN_node_reverse
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* axes,
    _IN int32_t axes_size
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_REVERSE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.reverse.axis = axes;
        node->nn_param.reverse.axis_num = axes_size;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_reverse() */

VSI_NN_error_e VSI_NN_node_transpose
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* perm
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_PERMUTE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.permute.perm = (const uint32_t *)perm;
        node->nn_param.permute.dim_num = input->attr.dim_num;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_transpose() */

VSI_NN_error_e VSI_NN_node_gather
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* indices,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !indices || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_GATHER, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input;
        node_inputs[1] = indices;
        node->nn_param.gather.axis = axis;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_gather() */

VSI_NN_error_e VSI_NN_node_neg
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_NEG, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_neg() */

VSI_NN_error_e VSI_NN_node_reduce_max
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* axes,
    _IN int32_t axes_size,
    _IN int32_t keep_dim
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_REDUCE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.reduce.type = VSI_NN_REDUCE_MAX;
        node->nn_param.reduce.axis = axes;
        node->nn_param.reduce.axis_num = axes_size;
        node->nn_param.reduce.keep_dim = keep_dim;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_reduce_max() */

VSI_NN_error_e VSI_NN_node_reduce_min
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* axes,
    _IN int32_t axes_size,
    _IN int32_t keep_dim
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_REDUCE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.reduce.type = VSI_NN_REDUCE_MIN;
        node->nn_param.reduce.axis = axes;
        node->nn_param.reduce.axis_num = axes_size;
        node->nn_param.reduce.keep_dim = keep_dim;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_reduce_min() */

VSI_NN_error_e VSI_NN_node_reduce_sum
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* axes,
    _IN int32_t axes_size,
    _IN int32_t keep_dim
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_REDUCE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.reduce.type = VSI_NN_REDUCE_SUM;
        node->nn_param.reduce.axis = axes;
        node->nn_param.reduce.axis_num = axes_size;
        node->nn_param.reduce.keep_dim = keep_dim;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_reduce_sum() */

VSI_NN_error_e VSI_NN_node_reduce_mean
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* axes,
    _IN int32_t axes_size,
    _IN int32_t keep_dim
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_REDUCE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.reduce.type = VSI_NN_REDUCE_MEAN;
        node->nn_param.reduce.axis = axes;
        node->nn_param.reduce.axis_num = axes_size;
        node->nn_param.reduce.keep_dim = keep_dim;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_reduce_mean() */

VSI_NN_error_e VSI_NN_node_sin
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SIN, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_sin() */

VSI_NN_error_e VSI_NN_node_tile
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* multiples,
    _IN int32_t multiples_size
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_TILE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.tile.multiples = multiples;
        node->nn_param.tile.multiples_num = multiples_size;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_tile() */

VSI_NN_error_e VSI_NN_node_topk
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_tensor* output_indices,
    _IN int32_t k
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output || !output_indices)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_TOPK, 1, 2, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_outputs[2] = {NULL};
        node_outputs[0] = output;
        node_outputs[1] = output_indices;
        node->nn_param.topk.k = k;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, node_outputs, 2);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_topk() */

/** Logical */
VSI_NN_error_e VSI_NN_node_equal
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RELATIONAL_OPS, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        node->nn_param.relational_ops.op = VSI_NN_RELATIONAL_OPS_EQUAL;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_equal() */

VSI_NN_error_e VSI_NN_node_greater
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RELATIONAL_OPS, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        node->nn_param.relational_ops.op = VSI_NN_RELATIONAL_OPS_GREAT;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_greater() */

VSI_NN_error_e VSI_NN_node_greater_equal
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RELATIONAL_OPS, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        node->nn_param.relational_ops.op = VSI_NN_RELATIONAL_OPS_GREAT_EQUAL;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_greater_equal() */

VSI_NN_error_e VSI_NN_node_less
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RELATIONAL_OPS, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        node->nn_param.relational_ops.op = VSI_NN_RELATIONAL_OPS_LESS;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_less() */

VSI_NN_error_e VSI_NN_node_less_equal
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RELATIONAL_OPS, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        node->nn_param.relational_ops.op = VSI_NN_RELATIONAL_OPS_LESS_EQUAL;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_less_equal() */

VSI_NN_error_e VSI_NN_node_logical_and
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_LOGICAL_OPS, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        node->nn_param.relational_ops.op = VSI_NN_LOGICAL_AND;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_logical_and() */

VSI_NN_error_e VSI_NN_node_logical_or
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_LOGICAL_OPS, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        node->nn_param.relational_ops.op = VSI_NN_LOGICAL_OR;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_logical_or() */

VSI_NN_error_e VSI_NN_node_logical_not
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_LOGICAL_NOT, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_logical_not() */

VSI_NN_error_e VSI_NN_node_not_equal
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RELATIONAL_OPS, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input1;
        node_inputs[1] = input2;
        node->nn_param.relational_ops.op = VSI_NN_RELATIONAL_OPS_NOT_EQUAL;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_not_equal() */

VSI_NN_error_e VSI_NN_node_select
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* condition,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !condition || !input1 || !input2 || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SELECT, 3, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[3] = {NULL};
        node_inputs[0] = condition;
        node_inputs[1] = input1;
        node_inputs[2] = input2;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 3, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_select() */

/** Activation */
VSI_NN_error_e VSI_NN_node_relu
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RELU, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_relu() */

VSI_NN_error_e VSI_NN_node_relu1
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RELU1, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_relu1() */

VSI_NN_error_e VSI_NN_node_relu6
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RELU6, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_relu6() */

VSI_NN_error_e VSI_NN_node_tanh
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN float scale_a,
    _IN float scale_b
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_TANH, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_tanh() */

VSI_NN_error_e VSI_NN_node_sigmoid
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SIGMOID, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_sigmoid() */

VSI_NN_error_e VSI_NN_node_hard_sigmoid
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_HARD_SIGMOID, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_hard_sigmoid() */

VSI_NN_error_e VSI_NN_node_leaky_relu
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN float ratio
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_LEAKY_RELU, 1, 1, NULL);
    if(node)
    {
        node->nn_param.activation.leaky_ratio = ratio;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_leaky_relu() */

VSI_NN_error_e VSI_NN_node_prelu
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* alpha,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_PRELU, 1, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = input;
        node_inputs[1] = alpha;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_prelu() */

VSI_NN_error_e VSI_NN_node_elu
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_ELU, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_elu() */

VSI_NN_error_e VSI_NN_node_soft_relu
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SOFTRELU, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_soft_relu() */

VSI_NN_error_e VSI_NN_node_mish
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_MISH, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_mish() */

/** Misc */
VSI_NN_error_e VSI_NN_node_pad
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_padding_mode_e mode,
    _IN const int32_t* pad_front,
    _IN const int32_t* pad_end,
    _IN int32_t pad_value
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_PAD, 1, 1, NULL);
    if(node)
    {
        node->nn_param.pad.mode = _convert_padding_mode(mode);
        node->nn_param.pad.const_val = pad_value;
        node->nn_param.pad.front_size = (const uint32_t*)pad_front;
        node->nn_param.pad.back_size = (const uint32_t*)pad_end;
        node->nn_param.pad.dim_num = (uint8_t)input->attr.dim_num;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_pad() */

VSI_NN_error_e VSI_NN_node_fully_connected
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !kernel || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_FCL, 3, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[3] = {NULL};
        node->nn_param.fcl.axis = axis;
        node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
        node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
        node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;
        node_inputs[0] = input;
        node_inputs[1] = kernel;
        node_inputs[2] = bias;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 3, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_fully_connected() */

VSI_NN_error_e VSI_NN_node_concate
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* const inputs[],
    _IN int32_t input_num,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    int32_t i;
    vsi_nn_node_t* node = NULL;
    if(!graph || !inputs || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    for(i = 0; i < input_num; i ++)
    {
        if(!inputs[i])
        {
            return VSI_NN_ERROR_UNEXPECTED_NULL;
        }
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_CONCAT, input_num, 1, NULL);
    if(node)
    {
        node->nn_param.concat.axis = axis;
        vsi_nn_SetNodeInputsAndOutputs(node, inputs, input_num, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_concatenation() */

VSI_NN_error_e VSI_NN_node_split
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* const outputs[],
    _IN int32_t output_num,
    _IN const int32_t* slices,
    _IN int32_t slices_size,
    _IN int32_t axis
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    int32_t i;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !outputs)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    for(i = 0; i < output_num; i ++)
    {
        if(!outputs[i])
        {
            return VSI_NN_ERROR_UNEXPECTED_NULL;
        }
    }
    if(output_num != slices_size - 1)
    {
        return VSI_NN_ERROR_VALUED_ERROR;
    }

    node = vsi_nn_AddNode(graph, VSI_NN_OP_SPLIT, 1, output_num, NULL);
    if(node)
    {
        node->nn_param.split.axis = axis;
        node->nn_param.split.slices = (const uint32_t*)slices;
        node->nn_param.split.slices_num = slices_size;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, outputs, output_num);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_split() */

VSI_NN_error_e VSI_NN_node_cast
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_DATACONVERT, 1, 1, NULL);
    if(node)
    {
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_cast() */

VSI_NN_error_e VSI_NN_node_quantize
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    return VSI_NN_node_cast(graph, input, output);
} /* VSI_NN_node_quantize() */

VSI_NN_error_e VSI_NN_node_dequantize
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    )
{
    return VSI_NN_node_cast(graph, input, output);
} /* VSI_NN_node_dequantize() */

VSI_NN_error_e VSI_NN_node_space_to_batch
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* block_size,
    _IN int32_t block_size_num,
    _IN const int32_t* pad_front,
    _IN const int32_t* pad_end
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output || !pad_front || !pad_end)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    if(input->attr.dim_num != 4 || block_size_num != 2)
    {
        return VSI_NN_ERROR_VALUED_ERROR;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SPACE2BATCH, 1, 1, NULL);
    if(node)
    {
        node->nn_param.space2batch.block_size = block_size;
        node->nn_param.space2batch.block_size_num = block_size_num;
        node->nn_param.space2batch.pad[0] = pad_front[0];
        node->nn_param.space2batch.pad[1] = pad_end[0];
        node->nn_param.space2batch.pad[2] = pad_front[1];
        node->nn_param.space2batch.pad[3] = pad_end[1];
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_space_to_batch() */

VSI_NN_error_e VSI_NN_node_batch_to_space
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* block_size,
    _IN int32_t block_size_num,
    _IN const int32_t* crop_front,
    _IN const int32_t* crop_end
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output || !crop_front || !crop_end)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    if(input->attr.dim_num != 4 || block_size_num != 2)
    {
        return VSI_NN_ERROR_VALUED_ERROR;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_BATCH2SPACE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.batch2space.block_size = block_size;
        node->nn_param.batch2space.block_size_num = block_size_num;
        node->nn_param.batch2space.crop[0] = crop_front[0];
        node->nn_param.batch2space.crop[1] = crop_end[0];
        node->nn_param.batch2space.crop[2] = crop_front[1];
        node->nn_param.batch2space.crop[3] = crop_end[1];
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_batch_to_space() */

VSI_NN_error_e VSI_NN_node_space_to_depth
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* block_size,
    _IN int32_t block_size_num,
    _IN const int32_t* pad_front,
    _IN const int32_t* pad_end
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output || !block_size)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    if(pad_front || pad_end)
    {
        // Not support padding yet
        assert(FALSE);
        return VSI_NN_ERROR_VALUED_ERROR;
    }
    if(block_size_num != 2)
    {
        return VSI_NN_ERROR_VALUED_ERROR;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SPACE2DEPTH, 1, 1, NULL);
    if(node)
    {
        node->nn_param.space2depth.block_size[0] = block_size[0];
        node->nn_param.space2depth.block_size[1] = block_size[1];
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_space_to_depth() */

VSI_NN_error_e VSI_NN_node_depth_to_space
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* block_size,
    _IN int32_t block_size_num,
    _IN const int32_t* crop_front,
    _IN const int32_t* crop_end
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    int32_t i;
    if(!graph || !input || !output || !block_size)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    if(crop_front || crop_end)
    {
        // Not support crop yet
        assert(FALSE);
        return VSI_NN_ERROR_VALUED_ERROR;
    }
    if(block_size_num != 2)
    {
        return VSI_NN_ERROR_VALUED_ERROR;
    }
    for(i = 1; i < block_size_num; i ++)
    {
        if(block_size[0] != block_size[i])
        {
            return VSI_NN_ERROR_VALUED_ERROR;
        }
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_DEPTH2SPACE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.depth2space.block_size = block_size[0];
        node->nn_param.depth2space.mode = VSI_NN_DEPTH2SPACE_DCR;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_depth_to_space() */

VSI_NN_error_e VSI_NN_node_channel_shuffle
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t group_number,
    _IN int32_t axis
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SHUFFLECHANNEL, 1, 1, NULL);
    if(node)
    {
        node->nn_param.shufflechannel.group_number = group_number;
        node->nn_param.shufflechannel.axis = axis;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_channel_shuffle() */

VSI_NN_error_e VSI_NN_node_expand_dims
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    )
{

    assert(FALSE);
    return VSI_NN_ERROR_API_FAIL;
} /* VSI_NN_node_expand_dims() */

VSI_NN_error_e VSI_NN_node_hashtable_lookup
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* lookups,
    _IN VSI_NN_tensor* keys,
    _IN VSI_NN_tensor* values,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_tensor* output_hits
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !lookups || !keys || !values || !output || !output_hits)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_HASHTABLE_LOOKUP, 3, 2, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[3] = {NULL};
        vsi_nn_tensor_t * node_outputs[2] = {NULL};
        node_inputs[0] = lookups;
        node_inputs[1] = keys;
        node_inputs[2] = values;
        node_outputs[0] = output;
        node_outputs[1] = output_hits;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 3, node_outputs, 2);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_hashtable_lookup() */

VSI_NN_error_e VSI_NN_node_embedding_lookup
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* lookups,
    _IN VSI_NN_tensor* values,
    _IN VSI_NN_tensor* output
     )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !lookups || !values || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_EMBEDDING_LOOKUP, 2, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[2] = {NULL};
        node_inputs[0] = lookups;
        node_inputs[1] = values;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 2, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_embedding_lookup() */

VSI_NN_error_e VSI_NN_node_lsh_projection
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* hash_func,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* weight,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_lsh_projection_type_e type
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !hash_func || !weight || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_LSH_PROJECTION, 3, 1, NULL);
    if(node)
    {
        vsi_nn_tensor_t * node_inputs[3] = {NULL};
        node_inputs[0] = hash_func;
        node_inputs[1] = input;
        node_inputs[2] = weight;
        node->nn_param.lsh_projection.type = type;
        vsi_nn_SetNodeInputsAndOutputs(node, node_inputs, 3, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_lsh_projection() */

VSI_NN_error_e VSI_NN_node_slice
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* begin,
    _IN const int32_t* size
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output || !begin || !size)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SLICE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.slice.dims = input->attr.dim_num;
        node->nn_param.slice.start = (const uint32_t*)begin;
        node->nn_param.slice.length = (const uint32_t*)size;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_slice() */

VSI_NN_error_e VSI_NN_node_strided_slice
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* begin,
    _IN const int32_t* end,
    _IN const int32_t* strides,
    _IN int32_t begin_mask,
    _IN int32_t end_mask,
    _IN int32_t shrink_axis_mask
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output || !begin || !end || !strides)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_STRIDED_SLICE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.strided_slice.begin_dims = begin;
        node->nn_param.strided_slice.begin_dims_num = input->attr.dim_num;
        node->nn_param.strided_slice.end_dims = end;
        node->nn_param.strided_slice.end_dims_num = input->attr.dim_num;
        node->nn_param.strided_slice.stride_dims = strides;
        node->nn_param.strided_slice.stride_dims_num = input->attr.dim_num;
        node->nn_param.strided_slice.begin_mask = begin_mask;
        node->nn_param.strided_slice.end_mask = end_mask;
        node->nn_param.strided_slice.shrink_axis_mask = shrink_axis_mask;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_strided_slice() */

/** Detection */
VSI_NN_error_e VSI_NN_node_roi_pool
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* feature_map,
    _IN VSI_NN_tensor* loc,
    _IN VSI_NN_tensor* batch_index,
    _IN VSI_NN_tensor* output,
    _IN int32_t output_h,
    _IN int32_t output_w,
    _IN float ratio_h,
    _IN float ratio_w
    )
{

    assert(FALSE);
    return VSI_NN_ERROR_API_FAIL;
} /* VSI_NN_node_roi_pool() */

VSI_NN_error_e VSI_NN_node_roi_align
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* feature_map,
    _IN VSI_NN_tensor* loc,
    _IN VSI_NN_tensor* batch_index,
    _IN VSI_NN_tensor* output,
    _IN int32_t output_h,
    _IN int32_t output_w,
    _IN float ratio_h,
    _IN float ratio_w,
    _IN int32_t sample_num_h,
    _IN int32_t sample_num_w
    )
{

    assert(FALSE);
    return VSI_NN_ERROR_API_FAIL;
} /* VSI_NN_node_roi_align() */

/** Image transform */
VSI_NN_error_e VSI_NN_node_resize_bilinear
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t scale_h,
    _IN int32_t scale_w
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RESIZE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.resize.type = VSI_NN_INTERPOLATION_BILINEAR;
        node->nn_param.resize.size[0] = scale_w;
        node->nn_param.resize.size[1] = scale_h;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_resize_bilinear() */

VSI_NN_error_e VSI_NN_node_resize_nearest
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t scale_h,
    _IN int32_t scale_w
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_OK;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_RESIZE, 1, 1, NULL);
    if(node)
    {
        node->nn_param.resize.type = VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR;
        node->nn_param.resize.size[0] = scale_w;
        node->nn_param.resize.size[1] = scale_h;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
    }
    else
    {
        error = VSI_NN_ERROR_API_FAIL;
    }
    return error;
} /* VSI_NN_node_resize_nearest() */

/** RNN */
VSI_NN_error_e VSI_NN_node_svdf
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* weights_feature,
    _IN VSI_NN_tensor* weights_time,
    _IN VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* input_state,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_tensor* output_state,
    _IN int32_t rank
    )
{

    assert(FALSE);
    return VSI_NN_ERROR_API_FAIL;
} /* VSI_NN_node_svdf() */

//EXPORT VSI_NN_error_e VSI_NN_node_rnn();

VSI_NN_error_e VSI_NN_node_rnn_unit
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* input_state,
    _IN VSI_NN_tensor* weight, _IN VSI_NN_tensor* recrrent_weight,
    _IN VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_tensor* output_state,
    _IN VSI_NN_activation_e activation
    )
{

    assert(FALSE);
    return VSI_NN_ERROR_API_FAIL;
} /* VSI_NN_node_rnn_unit() */

VSI_NN_error_e VSI_NN_node_lstm_unit
    (
    _IN VSI_NN_graph* graph
    )
{

    assert(FALSE);
    return VSI_NN_ERROR_API_FAIL;
} /* VSI_NN_node_lstm_unit() */

VSI_NN_error_e VSI_NN_node_argmin
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_ARGMIN, 1, 1, NULL);
    if(node)
    {
        node->nn_param.argmin.axis = axis;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_argmin() */

VSI_NN_error_e VSI_NN_node_argmax
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    )
{
    VSI_NN_error_e error = VSI_NN_ERROR_API_FAIL;
    vsi_nn_node_t* node = NULL;
    if(!graph || !input || !output)
    {
        return VSI_NN_ERROR_UNEXPECTED_NULL;
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_ARGMAX, 1, 1, NULL);
    if(node)
    {
        node->nn_param.argmin.axis = axis;
        vsi_nn_SetNodeInputsAndOutputs(node, &input, 1, &output, 1);
        error = VSI_NN_ERROR_OK;
    }
    return error;
} /* VSI_NN_node_argmin() */
