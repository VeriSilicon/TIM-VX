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
#include <stdio.h>
#include <string.h>

#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

static void _compute_stride
    (
    uint32_t * shape,
    uint32_t   dim_num,
    uint32_t * stride
    )
{
    uint32_t i;
    uint32_t s;
    s = 1;
    for( i = 0; i < dim_num; i ++ )
    {
        stride[i] = s;
        s *= shape[i];
    }
} /* _compute_stride() */

vsi_nn_tensor_t* vsi_nn_Concat
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t** tensors,
    uint32_t tensor_num,
    uint32_t axis
    )
{
    uint32_t i, j;
    int32_t k;
    uint8_t* buffer = NULL;
    uint8_t* tmp = NULL;
    size_t total_bytes = 0;
    size_t tensor_size = 0;
    size_t offset = 0, src = 0, dst = 0;
    uint32_t* strides = NULL;
    uint32_t* dst_strides = NULL;
    uint32_t type_bytes = 0;
    vsi_nn_tensor_attr_t output_attr;
    vsi_nn_tensor_t* tensor_out = NULL;
    // Validate inputs
    if( tensor_num < 2 || !graph )
    {
        return NULL;
    }
    for( i = 0; i < tensor_num; i ++ )
    {
        if( !tensors[i] )
        {
            VSILOGW("Concat tensor %u is null.", i);
            return NULL;
        }
    }
    memset( &output_attr, 0, sizeof(vsi_nn_tensor_attr_t) );
    memcpy( &output_attr.dtype, &tensors[0]->attr.dtype, sizeof(vsi_nn_dtype_t) );
    memcpy( output_attr.size, tensors[0]->attr.size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM );
    output_attr.dim_num = tensors[0]->attr.dim_num;

    for( i = 1; i < tensor_num; i ++ )
    {
        if( tensors[0]->attr.dim_num != tensors[i]->attr.dim_num )
        {
            VSILOGW("Concat tensor dim number mismatch.");
            return NULL;
        }
        for( j = 0; j < tensors[0]->attr.dim_num; j ++)
        {
            if( j == axis )
            {
                continue;
            }
            if( tensors[0]->attr.size[j] != tensors[i]->attr.size[j] )
            {
                vsi_nn_PrintTensor(tensors[0], 0);
                vsi_nn_PrintTensor(tensors[i], i);
                VSILOGW("Concat tensor shapes mismatch.");
                return NULL;
            }
        }
        output_attr.size[axis] += tensors[i]->attr.size[axis];
    }
    total_bytes = vsi_nn_GetTensorSize( output_attr.size, output_attr.dim_num,
            output_attr.dtype.vx_type );
    buffer = (uint8_t*)malloc( total_bytes );
    strides = (uint32_t*)malloc( sizeof(uint32_t) * tensors[0]->attr.dim_num );
    dst_strides = (uint32_t*)malloc( sizeof(uint32_t) * tensors[0]->attr.dim_num );
    if (!buffer || !strides || !dst_strides)
    {
        VSILOGW("Out of memroy.");
        goto concat_error;
    }
    type_bytes = vsi_nn_GetTypeBytes( output_attr.dtype.vx_type );
    _compute_stride(output_attr.size, output_attr.dim_num, dst_strides);
    offset = 0;
    for( i = 0; i < tensor_num; i ++ )
    {
        tmp = (uint8_t*)vsi_nn_ConvertTensorToData( graph, tensors[i] );
        tensor_size = vsi_nn_GetElementNum( tensors[i] );
        if( !tmp )
        {
            VSILOGW("Read tensor %u fail.", i);
            goto concat_error;
        }
        _compute_stride(tensors[i]->attr.size, tensors[i]->attr.dim_num, strides);
        for( j = 0; j < tensor_size; j ++ )
        {
            src = j;
            dst = 0;
            for( k = tensors[0]->attr.dim_num - 1; k >= 0; k -- )
            {
                dst += ( src / strides[k] ) * dst_strides[k];
                src %= strides[k];
            }
            dst += offset;
            src = j;
            memcpy( &buffer[dst * type_bytes], &tmp[src * type_bytes], type_bytes );
        }
        free(tmp);
        offset += strides[axis] * tensors[i]->attr.size[axis];
    }
    tensor_out = vsi_nn_CreateTensorFromData( graph, buffer, &output_attr );

concat_error:
    if( buffer )
    {
        free(buffer);
    }
    if( strides )
    {
        free(strides);
    }
    if( dst_strides )
    {
        free(dst_strides);
    }
    return tensor_out;
}

vsi_nn_tensor_t* vsi_nn_ConvertTensorDtype
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t* tensor,
    const vsi_nn_dtype_t* dst_dtype
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t i = 0, src_stride = 0, dst_stride = 0;
    uint32_t sz = 0;
    uint8_t* src_buf = NULL;
    uint32_t dst_buf_sz = 0;
    uint8_t* dst_buf = NULL;
    vsi_nn_tensor_attr_t dst_attr;
    vsi_nn_tensor_t* dst_tensor = NULL;

    if( NULL == graph || NULL == tensor || NULL == dst_dtype )
    {
        return NULL;
    }

    sz = vsi_nn_GetElementNum( tensor );
    src_stride = vsi_nn_TypeGetBytes( tensor->attr.dtype.vx_type );
    dst_stride = vsi_nn_TypeGetBytes( dst_dtype->vx_type );
    dst_buf_sz = sz * dst_stride;

    dst_buf = (uint8_t *)malloc( dst_buf_sz );
    if( NULL != dst_buf )
    {
        src_buf = vsi_nn_ConvertTensorToData( graph, tensor );
        if( NULL != src_buf )
        {
            for( i = 0; i < sz; i++ )
            {
                status = vsi_nn_DtypeConvert( &src_buf[src_stride * i],
                    &tensor->attr.dtype, &dst_buf[dst_stride * i], dst_dtype );
                if(status != VSI_SUCCESS)
                {
                    break;
                }
            }

            if( VSI_SUCCESS == status )
            {
                memcpy( &dst_attr, &tensor->attr, sizeof( dst_attr ) );
                memcpy( &dst_attr.dtype, dst_dtype, sizeof( dst_attr.dtype ) );
                dst_tensor = vsi_nn_CreateTensorFromData( graph, dst_buf, &dst_attr );
            }
        }
    }

    if( src_buf )
    {
        free( src_buf );
        src_buf = NULL;
    }
    if( dst_buf )
    {
        free( dst_buf );
        dst_buf = NULL;
    }

    return dst_tensor;
} /* vsi_nn_ConvertTensorDtype() */

vsi_nn_tensor_t* vsi_nn_TensorAdd
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t** tensors,
    uint32_t tensor_num,
    vsi_nn_tensor_attr_t output_attr
    )
{
#define MAX_TENSOR_NUM 16
    vsi_status status = VSI_SUCCESS;
    uint32_t i, j;
    uint8_t* buffer[MAX_TENSOR_NUM] = {NULL};
    uint8_t* tmp = NULL;
    size_t total_bytes = 0;
    size_t elements = 0;
    uint32_t type_bytes = 0;
    uint32_t out_bytes = 0;
    float data = 0.0f, sum = 0.0f;
    vsi_nn_tensor_t* tensor_out = NULL;

    // Validate inputs
    if( tensor_num < 2 || !graph || tensor_num > MAX_TENSOR_NUM)
    {
        return NULL;
    }
    for( i = 0; i < tensor_num; i ++ )
    {
        if( !tensors[i] )
        {
            VSILOGE("Tensor %u is null(TensorAdd).", i);
            return NULL;
        }
    }
    for( i = 1; i < tensor_num; i ++ )
    {
        if( tensors[0]->attr.dim_num != tensors[i]->attr.dim_num )
        {
            VSILOGE("Tensor dim number mismatch(TensorAdd).");
            return NULL;
        }
        for( j = 0; j < tensors[0]->attr.dim_num; j ++)
        {
            if( tensors[0]->attr.size[j] != tensors[i]->attr.size[j] )
            {
                vsi_nn_PrintTensor(tensors[0], 0);
                vsi_nn_PrintTensor(tensors[i], i);
                VSILOGE("Tensor shapes mismatch(TensorAdd).");
                return NULL;
            }
        }
    }
    for ( i = 0; i < tensor_num; i++ )
    {
        buffer[i] = vsi_nn_ConvertTensorToData(graph, tensors[i]);
        if ( !buffer[i] )
        {
            VSILOGE("Convert tensor to data failed.");
            goto error;
        }
    }

    elements = vsi_nn_GetElementNum( tensors[0] );
    out_bytes = vsi_nn_TypeGetBytes( output_attr.dtype.vx_type );
    total_bytes = vsi_nn_GetTensorSize( output_attr.size, output_attr.dim_num,
            output_attr.dtype.vx_type );
    tmp = (uint8_t*)malloc( total_bytes );
    if ( !tmp )
    {
        VSILOGE("Out of memroy.");
        goto error;
    }
    for ( i = 0; i < elements; i++ )
    {
        sum = 0.0f;
        for ( j = 0;  j < tensor_num;  j++ )
        {
            type_bytes = vsi_nn_TypeGetBytes( tensors[j]->attr.dtype.vx_type );
            status = vsi_nn_DtypeToFloat32(&(buffer[j][type_bytes * i]), &data, &tensors[j]->attr.dtype);
            if ( status != VSI_SUCCESS )
            {
                VSILOGE("Convert data failed.");
                goto error;
            }
            sum += data;
        }
        status = vsi_nn_Float32ToDtype(sum, &tmp[out_bytes * i], &output_attr.dtype);
        if ( status != VSI_SUCCESS )
        {
            VSILOGE("Convert data failed.");
            goto error;
        }
    }
    tensor_out = vsi_nn_CreateTensorFromData( graph, tmp, &output_attr );

#undef MAX_TENSOR_NUM
error:
    for ( i = 0; i < tensor_num; i++ )
    {
        if ( buffer[i] )
        {
            free(buffer[i]);
        }
    }
    if( tmp )
    {
        free(tmp);
    }

    return tensor_out;
} /* vsi_nn_ConstTensorAdd() */