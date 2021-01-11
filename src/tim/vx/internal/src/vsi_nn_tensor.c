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
#include <stdarg.h>

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_types.h"
#include "vsi_nn_test.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_dtype_util_prv.h"
#include "utils/vsi_nn_tensor_op.h"

static vsi_bool _try_set_const_tensor
    (
    vsi_nn_tensor_t *tensor
    );

static vsi_bool _auto_cal_shape
    (
    uint32_t * input_shape,
    uint32_t   input_dim,
    uint32_t * shape,
    uint32_t * dim_num
    );

static vsi_bool _init_tensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    uint8_t         * data
    );

static vsi_nn_tensor_t * _create_tensor
    (
    vsi_nn_graph_t       * graph,
    uint8_t              * data,
    vsi_nn_tensor_attr_t * attr
    );

static uint32_t get_tensor_elements_num
    (
    const uint32_t   * shape,
    uint32_t     dim_num,
    vsi_nn_type_e type
    )
{
    uint32_t num;
    uint32_t sz;
    uint32_t dsize;

    sz = vsi_nn_GetTensorSize( shape,
        dim_num, type );
    dsize = vsi_nn_GetTypeBytes( type );
    num = (uint32_t)(sz / dsize);
    return num;
} /* get_tensor_elements_num() */

static void print_tensor
    (
    vsi_nn_tensor_t *tensor,
    vsi_nn_tensor_id_t id,
    char *ext_str
    )
{
#define _SHAPE_BUF_SZ   (64)
#define _EXT_ATTR_BUF_SZ   (64)
#define _ATTR_BUF_SZ   (64)
    int count;
    char shape[_SHAPE_BUF_SZ] = { 0 };
    char ext_attr[_EXT_ATTR_BUF_SZ] = { 0 };
    char format[_ATTR_BUF_SZ] = {0};

    if( !tensor )
    {
        VSILOGD("%s None", ext_str);
        return;
    }
    vsi_nn_ShapeToString( tensor->attr.size, tensor->attr.dim_num,
        shape, _SHAPE_BUF_SZ, TRUE );
    vsi_nn_FormatToString( tensor, format, _SHAPE_BUF_SZ );

    /* Process quantize parameters */
    switch( tensor->attr.dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        count = snprintf( &ext_attr[0], _EXT_ATTR_BUF_SZ,
            "DFP fl=%3d", tensor->attr.dtype.fl );
        ext_attr[count] = 0;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        count = snprintf( &ext_attr[0], _EXT_ATTR_BUF_SZ,
            "ASM zp=%3d, scale=%.6f",
            tensor->attr.dtype.zero_point, tensor->attr.dtype.scale );
        ext_attr[count] = 0;
        break;
#ifdef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC:
        count = snprintf( &ext_attr[0], _EXT_ATTR_BUF_SZ,
            "SYM PERCHANNEL axis=%d, count=%d",
            tensor->attr.dtype.channel_dim, tensor->attr.dtype.scale_dim );
        ext_attr[count] = 0;
        break;
#endif
    default:
        strncpy(ext_attr, "NONE", _EXT_ATTR_BUF_SZ);
        break;
    }

    if(ext_str)
    {
        VSILOGD("%s id[%4u] vtl[%d] const[%d] shape[%-18s] fmt[%s] qnt[%s]",
            ext_str,
            id,
            tensor->attr.vtl,
            tensor->attr.is_const,
            shape,
            format,
            ext_attr);
    }
    else
    {
        VSILOGD("id[%4u] vtl[%d] const[%d] shape[%-18s] fmt[%s] qnt[%s]",
            id,
            tensor->attr.vtl,
            tensor->attr.is_const,
            shape,
            format,
            ext_attr);
    }
}

static vsi_nn_tensor_rel_t *_init_tensor_rel_buffer
    (
    vsi_nn_graph_t *graph,
    uint32_t max_io
    )
{
    uint32_t i,tensor_num;
    vsi_nn_tensor_rel_t *tensor_ref;

    tensor_num = graph->tensor_num;
    tensor_ref = (vsi_nn_tensor_rel_t *)malloc(tensor_num * sizeof(vsi_nn_tensor_rel_t));
    if(NULL == tensor_ref)
    {
        return NULL;
    }
    memset(tensor_ref, 0, sizeof(vsi_nn_tensor_rel_t) * tensor_num);

    for(i = 0; i < tensor_num; i++)
    {
        tensor_ref[i].input.num = 0;
        tensor_ref[i].output.num = 0;
        tensor_ref[i].input.table  = (vsi_nn_tensor_rel_table_t *)malloc(
            max_io * sizeof(vsi_nn_tensor_rel_table_t));
        tensor_ref[i].output.table = (vsi_nn_tensor_rel_table_t *)malloc(
            max_io * sizeof(vsi_nn_tensor_rel_table_t));
        if(NULL == tensor_ref[i].input.table || NULL == tensor_ref[i].output.table)
        {
            goto error;
        }
        memset(tensor_ref[i].input.table,  0, max_io * sizeof(vsi_nn_tensor_rel_table_t));
        memset(tensor_ref[i].output.table, 0, max_io * sizeof(vsi_nn_tensor_rel_table_t));
    }

    return tensor_ref;
error:
    if(tensor_ref)
    {
        for(i = 0; i < tensor_num; i++)
        {
            if(tensor_ref[i].input.table)
            {
                free(tensor_ref[i].input.table);
                tensor_ref[i].input.table = NULL;
            }
            if(tensor_ref[i].output.table)
            {
                free(tensor_ref[i].output.table);
                tensor_ref[i].output.table = NULL;
            }
        }
    free(tensor_ref);
    tensor_ref = NULL;
    }
    return NULL;
} /* _init_tensor_rel_buffer() */

static vsi_bool _try_set_const_tensor
    (
    vsi_nn_tensor_t *tensor
    )
{
    vsi_status status;
    vsi_bool ret;
    vsi_nn_vxtensor_attr_t attr;

    ret = TRUE;
    status = VSI_SUCCESS;
    if( TRUE == tensor->attr.is_const )
    {
        attr = VSI_NN_TENSOR_ATTR_CONST;
        status = vsi_nn_SetTensorAttr(tensor, attr);
    }
    if( VSI_FAILURE == status )
    {
        ret = FALSE;
    }

    return ret;
} /* _set_const_tensor() */

static vsi_bool _auto_cal_shape
    (
    uint32_t * input_shape,
    uint32_t   input_dim,
    uint32_t * shape,
    uint32_t * dim_num
    )
{
    vsi_bool   ret;
    int32_t  neg_idx;
    uint32_t i;
    uint32_t total_size;

    ret = TRUE;
    neg_idx = -1;
    total_size = vsi_nn_ShapeProduct( input_shape, input_dim );
    if (-1 == *dim_num)
    {
        *dim_num = 1;
        shape[0] = total_size;
        return ret;
    }

    for( i = 0; i < *dim_num; i ++ )
    {
        if( -1 != (int32_t)shape[i] )
        {
            if (0 == shape[i])
            {
                if (i >= input_dim)
                {
                    VSILOGE( "Wrong shape '%d' ", (int32_t)shape[i] );
                    ret = FALSE;
                    break;
                }
                shape[i] = input_shape[i];
            }
            total_size /= shape[i];
        }
        else if( -1 == neg_idx )
        {
            neg_idx = i;
        }
        else
        {
            VSILOGE( "Wrong shape '%d' ", (int32_t)shape[i] );
            ret = FALSE;
            break;
        }
    }
    if( FALSE == ret  )
    {
        shape[neg_idx] = -1;
    }
    else if(neg_idx != -1)
    {
        shape[neg_idx] = total_size;
    }
    return ret;
} /* _auto_cal_shape() */

static vsi_bool _init_tensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    uint8_t         * data
    )
{
    vsi_bool ret;
    vx_tensor_create_params_t params;
    float * scales = NULL;
    int32_t * null_zp = NULL;

    ret = TRUE;

    memset( &params, 0, sizeof( vx_tensor_create_params_t ) );
    params.num_of_dims = tensor->attr.dim_num;
    params.sizes = tensor->attr.size;
    params.data_format = (vsi_enum)tensor->attr.dtype.vx_type;
    params.quant_format = (vsi_enum)tensor->attr.dtype.qnt_type;
    switch( tensor->attr.dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        params.quant_data.dfp.fixed_point_pos = (uint8_t)tensor->attr.dtype.fl;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        params.quant_data.affine.scale = tensor->attr.dtype.scale;
        params.quant_data.affine.zeroPoint = (int32_t)tensor->attr.dtype.zero_point;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC:
#ifdef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
        // This is a hack that driver doesn't support const scales
        scales = (float*)malloc(sizeof(float) * tensor->attr.dtype.scale_dim);
        memcpy(scales, tensor->attr.dtype.scales, tensor->attr.dtype.scale_dim * sizeof(float));
        params.quant_data.affinePerChannel.channelDim = tensor->attr.dtype.channel_dim;
        params.quant_data.affinePerChannel.scaleCount = tensor->attr.dtype.scale_dim;
        params.quant_data.affinePerChannel.scales = scales;
        params.quant_data.affinePerChannel.zeroPoint = NULL;
        params.quant_data.affinePerChannel.zeroPointCount = 0;
        // TODO: This is a hack since driver will access a NULL pointer and cause a crash.
        // Remove me in the future.
        {
            null_zp = (int32_t*)malloc(sizeof(int32_t) * tensor->attr.dtype.scale_dim);
            memset(null_zp, 0, sizeof(int32_t) * tensor->attr.dtype.scale_dim);
            params.quant_data.affinePerChannel.zeroPoint = null_zp;
        }
        break;
#else
    VSILOGE( "can't support qnt_type VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC." );
#endif
    default:
        break;
    }

    if( NULL != tensor->t )
    {
        vxReleaseTensor( &tensor->t );
    }
    if( NULL != tensor->wb )
    {
        vxReleaseWeightsBiasesParameter( &tensor->wb );
    }

    if( TRUE == tensor->attr.is_created_from_handle )
    {
        vx_tensor_addressing addr;
        uint32_t stride_size[VSI_NN_MAX_DIM_NUM];
        uint32_t buf_sz;

        buf_sz = vsi_nn_GetStrideSize( &tensor->attr, stride_size );
        if( buf_sz > 0 )
        {
            uint32_t align_start_size = graph->handle_manager.align_start_size;
            uint32_t align_block_size = graph->handle_manager.align_block_size;
            if (data == NULL)
            {
                data = vsi_nn_MallocAlignedBuffer(buf_sz, align_start_size,
                    align_block_size);
                tensor->attr.is_handle_malloc_by_ovxlib = TRUE;
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
                tensor->attr.vsi_memory_type = VSI_MEMORY_TYPE_HOST;
#endif
            }
            else
            {
                tensor->attr.is_handle_malloc_by_ovxlib = FALSE;
                if (!vsi_nn_IsBufferAligned(data, align_start_size))
                {
                    VSILOGE( "vsi_nn_IsBufferAligned is FALSE." );
                    if( scales )
                    {
                        free(scales);
                    }
                    if(null_zp)
                    {
                        free(null_zp);
                        null_zp = NULL;
                    }
                    return FALSE;
                }
            }
            if( data )
            {
                addr = vxCreateTensorAddressing(graph->ctx->c,
                    tensor->attr.size, stride_size, (uint8_t)tensor->attr.dim_num);
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
#ifdef VX_13_NN_COMPATIBLITY
                tensor->t = vxCreateTensorFromHandle2(graph->ctx->c,
                    &params, sizeof(vx_tensor_create_params_t),
                    addr, data, tensor->attr.vsi_memory_type);
#else
                tensor->t = vxCreateTensorFromHandle(graph->ctx->c,
                    &params, sizeof(vx_tensor_create_params_t),
                    addr, data, tensor->attr.vsi_memory_type);
#endif
#else
#ifdef VX_13_NN_COMPATIBLITY
                tensor->t = vxCreateTensorFromHandle2(graph->ctx->c,
                    &params, sizeof(vx_tensor_create_params_t),
                    addr, data, VX_MEMORY_TYPE_HOST);
#else
                tensor->t = vxCreateTensorFromHandle(graph->ctx->c,
                    &params, sizeof(vx_tensor_create_params_t),
                    addr, data, VX_MEMORY_TYPE_HOST);
#endif

#endif
                //memset(data, 0x5A, buf_sz);
                vxReleaseTensorAddressing( &addr );
                vxFlushHandle( (vx_reference)tensor->t );
            }
        }
    }
    else if( FALSE == tensor->attr.vtl )
    {
        tensor->t = vxCreateTensor2( graph->ctx->c,
            &params, sizeof( vx_tensor_create_params_t ) );
    }
    else
    {
        tensor->t = vxCreateVirtualTensor2( graph->g,
            &params, sizeof( vx_tensor_create_params_t ) );
    }
    if( NULL == tensor->t )
    {
        VSILOGE( "Create vx tensor fail." );
        ret = FALSE;
    }

    if( !tensor->attr.vtl && !tensor->attr.is_const )
    {
        //norm tensor need to fill initial value
        if( ( !tensor->attr.is_created_from_handle ) || tensor->attr.is_handle_malloc_by_ovxlib )
        {
            vsi_nn_FillTensorWithValue( graph, tensor, 0.0f );
            if(tensor->attr.is_created_from_handle)
            {
                vxFlushHandle( (vx_reference)tensor->t );
            }
        }
    }

    ret = _try_set_const_tensor( tensor );

    if( scales )
    {
        free(scales);
    }
    if(null_zp)
    {
        free(null_zp);
        null_zp = NULL;
    }
    return ret;
} /* _init_tensor() */

vsi_bool vsi_nn_TensorReinit
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor
    )
{
    vsi_bool ret;
    ret = TRUE;

    if( NULL == graph || NULL == tensor )
    {
        return FALSE;
    }
    if( tensor->attr.dim_num != VSI_NN_DIM_AUTO )
    {
        ret = _init_tensor( graph, tensor, NULL );
    }
    return ret;
} /* vsi_nn_TensorReinit() */

static vsi_nn_tensor_t * _create_tensor
    (
    vsi_nn_graph_t       * graph,
    uint8_t              * data,
    vsi_nn_tensor_attr_t * attr
    )
{
    vsi_nn_tensor_t * tensor;

    tensor = NULL;
    if( NULL == graph || NULL == graph->g || NULL == attr )
    {
        return tensor;
    }

    tensor = (vsi_nn_tensor_t *)malloc( sizeof( vsi_nn_tensor_t ) );
    //vsi_nn_UpdateTensorDims( attr );

    if( NULL != tensor )
    {
        memset( tensor, 0, sizeof( vsi_nn_tensor_t ) );
        memcpy( &tensor->attr, attr, sizeof( vsi_nn_tensor_attr_t ) );
        tensor->is_swapped = FALSE;
        if( attr->dim_num != VSI_NN_DIM_AUTO )
        {
            _init_tensor( graph, tensor, data);
            if( NULL == tensor->t )
            {
                VSILOGE( "Create vx tensor fail." );
                free( tensor );
                tensor = NULL;
            }
        }
    }
    return tensor;
}

vsi_nn_tensor_t * vsi_nn_CreateTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr
    )
{
    attr->is_created_from_handle = FALSE;
    return _create_tensor(graph, NULL, attr);
} /* vsi_nn_CreateTensor() */

vsi_nn_tensor_t * vsi_nn_CreateTensorFromHandle
    (
    vsi_nn_graph_t       * graph,
    uint8_t              * data,
    vsi_nn_tensor_attr_t * attr
    )
{
    attr->is_created_from_handle = TRUE;
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
    if(attr->vsi_memory_type == VSI_MEMORY_TYPE_NONE || attr->vsi_memory_type == 0)
    {
        attr->vsi_memory_type = VSI_MEMORY_TYPE_HOST;
    }
#endif
    return _create_tensor(graph, data, attr);
} /* vsi_nn_CreateTensorFromHandle() */

vsi_nn_tensor_t * vsi_nn_CreateTensorWithDefault
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr,
    float                  defualt_value
    )
{
    vsi_nn_tensor_t* t = vsi_nn_CreateTensor( graph, attr );
    if( t )
    {
        uint32_t size = 0;
        uint32_t stride[VSI_NN_MAX_DIM_NUM] = { 0 };
        uint8_t* data = NULL;

        size = vsi_nn_GetStrideSize( &t->attr, stride );
        data = (uint8_t *)malloc( size );
        if( data )
        {
            uint32_t i = 0;
            uint32_t elements = size / stride[0];
            vsi_status status = VSI_SUCCESS;

            for( i = 0; i < elements; i ++ )
            {
                status = vsi_nn_Float32ToDtype( defualt_value, &data[stride[0] * i], &t->attr.dtype );
                if( VSI_FAILURE == status )
                {
                    VSILOGE("Convert default_value to dtype fail");
                    break;
                }
            }

            status = vsi_nn_CopyDataToTensor( graph, t, data );
            free( data );
            data = NULL;
            if( VSI_FAILURE == status )
            {
                VSILOGE("Copy data to tensor fail");
            }
        }
    }

    return t;
} /* vsi_nn_CreateTensorWithDefault() */

vsi_status vsi_nn_FillTensorWithValue
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_t      * tensor,
    float                  value
    )
{
    vsi_status status = VSI_FAILURE;

    if( tensor )
    {
        uint32_t size = 0;
        uint32_t stride[VSI_NN_MAX_DIM_NUM] = { 0 };
        uint8_t* data = NULL;

        size = vsi_nn_GetStrideSize( &tensor->attr, stride );
        data = (uint8_t *)malloc( size );
        if( data )
        {
            uint32_t i = 0;
            uint32_t elements = size / stride[0];

            for( i = 0; i < elements; i ++ )
            {
                status = vsi_nn_Float32ToDtype( value, &data[stride[0] * i], &tensor->attr.dtype );
                if( VSI_FAILURE == status )
                {
                    VSILOGE("Convert value to dtype fail");
                    break;
                }
            }

            status = vsi_nn_CopyDataToTensor( graph, tensor, data );
            free( data );
            data = NULL;
            if( VSI_FAILURE == status )
            {
                VSILOGE("Copy data to tensor fail");
            }
        }
    }

    return status;
} /* vsi_nn_FillTensorWithValue() */

void vsi_nn_ReleaseTensor
    (
    vsi_nn_tensor_t ** tensor
    )
{
    vsi_nn_tensor_t * ptr;
    ptr = *tensor;
    if( NULL != tensor && NULL != *tensor )
    {
        uint8_t * handle = NULL;
        if( NULL != ptr->t )
        {
            if (ptr->attr.is_created_from_handle &&
                ptr->attr.is_handle_malloc_by_ovxlib)
            {
                vxSwapTensorHandle( ptr->t, NULL, (void**)&handle);
                if ( handle == NULL )
                {
                    VSILOGE("vxSwapTensorHandle fail.");
                    return;
                }
            }
            vxReleaseTensor( &ptr->t );
            if (handle) vsi_nn_FreeAlignedBuffer(handle);
        }

        if (ptr->wb) {
            vxReleaseWeightsBiasesParameter(&ptr->wb);
        }

        free( ptr );
        *tensor = NULL;
    }
} /* vsi_nn_ReleaseTensor() */

vsi_status vsi_nn_SetTensorAttr
    (
    vsi_nn_tensor_t * tensor,
    const vsi_nn_vxtensor_attr_t attrs
    )
{
    vsi_status status;

    status = VSI_SUCCESS;
    if( NULL == tensor )
    {
        return VSI_FAILURE;
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_CONST ) )
    {
        vx_enum data_lifetime;
        if(tensor->attr.is_const == TRUE)
        {
            data_lifetime = VX_TENSOR_LIFE_TIME_STATIC;
        }
        else
        {
            data_lifetime = VX_TENSOR_LIFE_TIME_DYNAMIC;
        }
        status = vxSetTensorAttribute(tensor->t,
                                      VX_TENSOR_LIFETIME,
                                      &data_lifetime,
                                      sizeof(vx_enum));
    }
    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_HIGH_PRECISION ) )
    {
        vx_enum precision = VX_TENSOR_PRECISION_HIGH;
        status = vxSetTensorAttribute(tensor->t,
                                      VX_TENSOR_PRECISION,
                                      &precision,
                                      sizeof(vx_enum));
    }

    return status;
}

vsi_status vsi_nn_QueryTensorAttr
    (
    vsi_nn_tensor_t * tensor,
    const vsi_nn_vxtensor_attr_t attrs
    )
{
    vsi_status status;

    status = VSI_SUCCESS;
    if( NULL == tensor )
    {
        return VSI_FAILURE;
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_DIM_NUM ) )
    {
        status = vxQueryTensor( tensor->t, VX_TENSOR_NUM_OF_DIMS,
            &tensor->attr.dim_num, sizeof( tensor->attr.dim_num ) );
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_DTYPE ) )
    {
        status = vxQueryTensor( tensor->t, VX_TENSOR_DATA_TYPE,
            &tensor->attr.dtype.vx_type, sizeof( tensor->attr.dtype.vx_type ) );
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_SIZE ) )
    {
        status = vxQueryTensor( tensor->t, VX_TENSOR_DIMS,
            &tensor->attr.size, sizeof( tensor->attr.size ) );
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_FIXED_POINT_POS ) )
    {
        status = vxQueryTensor( tensor->t, VX_TENSOR_FIXED_POINT_POS,
            &tensor->attr.dtype.fl, sizeof( tensor->attr.dtype.fl ) );
    }

    return status;
} /* vsi_nn_QueryTensorAttr() */

uint32_t vsi_nn_CopyTensorToBuffer
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    void            * buffer
    )
{
    uint32_t     sz;
    uint32_t     stride_size[VSI_NN_MAX_DIM_NUM];
    vsi_status     status;
    if( NULL == tensor || NULL == buffer )
    {
        return 0;
    }
    sz = 0;
    status = VSI_FAILURE;

    status = vsi_nn_copy_tensor_patch(tensor->t, &tensor->attr, buffer, VX_READ_ONLY);
    if(VSI_SUCCESS == status)
    {
        sz = vsi_nn_GetStrideSize( &tensor->attr, stride_size );
    }
    return sz;
} /* vsi_nn_CopyTensorToData() */

float * vsi_nn_ConvertTensorToFloat32Data
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *tensor
    )
{
    vsi_status status;
    uint8_t *tensor_data = NULL;
    uint32_t elements;
    uint32_t i,stride;
    float *data;

    if(NULL == graph || NULL == tensor)
    {
        return NULL;
    }

    elements = vsi_nn_GetElementNum(tensor);
    stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);

    data = NULL;
    data = (float *)malloc(elements * sizeof(float));

    if( tensor->attr.is_created_from_handle )
    {
        vxSwapTensorHandle(tensor->t, NULL, (void**)&tensor_data);
        if ( tensor_data == NULL )
        {
            VSILOGE("vxSwapTensorHandle fail.");
            if( data )
            {
                free( data );
                data = NULL;
            }
            return NULL;
        }
    }
    else
    {
        tensor_data = vsi_nn_ConvertTensorToData(graph, tensor);
    }
    for(i = 0; i < elements; i++)
    {
        status = dtype_to_float32(&tensor_data[stride * i], &data[i], &tensor->attr.dtype);
        if(status != VSI_SUCCESS)
        {
            free(data);
            data = NULL;
            break;
        }
    }

    if( !tensor->attr.is_created_from_handle )
    {
        if(tensor_data)free(tensor_data);
    }
    return data;
} /* vsi_nn_ConvertTensorToFloat32Data() */

uint8_t * vsi_nn_ConvertTensorToData
    (
    const vsi_nn_graph_t * graph,
    vsi_nn_tensor_t * tensor
    )
{
    uint8_t    * data;
    uint32_t     buf_sz;
    uint32_t     stride_size[VSI_NN_MAX_DIM_NUM];
    vsi_status     status;
    if( NULL == tensor )
    {
        return NULL;
    }

    status = VSI_FAILURE;
    data = NULL;

    buf_sz = vsi_nn_GetStrideSize( &tensor->attr, stride_size );
    // TODO: Fix this to use copy tensor to buffer
    if( buf_sz > 0 )
    {
        data = (uint8_t *)malloc( buf_sz );
    }
    if( data && tensor->attr.is_created_from_handle )
    {
        uint8_t* tensor_data = NULL;
        vxSwapTensorHandle( tensor->t, NULL, (void **)&tensor_data );
        if ( tensor_data == NULL )
        {
            VSILOGE("vxSwapTensorHandle fail.");
            if( data )
            {
                free( data );
                data = NULL;
            }
            return NULL;
        }
        memcpy( data, tensor_data, buf_sz);
    }
    else
    {
        if( NULL != data )
        {
            status = vsi_nn_copy_tensor_patch(tensor->t, &tensor->attr, data, VX_READ_ONLY);
        }
        if(VSI_SUCCESS != status)
        {
            VSILOGE("Read tensor data fail");
            free(data);
            data = NULL;
        }
    }
    return data;

} /* vsi_nn_ConvertTensorToData() */

/*
* Deprecated: Use vsi_nn_ConvertRawTensorToData2() instead
* WARNING: This is a bad API,
*          please add a new API for WRITE_ONLY accessor.
*/
uint8_t * vsi_nn_ConvertRawTensorToData
    (
    vx_context context,
    vx_tensor tensor,
    uint32_t * dim,
    vx_enum  * data_format,
    uint32_t * size,
    uint32_t * stride_size,
    vx_tensor_addressing * addr,
    vx_enum accessor
    )
{
    uint8_t    * data;
    uint32_t     buf_sz;
    vsi_status     status;
    vsi_nn_tensor_attr_t attr;
    if( NULL == tensor || NULL == context )
    {
        return NULL;
    }

    status = VSI_FAILURE;
    data = NULL;

    status = vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS, dim, sizeof(uint32_t));
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS, size, sizeof(uint32_t) * (*dim));
    status = vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, data_format, sizeof(vsi_enum));
    attr.dim_num = *dim;
    memcpy(attr.size, size, sizeof(uint32_t) * attr.dim_num);

    buf_sz = vsi_nn_GetStrideSizeBySize(size, *dim, *data_format, stride_size);
    // TODO: Fix this to use copy tensor to buffer
    if( buf_sz > 0 )
    {
        data = (uint8_t *)malloc( buf_sz );
    }
    if( NULL != data )
    {
        if (accessor != VX_READ_ONLY)
        {
            return data;
        }
        status = vsi_nn_copy_tensor_patch(tensor, &attr, data, VX_READ_ONLY);
        if( VSI_SUCCESS != status )
        {
            VSILOGE("Read tensor data fail");
            free(data);
            data = NULL;
        }
    }
    return data;
} /* vsi_nn_ConvertRawTensorToData() */

/*
* WARNING: This is a bad API,
*          please add the new APIs for WRITE_ONLY and READ_ONLY.
*          Then deprecate this function.
*/
uint8_t * vsi_nn_ConvertRawTensorToData2
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t * attr,
    uint32_t * stride_size,
    vx_tensor_addressing * addr,
    vx_enum accessor
    )
{
    uint8_t * data;
    uint32_t buf_sz;
    vsi_status status;

    if( NULL == tensor || NULL == context )
    {
        return NULL;
    }

    status = VSI_FAILURE;
    data = NULL;

    status = vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS,
        &(attr->dim_num), sizeof(uint32_t));
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS,
        attr->size, sizeof(uint32_t) * (attr->dim_num));
    status = vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE,
        &(attr->dtype.vx_type), sizeof(vsi_enum));
    status = vxQueryTensor(tensor, VX_TENSOR_QUANT_FORMAT,
        &(attr->dtype.qnt_type), sizeof(uint32_t));
    switch( attr->dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        status = vxQueryTensor(tensor, VX_TENSOR_FIXED_POINT_POS,
            &(attr->dtype.fl), sizeof(int8_t));
        break;
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        status = vxQueryTensor(tensor, VX_TENSOR_ZERO_POINT,
            &(attr->dtype.zero_point), sizeof(int32_t));
        status = vxQueryTensor(tensor, VX_TENSOR_SCALE,
            &(attr->dtype.scale), sizeof(float));
        break;
    default:
        break;
    }

    buf_sz = vsi_nn_GetStrideSize( attr, stride_size );
    // TODO: Fix this to use copy tensor to buffer
    if( buf_sz > 0 )
    {
        data = (uint8_t *)malloc( buf_sz );
    }
    if( NULL != data )
    {
        if (accessor != VX_READ_ONLY)
        {
            return data;
        }
        status = vsi_nn_copy_tensor_patch(tensor, attr, data, VX_READ_ONLY);
        if( VSI_SUCCESS != status )
        {
            VSILOGE("Read tensor data fail");
            free(data);
            data = NULL;
        }
    }
    return data;
} /* vsi_nn_ConvertRawTensorToData2() */

void vsi_nn_SaveTensorToTextByFp32
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_t  * tensor,
    const char       * filename,
    char             * seperator
    )
{
#define _TENSOR_TMPBUF_SZ  (512)
    const float   c_flush_th = 0.7f;
    uint8_t    * data;
    uint8_t    * ptr;
    uint32_t     type_bytes;
    uint8_t      buf[_TENSOR_TMPBUF_SZ];
    FILE        * fp;
    float    write_data;
    uint32_t     sz;
    uint32_t     i;
    uint32_t     count;

    if( NULL == graph || NULL == tensor || NULL == filename )
    {
        return;
    }
    if( NULL == seperator )
    {
        seperator = "\n";
    }

    data = vsi_nn_ConvertTensorToData( graph, tensor );
    if( NULL == data )
    {
        VSILOGE( "Convert data fail." );
        return;
    }

    fp = fopen( filename, "w" );
    sz = vsi_nn_GetElementNum( tensor );

    ptr = data;
    type_bytes = vsi_nn_TypeGetBytes( tensor->attr.dtype.vx_type );
    count = 0;
    for( i = 0; i < sz; i ++ )
    {
        vsi_nn_DtypeToFloat32( ptr, &write_data, &tensor->attr.dtype );
        ptr += type_bytes;

        count += snprintf( (char *)&buf[count], _TENSOR_TMPBUF_SZ - count,
            "%f%s", write_data, seperator );
        if( ((float)count / _TENSOR_TMPBUF_SZ) > c_flush_th )
        {
            fwrite( buf, count, 1, fp );
            count = 0;
        }
    }
    fwrite( buf, count, 1, fp );
    fclose( fp );
    free( data );
} /* vsi_nn_SaveTensorToTextByFp32() */

void vsi_nn_SaveTensorToText
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_t  * tensor,
    const char       * filename,
    char             * seperator
    )
{
    uint8_t * data;
    uint32_t  sz;

    if( NULL == graph || NULL == tensor || NULL == filename )
    {
        return;
    }

    data = vsi_nn_ConvertTensorToData( graph, tensor );
    if( NULL == data )
    {
        VSILOGE( "Convert data fail." );
        return;
    }

    sz = vsi_nn_GetElementNum( tensor );
    vsi_nn_SaveDataToText( filename, data, sz,
        tensor->attr.dtype.vx_type, seperator );
    free( data );
} /* vsi_nn_SaveTensorToText() */

void vsi_nn_SaveDataToText
    (
    const char  * filename,
    uint8_t    * data,
    uint32_t     data_size,
    vsi_nn_type_e type,
    char        * seperator
    )
{
#define _TENSOR_TMPBUF_SZ  (512)
    const float   c_flush_th = 0.7f;
    uint8_t      buf[_TENSOR_TMPBUF_SZ];
    FILE        * fp;
    float    write_data;
    uint32_t     type_bytes;
    uint32_t     i;
    uint32_t     count;

    if(  NULL == filename )
    {
        return;
    }
    if( NULL == seperator )
    {
        seperator = "\n";
    }

    if( NULL == data )
    {
        return;
    }

    fp = fopen( filename, "w" );
    type_bytes = vsi_nn_GetTypeBytes( type );

    count = 0;
    for( i = 0; i < data_size; i ++ )
    {
        write_data = vsi_nn_DataAsFloat32( &data[type_bytes * i],
            type );
        if( type == VSI_NN_TYPE_UINT8 || type == VSI_NN_TYPE_INT8 )
        {
            count += snprintf( (char *)&buf[count], _TENSOR_TMPBUF_SZ - count,
                "%d%s", (int32_t)write_data, seperator );
        }
        else
        {
            count += snprintf( (char *)&buf[count], _TENSOR_TMPBUF_SZ - count,
                "%f%s", write_data, seperator );
        }
        if( ((float) count / _TENSOR_TMPBUF_SZ ) > c_flush_th )
        {
            fwrite( buf, count, 1, fp );
            count = 0;
        }
    }
    fwrite( buf, count, 1, fp );
    fclose( fp );
} /* vsi_nn_SaveDataToText() */

void vsi_nn_SaveTensorToBinary
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_t  * tensor,
    const char       * filename
    )
{
    uint8_t        * data;
    FILE            * fp;
    uint32_t         sz;
    uint32_t         i;

    if( NULL == graph || NULL == tensor || NULL == filename )
    {
        return;
    }

    data = vsi_nn_ConvertTensorToData( graph, tensor );
    if( NULL == data )
    {
        VSILOGE( "Convert data fail." );
        return;
    }

    fp = fopen( filename, "wb" );
    sz = vsi_nn_GetTypeBytes( tensor->attr.dtype.vx_type );
    for( i = 0; i < tensor->attr.dim_num; i ++ )
    {
        sz *= tensor->attr.size[i];
    }
    fwrite( data, sz, 1, fp );
    fclose( fp );
    free( data );
} /* vsi_nn_SaveTensorToBinary() */

vsi_nn_tensor_t * vsi_nn_CreateTensorFromData
    (
    vsi_nn_graph_t       * graph,
    uint8_t             * data,
    vsi_nn_tensor_attr_t * attr
    )
{
    vsi_status         status;
    vsi_nn_tensor_t * tensor;

    status = VSI_FAILURE;
    tensor = NULL;

    if( NULL == graph || NULL == data || NULL == attr )
    {
        return NULL;
    }

    tensor = vsi_nn_CreateTensor( graph, attr );

    status = vsi_nn_CopyDataToTensor( graph, tensor, data );

    if( VSI_SUCCESS != status )
    {
        VSILOGE("Create tensor from data fail.");
        if( NULL != tensor )
        {
            vsi_nn_ReleaseTensor( &tensor );
        }
    }
    return tensor;
} /* vsi_nn_CreateTensorFromData() */

vsi_status vsi_nn_CopyDataToTensor
    (
    const vsi_nn_graph_t * graph,
    vsi_nn_tensor_t      * tensor,
    void                 * data
    )
{
    vsi_status         status = VSI_FAILURE;
    if( NULL == data || NULL == tensor )
    {
        return status;
    }

    if( tensor->attr.is_created_from_handle )
    {
        uint8_t* ptr = NULL;
        vxSwapTensorHandle( tensor->t, NULL, (void **)&ptr);
        if ( ptr == NULL )
        {
            VSILOGE("vxSwapTensorHandle fail.");
            return VSI_FAILURE;
        }
        memcpy( ptr, data, vsi_nn_GetTensorSize(tensor->attr.size, tensor->attr.dim_num,
                    tensor->attr.dtype.vx_type));
        status = vxSwapTensorHandle( tensor->t, ptr, NULL );
        status |= vxFlushHandle( (vx_reference)tensor->t );
    }
    else
    {
        status = vsi_nn_copy_tensor_patch(tensor->t, &tensor->attr, data, VX_WRITE_ONLY);
    }
    return status;
} /* vsi_nn_CopyDataToTensor() */

vsi_status vsi_nn_FlushHandle
    (
    const vsi_nn_tensor_t * tensor
    )
{
    if ( NULL == tensor || NULL == tensor->t )
    {
        return VSI_FAILURE;
    }
    else
    {
        return vxFlushHandle( (vx_reference)tensor->t );
    }
} /* vsi_nn_FlushHandle() */

vsi_status vsi_nn_GetTensorHandle
    (
    vsi_nn_tensor_t      * tensor,
    void** ptr
    )
{
    if ( NULL == tensor || NULL == tensor->t )
    {
        return VSI_FAILURE;
    }
    else
    {
        return vxSwapTensorHandle(tensor->t, NULL, ptr);
    }
} /* vsi_nn_GetTensorHandle() */

vsi_status vsi_nn_CopyRawDataToTensor
    (
    vsi_nn_graph_t*         graph,
    uint8_t*                src_data,
    const vsi_nn_dtype_t*   src_dtype,
    vsi_nn_tensor_t*        tensor
    )
{
    vsi_status status           = VSI_FAILURE;
    uint32_t src_data_sz   = 0;
    uint8_t* buffer             = NULL;
    uint32_t target_tensor_size = 0; /* in bytes */

    src_data_sz = vsi_nn_GetElementNum(tensor) * vsi_nn_GetTypeBytes(src_dtype->vx_type);
    target_tensor_size = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num, tensor->attr.dtype.vx_type );
    buffer = (uint8_t *)malloc(target_tensor_size);

    vsi_nn_DtypeConvertRawData(src_data, src_data_sz, src_dtype, buffer, target_tensor_size, &tensor->attr.dtype);
    status = vsi_nn_CopyDataToTensor(graph, tensor, buffer);

    if( NULL != buffer )
    {
        free( buffer );
        buffer = NULL;
    }
    return status;
} /* vsi_nn_CopyRawDataToTensor */

vsi_bool vsi_nn_CalcReshapeTensor
    (
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    uint32_t        * shape,
    uint32_t          dim_num
    )
{
    vsi_bool ret;
    uint32_t i;
    uint32_t total_size;
    uint32_t dst_size;

    if( NULL == input || NULL == output
        || NULL == shape || 0 == dim_num )
    {
        VSILOGE( "Wrong reshape parameters." );
        return FALSE;
    }

    ret = _auto_cal_shape( input->attr.size, input->attr.dim_num, shape, &dim_num );
    if( FALSE == ret )
    {
        return ret;
    }

    /* Check total size */
    total_size = vsi_nn_ShapeProduct( input->attr.size, input->attr.dim_num );
    dst_size = vsi_nn_ShapeProduct( shape, dim_num );
    if( total_size != dst_size )
    {
        VSILOGE( "Cannot calculate the reshape tensor %u to %u.",
            total_size, dst_size );
        return FALSE;
    }

    if( TRUE == ret )
    {
        if( VSI_NN_DIM_AUTO == output->attr.dim_num )
        {
            for( i = 0; i < dim_num; i ++ )
            {
                output->attr.size[i] = shape[i];
            }
            output->attr.dim_num = dim_num;
        }
    }

    return ret;
} /* vsi_nn_CalcReshapeTensor() */

/*
    This function will create a new tensor,
    and reshape input to output.
*/
vsi_nn_tensor_t *vsi_nn_reshape_tensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * input,
    uint32_t        * shape,
    uint32_t          dim_num
    )
{
    vsi_bool ret;
    vsi_nn_tensor_t *output = NULL;
    vsi_nn_tensor_attr_t attr;
    if (NULL == graph || NULL == input || NULL == shape)
    {
        return NULL;
    }
    /* New a ovxlib tensor struct */
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    memcpy(&attr, &input->attr, sizeof(vsi_nn_tensor_attr_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    output = vsi_nn_CreateTensor(graph, &attr);
    if (NULL == output)
    {
        VSILOGW("Create tensor fail.");
        return NULL;
    }

    ret = vsi_nn_ReshapeTensor(graph, input, output, shape, dim_num);
    if (FALSE == ret)
    {
        VSILOGW("Reshape tensor fail.");
        vsi_nn_ReleaseTensor(&output);
        output = NULL;
    }

    return output;
} /* vsi_nn_reshape_tensor() */

vsi_bool vsi_nn_ReshapeTensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    const uint32_t  * shape,
    uint32_t         dim_num
    )
{
    vsi_bool ret;
    uint32_t new_shape[VSI_NN_MAX_DIM_NUM] = {0};
    memcpy(new_shape, shape, sizeof(uint32_t) * dim_num);

    ret = TRUE;
    ret = vsi_nn_CalcReshapeTensor(input, output, new_shape, dim_num);
    if( FALSE == ret )
    {
        return FALSE;
    }

    /* Create a openvx tensor if it is not exist */
    if( NULL == input->t )
    {
        ret = vsi_nn_TensorReinit( graph, input );
    }

    /* We can not reshape input to output if output->t is already exist */
    if( NULL != output->t )
    {
        VSILOGW( "Free tensor." );
    }

    /* Create reshape tensor */
    output->t = vxReshapeTensor( input->t, (int32_t *)new_shape, dim_num );
    if( NULL == output->t )
    {
        ret = FALSE;
    }

    if( FALSE == ret )
    {
        VSILOGW( "Reshape tensor error." );
    }

    return ret;
} /* vsi_nn_ReshapeTensor() */

void vsi_nn_TransposeTensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    uint32_t       * perm,
    uint32_t         dim_num,
    uint32_t       * as_shape
    )
{
    uint8_t * buf;
    uint8_t * dst;
    uint32_t  buf_sz;
    uint32_t  tensor_sz;
    uint32_t * shape_ptr;
    vsi_status  status;

    if( NULL == tensor || NULL == perm || 0 == dim_num )
    {
        VSILOGE( "Wrong perm dims." );
        return;
    }
    tensor_sz = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num,
        tensor->attr.dtype.vx_type );
    shape_ptr = tensor->attr.size;

    if( NULL != as_shape )
    {
        buf_sz = vsi_nn_GetTensorSize( as_shape, dim_num, tensor->attr.dtype.vx_type );
        if( buf_sz != tensor_sz )
        {
            VSILOGW( "The shape does not match origin tensor's shape." );
            return;
        }
        shape_ptr = as_shape;
    }
    buf = vsi_nn_ConvertTensorToData( graph, tensor );

    if( NULL == buf )
    {
        VSILOGE( "Create tensor buf fail." );
        return;
    }
    dst = (uint8_t *)malloc( tensor_sz * sizeof( uint8_t ) );
    // TODO: Check memory allocate.

    vsi_nn_Transpose( dst, buf, shape_ptr, dim_num, perm, tensor->attr.dtype.vx_type );
    status = vsi_nn_CopyDataToTensor( graph, tensor, dst );
    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Copy transpose data fail with code %#x.", status );
    }

    free( buf );
    free( dst );
} /* vsi_nn_TransposeTensor() */

void vsi_nn_PermuteTensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    uint32_t       * perm,
    uint32_t         dim_num
    )
{
    uint8_t * buf = NULL;
    uint8_t * dst = NULL;
    uint32_t  tensor_sz;
    uint32_t * shape_ptr;
    uint32_t   dst_shape[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t i;
    vsi_status  status;

    if( NULL == tensor || NULL == perm || 0 == dim_num )
    {
        VSILOGE( "Wrong perm parameters." );
        return;
    }
    tensor_sz = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num,
        tensor->attr.dtype.vx_type );
    shape_ptr = tensor->attr.size;

    buf = vsi_nn_ConvertTensorToData( graph, tensor );

    if( NULL == buf )
    {
        VSILOGE( "Create tensor buf fail." );
        return;
    }
    dst = (uint8_t *)malloc( tensor_sz * sizeof( uint8_t ) );
    if ( NULL == dst)
    {
        VSILOGE( "Malloc dst buf fail." );
        if( buf ) { free(buf); buf = NULL; }
        return;
    }

    for ( i = 0; i < dim_num; i++)
    {
        if( perm[i] >= dim_num )
        {
            VSILOGW( "Incorrect perm %d", perm[i] );
            if( buf ) { free(buf); buf = NULL; }
            if( dst ) { free(dst); dst = NULL; }
            return;
        }
        dst_shape[i] = shape_ptr[perm[i]];
    }
    vsi_nn_Permute( dst, buf, shape_ptr, dim_num, perm, tensor->attr.dtype.vx_type );
    memcpy(tensor->attr.size, dst_shape, sizeof(dst_shape));
    tensor->t = vxReshapeTensor(tensor->t, (int32_t *)tensor->attr.size, tensor->attr.dim_num);
    status = vsi_nn_CopyDataToTensor( graph, tensor, dst );
    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Copy permute data fail with code %#x.", status );
    }

    if( buf ) { free(buf); buf = NULL; }
    if( dst ) { free(dst); dst = NULL; }
} /* vsi_nn_PermuteTensor() */

uint32_t vsi_nn_GetElementNum
    (
    const vsi_nn_tensor_t * tensor
    )
{
    if( NULL == tensor )
    {
        return 0;
    }

    return get_tensor_elements_num(tensor->attr.size,
        tensor->attr.dim_num, tensor->attr.dtype.vx_type);
} /* vsi_nn_GetElementNum() */

uint32_t vsi_nn_GetTensorSize
    (
    const uint32_t * shape,
    uint32_t dim_num,
    vsi_nn_type_e type
    )
{
    uint32_t sz;
    uint32_t i;
    sz = 0;
    if( NULL == shape || 0 == dim_num )
    {
        return sz;
    }
    sz = 1;
    for( i = 0; i < dim_num; i ++ )
    {
        sz *= shape[i];
    }
    sz *= vsi_nn_GetTypeBytes( type );
    return sz;
} /* vsi_nn_GetTensorSize() */

vsi_nn_tensor_t * vsi_nn_VariableToTensor
    (
    vsi_nn_node_t * self,
    uint8_t * data,
    vsi_nn_type_e type
    )
{
    vsi_nn_tensor_t * tensor;
    vsi_nn_tensor_attr_t attr;

    if(NULL == data || NULL == self)
    {
        return NULL;
    }

    memset( &attr, 0, sizeof( attr ) );
    attr.size[0] = 1;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = type;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        data,
        &attr);
    if(NULL == tensor)
    {
        return NULL;
    }

    return tensor;
} /* vsi_nn_VariableToTensor() */

/*
    type 0x01: input
    type 0x02: output
    type 0x03: all
*/
void vsi_nn_print_node_io
    (
    vsi_nn_graph_t *graph,
    vsi_nn_node_t *node,
    int32_t type
    )
{
    uint32_t i;
    vsi_nn_tensor_id_t id;
    vsi_nn_tensor_t *tensor;
    char index[32];
#define _TYPE_INPUT 0x01
#define _TYPE_OUTPUT 0x02
    if (!(type & _TYPE_INPUT) && !(type & _TYPE_OUTPUT))
    {
        VSILOGW("Can't handle this node io type %d", type);
        return;
    }

    if (type & _TYPE_INPUT)
    {
        for (i = 0; i < node->input.num; i++)
        {
            id = node->input.tensors[i];
            tensor = vsi_nn_GetTensor(graph, id);
            snprintf(index, 32, "in(%d) :", i);
            print_tensor(tensor, id, index);
        }
    }
    if (type & _TYPE_OUTPUT)
    {
        for (i = 0; i < node->output.num; i++)
        {
            id = node->output.tensors[i];
            tensor = vsi_nn_GetTensor(graph, id);
            snprintf(index, 32, "out(%d):", i);
            print_tensor(tensor, id, index);
        }
    }
}

void vsi_nn_PrintNodeIO
    (
    vsi_nn_graph_t *graph,
    vsi_nn_node_t *node
    )
{
    vsi_nn_print_node_io(graph, node, 0x03);
} /* vsi_nn_PrintNodeIO() */

void vsi_nn_PrintTensor
    (
    vsi_nn_tensor_t * tensor,
    vsi_nn_tensor_id_t id
    )
{
    print_tensor(tensor, id, NULL);
} /* vsi_nn_PrintTensor() */

vx_tensor vsi_nn_CreateViewTensor
    (
    vsi_nn_graph_t *graph,
    uint32_t *start,
    uint32_t *end,
    vsi_nn_tensor_t *tensor
    )
{
    size_t i,view_dim;
    size_t view_start[VSI_NN_MAX_DIM_NUM] = {0};
    size_t view_end[VSI_NN_MAX_DIM_NUM] = {0};
    vx_tensor view_tensor;
    if(NULL == graph
        || NULL == start
        || NULL == end
        || NULL == tensor)
    {
        return NULL;
    }

    view_dim = (size_t)tensor->attr.dim_num;
    for(i = 0; i < view_dim; i++)
    {
        view_start[i] = (size_t)start[i];
        view_end[i] = (size_t)end[i];
    }
    view_tensor = vxCreateTensorFromView( tensor->t, view_dim, view_start, view_end );
    if( NULL == view_tensor )
    {
        VSILOGE("Call vxCreateTensorFromView fail.");
        return NULL;
    }

    return view_tensor;
} /* vsi_nn_CreateViewTensor() */

void *vsi_nn_Malloc
    (
    size_t size
    )
{
    void *mem = malloc(size);
    return mem;
} /* vsi_nn_Malloc() */

void vsi_nn_Free
    (
    void * data
    )
{
    if(NULL != data)
    {
        free(data);
        data = NULL;
    }
} /* vsi_nn_Free() */

void vsi_nn_ReleaseTensorRelevance
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_rel_t *tensor_ref
    )
{
    uint32_t i;
    if(NULL == tensor_ref || NULL == graph)
    {
        return ;
    }

    for(i = 0; i < graph->tensor_num; i++)
    {
        if(tensor_ref[i].input.table)
        {
            free(tensor_ref[i].input.table);
            tensor_ref[i].input.table = NULL;
        }
        if(tensor_ref[i].output.table)
        {
            free(tensor_ref[i].output.table);
            tensor_ref[i].output.table = NULL;
        }
    }

    if(tensor_ref)
    {
        free(tensor_ref);
        tensor_ref = NULL;
    }
} /* vsi_nn_ReleaseTensorRelevance() */

vsi_nn_tensor_rel_t *vsi_nn_CreateTensorRelevance
    (
    vsi_nn_graph_t *graph
    )
{
    uint32_t i,j,k;
    uint32_t in_num,out_num;
    uint32_t max_io,tensor_num;
    vsi_nn_tensor_rel_t *tensor_ref;
    vsi_nn_node_t *node;

#define _MAX_TENSOR_IO 32
    max_io = _MAX_TENSOR_IO;
    tensor_num = graph->tensor_num;
    tensor_ref = _init_tensor_rel_buffer(graph, max_io);
    if(NULL == tensor_ref)
    {
        VSILOGE("init tensor_ref buffer fail");
        return NULL;
    }

    for (i = 0; i < tensor_num; i++)
    {
        in_num = 0;
        out_num = 0;

        for(j = 0; j < graph->node_num; j++)
        {
            node = vsi_nn_GetNode( graph, (vsi_nn_node_id_t)j );
            for(k = 0; k < node->output.num; k++)
            {
                if(node->output.tensors[k] == i)
                {
                    if(in_num > max_io)
                    {
                        VSILOGW("tensor ref input num > max_io %u, stop build", max_io);
                        break;
                    }
                    tensor_ref[i].input.table[in_num].node  = j;
                    tensor_ref[i].input.table[in_num].index = k;
                    in_num++;
                }
            }
            for(k = 0; k < node->input.num; k++)
            {
                if(node->input.tensors[k] == i)
                {
                    if(out_num > max_io)
                    {
                        VSILOGW("tensor ref output num > max_io %u, stop build", max_io);
                        break;
                    }
                    tensor_ref[i].output.table[out_num].node  = j;
                    tensor_ref[i].output.table[out_num].index = k;
                    out_num++;
                }
            }
        }
        tensor_ref[i].input.num = in_num;
        tensor_ref[i].output.num = out_num;
    }

    return tensor_ref;
} /* vsi_nn_CreateTensorRelevance() */

vsi_status vsi_nn_SwapTensorHandle
    (
    vsi_nn_tensor_t * tensor0,
    vsi_nn_tensor_t * tensor1
    )
{
     uint32_t stride_size[VSI_NN_MAX_DIM_NUM];
     uint32_t buf_sz0, buf_sz1;
     vsi_status status = VSI_FAILURE;

    if( NULL == tensor0 || NULL == tensor1 )
    {
        VSILOGE("tensor0 or tensor1 is NULL.");
        return VSI_FAILURE;
    }

    if( !tensor0->attr.is_created_from_handle || !tensor1->attr.is_created_from_handle )
    {
        VSILOGE("tensor0 or tensor1 is not created form handle.");
        return VSI_FAILURE;
    }

    buf_sz0 = vsi_nn_GetStrideSize( &tensor0->attr, stride_size );
    buf_sz1 = vsi_nn_GetStrideSize( &tensor0->attr, stride_size );

    if( buf_sz0 != buf_sz1 )
    {
        VSILOGE("The memory size of tensor0 and tensor1 are not equal.");
        return VSI_FAILURE;
    }

    status = vxSwapTensor( tensor0->t, tensor1->t );
    if( VX_SUCCESS == status )
    {
        tensor0->is_swapped = TRUE;
        tensor1->is_swapped = TRUE;
    }

    return status;
} /* vsi_nn_SwapTensorHandle() */

uint32_t vsi_nn_vxGetTensorElementNum
    (
    vsi_nn_tensor_attr_t *attr
    )
{
    if( NULL == attr )
    {
        return 0;
    }

    return get_tensor_elements_num(attr->size,
        attr->dim_num, attr->dtype.vx_type);
}

vsi_status vsi_nn_vxGetTensorAttr
    (
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr
    )
{
    vsi_status status = VSI_FAILURE;

    if(NULL == tensor || NULL == attr)
    {
        return status;
    }

    status = vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS,
        &(attr->dim_num), sizeof(uint32_t));
    TEST_CHECK_STATUS( status, final );
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS,
        attr->size, sizeof(uint32_t) * (attr->dim_num));
    TEST_CHECK_STATUS( status, final );
    status = vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE,
        &(attr->dtype.vx_type), sizeof(vsi_enum));
    TEST_CHECK_STATUS( status, final );
    status = vxQueryTensor(tensor, VX_TENSOR_QUANT_FORMAT,
        &(attr->dtype.qnt_type), sizeof(uint32_t));
    TEST_CHECK_STATUS( status, final );
    switch( attr->dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        status = vxQueryTensor(tensor, VX_TENSOR_FIXED_POINT_POS,
            &(attr->dtype.fl), sizeof(int8_t));
        TEST_CHECK_STATUS( status, final );
        break;
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        status = vxQueryTensor(tensor, VX_TENSOR_ZERO_POINT,
            &(attr->dtype.zero_point), sizeof(int32_t));
        TEST_CHECK_STATUS( status, final );
        status = vxQueryTensor(tensor, VX_TENSOR_SCALE,
            &(attr->dtype.scale), sizeof(float));
        TEST_CHECK_STATUS( status, final );
        break;
    default:
        break;
    }

final:
    return status;
} /* vsi_nn_vxGetTensorAttr() */

uint8_t *vsi_nn_vxCopyTensorToData
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr
    )
{
    uint8_t *data;
    vsi_status status;
    uint32_t buf_sz;
    uint32_t stride_size[VSI_NN_MAX_DIM_NUM];

    memset(stride_size, 0, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    if(NULL == tensor || NULL == context || NULL == attr)
    {
        return NULL;
    }
    data = NULL;
    status = VSI_FAILURE;

    buf_sz = vsi_nn_GetStrideSize( attr, stride_size );
    if(0 < buf_sz)
    {
        data = (uint8_t *)malloc( buf_sz );
        if(NULL == data)
        {
            return NULL;
        }
    }

    status = vsi_nn_copy_tensor_patch(tensor, attr, data, VX_READ_ONLY);
    if(VSI_SUCCESS != status)
    {
        VSILOGE("Copy tensor to data fail");
        free(data);
        data = NULL;
    }
    return data;
} /* vsi_nn_vxCopyTensorToData() */

vsi_status vsi_nn_vxCopyDataToTensor
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr,
    uint8_t *data
    )
{
    vsi_status status;
    uint32_t stride_size[VSI_NN_MAX_DIM_NUM];

    status = VSI_FAILURE;
    if(NULL == tensor || NULL == attr ||
       NULL == context || NULL == data)
    {
        return status;
    }

    memset(stride_size, 0, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    vsi_nn_GetStrideSize(attr, stride_size);
    status = vsi_nn_copy_tensor_patch(tensor, attr, data, VX_WRITE_ONLY);
    if(VSI_SUCCESS != status)
    {
        VSILOGE("Copy data to tensor fail");
    }
    return status;
} /* vsi_nn_vxCopyDataToTensor() */

vsi_status vsi_nn_copy_tensor_veiw_patch
    (
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr,
    void *user_ptr,
    uint32_t *start,
    uint32_t *end,
    uint32_t *stride,
    vsi_enum usage,
    vsi_enum user_memory_type
    )
{
#define USE_OPENVX_1_2
    size_t dim,i;
    size_t vstart[VSI_NN_MAX_DIM_NUM],vend[VSI_NN_MAX_DIM_NUM],vstride[VSI_NN_MAX_DIM_NUM];
    vsi_status status = VSI_FAILURE;
    if(NULL == tensor || NULL == user_ptr || NULL == start || NULL == end || NULL == stride)
    {
        VSILOGE("Invalid parameter");
        return status;
    }
    dim = (size_t)attr->dim_num;
    for(i = 0; i < dim; i++)
    {
        vstart[i] = (size_t)start[i];
        vend[i] = (size_t)end[i];
        vstride[i] = (size_t)stride[i];
    }

#ifdef USE_OPENVX_1_2
    status = vxCopyTensorPatch(tensor, dim, vstart, vend, vstride, user_ptr, usage, user_memory_type);
#else
    {
        vx_context context = NULL;
        vx_tensor_addressing addr = NULL;
        uint32_t stride_size[VSI_NN_MAX_DIM_NUM];
        vsi_nn_tensor_attr_t t;

        memset(vstart, 0, sizeof(size_t) * VSI_NN_MAX_DIM_NUM);
        memset(vend, 0, sizeof(size_t) * VSI_NN_MAX_DIM_NUM);
        memset(vstride, 0, sizeof(size_t) * VSI_NN_MAX_DIM_NUM);
        status = vsi_nn_vxGetTensorAttr(tensor, &t);
        vsi_nn_GetStrideSize( attr, stride_size );
        context = vxGetContext((vx_reference)tensor);
        if( NULL == context )
        {
            VSILOGE("Call vxGetContext fail");
            return status;
        }
        addr = vxCreateTensorAddressing( context, attr->size,
            stride_size, attr->dim_num );
        if( NULL == addr )
        {
            VSILOGE("Call vxCreateTensorAddressing fail");
            return status;
        }
        status = vxCopyTensorPatch_11( tensor,
                                       NULL,
                                       addr,
                                       user_ptr,
                                       usage,
                                       user_memory_type
                                      );
        vxReleaseTensorAddressing( &addr );
        if( VSI_SUCCESS != status )
        {
            VSILOGE("Call vxCopyTensorPatch_11 fail");
            return status;
        }
    }
#endif
    return status;
} /* vsi_nn_copy_tensor_veiw_patch() */

vsi_status vsi_nn_copy_tensor_patch
    (
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr,
    void * user_ptr,
    vsi_enum usage
    )
{
    uint32_t start[VSI_NN_MAX_DIM_NUM],end[VSI_NN_MAX_DIM_NUM],stride[VSI_NN_MAX_DIM_NUM];
    vsi_status status = VSI_FAILURE;
    if(NULL == tensor || NULL == user_ptr)
    {
        VSILOGE("Invalid parameter");
        return status;
    }
    vsi_nn_GetStrideSize(attr, stride);
    memset(start, 0, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    memcpy(end, attr->size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    status = vsi_nn_copy_tensor_veiw_patch(tensor, attr, user_ptr, start, end, stride, usage, 0);
    return status;
} /* vsi_nn_copy_tensor_patch() */

uint32_t vsi_nn_GetOffsetByCoords
    (
    vsi_nn_tensor_attr_t *attr,
    uint32_t *coords
    )
{
    uint32_t i, res = 0, strides = 1;
    for (i = 0; i < attr->dim_num; i++)
    {
        res += coords[i] * strides;
        strides *= attr->size[i];
    }
    return res;
}

void vsi_nn_reshuffle_weight_data
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * weights
    )
{
    int32_t b, sy, sx, c, h, w;
    uint8_t* weight_data = NULL;
    uint8_t* reshuffled_weights = NULL;
    uint8_t* buffer = NULL;
    int32_t kernel_size_x = weights->attr.size[0];
    int32_t kernel_size_y = weights->attr.size[1];
    int32_t weight_size_c = weights->attr.size[2];
    int32_t weight_size_b = weights->attr.size[3];
    int32_t slice_size = kernel_size_x * kernel_size_y;
    int32_t item_size = vsi_nn_TypeGetBytes(weights->attr.dtype.vx_type);

    weight_data = vsi_nn_ConvertTensorToData(graph, weights);
    buffer = (uint8_t*)malloc(item_size * slice_size * weight_size_c * weight_size_b);
    memset(buffer, 0x00, item_size * slice_size * weight_size_c * weight_size_b);
    memcpy(buffer, weight_data, item_size * slice_size * weight_size_c * weight_size_b);
#if 0 // transpose whnc to whcn if need
    for (b = 0; b < weight_size_b; b++)
    {
        for (c = 0; c < weight_size_c; c++)
        {
            memcpy(buffer + kernel_size_x * kernel_size_y * (c * weight_size_b + b) * item_size,
                weight_data + kernel_size_x * kernel_size_y * (b * weight_size_c + c) * item_size,
                item_size * slice_size);
        }
    }
#endif
    reshuffled_weights = weight_data;
    for (b = 0; b < weight_size_b; b++)
    {
        for (sy = 0; sy < 1; sy++)
        {
            for (sx = 0; sx < 1; sx++)
            {
                for (c = 0; c < weight_size_c; c++)
                {
                    uint8_t* weight_output = reshuffled_weights +
                        (b * slice_size * weight_size_c + slice_size * c) * item_size;

                    uint8_t* data = buffer + (b * slice_size * weight_size_c + slice_size * c) * item_size;

                    for (h = 0; h < kernel_size_y; h++)
                    {
                        for (w = 0; w < kernel_size_x; w++)
                        {
                            uint8_t* reshuffled_output = weight_output + (h * kernel_size_x + w) * item_size;
                            int32_t input_index = ((kernel_size_y - 1 - h) + sy) * kernel_size_x +
                                ((kernel_size_x - 1 - w) + sx);

                            memcpy(reshuffled_output, data + input_index * item_size, item_size);
                        }
                    }
                }
            }
        }
    }
    vsi_nn_CopyDataToTensor( graph, weights, weight_data );
    vsi_nn_Free( buffer );
    vsi_nn_Free( weight_data );
}

vsi_nn_tensor_t* vsi_nn_ConcatTensor_impl
    (
    vsi_nn_graph_t* graph,
    uint32_t axis,
    ...
    )
{
    va_list args;
    vsi_nn_tensor_t* next = NULL;
    vsi_nn_tensor_t** tensors = NULL;
    int tensor_count = 0;

    va_start(args, axis);

    FOREACH_ARGS(args, next, vsi_nn_tensor_t*)
    {
        tensor_count++;
    }
    va_end(args);

    tensors = (vsi_nn_tensor_t**)malloc(sizeof(vsi_nn_tensor_t*) * tensor_count);
    tensor_count = 0;
    va_start(args, axis);

    FOREACH_ARGS(args, next, vsi_nn_tensor_t*)
    {
        tensors[tensor_count++] = next;
    }
    va_end(args);

    next = vsi_nn_Concat(graph, tensors, tensor_count, axis);

    vsi_nn_safe_free(tensors);

    return next;
}

vsi_nn_tensor_t* vsi_nn_ConstTensorAdd_impl
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_attr_t output_attr,
    ...
    )
{
    va_list args;
    vsi_nn_tensor_t* next = NULL;
    vsi_nn_tensor_t** tensors = NULL;
    int tensor_count = 0;

    va_start(args, output_attr);
    FOREACH_ARGS(args, next, vsi_nn_tensor_t*)
    {
        tensor_count++;
    }
    va_end(args);

    tensors = (vsi_nn_tensor_t**)malloc(sizeof(vsi_nn_tensor_t*) * tensor_count);
    tensor_count = 0;
    va_start(args, output_attr);
    FOREACH_ARGS(args, next, vsi_nn_tensor_t*)
    {
        tensors[tensor_count++] = next;
    }
    va_end(args);

    next = vsi_nn_TensorAdd(graph, tensors, tensor_count, output_attr);

    vsi_nn_safe_free(tensors);

    return next;
}

vsi_status vsi_nn_SwapHandle
    (
    vsi_nn_tensor_t * tensor,
    void * new_ptr,
    void ** old_ptr
    )
{
    if(!tensor)
    {
        return VSI_FAILURE;
    }
    vxSwapTensorHandle(tensor->t, new_ptr, old_ptr);
    return VSI_SUCCESS;
} /* vsi_nn_SwapHandle() */

