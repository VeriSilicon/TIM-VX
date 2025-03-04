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

#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_tensor_util_prv.h"
#include "vsi_nn_types.h"
#include "vsi_nn_types_prv.h"
#include "vsi_nn_test.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_dtype_util_prv.h"
#include "utils/vsi_nn_tensor_op.h"
#include "vsi_nn_error.h"

static vsi_bool _try_set_const_tensor
    (
    vsi_nn_tensor_t *tensor
    );

static vsi_bool _auto_cal_shape
    (
    vsi_size_t * input_shape,
    vsi_size_t   input_dim,
    vsi_size_t * shape,
    vsi_size_t * dim_num
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
    vsi_nn_tensor_attr_t * attr,
    int8_t                 is_from_axisram
    );

static vsi_size_t get_tensor_elements_num
    (
    const vsi_size_t   * shape,
    vsi_size_t     dim_num,
    vsi_nn_type_e type
    )
{
    vsi_size_t num;
    vsi_size_t sz;
    vsi_size_t dsize;

    sz = vsi_nn_GetTensorSize( shape,
        dim_num, type );
    dsize = vsi_nn_TypeGetBytesExt( type );
    num = sz / dsize;
    return num;
} /* get_tensor_elements_num() */

static void print_tensor
    (
    vsi_nn_tensor_t *tensor,
    vsi_nn_tensor_id_t id,
    char *ext_str
    )
{
#define _SHAPE_BUF_SZ   (128)
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
    vsi_nn_FormatToString( tensor, format, _ATTR_BUF_SZ );

    /* Process quantize parameters */
    switch( tensor->attr.dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        count = snprintf( &ext_attr[0], _EXT_ATTR_BUF_SZ,
            "DFP fl=%3d", tensor->attr.dtype.fl );
        ext_attr[count] = 0;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
    case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
    case VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8:
        count = snprintf( &ext_attr[0], _EXT_ATTR_BUF_SZ,
            "ASM zp=%3d, scale=%.6f",
            tensor->attr.dtype.zero_point, tensor->attr.dtype.scale );
        ext_attr[count] = 0;
        break;
#ifdef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC:
    case VSI_NN_QNT_TYPE_PERCHANNEL_SYMMETRIC_FLOAT8:
        count = snprintf( &ext_attr[0], _EXT_ATTR_BUF_SZ,
            "SYM PERCHANNEL axis=%d, count=%d",
            tensor->attr.dtype.channel_dim, tensor->attr.dtype.scale_dim );
        ext_attr[count] = 0;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_ASYMMETRIC:
        count = snprintf(&ext_attr[0],
                         _EXT_ATTR_BUF_SZ,
                         "ASYM PERCHANNEL axis=%d, count=%d",
                         tensor->attr.dtype.channel_dim,
                         tensor->attr.dtype.scale_dim);
        ext_attr[count] = 0;
        break;
#endif
#ifdef VSI_PER_GROUP_QUANTIZATION_SUPPORT
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_GROUP_SYMMETRIC:
        count = snprintf(&ext_attr[0],
                         _EXT_ATTR_BUF_SZ,
                         "SYM GPTQ axis=%d, count=%d, group_size=%d",
                         tensor->attr.dtype.group_channel_dim,
                         tensor->attr.dtype.group_count,
                         tensor->attr.dtype.group_size);
        ext_attr[count] = 0;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_GROUP_ASYMMETRIC:
        count = snprintf(&ext_attr[0],
                         _EXT_ATTR_BUF_SZ,
                         "ASYM GPTQ axis=%d, count=%d, group_size=%d",
                         tensor->attr.dtype.group_channel_dim,
                         tensor->attr.dtype.group_count,
                         tensor->attr.dtype.group_size);
        ext_attr[count] = 0;
        break;
#endif
    default:
        vsi_nn_strncpy(ext_attr, "NONE", _EXT_ATTR_BUF_SZ);
        break;
    }

    if(ext_str)
    {
        VSILOGD("%s id[%4u] vtl[%d] const[%d] shape[%-18s] is_scalar[%d] fmt[%s] qnt[%s]",
            ext_str,
            id,
            tensor->attr.vtl,
            tensor->attr.is_const,
            shape,
            vsi_nn_GetTensorIsScalar(tensor),
            format,
            ext_attr);
    }
    else
    {
        VSILOGD("id[%4u] vtl[%d] const[%d] shape[%-18s] is_scalar[%d] fmt[%s] qnt[%s]",
            id,
            tensor->attr.vtl,
            tensor->attr.is_const,
            shape,
            vsi_nn_GetTensorIsScalar(tensor),
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
    vsi_size_t * input_shape,
    vsi_size_t   input_dim,
    vsi_size_t * shape,
    vsi_size_t * dim_num
    )
{
    vsi_bool ret;
    vsi_ssize_t  neg_idx;
    vsi_size_t i = 0;
    vsi_size_t total_size = 1;

    ret = TRUE;
    neg_idx = -1;
    total_size = vsi_nn_ShapeProduct( input_shape, input_dim );
    if ((vsi_size_t)-1 == *dim_num)
    {
        *dim_num = 1;
        shape[0] = total_size;
        return ret;
    }

    for( i = 0; i < *dim_num; i ++ )
    {
        if( -1 != (vsi_ssize_t)shape[i] )
        {
            if (0 == shape[i])
            {
                if (i >= input_dim)
                {
                    VSILOGE( "Wrong shape '%"VSI_SSIZE_T_SPECIFIER"' ", (vsi_ssize_t)shape[i] );
                    ret = FALSE;
                    goto final;
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
            VSILOGE( "Wrong shape '%"VSI_SSIZE_T_SPECIFIER"' ", (vsi_ssize_t)shape[i] );
            ret = FALSE;
            goto final;
        }
    }

    if (-1 != neg_idx)
    {
        shape[neg_idx] = (vsi_size_t)total_size;
    }

final:
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
    int32_t * zeroPoints = NULL;
    int32_t * null_zp = NULL;
    vx_size size_vxsize[VSI_NN_MAX_DIM_NUM] = {0};
    vx_uint32 size_u32[VSI_NN_MAX_DIM_NUM] = {0};
    size_t i = 0;
    ret = TRUE;

    if (tensor->attr.dim_num > VSI_NN_MAX_DIM_NUM)
    {
        VSILOGE( "tensor rank greater than %d.", VSI_NN_MAX_DIM_NUM );
        return FALSE;
    }

    memset( &params, 0, sizeof( vx_tensor_create_params_t ) );
    params.num_of_dims = tensor->attr.dim_num;
    for(i = 0; i < tensor->attr.dim_num; i++)
    {
        size_vxsize[i] = (vsi_size_t)-1 == tensor->attr.size[i] ? (vx_size)-1 : (vx_size)tensor->attr.size[i];
    }
    for(i = 0; i < tensor->attr.dim_num; i++)
    {
        size_u32[i] = (vsi_size_t)-1 == tensor->attr.size[i] ? (vx_uint32)-1 : (vx_uint32)tensor->attr.size[i];
    }
#ifdef VSI_40BIT_VA_SUPPORT
    params.sizes = size_vxsize;
    (void)size_u32;
#else
    params.sizes = size_u32;
    (void)size_vxsize;
#endif
    params.data_format = (vsi_enum)tensor->attr.dtype.vx_type;
    switch( tensor->attr.dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        params.quant_format = (vsi_enum)VX_QUANT_DYNAMIC_FIXED_POINT;
        params.quant_data.dfp.fixed_point_pos = (uint8_t)tensor->attr.dtype.fl;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
    case VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8:
        params.quant_format = (vsi_enum)VX_QUANT_AFFINE_SCALE;
        params.quant_data.affine.scale = tensor->attr.dtype.scale;
        params.quant_data.affine.zeroPoint = (int32_t)tensor->attr.dtype.zero_point;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC:
    case VSI_NN_QNT_TYPE_PERCHANNEL_SYMMETRIC_FLOAT8:
#ifdef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
        #ifdef VX_QUANT_AFFINE_SCALE_PER_CHANNEL
            params.quant_format = (vsi_enum)VX_QUANT_AFFINE_SCALE_PER_CHANNEL;
        #else
            params.quant_format = (vsi_enum)VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC;
        #endif
        // This is a hack that driver doesn't support const scales
        scales = (float*)malloc(sizeof(float) * tensor->attr.dtype.scale_dim);
        CHECK_PTR_FAIL_GOTO( scales, "Create buffer fail.", final );
        memcpy(scales, tensor->attr.dtype.scales, tensor->attr.dtype.scale_dim * sizeof(float));
        params.quant_data.affinePerChannel.channelDim = tensor->attr.dtype.channel_dim;
        params.quant_data.affinePerChannel.scaleCount = tensor->attr.dtype.scale_dim;
        params.quant_data.affinePerChannel.scales = scales;
        params.quant_data.affinePerChannel.zeroPoint = NULL;
        params.quant_data.affinePerChannel.zeroPointCount = 0;
        {
            // Low-level driver only support asymmetric. Application doesn't provide zp information if
            // it's symmetric quantized tensor. Fake a zp information filled with zero to meet low-level's
            // requirement
            null_zp = (int32_t*)malloc(sizeof(int32_t) * tensor->attr.dtype.scale_dim);
            CHECK_PTR_FAIL_GOTO( null_zp, "Create buffer fail.", final );
            memset(null_zp, 0, sizeof(int32_t) * tensor->attr.dtype.scale_dim);
            params.quant_data.affinePerChannel.zeroPoint = null_zp;
            params.quant_data.affinePerChannel.zeroPointCount= tensor->attr.dtype.scale_dim;
        }
        break;
#else
    VSILOGE( "can't support qnt_type VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC." );
#endif
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_ASYMMETRIC:
#ifdef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
        #ifdef VX_QUANT_AFFINE_SCALE_PER_CHANNEL
            params.quant_format = (vsi_enum)VX_QUANT_AFFINE_SCALE_PER_CHANNEL;
        #else
            params.quant_format = (vsi_enum)VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC;
        #endif
        // This is a hack that driver doesn't support const scales
        scales = (float*)malloc(sizeof(float) * tensor->attr.dtype.scale_dim);
        CHECK_PTR_FAIL_GOTO( scales, "Create buffer fail.", final );
        memcpy(scales,
               tensor->attr.dtype.scales,
               tensor->attr.dtype.scale_dim * sizeof(float));
        zeroPoints = (int32_t*)malloc(sizeof(int32_t) * tensor->attr.dtype.zero_points_dim);
        CHECK_PTR_FAIL_GOTO( zeroPoints, "Create buffer fail.", final );
        memcpy(zeroPoints,
               tensor->attr.dtype.zero_points,
               tensor->attr.dtype.zero_points_dim * sizeof(int32_t));
        params.quant_data.affinePerChannel.channelDim =
            tensor->attr.dtype.channel_dim;
        params.quant_data.affinePerChannel.scaleCount =
            tensor->attr.dtype.scale_dim;
        params.quant_data.affinePerChannel.scales = scales;
        params.quant_data.affinePerChannel.zeroPoint = zeroPoints;
        params.quant_data.affinePerChannel.zeroPointCount = tensor->attr.dtype.zero_points_dim;
        break;
#else
        VSILOGE(
            "can't support qnt_type "
            "VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_ASYMMETRIC.");
#endif
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_GROUP_SYMMETRIC:
#ifdef VSI_PER_GROUP_QUANTIZATION_SUPPORT
        params.quant_format = (vsi_enum)VX_QUANT_AFFINE_SCALE_PER_GROUP;
        // This is a hack that driver doesn't support const scales
        scales = (float*)malloc(sizeof(float) * tensor->attr.dtype.group_count);
        CHECK_PTR_FAIL_GOTO( scales, "Create buffer fail.", final );
        memcpy(scales, tensor->attr.dtype.group_scales, tensor->attr.dtype.group_count * sizeof(float));
        zeroPoints = (int32_t*)malloc(sizeof(int32_t) * tensor->attr.dtype.zero_points_dim);
        CHECK_PTR_FAIL_GOTO( zeroPoints, "Create buffer fail.", final );
        memcpy(zeroPoints,
               tensor->attr.dtype.zero_points,
               tensor->attr.dtype.zero_points_dim * sizeof(int32_t));
        params.quant_data.affinePerGroup.channel_dim = tensor->attr.dtype.group_channel_dim;
        params.quant_data.affinePerGroup.group_size = tensor->attr.dtype.group_size;
        params.quant_data.affinePerGroup.scale_group_count = tensor->attr.dtype.group_count;
        params.quant_data.affinePerGroup.scales = scales;
        params.quant_data.affinePerGroup.zero_points = NULL;
        params.quant_data.affinePerGroup.zero_point_group_count = 0;
        break;
#else
        VSILOGE(
            "can't support qnt_type "
            "VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_GROUP_SYMMETRIC.");
        break;
#endif
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_GROUP_ASYMMETRIC:
#ifdef VSI_PER_GROUP_QUANTIZATION_SUPPORT
        params.quant_format = (vsi_enum)VX_QUANT_AFFINE_SCALE_PER_GROUP;
        // This is a hack that driver doesn't support const scales
        scales = (float*)malloc(sizeof(float) * tensor->attr.dtype.group_count);
        CHECK_PTR_FAIL_GOTO( scales, "Create buffer fail.", final );
        memcpy(scales, tensor->attr.dtype.group_scales, tensor->attr.dtype.group_count * sizeof(float));
        zeroPoints = (int32_t*)malloc(sizeof(int32_t) * tensor->attr.dtype.zero_points_dim);
        CHECK_PTR_FAIL_GOTO( zeroPoints, "Create buffer fail.", final );
        memcpy(zeroPoints,
               tensor->attr.dtype.group_zero_points,
               tensor->attr.dtype.group_count * sizeof(int32_t));
        params.quant_data.affinePerGroup.channel_dim = tensor->attr.dtype.group_channel_dim;
        params.quant_data.affinePerGroup.group_size = tensor->attr.dtype.group_size;
        params.quant_data.affinePerGroup.scale_group_count = tensor->attr.dtype.group_count;
        params.quant_data.affinePerGroup.scales = scales;
        params.quant_data.affinePerGroup.zero_points = zeroPoints;
        params.quant_data.affinePerGroup.zero_point_group_count = tensor->attr.dtype.group_count;
        break;
#else
        VSILOGE(
            "can't support qnt_type "
            "VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_GROUP_ASYMMETRIC.");
        break;
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
        vx_tensor_addressing addr = NULL;
        vsi_size_t stride_size[VSI_NN_MAX_DIM_NUM];
        vsi_size_t buf_sz;

        buf_sz = vsi_nn_GetStrideSize( &tensor->attr, stride_size );
        if( buf_sz > 0 )
        {
            vsi_size_t align_start_size = graph->handle_manager.align_start_size;
            vsi_size_t align_block_size = graph->handle_manager.align_block_size;
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
                if (TRUE == tensor->attr.is_handle_malloc_by_ovxlib)
                {
                    VSILOGE("Data allocated by OVXLIB should not be shared by other OVXLIB tensors.");
                    ret = FALSE;
                    goto final;
                }
                if (!vsi_nn_IsBufferAligned(data, align_start_size))
                {
                    VSILOGE( "vsi_nn_IsBufferAligned is FALSE." );
                    ret = FALSE;
                    goto final;
                }
            }
            if( data )
            {
                vsi_status status = VSI_FAILURE;
#ifdef VSI_40BIT_VA_SUPPORT
                {
                    vx_size size_vxsize2[_cnt_of_array(tensor->attr.size)] = {0};
                    vx_size stride_size_vxsize[_cnt_of_array(stride_size)] = {0};
                    for(i = 0; i < _cnt_of_array(tensor->attr.size); i++)
                    {
                        size_vxsize2[i] = (vsi_size_t)-1 == tensor->attr.size[i] ? \
                            (vx_size)-1 : (vx_size)tensor->attr.size[i];
                    }
                    for(i = 0; i < _cnt_of_array(stride_size); i++)
                    {
                        stride_size_vxsize[i] = (vsi_size_t)-1 == stride_size[i] ? \
                            (vx_size)-1 : (vx_size)stride_size[i];
                    }
                    addr = vxCreateTensorAddressing(graph->ctx->c,
                        size_vxsize2, stride_size_vxsize, (vx_size)tensor->attr.dim_num);
                    CHECK_PTR_FAIL_GOTO( addr, "Create tensor address fail.", final );
                }
#else
                {
                    uint32_t size_32bit[_cnt_of_array(tensor->attr.size)] = {0};
                    uint32_t stride_size_32bit[_cnt_of_array(stride_size)] = {0};
                    for(i = 0; i < _cnt_of_array(tensor->attr.size); i++)
                    {
                        size_32bit[i] = (vsi_size_t)-1 == tensor->attr.size[i] ? \
                            (uint32_t)-1 : (uint32_t)tensor->attr.size[i];
                    }
                    for(i = 0; i < _cnt_of_array(stride_size); i++)
                    {
                        stride_size_32bit[i] = (vsi_size_t)-1 == stride_size[i] ? \
                            (uint32_t)-1 : (uint32_t)stride_size[i];
                    }
                    addr = vxCreateTensorAddressing(graph->ctx->c,
                        size_32bit, stride_size_32bit, (uint8_t)tensor->attr.dim_num);
                    CHECK_PTR_FAIL_GOTO( addr, "Create tensor address fail.", final );
                }
#endif
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
                if (addr)
                {
                    vxReleaseTensorAddressing( &addr );
                }

                if ( NULL == tensor->t )
                {
                    ret = FALSE;
                    goto final;
                }
                status = vxFlushHandle( (vx_reference)tensor->t );
                if (VSI_SUCCESS != status)
                {
                    VSILOGE("Flush handle fail.");
                    ret = FALSE;
                    goto final;
                }
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
                _set_tensor_handle((vsi_nn_tensor_prv_t*)tensor, data);
#endif
            }
        }
    }
    else if( FALSE == tensor->attr.vtl )
    {
        tensor->t = vxCreateTensor2( graph->ctx->c,
            &params, sizeof( vx_tensor_create_params_t ) );
#ifdef VSI_CREATE_TENSOR_FROM_AXISRAM_SUPPORT
        if (TRUE == _get_tensor_is_from_axisram((vsi_nn_tensor_prv_t*)tensor))
        {
            vx_enum pool_type = VX_VIV_MEM_POOL_TYPE_AXI_SRAM;
            vxSetTensorAttribute(tensor->t,
                                 VX_TENSOR_MEMORY_POOL_TYPE,
                                 &pool_type,
                                 sizeof(vx_enum));
        }
#endif
    }
    else
    {
        tensor->t = vxCreateVirtualTensor2( graph->g,
            &params, sizeof( vx_tensor_create_params_t ) );

        if ((!vsi_nn_IsGraphFastMode(graph))
            && (tensor->t != NULL)
            && (params.data_format == VX_TYPE_FLOAT32))
        {

            vx_enum precision = VX_TENSOR_PRECISION_HIGH;
            vxSetTensorAttribute(tensor->t,
                                      VX_TENSOR_PRECISION,
                                      &precision,
                                      sizeof(vx_enum));
        }
    }
    if ( NULL == tensor->t )
    {
        VSILOGE( "Create vx tensor fail." );
        ret = FALSE;
        goto final;
    }

    if( !tensor->attr.vtl && !tensor->attr.is_const )
    {
        //norm tensor need to fill initial value
#ifdef VSI_CREATE_TENSOR_FROM_AXISRAM_SUPPORT
        if (TRUE != _get_tensor_is_from_axisram((vsi_nn_tensor_prv_t*)tensor))
#endif
        {
            if( ( !tensor->attr.is_created_from_handle ) || tensor->attr.is_handle_malloc_by_ovxlib)
            {
                vsi_nn_FillTensorWithValue( graph, tensor, 0.0f );
                if(tensor->attr.is_created_from_handle)
                {
                    vsi_status status = vxFlushHandle( (vx_reference)tensor->t );
                    if (VSI_SUCCESS != status)
                    {
                        ret = FALSE;
                        goto final;
                    }
                }
            }
        }
    }

    ret = _try_set_const_tensor( tensor );

final:
    if( scales )
    {
        free(scales);
    }
    if (zeroPoints)
    {
        free(zeroPoints);
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
    vsi_nn_tensor_attr_t * attr,
    int8_t                 is_from_axisram
    )
{
    vsi_nn_tensor_prv_t * tensor;

    tensor = NULL;
    if( NULL == graph || NULL == graph->g || NULL == attr )
    {
        return NULL;
    }

    tensor = (vsi_nn_tensor_prv_t *)malloc( sizeof( vsi_nn_tensor_prv_t ) );
    //vsi_nn_UpdateTensorDims( attr );

    if( NULL != tensor )
    {
        memset( tensor, 0, sizeof( vsi_nn_tensor_prv_t ) );
        memcpy( &tensor->pot.attr, attr, sizeof( vsi_nn_tensor_attr_t ) );
        tensor->pot.is_swapped = FALSE;
        if (TRUE == is_from_axisram)
        {
            tensor->is_from_axisram = is_from_axisram;
        }
        if( attr->dim_num != VSI_NN_DIM_AUTO )
        {
            _init_tensor( graph, &tensor->pot, data);
            if( NULL == tensor->pot.t )
            {
                VSILOGE( "Create vx tensor fail." );
                free( tensor );
                tensor = NULL;
            }
        }
    }
    return (vsi_nn_tensor_t*)tensor;
}

vsi_nn_tensor_t * vsi_nn_CreateTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr
    )
{
    attr->is_created_from_handle = FALSE;
    return _create_tensor(graph, NULL, attr, FALSE);
} /* vsi_nn_CreateTensor() */

vsi_nn_tensor_t * vsi_nn_CreateTensorFromHandle
    (
    vsi_nn_graph_t       * graph,
    uint8_t              * data,
    vsi_nn_tensor_attr_t * attr
    )
{
    vsi_nn_tensor_t* ptensor = NULL;
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
    if(attr->vsi_memory_type == VSI_MEMORY_TYPE_NONE || attr->vsi_memory_type == 0)
    {
        attr->vsi_memory_type = VSI_MEMORY_TYPE_HOST;
    }
#endif
    if (TRUE != attr->is_created_from_handle)
    {
        VSILOGE("Could only create tensor with flag 'is_created_from_handle == TRUE'.");
        ptensor = NULL;
        goto final;
    }
    /* 'attr' should contain correct flag is_handle_malloc_by_ovxlib to indicate the if 'data' is
    allocated by OVXLIB. And 'data' allocated by OVXLIB shouldn't be shared by other ovxlib tensors */
    if (NULL != data && TRUE == attr->is_handle_malloc_by_ovxlib)
    {
        VSILOGE("Handle allocated by OVXLIB should not be shared by other OVXLIB tensors.");
        ptensor = NULL;
        goto final;
    }
    else
    {
        ptensor = _create_tensor(graph, data, attr, FALSE);
    }

 final:
    return ptensor;
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
        vsi_size_t size = 0;
        vsi_size_t stride[VSI_NN_MAX_DIM_NUM] = { 0 };
        uint8_t* data = NULL;

        size = vsi_nn_GetStrideSize( &t->attr, stride );
        if( stride[0] == 0 )
        {
            size = vsi_nn_GetElementNum(t);
        }
        data = (uint8_t *)malloc( size );
        if( data )
        {
            vsi_size_t i = 0, j = 0;
            vsi_size_t elements = 0;
            vsi_status status = VSI_FAILURE;

            if(stride[0] != 0)
            {
                elements = size / stride[0];
            }
            status = vsi_nn_Float32ToDtype( defualt_value, &data[0], &t->attr.dtype );
            if(stride[0] == 1 || stride[0] == 0)
            {
                 memset(data, data[0], size);
            }
            else
            {
                for( i = 1; i < elements; i ++ )
                {
                    for(j=0;j<stride[0];j++)
                    {
                        data[stride[0] * i + j] = data[j];
                    }
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
        vsi_size_t size = 0;
        vsi_size_t stride[VSI_NN_MAX_DIM_NUM] = { 0 };
        uint8_t* data = NULL;

        size = vsi_nn_GetStrideSize( &tensor->attr, stride );
        if( stride[0] == 0)
        {
            size = vsi_nn_GetElementNum(tensor);
        }
        data = (uint8_t *)malloc( size );
        if( data )
        {
            vsi_size_t i = 0, j = 0;
            vsi_size_t elements = 0;
            if(stride[0] != 0)
            {
                elements = size / stride[0];
            }
            status = vsi_nn_Float32ToDtype( value, &data[0], &tensor->attr.dtype );

            if(stride[0] == 1 || stride[0] == 0)
            {
                 memset(data, data[0], size);
            }
            else
            {
                for( i = 1; i < elements; i ++ )
                {
                    for(j=0;j<stride[0];j++)
                    {
                        data[stride[0] * i + j] = data[j];
                    }
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
    vsi_nn_tensor_prv_t * ptr;
    ptr = (NULL != tensor) ? (vsi_nn_tensor_prv_t*)(*tensor) : NULL;
    if( NULL != ptr)
    {
        if( NULL != ptr->pot.t )
        {
            uint8_t* handle = NULL;
            if (ptr->pot.attr.is_created_from_handle &&
                ptr->pot.attr.is_handle_malloc_by_ovxlib)
            {
                vxSwapTensorHandle(ptr->pot.t, NULL, (void**)&handle);
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
                if(handle != _get_tensor_handle(ptr))
                {
                    VSILOGE("Tensor handle maybe swapped by accident!");
                }
#endif
                if ( handle == NULL )
                {
                    VSILOGE("Tensor handle is NULL.");
                    return;
                }
            }
            vxReleaseTensor( &ptr->pot.t );
            if (handle)
            {
                vsi_nn_FreeAlignedBuffer(handle);
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
                handle = NULL;
                _set_tensor_handle(ptr, NULL);
#endif
            }
        }

        if (ptr->pot.wb) {
            vxReleaseWeightsBiasesParameter(&ptr->pot.wb);
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

vsi_size_t vsi_nn_CopyTensorToBuffer
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    void            * buffer
    )
{
    vsi_size_t     sz;
    vsi_size_t     stride_size[VSI_NN_MAX_DIM_NUM];
    vsi_status     status;

    VSI_UNREFERENCED(graph);

    if( NULL == tensor || NULL == buffer )
    {
        return 0;
    }
    sz = 0;
    status = VSI_FAILURE;

    status = vsi_nn_copy_tensor_patch(tensor->t, &tensor->attr, buffer, VX_READ_ONLY, NULL, NULL);
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
    vsi_size_t elements;
    vsi_size_t i;
    vsi_size_t stride;
    float *data = NULL;

    if(NULL == graph || NULL == tensor)
    {
        return NULL;
    }

    elements = vsi_nn_GetElementNum(tensor);
    stride = vsi_nn_TypeGetBytesExt(tensor->attr.dtype.vx_type);

    data = NULL;
    data = (float *)malloc(elements * sizeof(float));
    CHECK_PTR_FAIL_GOTO( data, "Create buffer fail.", final );
    if( tensor->attr.is_created_from_handle )
    {
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
        tensor_data = _get_tensor_handle((vsi_nn_tensor_prv_t*)tensor);
        vxInvalidateHandleVSI((vx_reference)tensor->t);
#else
        vxSwapTensorHandle(tensor->t, NULL, (void**)&tensor_data);
#endif
        if ( tensor_data == NULL )
        {
            VSILOGE("Tensor handle is NULL.");
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
        if ( tensor_data == NULL )
        {
            VSILOGE("tensor_data is NULL.");
            vsi_nn_safe_free(data);
            return NULL;
        }
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

final:
    if( !tensor->attr.is_created_from_handle )
    {
        vsi_nn_safe_free( tensor_data );
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
    uint8_t    * new_data;
    vsi_size_t     buf_sz;
    vsi_size_t     stride_size[VSI_NN_MAX_DIM_NUM];
    vsi_status     status;

    VSI_UNREFERENCED(graph);

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
        if (data == NULL)
        {
            VSILOGE("Create buffer fail");

            return NULL;
        }
    }
    if( data && tensor->attr.is_created_from_handle )
    {
        uint8_t* tensor_data = NULL;
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
        tensor_data = _get_tensor_handle((vsi_nn_tensor_prv_t*)tensor);
        vxInvalidateHandleVSI((vx_reference)tensor->t);
#else
        vxSwapTensorHandle(tensor->t, NULL, (void**)&tensor_data);
#endif
        if ( tensor_data == NULL )
        {
            VSILOGE("Tensor handle is NULL.");
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
            status = vsi_nn_copy_tensor_patch(tensor->t, &tensor->attr, data, VX_READ_ONLY, NULL, NULL);
        }
        if(VSI_SUCCESS != status)
        {
            VSILOGE("Read tensor data fail");
            free(data);
            data = NULL;
            return NULL;
        }
    }
    if(tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT4 ||
        tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT4)
    {
        vsi_size_t dest_size = vsi_nn_GetElementNum(tensor);
        new_data = (uint8_t*)malloc(dest_size);
        if (new_data == NULL)
        {
            VSILOGE("Create buffer fail");
            vsi_nn_safe_free(data);

            return NULL;
        }

        status = vsi_nn_Unpack4bitData(tensor, data, new_data, tensor->attr.dtype.vx_type);
        vsi_nn_safe_free(data);
        return new_data;
    }
    else
    {
        return data;
    }
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
    vsi_size_t * dim,
    vx_enum  * data_format,
    vsi_size_t * size,
    vsi_size_t * stride_size,
    vx_tensor_addressing * addr,
    vx_enum accessor
    )
{
    uint8_t    * data;
    vsi_size_t     buf_sz;
    vsi_status     status;
    vsi_nn_tensor_attr_t attr;

    VSI_UNREFERENCED(addr);

    if( NULL == tensor || NULL == context )
    {
        return NULL;
    }

    status = VSI_FAILURE;
    data = NULL;

    status = vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS, dim, sizeof(vsi_size_t));
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS, size, sizeof(vsi_size_t) * (*dim));
    status = vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, data_format, sizeof(vsi_enum));
    attr.dim_num = (uint32_t)(*dim);
    memcpy(attr.size, size, sizeof(vsi_size_t) * attr.dim_num);

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
        status = vsi_nn_copy_tensor_patch(tensor, &attr, data, VX_READ_ONLY, NULL, NULL);
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
    vsi_size_t * stride_size,
    vx_tensor_addressing * addr,
    vx_enum accessor
    )
{
    uint8_t * data;
    vsi_size_t buf_sz;
    vsi_status status;

    VSI_UNREFERENCED(addr);

    if( NULL == tensor || NULL == context )
    {
        return NULL;
    }

    status = VSI_FAILURE;
    data = NULL;

    status = vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS,
        &(attr->dim_num), sizeof(attr->dim_num));
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS,
        attr->size, sizeof(attr->size[0]) * (attr->dim_num));
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
    case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
    case VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8:
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
        status = vsi_nn_copy_tensor_patch(tensor, attr, data, VX_READ_ONLY, NULL, NULL);
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
    vsi_size_t     stride;
    uint8_t      buf[_TENSOR_TMPBUF_SZ];
    FILE        * fp;
    float    write_data;
    vsi_size_t     sz;
    vsi_size_t     i;
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

    fp = vsi_nn_fopen( filename, "w" );
    if( NULL == fp )
    {
        VSILOGW( "Write file %s fail. Please check...", filename );
        goto final;
    }
    sz = vsi_nn_GetElementNum( tensor );
    ptr = data;
    stride = vsi_nn_TypeGetBytesExt( tensor->attr.dtype.vx_type );
    count = 0;
    for( i = 0; i < sz; i ++ )
    {
        vsi_nn_DtypeToFloat32( ptr, &write_data, &tensor->attr.dtype );
        ptr += stride;

        count += snprintf( (char *)&buf[count], _TENSOR_TMPBUF_SZ - count,
            "%.16f%s", write_data, seperator );
        if ( count > _TENSOR_TMPBUF_SZ )
        {
            VSILOGW( "tensor buffer overflow!" );
            break;
        }
        if( ((float)count / _TENSOR_TMPBUF_SZ) > c_flush_th )
        {
            fwrite( buf, count, 1, fp );
            count = 0;
        }
    }
    fwrite( buf, count, 1, fp );
    fclose( fp );

final:
    vsi_nn_safe_free( data );
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
    vsi_size_t  sz;

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
    vsi_nn_safe_free( data );
} /* vsi_nn_SaveTensorToText() */

void vsi_nn_SaveDataToText
    (
    const char  * filename,
    uint8_t    * data,
    vsi_size_t     data_size,
    vsi_nn_type_e type,
    char        * seperator
    )
{
#define _TENSOR_TMPBUF_SZ  (512)
    const float   c_flush_th = 0.7f;
    uint8_t      buf[_TENSOR_TMPBUF_SZ];
    FILE        * fp;
    float    write_data;
    vsi_size_t     stride;
    vsi_size_t     i;
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

    fp = vsi_nn_fopen( filename, "w" );
    if( NULL == fp )
    {
        VSILOGW( "Write file %s fail. Please check...", filename );
        return;
    }
    stride = vsi_nn_TypeGetBytesExt( type );

    count = 0;
    for( i = 0; i < data_size; i ++ )
    {
        write_data = vsi_nn_DataAsFloat32( &data[stride * i],
            type );
        if( type == VSI_NN_TYPE_UINT8 || type == VSI_NN_TYPE_INT8 ||
            type == VSI_NN_TYPE_UINT4 || type == VSI_NN_TYPE_INT4 ||
            type == VSI_NN_TYPE_FLOAT8_E4M3 || type == VSI_NN_TYPE_FLOAT8_E5M2 )
        {
            count += snprintf( (char *)&buf[count], _TENSOR_TMPBUF_SZ - count,
                "%d%s", (int32_t)write_data, seperator );
            if ( count > _TENSOR_TMPBUF_SZ )
            {
            VSILOGW( "tensor buffer overflow!" );
            break;
            }
        }
        else
        {
            count += snprintf( (char *)&buf[count], _TENSOR_TMPBUF_SZ - count,
                "%f%s", write_data, seperator );
            if ( count > _TENSOR_TMPBUF_SZ )
            {
            VSILOGW( "tensor buffer overflow!" );
            break;
            }
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
    uint8_t        * data = NULL;
    FILE            * fp = NULL;
    vsi_size_t         sz;
    uint32_t         i;
    uint8_t        * packed_data = NULL;
    vsi_size_t     packed_size;

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

    fp = vsi_nn_fopen( filename, "wb" );
    if( NULL == fp )
    {
        VSILOGW( "Write file %s fail. Please check...", filename );
        goto final;
    }
    sz = (vsi_size_t)vsi_nn_GetTypeBytes( tensor->attr.dtype.vx_type );
    if( tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT4 ||
        tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT4 )
    {
        packed_size = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num,
                                                         tensor->attr.dtype.vx_type);
        packed_data = (uint8_t*)malloc(packed_size);
        if ( NULL == packed_data )
        {
            VSILOGW( "malloc packed data failed" );
            goto final;
        }

        vsi_nn_Pack4bitData(tensor, data, packed_data);
        fwrite( packed_data, packed_size, 1, fp );
        if( packed_data )
        {
            free(packed_data);
            packed_data = NULL;
        }
    }
    else
    {
        for( i = 0; i < tensor->attr.dim_num; i ++ )
        {
            sz *= tensor->attr.size[i];
        }
        fwrite( data, sz, 1, fp );
    }

final:
    if (fp)
    {
        fclose( fp );
    }
    vsi_nn_safe_free( data );
    vsi_nn_safe_free( packed_data );
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
    uint8_t* new_data = NULL;

    VSI_UNREFERENCED(graph);

    if( NULL == data || NULL == tensor )
    {
        return status;
    }

    if( tensor->attr.is_created_from_handle )
    {
        uint8_t* ptr = NULL;
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
        ptr = _get_tensor_handle((vsi_nn_tensor_prv_t*)tensor);
#else
        vxSwapTensorHandle(tensor->t, NULL, (void**)&ptr);
#endif
        if ( ptr == NULL )
        {
            VSILOGE("Tensor handle is NULL.");
            return VSI_FAILURE;
        }
        memcpy( ptr, data, vsi_nn_GetTensorSize(tensor->attr.size, tensor->attr.dim_num,
                    tensor->attr.dtype.vx_type));
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
        status = vxFlushHandle((vx_reference)tensor->t);
#else
        status = vxSwapTensorHandle(tensor->t, ptr, NULL);
        status |= vxFlushHandle((vx_reference)tensor->t);
#endif
    }
    else
    {
        if( tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT4 ||
            tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT4 )
        {
            vsi_size_t dest_size = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num,
                                                         tensor->attr.dtype.vx_type);
            new_data = (uint8_t*)malloc( dest_size );
            CHECK_PTR_FAIL_GOTO( new_data, "Create buffer fail.", final );
            status = vsi_nn_Pack4bitData(tensor, (uint8_t*)data, new_data);
            status = vsi_nn_copy_tensor_patch( tensor->t, &tensor->attr, new_data, VX_WRITE_ONLY, NULL, NULL );
        }
        else
        {
            status = vsi_nn_copy_tensor_patch( tensor->t, &tensor->attr, data, VX_WRITE_ONLY, NULL, NULL );
        }
    }

final:
    vsi_nn_safe_free(new_data);

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

vsi_status vsi_nn_InvalidateHandle
(
    const vsi_nn_tensor_t* tensor
)
{
    if (NULL == tensor || NULL == tensor->t)
    {
        return VSI_FAILURE;
    }
    else
    {
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
        return vxInvalidateHandleVSI((vx_reference)tensor->t);
#else
        return VSI_SUCCESS;
#endif
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
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
        if (NULL != _get_tensor_handle((vsi_nn_tensor_prv_t*)tensor))
        {
            *ptr = _get_tensor_handle((vsi_nn_tensor_prv_t*)tensor);
            return VSI_SUCCESS;
        }
        else
        {
            return VSI_FAILURE;
        }
#else
        return vxSwapTensorHandle(tensor->t, NULL, ptr);
#endif
    }
} /* vsi_nn_GetTensorHandle() */

vsi_status vsi_nn_SetTensorIsScalar
(
    vsi_nn_tensor_t* tensor,
    int8_t is_scalar
)
{
    return _set_tensor_is_scalar((vsi_nn_tensor_prv_t*)tensor, is_scalar);
}

int8_t vsi_nn_GetTensorIsScalar
(
    vsi_nn_tensor_t* tensor
)
{
    return _get_tensor_is_scalar((vsi_nn_tensor_prv_t*)tensor);
}

int32_t _get_tensor_is_sparsity
(
    vsi_nn_tensor_prv_t* tensor
)
{
    int32_t is_sparsity = FALSE;
    if (NULL == tensor)
    {
        VSILOGE("To get is_sparsity, tensor pointer SHOULD NOT be NULL.");
        goto final;
    }
#if defined(VSI_TENSOR_SPARSITY_SUPPORT)
    is_sparsity = tensor->sparsity_type;
#endif
final:
    return is_sparsity;
}

int32_t vsi_nn_GetTensorIsSparsity
(
    vsi_nn_tensor_t* tensor
)
{
    return _get_tensor_is_sparsity((vsi_nn_tensor_prv_t*)tensor);
}

vsi_status vsi_nn_SetTensorIsSparsity
(
    vsi_nn_tensor_t* tensor,
    int32_t is_sparsity
)
{
    VSI_UNREFERENCED(is_sparsity);
    vsi_status status = VSI_SUCCESS;
    if (NULL == tensor) {
        status = VSI_FAILURE;
        goto final;
    }
#if defined(VSI_TENSOR_SPARSITY_SUPPORT)
    vxSetTensorAttribute(tensor->t,
                        VX_TENSOR_SPARSITY_TYPE,
                        &is_sparsity,
                        sizeof(vx_enum));
    status = VSI_SUCCESS;
    ((vsi_nn_tensor_prv_t*)tensor)->sparsity_type = is_sparsity;
#endif
final:
    return status;
}


vsi_status vsi_nn_CopyRawDataToTensor
    (
    vsi_nn_graph_t*         graph,
    uint8_t*                src_data,
    const vsi_nn_dtype_t*   src_dtype,
    vsi_nn_tensor_t*        tensor
    )
{
    vsi_status status           = VSI_FAILURE;
    vsi_size_t src_data_sz   = 0;
    uint8_t* buffer             = NULL;
    vsi_size_t target_tensor_size = 0; /* in bytes */

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
    vsi_size_t        * shape,
    vsi_size_t          dim_num
    )
{
    vsi_bool ret;
    uint32_t i;
    vsi_size_t total_size;
    vsi_size_t dst_size;

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
        VSILOGE( "Cannot calculate the reshape tensor %"VSI_SIZE_T_SPECIFIER" to %"VSI_SIZE_T_SPECIFIER".",
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
            output->attr.dim_num = (uint32_t)dim_num;
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
    vsi_size_t        * shape,
    vsi_size_t          dim_num
    )
{
    vsi_bool ret;
    vsi_nn_tensor_t *output = NULL;
    vsi_nn_tensor_attr_t attr;
    if (NULL == graph || NULL == input || NULL == shape)
    {
        return NULL;
    }

    if (dim_num > VSI_NN_MAX_DIM_NUM)
    {
        VSILOGE( "tensor rank greater than %d.", VSI_NN_MAX_DIM_NUM );
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
    const vsi_size_t  * shape,
    vsi_size_t         dim_num
    )
{
    vsi_bool ret;
    vsi_size_t new_shape[VSI_NN_MAX_DIM_NUM] = {0};

    if (dim_num > VSI_NN_MAX_DIM_NUM)
    {
        VSILOGE( "tensor rank greater than %d.", VSI_NN_MAX_DIM_NUM );
        return FALSE;
    }

    memcpy(new_shape, shape, sizeof(vsi_size_t) * dim_num);

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
    output->t = vsi_nn_safe_reshape_tensor( input->t, (void*)new_shape, (vsi_size_t)dim_num, sizeof(new_shape[0]) );
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
    vsi_size_t       * perm,
    vsi_size_t         dim_num,
    vsi_size_t       * as_shape
    )
{
    uint8_t * buf;
    uint8_t * dst;
    vsi_size_t  buf_sz;
    vsi_size_t  tensor_sz;
    vsi_size_t * shape_ptr;
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

    vsi_nn_safe_free( buf );
    free( dst );
} /* vsi_nn_TransposeTensor() */

vx_tensor vsi_nn_safe_reshape_tensor
    (
    vx_tensor         tensor,
    void            * num_of_dims,
    vsi_size_t        sizes,
    vsi_size_t        size_of_shape_element
    )
{
    if (sizes > VSI_NN_MAX_DIM_NUM)
    {
        VSILOGE( "tensor rank greater than %d.", VSI_NN_MAX_DIM_NUM );
        return NULL;
    }

    if(sizeof(vx_size) == size_of_shape_element)
    {
        vx_size* num_of_dims_vxsize = (vx_size*)num_of_dims;
        #ifdef VSI_40BIT_VA_SUPPORT
            return vxReshapeTensor( tensor, num_of_dims_vxsize, (vx_size)sizes );
        #else
            {
                int32_t new_shape_int32[VSI_NN_MAX_DIM_NUM] = { 0 };
                vsi_size_t i = 0;
                for(i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
                {
                    new_shape_int32[i] = (vx_size)-1 == num_of_dims_vxsize[i] ? \
                        (int32_t)-1 : (int32_t)num_of_dims_vxsize[i];
                }
                return vxReshapeTensor( tensor, new_shape_int32, (uint32_t)sizes );
            }
        #endif
    }
    else if(sizeof(int32_t) == size_of_shape_element)
    {
        int32_t* num_of_dims_int32 = (int32_t*)num_of_dims;
        #ifdef VSI_40BIT_VA_SUPPORT
            {
                vx_size new_shape_vxsize[VSI_NN_MAX_DIM_NUM] = { 0 };
                vsi_size_t i = 0;
                for(i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
                {
                    new_shape_vxsize[i] = -1 == num_of_dims_int32[i] ? (vx_size)-1 : (vx_size)num_of_dims_int32[i];
                }
                return vxReshapeTensor( tensor, new_shape_vxsize, (vx_size)sizes );
            }
        #else
            return vxReshapeTensor( tensor, num_of_dims_int32, (uint32_t)sizes );
        #endif
    }
    else
    {
        VSILOGE("couldn't handle tensor shape element with length of %"VSI_SIZE_T_SPECIFIER"", size_of_shape_element);
        return NULL;
    }
} /* vsi_nn_safe_reshape_tensor() */

void vsi_nn_PermuteTensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    vsi_size_t       * perm,
    vsi_size_t         dim_num
    )
{
    uint8_t * buf = NULL;
    uint8_t * dst = NULL;
    vsi_size_t  tensor_sz;
    vsi_size_t * shape_ptr;
    vsi_size_t   dst_shape[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t i;
    vsi_status  status;

    if( NULL == tensor || NULL == perm || 0 == dim_num || dim_num > VSI_NN_MAX_DIM_NUM )
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
            vsi_nn_safe_free( buf );
            if( dst ) { free(dst); dst = NULL; }
            return;
        }
        dst_shape[i] = shape_ptr[perm[i]];
    }
    vsi_nn_Permute( dst, buf, shape_ptr, dim_num, perm, tensor->attr.dtype.vx_type );
    memcpy(tensor->attr.size, dst_shape, sizeof(dst_shape));
    tensor->t = vsi_nn_safe_reshape_tensor(tensor->t, (void*)tensor->attr.size,
        (vsi_size_t)tensor->attr.dim_num, sizeof(tensor->attr.size[0]));
    status = vsi_nn_CopyDataToTensor( graph, tensor, dst );
    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Copy permute data fail with code %#x.", status );
    }

    vsi_nn_safe_free( buf );
    if( dst ) { free(dst); dst = NULL; }
} /* vsi_nn_PermuteTensor() */

vsi_size_t vsi_nn_GetElementNum
    (
    const vsi_nn_tensor_t * tensor
    )
{
    if( NULL == tensor )
    {
        return 0;
    }

    return vsi_nn_ShapeProduct((vsi_size_t*)tensor->attr.size, tensor->attr.dim_num);
} /* vsi_nn_GetElementNum() */

vsi_size_t vsi_nn_GetTensorSize
    (
    const vsi_size_t * shape,
    vsi_size_t dim_num,
    vsi_nn_type_e type
    )
{
    vsi_size_t sz;
    vsi_size_t i;
    vsi_size_t bits_num;
    sz = 0;
    if( NULL == shape || 0 == dim_num )
    {
        return sz;
    }
    bits_num = vsi_nn_TypeGetBits( type );
    if( bits_num < BITS_PER_BYTE )
    {
        if(shape[0] % 2 == 0)
        {
            sz = shape[0] / 2;
        }
        else
        {
            sz = shape[0] / 2 + shape[0] % 2;
        }
    }
    else
    {
        sz = shape[0] * bits_num / BITS_PER_BYTE;
    }
    for( i = 1; i < dim_num; i ++ )
    {
        sz *= shape[i];
    }
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
    vsi_size_t *start,
    vsi_size_t *end,
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
    if (NULL == tensor_ref || NULL == graph)
    {
        vsi_nn_safe_free(tensor_ref);

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

    vsi_nn_safe_free(tensor_ref);
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

#define _MAX_TENSOR_IO 128
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
            if (node == NULL)
            {
                continue;
            }

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
     vsi_size_t stride_size[VSI_NN_MAX_DIM_NUM];
     vsi_size_t buf_sz0, buf_sz1;
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
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
        uint8_t* temp_handle = NULL;
        vsi_bool temp_is_handle_malloc_by_ovxlib = TRUE;

        temp_handle = _get_tensor_handle((vsi_nn_tensor_prv_t*)tensor0);
        _set_tensor_handle((vsi_nn_tensor_prv_t*)tensor0, _get_tensor_handle((vsi_nn_tensor_prv_t*)tensor1));
        _set_tensor_handle((vsi_nn_tensor_prv_t*)tensor1, temp_handle);

        temp_is_handle_malloc_by_ovxlib = tensor0->attr.is_handle_malloc_by_ovxlib;
        tensor0->attr.is_handle_malloc_by_ovxlib = tensor1->attr.is_handle_malloc_by_ovxlib;
        tensor1->attr.is_handle_malloc_by_ovxlib = temp_is_handle_malloc_by_ovxlib;
#endif
        tensor0->is_swapped = TRUE;
        tensor1->is_swapped = TRUE;
    }

    return status;
} /* vsi_nn_SwapTensorHandle() */

vsi_status vsi_nn_SwapTensorHandleWithCache
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t * tensor0,
    vsi_nn_tensor_t * tensor1
    )
{
    ((vsi_nn_graph_prv_t*)graph)->swap_handle_cache.is_feature_on = TRUE;
    return vsi_nn_SwapTensorHandle(tensor0, tensor1);
} /* vsi_nn_SwapTensorHandleWithCache() */

vsi_size_t vsi_nn_vxGetTensorElementNum
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
        &(attr->dim_num), sizeof(attr->dim_num));
    TEST_CHECK_STATUS( status, final );
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS,
        attr->size, sizeof(attr->size[0]) * (attr->dim_num));
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
    case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
    case VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8:
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
    vsi_size_t buf_sz;
    vsi_size_t stride_size[VSI_NN_MAX_DIM_NUM];

    memset(stride_size, 0, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
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

    status = vsi_nn_copy_tensor_patch(tensor, attr, data, VX_READ_ONLY, NULL, NULL);
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
    vsi_size_t stride_size[VSI_NN_MAX_DIM_NUM];

    status = VSI_FAILURE;
    if(NULL == tensor || NULL == attr ||
       NULL == context || NULL == data)
    {
        return status;
    }

    memset(stride_size, 0, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
    vsi_nn_GetStrideSize(attr, stride_size);
    status = vsi_nn_copy_tensor_patch(tensor, attr, data, VX_WRITE_ONLY, NULL, NULL);
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
    vsi_size_t *start,
    vsi_size_t *end,
    vsi_size_t *stride,
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

#ifdef VX_TENSOR_STRIDE_X_BITS_SUPPORT
    {
        vx_trensor_addressing addr = NULL;
        vx_size dim_sizes[VSI_NN_MAX_DIM_NUM], strides[VSI_NN_MAX_DIM_NUM];
        addr = (vx_trensor_addressing)malloc(sizeof(vx_tensorpatch_addressing_t));
        if( NULL == addr )
        {
            VSILOGE("Call malloc fail");
            return status;
        }
        addr->num_of_dims = (vx_uint32)attr->dim_num;
        for(i = 0; i < dim; i++)
        {
            strides[i] = (vx_size)vstride[i];
            dim_sizes[i] = (vx_size)attr->size[i];
        }
        addr->strides = strides;
        addr->dim_sizes = dim_sizes;
        if(attr->dtype.vx_type == VSI_NN_TYPE_INT4 || attr->dtype.vx_type == VSI_NN_TYPE_UINT4)
        {
           addr->strides[0] = 0;
           addr->stride_x_bits = 4;
        }
        status = vxCopyTensorPatch2(tensor, dim, vstart, vend, addr,sizeof(vx_tensorpatch_addressing_t),
                                    user_ptr, usage, user_memory_type);
        if(addr)
        {
            free(addr);
            addr = NULL;
        }
    }
#else
    status = vxCopyTensorPatch(tensor, dim, vstart, vend, vstride, user_ptr, usage, user_memory_type);
#endif
#else
    {
        vx_context context = NULL;
        vx_tensor_addressing addr = NULL;
        size_t stride_size[VSI_NN_MAX_DIM_NUM];
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
            (vx_uint32*)stride_size, attr->dim_num );
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
    vsi_enum usage,
    vsi_size_t* start,
    vsi_size_t* end
    )
{
    vsi_size_t tmp_start[VSI_NN_MAX_DIM_NUM],tmp_end[VSI_NN_MAX_DIM_NUM],stride[VSI_NN_MAX_DIM_NUM];
    vsi_status status = VSI_FAILURE;

    if(NULL == tensor || NULL == user_ptr)
    {
        VSILOGE("Invalid parameter");
        return status;
    }
    vsi_nn_GetStrideSize(attr, stride);
    if (NULL == start)
    {
        memset(tmp_start, 0, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
    }
    else
    {
        memcpy(tmp_start, start, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
    }

    if (NULL == end)
    {
        memcpy(tmp_end, attr->size, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
    }
    else
    {
        memcpy(tmp_end, end, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
    }

    status = vsi_nn_copy_tensor_veiw_patch(tensor, attr, user_ptr, tmp_start, tmp_end, stride, usage, 0);
    return status;
} /* vsi_nn_copy_tensor_patch() */

vsi_size_t vsi_nn_GetOffsetByCoords
    (
    vsi_nn_tensor_attr_t *attr,
    uint32_t *coords
    )
{
    vsi_size_t i, res = 0, strides = 1;
    for (i = 0; i < (vsi_size_t)attr->dim_num; i++)
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
    vsi_ssize_t b, sy, sx, c, h, w;
    uint8_t* weight_data = NULL;
    uint8_t* reshuffled_weights = NULL;
    uint8_t* buffer = NULL;
    vsi_ssize_t kernel_size_x = weights->attr.size[0];
    vsi_ssize_t kernel_size_y = weights->attr.size[1];
    vsi_ssize_t weight_size_c = weights->attr.size[2];
    vsi_ssize_t weight_size_b = weights->attr.size[3];
    vsi_ssize_t slice_size = kernel_size_x * kernel_size_y;
    int32_t item_size = vsi_nn_TypeGetBytes(weights->attr.dtype.vx_type);

    weight_data = vsi_nn_ConvertTensorToData(graph, weights);
    CHECK_PTR_FAIL_GOTO( weight_data, "Create weight_data fail.", final );
    buffer = (uint8_t*)malloc(item_size * slice_size * weight_size_c * weight_size_b);
    CHECK_PTR_FAIL_GOTO( buffer, "Create buffer fail.", final );
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
                            vsi_ssize_t input_index = ((kernel_size_y - 1 - h) + sy) * kernel_size_x +
                                ((kernel_size_x - 1 - w) + sx);

                            memcpy(reshuffled_output, data + input_index * item_size, item_size);
                        }
                    }
                }
            }
        }
    }
    vsi_nn_CopyDataToTensor( graph, weights, weight_data );

final:
    vsi_nn_Free( buffer );
    vsi_nn_safe_free( weight_data );
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
    TEST_CHECK_PTR( tensors, final );
    tensor_count = 0;
    va_start(args, axis);

    FOREACH_ARGS(args, next, vsi_nn_tensor_t*)
    {
        tensors[tensor_count++] = next;
    }
    va_end(args);

    next = vsi_nn_Concat(graph, tensors, tensor_count, axis);

final:
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
    TEST_CHECK_PTR( tensors, final );
    tensor_count = 0;
    va_start(args, output_attr);
    FOREACH_ARGS(args, next, vsi_nn_tensor_t*)
    {
        tensors[tensor_count++] = next;
    }
    va_end(args);

    next = vsi_nn_TensorAdd(graph, tensors, tensor_count, output_attr);

final:
    vsi_nn_safe_free(tensors);

    return next;
}

vsi_status vsi_nn_SwapHandle
    (
    vsi_nn_tensor_t* tensor,
    void* new_ptr,
    vsi_bool is_new_ptr_malloc_by_ovxlib,
    void** old_ptr
    )
{
    vsi_status status = VSI_FAILURE;
    VSI_UNREFERENCED(is_new_ptr_malloc_by_ovxlib);
    if (!tensor)
    {
        return VSI_FAILURE;
    }
    status = vxSwapTensorHandle(tensor->t, new_ptr, old_ptr);
#ifdef VSI_INVALIDATE_HANDLE_SUPPORT
    if (VSI_SUCCESS == status)
    {
        _set_tensor_handle((vsi_nn_tensor_prv_t*)tensor, new_ptr);
        tensor->attr.is_handle_malloc_by_ovxlib = is_new_ptr_malloc_by_ovxlib;
    }
#endif
    return status;
} /* vsi_nn_SwapHandle() */

vsi_bool vsi_nn_ConvertTensor
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t* input,
    vsi_nn_tensor_t* output
    )
{
    vsi_bool ret = TRUE;
    uint8_t* src_buf = NULL;
    vsi_size_t sz = 0;
    uint32_t src_stride = 0;
    uint32_t dst_stride = 0;
    vsi_size_t dst_buf_sz = 0;
    uint8_t* dst_buf = NULL;

    if( NULL == graph || NULL == input || NULL == output )
    {
        return FALSE;
    }

    src_buf = vsi_nn_ConvertTensorToData( graph, input );
    if ( NULL == src_buf )
    {
        VSILOGE( "Convert data fail." );
        return FALSE;
    }

    sz = vsi_nn_GetElementNum( output );
    src_stride = vsi_nn_TypeGetBytes( input->attr.dtype.vx_type );
    dst_stride = vsi_nn_TypeGetBytes( output->attr.dtype.vx_type );
    dst_buf_sz = sz * dst_stride;
    dst_buf = (uint8_t *)malloc( dst_buf_sz );

    if ( dst_buf )
    {
        vsi_size_t i = 0;
        vsi_status status = VSI_SUCCESS;

        for ( i = 0; i < sz; i ++ )
        {
            status = vsi_nn_DtypeConvert( &src_buf[src_stride * i],
                &input->attr.dtype, &dst_buf[dst_stride * i], &output->attr.dtype );
            if( VSI_FAILURE == status )
            {
                ret = FALSE;
                VSILOGE("Convert default_value to dtype fail");
                break;
            }
        }

        status = vsi_nn_CopyDataToTensor( graph, output, dst_buf );
        if ( VSI_FAILURE == status )
        {
            ret = FALSE;
            VSILOGE("Copy data to tensor fail");
        }
    }

    vsi_nn_safe_free( src_buf );
    vsi_nn_safe_free( dst_buf );

    return ret;
}

vsi_nn_tensor_t * vsi_nn_dropout_tensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * input,
    float             rate
    )
{
    vsi_nn_tensor_t *output = NULL;
    vsi_size_t size = 0;
    vsi_size_t i = 0;
    float* data   = NULL;
    vsi_nn_tensor_attr_t attr;

    if (NULL == input || NULL == graph)
    {
        return NULL;
    }

    memset(&attr, 0, sizeof(attr));
    memcpy(&attr, &input->attr, sizeof(attr));
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;

    output = vsi_nn_CreateTensor(graph, &attr);
    if ( !output )
    {
        VSILOGE("create tensor failed.");
        goto final;
    }

    data = vsi_nn_ConvertTensorToFloat32Data(graph, input);
    if (NULL == data)
    {
        goto final;
    }

    size = vsi_nn_vxGetTensorElementNum(&input->attr);

    for (i = 0; i < size; i++)
    {
        data[i] = data[i] * rate;
    }

    vsi_nn_CopyRawDataToTensor( graph, (uint8_t *)data, &attr.dtype, output );

final:
    vsi_nn_safe_free(data);

    return output;
}

uint8_t* _get_tensor_handle
    (
    vsi_nn_tensor_prv_t* tensor
    )
{
    uint8_t* handle = NULL;
    if (NULL == tensor)
    {
        goto final;
    }
    handle = tensor->handle;

final:
    return handle;
}

vsi_status _set_tensor_handle
    (
    vsi_nn_tensor_prv_t* tensor,
    uint8_t*             handle
    )
{
    vsi_status status = VSI_SUCCESS;
    if (NULL == tensor)
    {
        status = VSI_FAILURE;
        goto final;
    }
    tensor->handle = handle;

final:
    return status;
}

int8_t _get_tensor_is_scalar
(
    vsi_nn_tensor_prv_t* tensor
)
{
    int8_t is_scalar = FALSE;
    if (NULL == tensor)
    {
        VSILOGE("To get is_scalar, tensor pointer SHOULD NOT be NULL.");
        goto final;
    }
    is_scalar = tensor->is_scalar;

    final:
    return is_scalar;
}

vsi_status _set_tensor_is_scalar
(
    vsi_nn_tensor_prv_t* tensor,
    int8_t is_salar
)
{
    vsi_status status = VSI_SUCCESS;
    if (NULL == tensor)
    {
        status = VSI_FAILURE;
        goto final;
    }
    tensor->is_scalar = is_salar;

    final:
    return status;
}

int8_t _get_tensor_is_from_axisram
    (
    vsi_nn_tensor_prv_t* tensor
    )
{
    int8_t is_from_axisram = FALSE;
    if (NULL == tensor) {
        VSILOGE("To get is_scalar, tensor pointer SHOULD NOT be NULL.");
        goto final;
    }
    is_from_axisram = tensor->is_from_axisram;

final:
    return is_from_axisram;
}

vsi_status _set_tensor_is_from_axisram
    (
    vsi_nn_tensor_prv_t* tensor,
    int8_t is_from_axisram
    )
{
    vsi_status status = VSI_SUCCESS;
    if (NULL == tensor) {
        status = VSI_FAILURE;
        goto final;
    }
    tensor->is_from_axisram = is_from_axisram;

final:
    return status;
}

static vsi_bool _init_dummy_tensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor
    )
{
    vsi_bool ret;
    vx_tensor_create_params_t params;
    float * scales = NULL;
    int32_t * zeroPoints = NULL;
    int32_t * null_zp = NULL;
    vx_size size_vxsize[VSI_NN_MAX_DIM_NUM] = {0};
    vx_uint32 size_u32[VSI_NN_MAX_DIM_NUM] = {0};
    size_t i = 0;
    ret = TRUE;

    VSI_UNREFERENCED(graph);

    memset( &params, 0, sizeof( vx_tensor_create_params_t ) );
    params.num_of_dims = tensor->attr.dim_num;
    for(i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        size_vxsize[i] = (vsi_size_t)-1 == tensor->attr.size[i] ? (vx_size)-1 : (vx_size)tensor->attr.size[i];
    }
    for(i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        size_u32[i] = (vsi_size_t)-1 == tensor->attr.size[i] ? (vx_uint32)-1 : (vx_uint32)tensor->attr.size[i];
    }
#ifdef VSI_40BIT_VA_SUPPORT
    params.sizes = size_vxsize;
    (void)size_u32;
#else
    params.sizes = size_u32;
    (void)size_vxsize;
#endif
    params.data_format = (vsi_enum)tensor->attr.dtype.vx_type;
    switch( tensor->attr.dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        params.quant_format = (vsi_enum)VX_QUANT_DYNAMIC_FIXED_POINT;
        params.quant_data.dfp.fixed_point_pos = (uint8_t)tensor->attr.dtype.fl;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
    case VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8:
        params.quant_format = (vsi_enum)VX_QUANT_AFFINE_SCALE;
        params.quant_data.affine.scale = tensor->attr.dtype.scale;
        params.quant_data.affine.zeroPoint = (int32_t)tensor->attr.dtype.zero_point;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC:
    case VSI_NN_QNT_TYPE_PERCHANNEL_SYMMETRIC_FLOAT8:
#ifdef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
        #ifdef VX_QUANT_AFFINE_SCALE_PER_CHANNEL
            params.quant_format = (vsi_enum)VX_QUANT_AFFINE_SCALE_PER_CHANNEL;
        #else
            params.quant_format = (vsi_enum)VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC;
        #endif
        // This is a hack that driver doesn't support const scales
        scales = (float*)malloc(sizeof(float) * tensor->attr.dtype.scale_dim);
        CHECK_PTR_FAIL_GOTO( scales, "Create buffer fail.", final );
        memcpy(scales, tensor->attr.dtype.scales, tensor->attr.dtype.scale_dim * sizeof(float));
        params.quant_data.affinePerChannel.channelDim = tensor->attr.dtype.channel_dim;
        params.quant_data.affinePerChannel.scaleCount = tensor->attr.dtype.scale_dim;
        params.quant_data.affinePerChannel.scales = scales;
        params.quant_data.affinePerChannel.zeroPoint = NULL;
        params.quant_data.affinePerChannel.zeroPointCount = 0;
        {
            // Low-level driver only support asymmetric. Application doesn't provide zp information if
            // it's symmetric quantized tensor. Fake a zp information filled with zero to meet low-level's
            // requirement
            null_zp = (int32_t*)malloc(sizeof(int32_t) * tensor->attr.dtype.scale_dim);
            CHECK_PTR_FAIL_GOTO( null_zp, "Create buffer fail.", final );
            memset(null_zp, 0, sizeof(int32_t) * tensor->attr.dtype.scale_dim);
            params.quant_data.affinePerChannel.zeroPoint = null_zp;
            params.quant_data.affinePerChannel.zeroPointCount= tensor->attr.dtype.scale_dim;
        }
        break;
#else
    VSILOGE( "can't support qnt_type VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC." );
#endif
    case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_ASYMMETRIC:
#ifdef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
        #ifdef VX_QUANT_AFFINE_SCALE_PER_CHANNEL
            params.quant_format = (vsi_enum)VX_QUANT_AFFINE_SCALE_PER_CHANNEL;
        #else
            params.quant_format = (vsi_enum)VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC;
        #endif
        // This is a hack that driver doesn't support const scales
        scales = (float*)malloc(sizeof(float) * tensor->attr.dtype.scale_dim);
        CHECK_PTR_FAIL_GOTO( scales, "Create buffer fail.", final );
        memcpy(scales,
               tensor->attr.dtype.scales,
               tensor->attr.dtype.scale_dim * sizeof(float));
        zeroPoints = (int32_t*)malloc(sizeof(int32_t) * tensor->attr.dtype.zero_points_dim);
        CHECK_PTR_FAIL_GOTO( zeroPoints, "Create buffer fail.", final );
        memcpy(zeroPoints,
               tensor->attr.dtype.zero_points,
               tensor->attr.dtype.zero_points_dim * sizeof(int32_t));
        params.quant_data.affinePerChannel.channelDim =
            tensor->attr.dtype.channel_dim;
        params.quant_data.affinePerChannel.scaleCount =
            tensor->attr.dtype.scale_dim;
        params.quant_data.affinePerChannel.scales = scales;
        params.quant_data.affinePerChannel.zeroPoint = zeroPoints;
        params.quant_data.affinePerChannel.zeroPointCount = tensor->attr.dtype.zero_points_dim;
        break;
#else
        VSILOGE(
            "can't support qnt_type "
            "VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_ASYMMETRIC.");
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

#if (VX_STREAM_PROCESSOR_SUPPORT)
    tensor->t = vxCreateDummyTensor( graph->ctx->c,
        (vsi_size_t)tensor->attr.dim_num, size_vxsize, (vsi_enum)tensor->attr.dtype.vx_type );
#else
    tensor->t = NULL;
#endif
    if ( NULL == tensor->t )
    {
        VSILOGE( "Create vx tensor fail." );
        ret = FALSE;
        goto final;
    }

final:
    if( scales )
    {
        free(scales);
    }
    if (zeroPoints)
    {
        free(zeroPoints);
    }
    if(null_zp)
    {
        free(null_zp);
        null_zp = NULL;
    }
    return ret;
} /* _init_dummy_tensor() */

static vsi_nn_tensor_t * _create_dummy_tensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr
    )
{
    vsi_nn_tensor_prv_t * tensor;

    tensor = NULL;
    if( NULL == graph || NULL == graph->g || NULL == attr )
    {
        return NULL;
    }

    tensor = (vsi_nn_tensor_prv_t *)malloc( sizeof( vsi_nn_tensor_prv_t ) );
    //vsi_nn_UpdateTensorDims( attr );

    if ( NULL != tensor )
    {
        memset( tensor, 0, sizeof( vsi_nn_tensor_prv_t ) );
        memcpy( &tensor->pot.attr, attr, sizeof( vsi_nn_tensor_attr_t ) );
        tensor->pot.is_swapped = FALSE;
        if( attr->dim_num != VSI_NN_DIM_AUTO )
        {
            _init_dummy_tensor( graph, &tensor->pot);
            if( NULL == tensor->pot.t )
            {
                VSILOGE( "Create vx tensor fail." );
                free( tensor );
                tensor = NULL;
            }
        }
    }

    return (vsi_nn_tensor_t*)tensor;
}

vsi_nn_tensor_t * vsi_nn_create_dummy_tensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr
    )
{
    attr->is_created_from_handle = FALSE;
    return _create_dummy_tensor(graph, attr);
} /* vsi_nn_create_dummy_tensor() */

vsi_status vsi_nn_MapTensorPatch
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t* tensor,
    void** ptr,
    vsi_nn_accessor_type_e usage
    )
{
    vsi_status status = VSI_FAILURE;
#ifdef VSI_MAP_TENSOR_PATCH_SUPPORT
    size_t dim, i;
    vsi_size_t tem_stride[VSI_NN_MAX_DIM_NUM];
    vx_size start[VSI_NN_MAX_DIM_NUM], end[VSI_NN_MAX_DIM_NUM],
        stride[VSI_NN_MAX_DIM_NUM];
    vx_map_id map_id = 0;

    if (NULL == graph || NULL == tensor || NULL == ptr)
    {
        VSILOGE("Invalid parameter");
        return status;
    }
    if (TRUE == tensor->attr.vtl)
    {
        VSILOGE("Can not access a virtual tensor.");
        return status;
    }
    vsi_nn_GetStrideSize(&tensor->attr, tem_stride);

    memset(start, 0, sizeof(vx_size) * VSI_NN_MAX_DIM_NUM);
    dim = (size_t)tensor->attr.dim_num;
    for (i = 0; i < dim; i++)
    {
        end[i] = (size_t)tensor->attr.size[i];
        stride[i] = (size_t)tem_stride[i];
    }

    status = vxMapTensorPatch(tensor->t,dim,start,end,
                         &map_id,stride,ptr,usage, VX_MEMORY_TYPE_HOST);
    ((vsi_nn_tensor_prv_t*)tensor)->map_id = map_id;
#else
    VSI_UNREFERENCED(graph);
    VSI_UNREFERENCED(tensor);
    VSI_UNREFERENCED(ptr);
    VSI_UNREFERENCED(usage);
    VSILOGE("Function unspported, please upgrade OpenVX driver to 1.3.0!");
#endif
    return status;
} /* vsi_nn_MapTensorPatch() */

vsi_status vsi_nn_UnmapTensorPatch
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t* tensor
    )
{
    vsi_status status = VSI_FAILURE;
#ifdef VSI_MAP_TENSOR_PATCH_SUPPORT
    vx_map_id map_id = 0;

    if (NULL == graph || NULL == tensor)
    {
        VSILOGE("Invalid parameter");
        return status;
    }
    if (TRUE == tensor->attr.vtl)
    {
        VSILOGE("Can not access a virtual tensor.");
        return status;
    }

    map_id = ((vsi_nn_tensor_prv_t*)tensor)->map_id;
    status = vxUnmapTensorPatch(tensor->t, map_id);
#else
    VSI_UNREFERENCED(graph);
    VSI_UNREFERENCED(tensor);
    VSILOGE("Function unspported, please upgrade OpenVX driver to 1.3.0!");
#endif
    return status;
} /* vsi_nn_UnmapTensorPatch() */

vsi_nn_tensor_t * vsi_nn_CreateTensorFromAXISRAM
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr
    )
{

    if (NULL == graph || NULL == attr) {
        VSILOGE("Invalid parameter");
        return NULL;
    }
    if (TRUE == attr->vtl) {
        VSILOGE("Can not create tensor from AXI-SRAM for a virtual tensor.");
        return NULL;
    }
#ifdef VSI_CREATE_TENSOR_FROM_AXISRAM_SUPPORT
    attr->is_created_from_handle = FALSE;
    return _create_tensor(graph, NULL, attr, TRUE);
#else
    return NULL;
#endif
} /*vsi_nn_CreateTensorFromAXISRAM*/
