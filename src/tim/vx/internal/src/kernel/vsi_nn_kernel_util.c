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

#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"

typedef enum
{
    MEMORY_ACCESSOR_READ_ONLY = 0,
    MEMORY_ACCESSOR_WRITE_ONLY = 1,
} mem_accessor_e;

vsi_status vsi_nn_kernel_copy_tensor_veiw_patch
    (
    vx_tensor tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
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
    if (NULL == tensor || NULL == user_ptr || NULL == start || NULL == end || NULL == stride)
    {
        VSILOGE("Invalid parameter");
        return status;
    }
    dim = (size_t)attr->shape->size;
    for (i = 0; i < dim; i++)
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
        addr->num_of_dims = (vx_uint32)attr->shape->size;

        for (i = 0; i < dim; i++)
        {
            strides[i] = (vx_size)vstride[i];
            dim_sizes[i] = (vx_size)attr->shape->data[i];
        }
        addr->strides = strides;
        addr->dim_sizes = dim_sizes;
        if ( attr->dtype == I4 || attr->dtype == U4 )
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
        vsi_nn_kernel_tensor_attr_get_stride( attr, stride_size );
        context = vxGetContext((vx_reference)tensor);
        if( NULL == context )
        {
            VSILOGE("Call vxGetContext fail");
            return status;
        }
        addr = vxCreateTensorAddressing( context, attr->shape->data,
            (vx_uint32*)stride_size, attr->shape->size );
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
} /* vsi_nn_kernel_copy_tensor_veiw_patch() */

vsi_status vsi_nn_kernel_copy_tensor_patch
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    mem_accessor_e accessor,
    void * user_ptr,
    size_t buffer_size
    )
{
    vsi_size_t start[VSI_NN_MAX_DIM_NUM],end[VSI_NN_MAX_DIM_NUM],stride[VSI_NN_MAX_DIM_NUM];
    vsi_status status = VSI_FAILURE;
    uint32_t i;
    if (NULL == tensor || NULL == user_ptr)
    {
        VSILOGE("Invalid parameter");
        return status;
    }

    vsi_nn_kernel_tensor_attr_get_stride( attr, stride );
    memset(start, 0, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM);
    for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        end[i] = attr->shape->data[i];
        if ( attr->dtype != I4 && attr->dtype != U4 )
        {
            size_t type_bytes = vsi_nn_kernel_dtype_get_bytes( attr->dtype );
            stride[i] = stride[i] * (vsi_size_t)type_bytes;
        }
    }

    switch( accessor )
    {
        case MEMORY_ACCESSOR_READ_ONLY:
            status = vsi_nn_kernel_copy_tensor_veiw_patch( (vx_tensor)tensor, attr,
                    user_ptr, start, end, stride, VX_READ_ONLY, 0);
            break;
        case MEMORY_ACCESSOR_WRITE_ONLY:
            status = vsi_nn_kernel_copy_tensor_veiw_patch( (vx_tensor)tensor, attr,
                    user_ptr, start, end, stride, VX_WRITE_ONLY, 0);
            break;
        default:
            VSI_ASSERT( FALSE );
            break;
    }

    return status;
} /* vsi_nn_kernel_copy_tensor_patch() */

void * vsi_nn_kernel_tensor_create_buffer
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    vsi_bool convert_to_float
    )
{
    vsi_status status = VSI_FAILURE;
    void * buffer = NULL;
    void * out_buffer = NULL;
    void * tensor_buffer = NULL;
    void * new_data = NULL;
    size_t bytes;
    size_t float_bytes;
    size_t tensor_size = 0;
    vsi_nn_kernel_tensor_attr_t * internal_attr = NULL;

    if ( !tensor )
    {
        return NULL;
    }

    if ( !attr )
    {
        internal_attr = vsi_nn_kernel_tensor_attr_create( tensor );
        CHECK_PTR_FAIL_GOTO( internal_attr, "Create tensor attr fail.", final );
        attr = internal_attr;
    }
    bytes = vsi_nn_kernel_tensor_attr_get_bytes( attr );
    tensor_buffer = malloc( bytes );
    CHECK_PTR_FAIL_GOTO( tensor_buffer, "Out of memory, create buffer fail.", final );

    status = vsi_nn_kernel_tensor_read( tensor, attr, tensor_buffer, bytes );
    if ( status != VSI_SUCCESS )
    {
        VSILOGE("Read tensor fail with error \"%s\".", vsi_nn_DescribeStatus(status));
        vsi_nn_safe_free( tensor_buffer );
        goto final;
    }

    if ( attr->dtype == I4 || attr->dtype == U4 )
    {
        vsi_size_t dest_size = vsi_nn_kernel_tensor_attr_get_size( attr );
        new_data = (uint8_t*)malloc(dest_size);
        if ( !new_data )
        {
            VSILOGE("Out of memory, create buffer fail");
            vsi_nn_safe_free( tensor_buffer );
            goto final;
        }
        CHECK_PTR_FAIL_GOTO( new_data, "Out of memory, create buffer fail.", final );
        status = vsi_nn_kernel_unpack_4bit_data(attr, (uint8_t *)tensor_buffer, (uint8_t *)new_data, attr->dtype);
        if ( status != VSI_SUCCESS )
        {
            VSILOGE("Read tensor fail with error \"%s\".", vsi_nn_DescribeStatus(status));
            vsi_nn_safe_free( tensor_buffer );
            vsi_nn_safe_free( new_data );
            goto final;
        }
        vsi_nn_safe_free( tensor_buffer );
        out_buffer = new_data;
    }
    else
    {
        out_buffer = tensor_buffer;
    }

    if ( convert_to_float && F32 != attr->dtype )
    {
        buffer = out_buffer;
        tensor_size = vsi_nn_kernel_tensor_attr_get_size( attr );
        float_bytes = tensor_size * sizeof(float);
        out_buffer = malloc( float_bytes );
        if ( !out_buffer )
        {
            VSILOGE("Out of memory, create float buffer fail.");
            vsi_nn_safe_free( buffer );
            goto final;
        }
        if ( vsi_nn_kernel_tensor_attr_is_quantized( attr ) )
        {
            switch( attr->quant )
            {
                case VSI_NN_KERNEL_QUANT_DFP:
                    vsi_nn_dtype_convert_quantize_dfp_to_float(
                            buffer, tensor_size, attr->dtype,
                            attr->dfp.fl, (float*)out_buffer );
                    break;
                case VSI_NN_KERNEL_QUANT_ASYMM:
                    vsi_nn_dtype_convert_quantize_asymm_to_float(
                            buffer, tensor_size, attr->dtype,
                            attr->asymm.scale, attr->asymm.zero_point,
                            (float*)out_buffer );
                    break;
                case VSI_NN_KERNEL_QUANT_SYMM_PERCHANNEL:
                    vsi_nn_dtype_convert_quantize_symm_perchannel_to_float(
                            buffer, tensor_size, attr->dtype,
                            attr->shape->data, attr->shape->size,
                            attr->asymm_v.scale->data,
                            attr->asymm_v.scale->size,
                            attr->asymm_v.zero_point->data,
                            attr->asymm_v.zero_point->size,
                            attr->asymm_v.channel_dim,
                            (float*)out_buffer );
                    break;
                default:
                    VSILOGE("Donot support quantize type %d", attr->quant);
                    VSI_ASSERT( FALSE );
                    break;
            }
        }
        else
        {
            vsi_nn_dtype_convert_dtype_to_float( buffer, tensor_size,
                    attr->dtype, (float*)out_buffer );
        }
        vsi_nn_safe_free( buffer );
    }

final:
    if ( internal_attr )
    {
        vsi_nn_kernel_tensor_attr_release( &internal_attr );
    }

    return out_buffer;
} /* vsi_nn_kernel_tensor_create_buffer() */

vsi_status vsi_nn_kernel_tensor_read
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    void * out_buffer,
    size_t out_buffer_size
    )
{
    return  vsi_nn_kernel_copy_tensor_patch( tensor, attr, MEMORY_ACCESSOR_READ_ONLY,
            out_buffer, out_buffer_size );
} /* vsi_nn_kernel_tensor_read() */

vsi_status vsi_nn_kernel_tensor_write
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    const void * buffer,
    size_t size
    )
{
    // NOTE: openvx api vxCopyTensorPatch access non-const buffer pointer,
    // so here we convert const to non-const ptr.
    return vsi_nn_kernel_copy_tensor_patch( tensor, attr, MEMORY_ACCESSOR_WRITE_ONLY,
            (void*)buffer, size );
} /* vsi_nn_kernel_tensor_write() */

vsi_status vsi_nn_kernel_tensor_write_from_float
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    const float * float_buffer,
    size_t size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_attr_t * internal_attr = NULL;
    size_t bytes;
    const void * buffer = NULL;
    void * internal_buffer = NULL;
    void * internal_buffer0 = NULL;
    size_t tensor_size = 0;
    if ( !attr )
    {
        internal_attr = vsi_nn_kernel_tensor_attr_create( tensor );
        CHECK_PTR_FAIL_GOTO( attr, "Create tensor attr fail.", final );
        attr = internal_attr;
    }
    bytes = vsi_nn_kernel_tensor_attr_get_bytes( attr );
    tensor_size = vsi_nn_kernel_tensor_attr_get_size( attr );
    if ( tensor_size != size )
    {
        VSILOGE("Tensor and buffer size mismatch %d vs %d", tensor_size, size);
        goto final;
    }

    if ( attr->dtype == I4 || attr->dtype == U4 )
    {
        vsi_size_t sz = 0;
        sz = vsi_nn_kernel_tensor_attr_get_size( attr );
        internal_buffer0 = malloc( sz );
    }
    else
    {
        internal_buffer0 = malloc( bytes );
        internal_buffer = internal_buffer0;
    }

    if( attr->dtype != F32 )
    {
        CHECK_PTR_FAIL_GOTO( internal_buffer0, "Create buffer fail.", final );
        if ( vsi_nn_kernel_tensor_attr_is_quantized( attr ) )
        {
            switch( attr->quant )
            {
                case VSI_NN_KERNEL_QUANT_DFP:
                    vsi_nn_dtype_convert_float_to_quantize_dfp(
                            float_buffer, size, attr->dtype,
                            attr->dfp.fl, internal_buffer0 );
                    break;
                case VSI_NN_KERNEL_QUANT_ASYMM:
                    vsi_nn_dtype_convert_float_to_quantize_asymm(
                            float_buffer, size, attr->dtype,
                            attr->asymm.scale, attr->asymm.zero_point,
                            internal_buffer0 );
                    break;
                case VSI_NN_KERNEL_QUANT_SYMM_PERCHANNEL:
                    vsi_nn_dtype_convert_float_to_quantize_symm_perchannel(
                            float_buffer, size, attr->dtype,
                            attr->shape->data, attr->shape->size,
                            attr->asymm_v.scale->data,
                            attr->asymm_v.scale->size,
                            attr->asymm_v.zero_point->data,
                            attr->asymm_v.zero_point->size,
                            attr->asymm_v.channel_dim,
                            internal_buffer0 );
                    break;
                default:
                    VSILOGE("Donot support quantize type %d", attr->quant);
                    VSI_ASSERT( FALSE );
                    break;
            }

            if ( attr->dtype == I4 || attr->dtype == U4 )
            {
                internal_buffer = malloc( bytes );
                status = vsi_nn_kernel_pack_4bit_data(attr, (uint8_t*)internal_buffer0, (uint8_t*)internal_buffer);
            }
        }
        else
        {
            vsi_nn_dtype_convert_float_to_dtype( float_buffer, size,
                    attr->dtype, internal_buffer );
        }
        buffer = (const void*)internal_buffer;
    }
    else
    {
        buffer = (const void*)float_buffer;
    }
    status = vsi_nn_kernel_tensor_write( tensor, attr, buffer, bytes );
final:
    if ( internal_attr )
    {
        vsi_nn_kernel_tensor_attr_release( &internal_attr );
    }
    if ( attr->dtype == I4 || attr->dtype == U4 )
    {
        vsi_nn_safe_free(internal_buffer0);
    }
    vsi_nn_safe_free(internal_buffer);

    return status;
} /* vsi_nn_kernel_tensor_write_from_float() */

vsi_status vsi_nn_kernel_scalar_get_dtype
    (
    vsi_nn_kernel_scalar_t scalar,
    vsi_nn_kernel_dtype_e * dtype
    )
{
    vsi_status status;
    vx_enum type;
    if( !dtype )
    {
        VSILOGW("Pointer to dtype is NULL");
        return VSI_FAILURE;
    }
    status = vxQueryScalar( (vx_scalar)scalar, VX_SCALAR_TYPE, &type, sizeof(vx_enum) );
    if( status == VSI_SUCCESS )
    {
        *dtype = vsi_nn_kernel_map_dtype( (vsi_nn_type_e)type );
    }
    return status;
} /* vsi_nn_kernel_scalar_get_dtype() */

#define DEF_KERNEL_SCALAR_FUNC( READ_FUNC_NAME, WRITE_FUNC_NAME, DTYPE, DTYPE_ID ) \
    vsi_status READ_FUNC_NAME \
        ( vsi_nn_kernel_scalar_t scalar, DTYPE * ptr  ) \
    { \
        vsi_status status; \
        vsi_nn_kernel_dtype_e dtype; \
        if( !ptr ) \
        { \
            VSILOGE("Pointer to store scalar is null"); \
            return VSI_FAILURE; \
        } \
        status = vsi_nn_kernel_scalar_get_dtype( scalar, &dtype ); \
        if( dtype != DTYPE_ID ) \
        { \
            VSILOGE("Try read scalar type %d as %d", dtype, DTYPE_ID); \
            return VSI_FAILURE; \
        } \
        if( status == VSI_SUCCESS ) \
        { \
            status = vxCopyScalarWithSize( (vx_scalar)scalar, sizeof(DTYPE), \
                    ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST ); \
        } \
        return status; \
    } \
    vsi_status WRITE_FUNC_NAME \
        ( vsi_nn_kernel_scalar_t scalar, DTYPE data  ) \
    { \
        vsi_status status; \
        status = vxCopyScalarWithSize( (vx_scalar)scalar, sizeof(DTYPE), \
                &data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ); \
        return status; \
    }

DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_int4,
                        vsi_nn_kernel_scalar_write_int4,
                        int8_t,   I4 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_int8,
                        vsi_nn_kernel_scalar_write_int8,
                        int8_t,   I8 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_int32,
                        vsi_nn_kernel_scalar_write_int32,
                        int32_t,  I32 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_uint8,
                        vsi_nn_kernel_scalar_write_uint8,
                        uint8_t,  U8 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_uint32,
                        vsi_nn_kernel_scalar_write_uint32,
                        uint32_t, U32 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_int64,
                        vsi_nn_kernel_scalar_write_int64,
                        int64_t,  I64 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_float32,
                        vsi_nn_kernel_scalar_write_float32,
                        float,    F32 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_float64,
                        vsi_nn_kernel_scalar_write_float64,
                        double,   F64 )
#undef DEF_KERNEL_SCALAR_FUNC

static void _convert_tensor_attr_to_vx_tensor_param
    (
    vx_tensor_create_params_t* p,
    const vsi_nn_kernel_tensor_attr_t* attr
    )
{
    memset( p, 0, sizeof( vx_tensor_create_params_t ) );

    p->num_of_dims = (uint32_t)attr->shape->size;
#define MAP_TYPE( var, src_type, dst_type ) \
    case src_type: \
        var = dst_type; \
        break;

    switch( attr->dtype )
    {
        MAP_TYPE( p->data_format, U4,  VSI_NN_TYPE_UINT4 );
        MAP_TYPE( p->data_format, I4,  VSI_NN_TYPE_INT4 );
        MAP_TYPE( p->data_format, I8,  VSI_NN_TYPE_INT8 );
        MAP_TYPE( p->data_format, I16, VSI_NN_TYPE_INT16 );
        MAP_TYPE( p->data_format, I32, VSI_NN_TYPE_INT32 );
        MAP_TYPE( p->data_format, I64, VSI_NN_TYPE_INT64 );
        MAP_TYPE( p->data_format, U8,  VSI_NN_TYPE_UINT8 );
        MAP_TYPE( p->data_format, U16, VSI_NN_TYPE_UINT16 );
        MAP_TYPE( p->data_format, U32, VSI_NN_TYPE_UINT32 );
        MAP_TYPE( p->data_format, U64, VSI_NN_TYPE_UINT64 );
        MAP_TYPE( p->data_format, F16, VSI_NN_TYPE_FLOAT16 );
        MAP_TYPE( p->data_format, F32, VSI_NN_TYPE_FLOAT32 );
        MAP_TYPE( p->data_format, F64, VSI_NN_TYPE_FLOAT64 );
        MAP_TYPE( p->data_format, BF16, VSI_NN_TYPE_BFLOAT16 );
        MAP_TYPE( p->data_format, BOOL8, VSI_NN_TYPE_BOOL8 );
        default:
            VSI_ASSERT( FALSE );
            break;
    }
    switch( attr->quant )
    {
        MAP_TYPE( p->quant_format,
                VSI_NN_KERNEL_QUANT_DFP,
                VSI_NN_QNT_TYPE_DFP );
        MAP_TYPE( p->quant_format,
                VSI_NN_KERNEL_QUANT_ASYMM,
                VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC );
        MAP_TYPE( p->quant_format,
                VSI_NN_KERNEL_QUANT_SYMM_PERCHANNEL,
                VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC );
        default:
            VSI_ASSERT( FALSE );
            break;
    }
    switch( attr->quant )
    {
        case VSI_NN_KERNEL_QUANT_DFP:
            p->quant_data.dfp.fixed_point_pos = (uint8_t)attr->dfp.fl;
            break;
        case VSI_NN_KERNEL_QUANT_ASYMM:
            p->quant_data.affine.scale = attr->asymm.scale;
            p->quant_data.affine.zeroPoint = attr->asymm.zero_point;
            break;
        //case VSI_NN_KERNEL_QUANT_SYMM_PERCHANNEL:
        //    break;
        default:
            VSI_ASSERT( FALSE );
            break;
    }
} /* _convert_tensor_attr_to_vx_tensor_param() */

vsi_nn_kernel_tensor_t vsi_nn_kernel_tensor_create
    (
    vsi_nn_kernel_graph_t graph,
    const vsi_nn_kernel_tensor_attr_t* attr,
    vsi_bool is_virtual
    )
{
    vsi_nn_kernel_tensor_t tensor = NULL;
    vx_tensor_create_params_t params;
    vx_size size_vxsize[VSI_NN_MAX_DIM_NUM] = {0};
    vx_uint32 size_u32[VSI_NN_MAX_DIM_NUM] = {0};
    size_t i = 0;

    _convert_tensor_attr_to_vx_tensor_param( &params, attr );
    //convert attr->shape->data to correct data type
    for(i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        size_vxsize[i] = -1 == attr->shape->data[i] ? -1 : (vx_size)attr->shape->data[i];
    }
    for(i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        size_u32[i] = -1 == attr->shape->data[i] ? -1 : (vx_uint32)attr->shape->data[i];
    }
#ifdef VSI_40BIT_VA_SUPPORT
    params.sizes = size_vxsize;
    (void)size_u32;
#else
    params.sizes = size_u32;
    (void)size_vxsize;
#endif
    if( is_virtual )
    {
        tensor = (vsi_nn_kernel_tensor_t)vxCreateVirtualTensor2(
                (vx_graph)graph, &params, sizeof( vx_tensor_create_params_t ) );
    }
    else
    {
        vx_context context = NULL;
        context = vxGetContext((vx_reference)graph);
        tensor = (vsi_nn_kernel_tensor_t)vxCreateTensor2(
                context, &params, sizeof( vx_tensor_create_params_t ) );
    }
    return tensor;
} /* vsi_nn_kernel_tensor_create() */

vsi_nn_tensor_t* vsi_nn_pad_tensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * input,
    vsi_size_t * pad_front,
    vsi_size_t * pad_end,
    vsi_size_t pad_size,
    vsi_nn_pad_mode_e mode,
    float pad_value
    )
{
    vsi_size_t sz = 0;
    vsi_nn_tensor_attr_t attr;
    float *input_data_ptr = NULL;
    float *output_data_ptr = NULL;
    float *src_ptr = NULL;
    float *dst_ptr = NULL;
    vsi_size_t i = 0;
    vsi_size_t out_w = 0;
    vsi_size_t out_h = 0;
    vsi_size_t out_d = 0;
    vsi_size_t out_b = 0;
    vsi_size_t output_width = 1;
    vsi_size_t output_height = 1;
    vsi_size_t output_depth = 1;
    vsi_size_t output_batch = 1;
    vsi_nn_dtype_t  dst_type;
    vsi_nn_tensor_t *output = NULL;

    input_data_ptr = vsi_nn_ConvertTensorToFloat32Data(graph, input);
    CHECK_PTR_FAIL_GOTO( input_data_ptr, "Create data ptr fail.", final );

    memcpy(&attr, &input->attr, sizeof(vsi_nn_tensor_attr_t));

    for(i = 0; i < pad_size; i ++)
    {
        vsi_size_t front = pad_front[i];
        vsi_size_t back  = pad_end[i];

        attr.size[i] = front + back + attr.size[i];
    }

    output_width = attr.size[0];
    output_height = attr.dim_num > 1 ? attr.size[1] : 1;
    output_depth = attr.dim_num > 2 ? attr.size[2] : 1;
    output_batch = attr.dim_num > 3 ? attr.size[3] : 1;

    sz = vsi_nn_GetTensorSize( attr.size, attr.dim_num, VSI_NN_TYPE_UINT8);
    output_data_ptr = (float *)malloc( sz * sizeof(float));
    CHECK_PTR_FAIL_GOTO( output_data_ptr, "Create data ptr fail.", final );

    dst_ptr = output_data_ptr;
    src_ptr = input_data_ptr;

    for (out_b = 0; out_b < output_batch; ++out_b)
    {
        for (out_d = 0; out_d < output_depth; ++out_d)
        {
            for (out_h = 0; out_h < output_height; ++out_h)
            {
                for (out_w = 0; out_w < output_width; ++out_w)
                {
                    if (out_b < pad_front[3] ||
                        out_b >= output_batch - pad_end[3] ||
                        out_d < pad_front[2] ||
                        out_d >= output_depth - pad_end[2] ||
                        out_h < pad_front[1] ||
                        out_h >= output_height - pad_end[1] ||
                        out_w < pad_front[0] ||
                        out_w >= output_width - pad_end[0])
                    {
                        *dst_ptr++ = pad_value;
                    }
                    else
                    {
                        *dst_ptr++ = *src_ptr++;
                    }
                }
            }
        }
    }

    output = vsi_nn_CreateTensor(graph, &attr);
    CHECK_PTR_FAIL_GOTO( output, "Create tensor fail.", final );

    memcpy(&dst_type, &attr.dtype, sizeof(vsi_nn_dtype_t));
    dst_type.vx_type = VSI_NN_TYPE_FLOAT32;
    vsi_nn_CopyRawDataToTensor( graph, (uint8_t *)output_data_ptr, &dst_type, output );
final:
    if (input_data_ptr)
    {
        free(input_data_ptr);
        input_data_ptr = NULL;
    }

    if (output_data_ptr)
    {
        free(output_data_ptr);
        output_data_ptr = NULL;
    }

    return output;
}


vsi_nn_tensor_t* vsi_nn_merge_input_zeropoint_to_bias
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias
    )
{
    vsi_nn_tensor_t * new_bias   = NULL;
    vsi_nn_tensor_attr_t attr;
    int32_t  *new_bias_data_ptr  = NULL;
    uint8_t  *weight_data        = NULL;
    int32_t  *bias_data          = NULL;
    uint32_t  i, j;
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    weight_data = vsi_nn_ConvertTensorToData(graph, weight);

    if (bias == NULL)
    {
        memcpy(&attr, &weight->attr, sizeof(vsi_nn_tensor_attr_t));
        attr.dim_num = 2;
        attr.size[0] = weight->attr.size[1];
        attr.size[1] = 1;
        if (weight->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
        {
            attr.dtype.scale = input->attr.dtype.scale * weight->attr.dtype.scale;
            attr.dtype.zero_point = 0;
            attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        }
    }
    else
    {
        memcpy(&attr, &bias->attr, sizeof(vsi_nn_tensor_attr_t));
        if (attr.dim_num == 1)
        {
            attr.size[1]  = 1;
            attr.dim_num  = 2;
        }
        bias_data = (int32_t *)vsi_nn_ConvertTensorToData(graph, bias);
    }

    new_bias_data_ptr = (int32_t *)malloc(attr.size[0] * sizeof(int32_t));
    memset((void *)new_bias_data_ptr, 0, sizeof(int32_t) * attr.size[0]);

    if (input->attr.dtype.zero_point != 0)
    {
        for (i = 0; i < weight->attr.size[1]; i++)
        {
            uint8_t *weight_ptr = weight_data + i * weight->attr.size[0];
            for (j = 0; j < weight->attr.size[0]; j++)
            {
                 new_bias_data_ptr[i] += -((int32_t)weight_ptr[j] - weight->attr.dtype.zero_point) \
                                         * input->attr.dtype.zero_point;
            }
        }
    }

    if (bias_data != NULL)
    {
        for (i = 0; i < weight->attr.size[1]; i++)
        {
            new_bias_data_ptr[i] += bias_data[i];
        }
    }

    new_bias = vsi_nn_CreateTensorFromData(graph, (uint8_t *)new_bias_data_ptr, &attr);

    vsi_nn_safe_free( new_bias_data_ptr );
    vsi_nn_safe_free( bias_data );
    vsi_nn_safe_free( weight_data );

    return new_bias;
}

vsi_status vsi_nn_set_sp_kernel_name
    (
        vsi_nn_kernel_node_t node,
        char* kernel_name
    )
{
    vsi_status status = VSI_SUCCESS;

    if (node == NULL || kernel_name == NULL)
    {
        return VSI_FAILURE;
    }

#if VX_STREAM_PROCESSOR_SUPPORT
    status = vxSetNodeAttribute((vx_node)node, VX_NODE_SP_NAME, kernel_name, sizeof(kernel_name));
#endif

    return status;
}

