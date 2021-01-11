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
#ifndef _VSI_NN_DTYPE_UTIL_PRV_H
#define _VSI_NN_DTYPE_UTIL_PRV_H

#include "vsi_nn_types.h"
#include "vsi_nn_math.h"
#include "vsi_nn_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline vsi_bool type_is_integer
    (
    const vsi_nn_type_e type
    )
{
    vsi_bool ret;
    ret = FALSE;
    switch( type )
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_INT32:
    case VSI_NN_TYPE_INT64:
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_UINT16:
    case VSI_NN_TYPE_UINT32:
    case VSI_NN_TYPE_UINT64:
    case VSI_NN_TYPE_BOOL8:
        ret = TRUE;
        break;
    default:
        break;
    }
    return ret;
} /* type_is_integer() */

static inline vsi_bool type_is_signed
    (
    const vsi_nn_type_e type
    )
{
    vsi_bool ret;
    ret = FALSE;
    switch( type )
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_INT32:
    case VSI_NN_TYPE_INT64:
    case VSI_NN_TYPE_FLOAT16:
    case VSI_NN_TYPE_FLOAT32:
    case VSI_NN_TYPE_FLOAT64:
    case VSI_NN_TYPE_BFLOAT16:
        ret = TRUE;
        break;
    default:
        break;
    }
    return ret;
} /* type_is_signed() */

static inline uint32_t type_get_bytes
    (
    const vsi_nn_type_e type
    )
{
    switch( type )
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_BOOL8:
        return 1;
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_UINT16:
    case VSI_NN_TYPE_FLOAT16:
    case VSI_NN_TYPE_BFLOAT16:
        return 2;
    case VSI_NN_TYPE_INT32:
    case VSI_NN_TYPE_UINT32:
    case VSI_NN_TYPE_FLOAT32:
        return 4;
    case VSI_NN_TYPE_INT64:
    case VSI_NN_TYPE_UINT64:
    case VSI_NN_TYPE_FLOAT64:
        return 8;
    default:
        return 0;
    }
} /* type_get_bytes() */

static inline void type_get_range
    (
    vsi_nn_type_e type,
    double  * max_range,
    double  * min_range
    )
{
    int32_t bits;
    double from, to;
    from = 0.0;
    to = 0.0;
    bits = type_get_bytes( type ) * 8;
    if( type_is_integer( type ) )
    {
        if( type_is_signed( type ) )
        {
            from = (double)(-(1L << (bits - 1)));
            to = (double)((1UL << (bits - 1)) - 1);
        }
        else
        {
            from = 0.0;
            to = (double)((1UL << bits) - 1);
        }
    }
    else
    {
        //  TODO: Add float
    }
    if( NULL != max_range )
    {
        *max_range = to;
    }
    if( NULL != min_range )
    {
        *min_range = from;
    }
} /* type_get_range() */

static inline int32_t fp32_to_affine
    (
    const float  in,
    const float  scale,
    const int32_t    zero_point,
    const vsi_nn_type_e type
    )
{
    int32_t data;
    double max_range;
    double min_range;
    type_get_range( type, &max_range, &min_range );
    data = (int32_t)(vsi_rint( in / scale ) + zero_point );
    data = vsi_nn_max( (int32_t)min_range, vsi_nn_min( (int32_t)max_range , data ) );
    return data;
} /* fp32_to_affine() */

static inline float affine_to_fp32
    (
    const int32_t    val,
    const float  scale,
    const int32_t    zero_point,
    const vsi_nn_type_e type
    )
{
    float data;
    data = ( (float)val - zero_point ) * scale;
    return data;
} /* affine_to_fp32() */

static inline int32_t fp32_to_dfp
    (
    const float in,
    const int8_t    fl,
    const vsi_nn_type_e type
    )
{
    int32_t data;
    double max_range;
    double min_range;
    type_get_range( type, &max_range, &min_range );
    if( fl > 0 )
    {
        data = (int32_t)vsi_rint( in * (float)( (int64_t)1 << fl ) );
    }
    else
    {
        data = (int32_t)vsi_rint( in * ( 1.0f / (float)( (int64_t)1 << -fl ) ) );
    }
    data = vsi_nn_min( data, (int32_t)max_range );
    data = vsi_nn_max( data, (int32_t)min_range );
    return data;
} /* fp32_to_dfp() */

static inline float dfp_to_fp32
    (
    const int32_t val,
    const int8_t  fl,
    const vsi_nn_type_e type
    )
{
    float result;
    if( fl > 0 )
    {
        result = (float)val * ( 1.0f / ( (float) ( (int64_t)1 << fl ) ) );
    }
    else
    {
        result = (float)val * ( (float) ( (int64_t)1 << -fl ) );
    }
    return result;
} /* dfp_to_fp32() */

static inline vsi_status integer_convert
    (
    const void *    src,
    vsi_nn_type_e   src_type,
    void *          dest,
    vsi_nn_type_e   dest_type
    )
{
    vsi_status status = VSI_SUCCESS;
    if( type_is_integer( src_type ) && type_is_integer( dest_type ) )
    {
        uint8_t    all_zeros[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
        uint8_t    all_ones[] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
        uint32_t   src_sz = type_get_bytes( src_type );
        uint32_t   dest_sz = type_get_bytes( dest_type );
        uint8_t*   buffer = all_zeros;
        if( type_is_signed( src_type ) && (((int8_t *)src)[src_sz - 1] & 0x80) )
        {
            buffer = all_ones;
        }
        memcpy( buffer, src, src_sz );
        memcpy( dest, buffer, dest_sz );
    }
    else
    {
        status = VSI_FAILURE;
    }
    return status;
} /* integer_convert() */

typedef union
{
    uint32_t u;
    float f;
} _fp32_t;

static inline float fp16_to_fp32
    (
    int16_t in
    )
{
    const _fp32_t magic = { (254 - 15) << 23 };
    const _fp32_t infnan = { (127 + 16) << 23 };
    _fp32_t o;
    // Non-sign bits
    o.u = ( in & 0x7fff ) << 13;
    o.f *= magic.f;
    if(o.f  >= infnan.f)
    {
        o.u |= 255 << 23;
    }
    //Sign bit
    o.u |= ( in & 0x8000 ) << 16;
    return o.f;
} /* fp16_to_fp32() */

static inline float bfp16_to_fp32
    (
    int16_t in
    )
{
    int32_t t1, t2, t3;
    float out;

    t1 = in & 0x00FF;                       // Mantissa
    t2 = in & 0xFF00;                       // Sign bit + Exponent
    t3 = in & 0x7F00;                       // Exponent

    t1 <<= 16;
    t2 <<= 16;                              // Shift (sign + Exponent) bit into position
    t1 |= t2;                               // Re-insert (sign + Exponent) bit

    *((uint32_t*)&out) = t1;

    return t3 == 0 ? 0 : out;
} /* bfp16_to_fp32() */

static inline uint16_t fp32_to_fp16
    (
    float in
    )
{
    uint32_t fp32 = *((uint32_t *) &in);
    uint32_t t1 = (fp32 & 0x80000000u) >> 16;  /* sign bit. */
    uint32_t t2 = (fp32 & 0x7F800000u) >> 13;  /* Exponent bits */
    uint32_t t3 = (fp32 & 0x007FE000u) >> 13;  /* Mantissa bits, no rounding */
    uint32_t fp16 = 0u;
    if( t2 >= 0x023c00u )
    {
        fp16 = t1 | 0x7BFF;     /* Don't round to infinity. */
    }
    else if( t2 <= 0x01c000u )
    {
        fp16 = t1;
    }
    else
    {
        t2 -= 0x01c000u;
        fp16 = t1 | t2 | t3;
    }
    return (uint16_t) fp16;
} /* fp32_to_fp16() */

static inline uint16_t fp32_to_bfp16
    (
    float in
    )
{
    uint32_t fp32 = *((unsigned int *) &in);
    uint32_t t1 = fp32 >> 16;

    return (uint16_t) t1;
} /* fp32_to_bfp16() */

static inline uint16_t fp32_to_bfp16_rtne
    (
    float in
    )
{
    /*
    Convert a float point to bfloat16, with round-nearest-to-even as rounding method.
    */

    uint32_t fp32 = *((unsigned int *) &in);
    uint16_t out;

    uint32_t lsb = (fp32 >> 16) & 1;    /* Least significant bit of resulting bfloat. */
    uint32_t rounding_bias = 0x7fff + lsb;

    if ( VSI_NN_FLOAT32_NAN == in )
    {
        out = 0x7fc0;
    }
    else
    {
        fp32 += rounding_bias;
        out = (uint16_t) (fp32 >> 16);
    }

    return out;
} /* fp32_to_bfp16_rtne */

static inline vsi_status dtype_to_float32
    (
    uint8_t *src,
    float   *dst,
    const vsi_nn_dtype_t * src_dtype
    )
{
    switch( src_dtype->vx_type )
    {
    case VSI_NN_TYPE_FLOAT32:
        *dst = *(float *)src;
        break;
    case VSI_NN_TYPE_FLOAT16:
        *dst = fp16_to_fp32( *(int16_t *)src );
        break;
    case VSI_NN_TYPE_BFLOAT16:
        *dst = bfp16_to_fp32( *(int16_t *)src );
        break;
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_BOOL8:
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_INT32:
        {
            int32_t src_value = 0;
            integer_convert(src, src_dtype->vx_type, &src_value, VSI_NN_TYPE_INT32 );
            switch( src_dtype->qnt_type )
            {
            case VSI_NN_QNT_TYPE_DFP:
                *dst = dfp_to_fp32( src_value, src_dtype->fl, src_dtype->vx_type );
                break;
            case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
                *dst = affine_to_fp32( src_value,
                    src_dtype->scale, src_dtype->zero_point, src_dtype->vx_type );
                break;
            case VSI_NN_QNT_TYPE_NONE:
                *dst = (float)src_value;
                break;
            default:
                break;
            }
        }
        break;
    default:
        return VSI_FAILURE;
    }
    return VSI_SUCCESS;
}

static inline vsi_status float32_to_dtype
    (
    float   src,
    uint8_t *dst,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    switch( dst_dtype->vx_type )
    {
    case VSI_NN_TYPE_FLOAT32:
        *(float *)dst = src;
        break;
    case VSI_NN_TYPE_FLOAT16:
        *(int16_t *)dst = fp32_to_fp16( src );
        break;
    case VSI_NN_TYPE_BFLOAT16:
        *(int16_t *)dst = fp32_to_bfp16_rtne( src );
        break;
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_BOOL8:
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_INT32:
        {
            int32_t dst_value = 0;
            switch( dst_dtype->qnt_type )
            {
            case VSI_NN_QNT_TYPE_DFP:
                dst_value = fp32_to_dfp( src, dst_dtype->fl, dst_dtype->vx_type );
                break;
            case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
                dst_value = fp32_to_affine( src,
                    dst_dtype->scale, dst_dtype->zero_point, dst_dtype->vx_type );
                break;
            case VSI_NN_QNT_TYPE_NONE:
                dst_value = (int32_t)src;
                break;
            default:
                break;
            }
            integer_convert( &dst_value, VSI_NN_TYPE_INT32, dst, dst_dtype->vx_type );
        }
        break;
    default:
        return VSI_FAILURE;
    }
    return VSI_SUCCESS;
}

#ifdef __cplusplus
}
#endif

vsi_bool vsi_nn_dtype_convert_float_to_quantize_symm8
    (
    const float * buffer, size_t size,
    float scale, int32_t zero_point,
    int8_t * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_float_to_quantize_symm16
    (
    const float * buffer, size_t size,
    float scale, int32_t zero_point,
    int16_t * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_float_to_quantize_symm32
    (
    const float * buffer, size_t size,
    float scale, int32_t zero_point,
    int32_t * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_float_to_quantize_symm64
    (
    const float * buffer, size_t size,
    float scale, int32_t zero_point,
    int64_t * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_float_to_quantize_asymm8
    (
    const float * buffer, size_t size,
    float scale, int32_t zero_point,
    uint8_t * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_float_to_quantize_symm8_perchannel
    (
    const float * buffer, size_t size,
    const int32_t * shape, size_t rank,
    const float * scale, size_t scale_size,
    const int32_t * zero_point, size_t zero_point_size,
    int32_t channel_dim,
    int8_t * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_quantize_symm8_to_float
    (
    const int8_t * buffer, size_t size,
    float scale, int32_t zero_point,
    float * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_quantize_symm16_to_float
    (
    const int16_t * buffer, size_t size,
    float scale, int32_t zero_point,
    float * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_quantize_symm32_to_float
    (
    const int32_t * buffer, size_t size,
    float scale, int32_t zero_point,
    float * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_quantize_symm64_to_float
    (
    const int64_t * buffer, size_t size,
    float scale, int32_t zero_point,
    float * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_quantize_asymm8_to_float
    (
    const uint8_t * buffer, size_t size,
    float scale, int32_t zero_point,
    float * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_quantize_symm8_perchannel_to_float
    (
    const int8_t * buffer, size_t size,
    const int32_t * shape, size_t rank,
    const float * scale, size_t scale_size,
    const int32_t * zero_point, size_t zero_point_size,
    int32_t channel_dim,
    float * out_buffer
    );


#endif
