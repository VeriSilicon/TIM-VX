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
#include "vsi_nn_error.h"
#include "utils/vsi_nn_dtype_util_prv.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel.h"

#define DEF_DTYPE_CONVERT_NORMAL(SRC_NAME, SRC_DTYPE, DST_NAME, DST_DTYPE) \
static inline void _convert_##SRC_NAME##_to_##DST_NAME \
        ( \
        const SRC_DTYPE * buffer, \
        size_t size, \
        DST_DTYPE * out_buffer \
        ) \
    { \
        uint32_t i; \
        for( i = 0; i < size; i ++ ) \
        { \
            out_buffer[i] = (DST_DTYPE)buffer[i]; \
        } \
    }
//DEF_DTYPE_CONVERT_NORMAL( bool8,  int8_t,   float, float )
DEF_DTYPE_CONVERT_NORMAL( int8,   int8_t,   float, float )
DEF_DTYPE_CONVERT_NORMAL( int16,  int16_t,  float, float )
DEF_DTYPE_CONVERT_NORMAL( int32,  int32_t,  float, float )
DEF_DTYPE_CONVERT_NORMAL( uint8,  uint8_t,  float, float )
DEF_DTYPE_CONVERT_NORMAL( uint32, uint32_t, float, float )
DEF_DTYPE_CONVERT_NORMAL( uint16, uint16_t, float, float )
//DEF_DTYPE_CONVERT_NORMAL( float, float, bool8,   int8_t   )
DEF_DTYPE_CONVERT_NORMAL( float, float, int8,   int8_t   )
DEF_DTYPE_CONVERT_NORMAL( float, float, int16,  int16_t  )
DEF_DTYPE_CONVERT_NORMAL( float, float, int32,  int32_t  )
DEF_DTYPE_CONVERT_NORMAL( float, float, uint8,  uint8_t  )
DEF_DTYPE_CONVERT_NORMAL( float, float, uint32, uint32_t )
DEF_DTYPE_CONVERT_NORMAL( float, float, uint16, uint16_t )
#undef DEF_DTYPE_CONVERT_NORMAL

static inline void _convert_float16_to_float
    (
    const vsi_float16 * buffer,
    size_t size,
    float * out_buffer
    )
{
    uint32_t i;
    for( i = 0; i < size; i ++ )
    {
        out_buffer[i] = fp16_to_fp32( (int16_t)buffer[i] );
    }
} /* _convert_float16_to_float */

static inline void _convert_float_to_float16
    (
    const float * buffer,
    size_t size,
    vsi_float16 * out_buffer
    )
{
    uint32_t i;
    for( i = 0; i < size; i ++ )
    {
        out_buffer[i] = (vsi_float16)fp32_to_fp16( buffer[i] );
    }
} /* _convert_float_to_float16 */

static inline void _convert_bfloat16_to_float
    (
    const vsi_bfloat16 * buffer,
    size_t size,
    float * out_buffer
    )
{
    uint32_t i;
    for( i = 0; i < size; i ++ )
    {
        out_buffer[i] = bfp16_to_fp32( (int16_t)buffer[i] );
    }
} /* _convert_bfloat16_to_float */

static inline void _convert_float_to_bfloat16
    (
    const float * buffer,
    size_t size,
    vsi_bfloat16 * out_buffer
    )
{
    uint32_t i;
    for( i = 0; i < size; i ++ )
    {
        out_buffer[i] = (vsi_bfloat16)fp32_to_bfp16( buffer[i] );
    }
} /* _convert_float_to_bfloat16 */

#define DEF_DTYPE_CONVERT_QUANTIZE( SRC_NAME, SRC_DTYPE, ROUND, MIN, MAX ) \
    vsi_bool vsi_nn_dtype_convert_quantize_##SRC_NAME##_to_float \
        ( \
        const SRC_DTYPE * buffer, size_t size, \
        float scale, int32_t zero_point, \
        float * out_buffer \
        ) \
    { \
        uint32_t i; \
        if( !buffer || !out_buffer ) \
        { \
            return FALSE; \
        } \
        for( i = 0; i < size; i ++ ) \
        { \
            out_buffer[i] = (float)(((double)buffer[i] - (double)zero_point) * scale); \
        } \
        return TRUE; \
    } \
    vsi_bool vsi_nn_dtype_convert_float_to_quantize_##SRC_NAME \
        ( \
        const float * buffer, size_t size, \
        float scale, int32_t zero_point, \
        SRC_DTYPE * out_buffer \
        ) \
    { \
        uint32_t i; \
        if( !buffer || !out_buffer ) \
        { \
            return FALSE; \
        } \
        for( i = 0; i < size; i ++ ) \
        { \
            out_buffer[i] = (SRC_DTYPE)vsi_clamp(\
                    ROUND( buffer[i] / scale ) + zero_point, \
                    (double)MIN, (double)MAX ); \
        } \
        return TRUE; \
    }

DEF_DTYPE_CONVERT_QUANTIZE( symm8,   int8_t,   vsi_rtne, SCHAR_MIN, SCHAR_MAX )
DEF_DTYPE_CONVERT_QUANTIZE( symm16,  int16_t,  vsi_rtne, SHRT_MIN,  SHRT_MAX  )
DEF_DTYPE_CONVERT_QUANTIZE( symm32,  int32_t,  vsi_rtne, INT_MIN,   INT_MAX   )
DEF_DTYPE_CONVERT_QUANTIZE( symm64,  int64_t,  vsi_rtne, LLONG_MIN, LLONG_MAX )
DEF_DTYPE_CONVERT_QUANTIZE( asymm8,  uint8_t,  vsi_rtne, 0,         UCHAR_MAX )
//DEF_DTYPE_CONVERT_QUANTIZE( asymm16, uint16_t, vsi_rtne, 0,         USHRT_MAX )
//DEF_DTYPE_CONVERT_QUANTIZE( asymm32, uint32_t, vsi_rtne, 0,         UINT_MAX  )
#undef DEF_DTYPE_CONVERT_QUANTIZE

vsi_bool vsi_nn_dtype_convert_float_to_quantize_symm8_perchannel
    (
    const float * buffer, size_t size,
    const int32_t * shape, size_t rank,
    const float * scale, size_t scale_size,
    const int32_t * zero_point, size_t zero_point_size,
    int32_t channel_dim,
    int8_t * out_buffer
    )
{
    if( !buffer || !out_buffer )
    {
        return FALSE;
    }
    VSI_ASSERT( FALSE );
    return TRUE;
} /* vsi_nn_dtype_convert_float_to_quantize_symm8_perchannel() */

vsi_bool vsi_nn_dtype_convert_quantize_symm8_perchannel_to_float
    (
    const int8_t * buffer, size_t size,
    const int32_t * shape, size_t rank,
    const float * scale, size_t scale_size,
    const int32_t * zero_point, size_t zero_point_size,
    int32_t channel_dim,
    float * out_buffer
    )
{
    if( !buffer || !out_buffer )
    {
        return FALSE;
    }
    VSI_ASSERT( FALSE );
    return TRUE;
} /* vsi_nn_dtype_convert_quantize_symm8_perchannel_to_float() */

vsi_bool vsi_nn_dtype_convert_float_to_dtype
    (
    const float * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    void * out_buffer
    )
{
    if( !buffer || !out_buffer )
    {
        return FALSE;
    }
    switch( dtype )
    {
        case I8:
        case BOOL8:
            _convert_float_to_int8( buffer, size, (int8_t*)out_buffer );
            break;
        case I16:
            _convert_float_to_int16( buffer, size, (int16_t*)out_buffer );
            break;
        case I32:
            _convert_float_to_int32( buffer, size, (int32_t*)out_buffer );
            break;
        case U8:
            _convert_float_to_uint8( buffer, size, (uint8_t*)out_buffer );
            break;
        case U16:
            _convert_float_to_uint16( buffer, size, (uint16_t*)out_buffer );
            break;
        case U32:
            _convert_float_to_uint32( buffer, size, (uint32_t*)out_buffer );
            break;
        case F16:
            _convert_float_to_float16( buffer, size, (vsi_float16*)out_buffer );
            break;
        case BF16:
            _convert_float_to_bfloat16( buffer, size, (vsi_bfloat16*)out_buffer );
            break;
        default:
            VSILOGE("Don't support convert float to dtype %d.", dtype);
            return FALSE;
    }
    return TRUE;
} /* vsi_nn_dtype_convert_float_to_dtype() */

vsi_bool vsi_nn_dtype_convert_float_to_quantize_asymm
    (
    const float * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    float scale, int32_t zero_point,
    void * out_buffer
    )
{
    switch( dtype )
    {
        case U8:
            return vsi_nn_dtype_convert_float_to_quantize_asymm8(
                    buffer, size, scale, zero_point, (uint8_t*)out_buffer );
        default:
            VSILOGE("Don't support convert float to asymm quant %d.", dtype);
            break;
    }
    return FALSE;
} /* vsi_nn_dtype_convert_float_to_quantize_aysmm() */

vsi_bool vsi_nn_dtype_convert_float_to_quantize_dfp
    (
    const float * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    int32_t fl,
    void * out_buffer
    )
{
    float scale;
    if( !buffer || !out_buffer )
    {
        return FALSE;
    }
    scale = powf( 2.0f, (float)(-fl) );
    return vsi_nn_dtype_convert_float_to_quantize_symm(
            buffer, size, dtype, scale, 0, out_buffer );
} /* vsi_nn_dtype_convert_float_to_quantize_dfp() */

vsi_bool vsi_nn_dtype_convert_float_to_quantize_symm
    (
    const float * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    float scale, int32_t zero_point,
    void * out_buffer
    )
{
    switch( dtype )
    {
        case I8:
            return vsi_nn_dtype_convert_float_to_quantize_symm8(
                    buffer, size, scale, zero_point, (int8_t*)out_buffer );
        case I16:
            return vsi_nn_dtype_convert_float_to_quantize_symm16(
                    buffer, size, scale, zero_point, (int16_t*)out_buffer );
        case I32:
            return vsi_nn_dtype_convert_float_to_quantize_symm32(
                    buffer, size, scale, zero_point, (int32_t*)out_buffer );
        case I64:
            return vsi_nn_dtype_convert_float_to_quantize_symm64(
                    buffer, size, scale, zero_point, (int64_t*)out_buffer );
        default:
            VSILOGE("Don't support convert float to symm quant %d.", dtype);
            break;
    }
    return FALSE;
} /* vsi_nn_dtype_convert_float_to_quantize_symm() */

vsi_bool vsi_nn_dtype_convert_float_to_quantize_symm_perchannel
    (
    const float * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    const int32_t * shape, size_t rank,
    const float * scale, size_t scale_size,
    const int32_t * zero_point, size_t zero_point_size,
    int32_t channel_dim,
    void * out_buffer
    )
{
    switch( dtype )
    {
        case I8:
            vsi_nn_dtype_convert_float_to_quantize_symm8_perchannel(
                    buffer, size, shape, rank,
                    scale, scale_size, zero_point, zero_point_size,
                    channel_dim, (int8_t*)out_buffer );
            break;
        default:
            VSILOGE("Don't support convert float to symm perchannel quant %d.",
                    dtype);
            return FALSE;
    }
    return TRUE;
} /* vsi_nn_dtype_convert_float_to_quantize_symm_perchannel() */

vsi_bool vsi_nn_dtype_convert_dtype_to_float
    (
    const void * buffer,
    size_t size,
    vsi_nn_kernel_dtype_e dtype,
    float * out_buffer
    )
{
    if( !buffer || !out_buffer )
    {
        return FALSE;
    }
    switch( dtype )
    {
        case I8:
        case BOOL8:
            _convert_int8_to_float( (const int8_t*)buffer, size, out_buffer );
            break;
        case I16:
            _convert_int16_to_float( (const int16_t*)buffer, size, out_buffer );
            break;
        case I32:
            _convert_int32_to_float( (const int32_t*)buffer, size, out_buffer );
            break;
        case U8:
            _convert_uint8_to_float( (const uint8_t*)buffer, size, out_buffer );
            break;
        case U16:
            _convert_uint16_to_float( (const uint16_t*)buffer, size, out_buffer );
            break;
        case U32:
            _convert_uint32_to_float( (const uint32_t*)buffer, size, out_buffer );
            break;
        case F16:
            _convert_float16_to_float( (const vsi_float16*)buffer, size, out_buffer );
            break;
        case BF16:
            _convert_bfloat16_to_float( (const vsi_bfloat16*)buffer, size, out_buffer );
            break;
        default:
            VSILOGE("Don't support convert dtype %d to float.", dtype);
            return FALSE;
    }
    return TRUE;
} /* vsi_nn_dtype_convert_dtype_to_float() */

vsi_bool vsi_nn_dtype_convert_quantize_asymm_to_float
    (
    const void * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    float scale, int32_t zero_point,
    float * out_buffer
    )
{
    switch( dtype )
    {
        case U8:
            return vsi_nn_dtype_convert_quantize_asymm8_to_float(
                    (const uint8_t *)buffer, size, scale, zero_point, out_buffer );
        case I32:
            return vsi_nn_dtype_convert_quantize_symm32_to_float(
                    (const int *)buffer, size, scale, zero_point, out_buffer );
        default:
            VSILOGE("Don't support convert asymm quant %d to float.", dtype);
            break;
    }
    return FALSE;
} /* vsi_nn_dtype_convert_quantize_aysmm_to_float() */

vsi_bool vsi_nn_dtype_convert_quantize_dfp_to_float
    (
    const void * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    int32_t fl,
    float * out_buffer
    )
{
    float scale;
    if( !buffer || !out_buffer )
    {
        return FALSE;
    }
    scale = powf( 2.0f, (float)(-fl) );
    return vsi_nn_dtype_convert_quantize_symm_to_float(
            buffer, size, dtype, scale, 0, out_buffer );
} /* vsi_nn_dtype_convert_quantize_dfp_to_float() */

vsi_bool vsi_nn_dtype_convert_quantize_symm_to_float
    (
    const void * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    float scale, int32_t zero_point,
    float * out_buffer
    )
{
    switch( dtype )
    {
        case I8:
            return vsi_nn_dtype_convert_quantize_symm8_to_float(
                    (const int8_t *)buffer, size, scale, zero_point, out_buffer );
        case I16:
            return vsi_nn_dtype_convert_quantize_symm16_to_float(
                    (const int16_t *)buffer, size, scale, zero_point, out_buffer );
        case I32:
            return vsi_nn_dtype_convert_quantize_symm32_to_float(
                    (const int32_t *)buffer, size, scale, zero_point, out_buffer );
        case I64:
            return vsi_nn_dtype_convert_quantize_symm64_to_float(
                    (const int64_t *)buffer, size, scale, zero_point, out_buffer );
        default:
            VSILOGE("Don't support convert symm quant %d to float.", dtype);
            break;
    }
    return FALSE;
} /* vsi_nn_dtype_convert_quantize_symm_to_float() */

vsi_bool vsi_nn_dtype_convert_quantize_symm_perchannel_to_float
    (
    const void * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    const int32_t * shape, size_t rank,
    const float * scale, size_t scale_size,
    const int32_t * zero_point, size_t zero_point_size,
    int32_t channel_dim,
    float * out_buffer
    )
{
    switch( dtype )
    {
        case I8:
            vsi_nn_dtype_convert_quantize_symm8_perchannel_to_float(
                    (const int8_t*)buffer, size, shape, rank,
                    scale, scale_size, zero_point, zero_point_size,
                    channel_dim, out_buffer );
            break;
        default:
            VSILOGE("Don't support convert symm perchannel quant %d to float.", dtype);
            return FALSE;
    }
    return TRUE;
} /* vsi_nn_dtype_convert_quantize_symm_perchannel_to_float() */

