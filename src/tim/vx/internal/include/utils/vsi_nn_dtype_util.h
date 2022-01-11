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

#ifndef _VSI_NN_DTYPE_UTIL_H
#define _VSI_NN_DTYPE_UTIL_H

#include "vsi_nn_platform.h"
#include "vsi_nn_types.h"
#include "vsi_nn_log.h"
#include "vsi_nn_math.h"

#ifdef __cplusplus
extern "C" {
#endif

OVXLIB_API void vsi_nn_TypeGetRange
    (
    vsi_nn_type_e type,
    double  * max_range,
    double  * min_range
    );

OVXLIB_API vsi_bool vsi_nn_TypeIsInteger
    (
    const vsi_nn_type_e type
    );

OVXLIB_API vsi_bool vsi_nn_TypeIsSigned
    (
    const vsi_nn_type_e type
    );

OVXLIB_API vsi_status vsi_nn_DtypeConvert
    (
    uint8_t * src,
    const vsi_nn_dtype_t * src_dtype,
    uint8_t * dst,
    const vsi_nn_dtype_t * dst_dtype
    );

/*
 * Deprecated: use vsi_nn_TypeGetBytes() insteatd.
 */
OVXLIB_API uint32_t vsi_nn_GetTypeBytes
    (
    const vsi_nn_type_e type
    );

OVXLIB_API uint32_t vsi_nn_TypeGetBytes
    (
    const vsi_nn_type_e type
    );

OVXLIB_API uint32_t vsi_nn_TypeGetBytesExt
    (
    const vsi_nn_type_e type
    );

OVXLIB_API uint32_t vsi_nn_TypeGetBits
    (
    const vsi_nn_type_e type
    );

OVXLIB_API uint16_t vsi_nn_Fp32ToFp16
    (
    float in
    );
/*
 * Deprecated: Use vsi_nn_Fp32ToFp16() instead
 */
#define vsi_nn_Fp32toFp16(in) vsi_nn_Fp32ToFp16( in )

OVXLIB_API float vsi_nn_Fp16ToFp32
    (
    int16_t in
    );
/*
 * Deprecated: Use vsi_nn_Fp16ToFp32() instead
 */
#define vsi_nn_Fp16toFp32(in) vsi_nn_Fp16ToFp32( in )

OVXLIB_API float vsi_nn_BFp16ToFp32
    (
    int16_t in
    );

OVXLIB_API uint16_t vsi_nn_Fp32ToBFp16
    (
    float in
    );

OVXLIB_API vsi_status vsi_nn_IntegerConvert
    (
    const void *    src,
    vsi_nn_type_e   src_type,
    void *          dest,
    vsi_nn_type_e   dest_type
    );

OVXLIB_API int32_t vsi_nn_Fp32ToDFP
    (
    const float in,
    const int8_t    fl,
    const vsi_nn_type_e type
    );

OVXLIB_API float vsi_nn_DFPToFp32
    (
    const int32_t val,
    const int8_t  fl,
    const vsi_nn_type_e type
    );

OVXLIB_API int32_t vsi_nn_Fp32ToAffine
    (
    const float  in,
    const float  scale,
    const int32_t    zero_point,
    const vsi_nn_type_e type
    );

OVXLIB_API float vsi_nn_AffineToFp32
    (
    const int32_t    val,
    const float  scale,
    const int32_t    zero_point,
    const vsi_nn_type_e type
    );

/*
 * Deprecated: Use vsi_nn_DtypeToFloat32() instead
 */
OVXLIB_API vsi_status vsi_nn_DtypeToFp32
    (
    void       * src,
    float * dst,
    uint32_t    index,
    const vsi_nn_dtype_t * src_dtype
    );

/*
 * Deprecated: Use vsi_nn_Float32ToDtype() instead
 */
OVXLIB_API vsi_status vsi_nn_Fp32toDtype
    (
    float   src,
    void       * dst,
    uint32_t    index,
    const vsi_nn_dtype_t * dst_dtype
    );

OVXLIB_API vsi_status vsi_nn_DtypeToFloat32
    (
    uint8_t   * src,
    float * dst,
    const vsi_nn_dtype_t * src_dtype
    );

OVXLIB_API vsi_status vsi_nn_Float32ToDtype
    (
    float   src,
    uint8_t   * dst,
    const vsi_nn_dtype_t * dst_dtype
    );

OVXLIB_API vsi_size_t vsi_nn_DtypeConvertRawData
    (
    uint8_t * src,
    vsi_size_t   src_bytes,
    const vsi_nn_dtype_t * src_dtype,
    uint8_t * dst,
    vsi_size_t   dst_bytes,
    const vsi_nn_dtype_t * dst_dtype
    );

OVXLIB_API vsi_size_t vsi_nn_DtypeConvertRawDataToFloat32
    (
    uint8_t   * src,
    vsi_size_t     src_bytes,
    const vsi_nn_dtype_t * src_dtype,
    float * dst,
    vsi_size_t     dst_size
    );

OVXLIB_API vsi_size_t vsi_nn_DtypeConvertFloat32ToRawData
    (
    float * src,
    vsi_size_t     src_size,
    uint8_t   * dst,
    vsi_size_t     dst_bytes,
    const vsi_nn_dtype_t * dst_dtype
    );

OVXLIB_API vsi_bool vsi_nn_QuantCheck
    (
    vsi_nn_tensor_t *input,
    vsi_nn_tensor_t *weight,
    vsi_nn_tensor_t *bias
    );

OVXLIB_API vsi_bool vsi_nn_DtypeCompare
    (
    vsi_nn_dtype_t *dtype0,
    vsi_nn_dtype_t *dtype1
    );

OVXLIB_API vsi_status vsi_nn_vxConvertTensorToFloat32Data
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr,
    float *f32_data,
    vsi_size_t f32_data_sz
    );

OVXLIB_API vsi_status vsi_nn_vxConvertFloat32DataToTensor
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr,
    float *f32_data,
    vsi_size_t f32_data_sz
    );

#ifdef __cplusplus
}
#endif

#endif

