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
#include <string.h>
#include <stdlib.h>

#include "vsi_nn_tensor_util.h"
#include "vsi_nn_test.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_dtype_util_prv.h"
#include "quantization/vsi_nn_asymmetric_affine.h"
#include "quantization/vsi_nn_dynamic_fixed_point.h"
#include "quantization/vsi_nn_perchannel_symmetric_affine.h"

vsi_bool vsi_nn_TypeIsInteger
    (
    const vsi_nn_type_e type
    )
{
    return type_is_integer(type);
} /* vsi_nn_TypeIsInteger() */

vsi_bool vsi_nn_TypeIsSigned
    (
    const vsi_nn_type_e type
    )
{
    return type_is_signed(type);
} /* vsi_nn_TypeIsSigned() */

uint32_t vsi_nn_TypeGetBitWidth
    (
    const vsi_nn_type_e type
    )
{
    uint32_t bw;
    bw = 8 * vsi_nn_TypeGetBytes( type );
    if( type_is_signed( type ) )
    {
        bw --;
    }
    return bw;
} /* vsi_nn_TypeGetBitWidth() */

int32_t vsi_nn_Fp32ToDFP
    (
    const float in,
    const int8_t    fl,
    const vsi_nn_type_e type
    )
{
    return fp32_to_dfp(in, fl, type);
} /* vsi_nn_Fp32ToDPF() */

float vsi_nn_DFPToFp32
    (
    const int32_t val,
    const int8_t  fl,
    const vsi_nn_type_e type
    )
{
    return dfp_to_fp32(val, fl, type);
} /* vsi_nn_DFPToFp32() */

int32_t vsi_nn_Fp32ToAffine
    (
    const float  in,
    const float  scale,
    const int32_t    zero_point,
    const vsi_nn_type_e type
    )
{
    return fp32_to_affine(in, scale, zero_point, type);
} /* vsi_nn_Fp32ToAffine() */

float vsi_nn_AffineToFp32
    (
    const int32_t    val,
    const float  scale,
    const int32_t    zero_point,
    const vsi_nn_type_e type
    )
{
    return affine_to_fp32(val, scale, zero_point, type);
} /* vsi_nn_AffineToFp32() */

uint16_t vsi_nn_Fp32ToFp16
    (
    float in
    )
{
    return fp32_to_fp16(in);
} /* vsi_nn_Fp32ToFp16() */

float vsi_nn_Fp16ToFp32
    (
    int16_t in
    )
{
    return fp16_to_fp32(in);
} /* vsi_nn_Fp16ToFp32() */

float vsi_nn_BFp16ToFp32
    (
    int16_t in
    )
{
    return bfp16_to_fp32(in);
} /* vsi_nn_Fp16ToFp32() */

uint16_t vsi_nn_Fp32ToBFp16
    (
    float in
    )
{
    return fp32_to_bfp16(in);
} /* vsi_nn_Fp32ToFp16() */

vsi_status vsi_nn_IntegerConvert
    (
    const void *    src,
    vsi_nn_type_e   src_type,
    void *          dest,
    vsi_nn_type_e   dest_type
    )
{
    return integer_convert(src, src_type, dest, dest_type);
} /* vsi_nn_IntegerConvert() */

vsi_status vsi_nn_DtypeConvert
    (
    uint8_t * src,
    const vsi_nn_dtype_t * src_dtype,
    uint8_t * dst,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    vsi_status status;
    float data;

    data = 0.0f;
    status = dtype_to_float32(src, &data, src_dtype);
    if(status != VSI_SUCCESS)
    {
        VSILOGE("dtype data convert to float32 fail");
        return status;
    }
    status = float32_to_dtype(data, dst, dst_dtype);
    if(status != VSI_SUCCESS)
    {
        VSILOGE("float32 data convert to dtype fail");
        return status;
    }
    return status;
} /* vsi_nn_DtypeConvert */

/*
* Deprated: Use vsi_nn_DtypeToFloat32() instead
*/
vsi_status vsi_nn_DtypeToFp32
    (
    void       * src,
    float * dst,
    uint32_t    index,     /* index to src buffer */
    const vsi_nn_dtype_t * src_dtype
    )
{
    uint8_t * ptr;
    ptr = (uint8_t *)src;

    //VSILOGW("Deprecated API, use vsi_nn_DtypeToFloat32 instead.");
    ptr += vsi_nn_TypeGetBytes( src_dtype->vx_type ) * index;

    return vsi_nn_DtypeToFloat32( ptr, dst, src_dtype );
} /* vsi_nn_DtypeToFp32() */

/*
* Deprated: Use vsi_nn_Float32ToDtype() instead
*/
vsi_status vsi_nn_Fp32toDtype
    (
    float   src,
    void       * dst,
    uint32_t    index,     /* index to dst buffer */
    const vsi_nn_dtype_t * dst_dtype
    )
{
    uint8_t * ptr;
    ptr = (uint8_t *)dst;

    //VSILOGW("Deprecated API, use vsi_nn_Float32ToDtype instead.");
    ptr += vsi_nn_TypeGetBytes( dst_dtype->vx_type ) * index;

    return vsi_nn_Float32ToDtype( src, ptr, dst_dtype );
} /* vsi_nn_Fp32toDtype */

vsi_status vsi_nn_DtypeToFloat32
    (
    uint8_t   * src,
    float * dst,
    const vsi_nn_dtype_t * src_dtype
    )
{
    return dtype_to_float32(src, dst, src_dtype);
} /* vsi_nn_DtypeToFloat32() */

vsi_status vsi_nn_Float32ToDtype
    (
    float   src,
    uint8_t   * dst,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    return float32_to_dtype(src, dst, dst_dtype);
} /* vsi_nn_Float32ToDtype() */

vsi_size_t vsi_nn_DtypeConvertRawData
    (
    uint8_t * src,
    vsi_size_t   src_bytes,
    const vsi_nn_dtype_t * src_dtype,
    uint8_t * dst,
    vsi_size_t   dst_bytes,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    uint8_t * src_iter;
    uint8_t * dst_iter;
    vsi_size_t count;
    vsi_size_t elements;
    vsi_size_t src_type_bytes;
    vsi_size_t dst_type_bytes;
    vsi_size_t target_bytes;
    vsi_size_t i;
    vsi_status status;
    count = 0;
    if( NULL == src || NULL == dst || NULL == src_dtype )
    {
        return count;
    }

    src_type_bytes = vsi_nn_TypeGetBytes( src_dtype->vx_type );
    dst_type_bytes = vsi_nn_TypeGetBytes( dst_dtype->vx_type );
    elements = src_bytes / src_type_bytes;
    target_bytes = dst_type_bytes * elements;
    if( dst_bytes < target_bytes )
    {
        VSILOGW("Wrong dest buffer size: %"VSI_SIZE_T_SPECIFIER", require: %"VSI_SIZE_T_SPECIFIER"",
            dst_bytes, target_bytes);
        return count;
    }
    src_iter = src;
    dst_iter = dst;
    for( i = 0; i < elements; i ++ )
    {
        status = vsi_nn_DtypeConvert( src_iter, src_dtype, dst_iter, dst_dtype );
        if( VSI_FAILURE == status )
        {
            break;
        }
        src_iter += src_type_bytes;
        dst_iter += dst_type_bytes;
    }
    count = i;
    return count;
} /* vsi_nn_DtypeConvertRawData() */

vsi_size_t vsi_nn_DtypeConvertRawDataToFloat32
    (
    uint8_t   * src,
    vsi_size_t     src_bytes,
    const vsi_nn_dtype_t * src_dtype,
    float * dst,
    vsi_size_t     dst_size
    )
{
    vsi_nn_dtype_t dst_dtype;
    memset( &dst_dtype, 0, sizeof( vsi_nn_dtype_t ) );
    dst_dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    return vsi_nn_DtypeConvertRawData(
        src, src_bytes, src_dtype,
        (uint8_t *)dst, dst_size * sizeof( float ), &dst_dtype );
} /*vsi_nn_DtypeConvertRawDataToFloat32()*/

vsi_size_t vsi_nn_DtypeConvertFloat32ToRawData
    (
    float * src,
    vsi_size_t     src_size,
    uint8_t   * dst,
    vsi_size_t     dst_bytes,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    vsi_nn_dtype_t src_dtype;
    memset( &src_dtype, 0, sizeof( vsi_nn_dtype_t ) );
    src_dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    return vsi_nn_DtypeConvertRawData(
        (uint8_t *)src, src_size * sizeof( float ), &src_dtype,
        dst, dst_bytes, dst_dtype );
} /*vsi_nn_DtypeConvertFloat32ToRawData()*/

uint32_t vsi_nn_TypeGetBytes
    (
    const vsi_nn_type_e type
    )
{
    return type_get_bytes( type );
} /* vsi_nn_TypeGetBytes() */

uint32_t vsi_nn_TypeGetBytesExt
    (
    const vsi_nn_type_e type
    )
{
    uint32_t bits_num = 0;
    bits_num = vsi_nn_TypeGetBits(type);
    if(bits_num < BITS_PER_BYTE)
    {
        return 1;
    }
    else
    {
        return bits_num / BITS_PER_BYTE;
    }
}

/*
* Deprecated: use vsi_nn_TypeGetBytes() insteatd.
*/
uint32_t vsi_nn_GetTypeBytes
    (
    const vsi_nn_type_e type
    )
{
    return type_get_bytes( type );
} /* vsi_nn_GetTypeBytes() */

uint32_t vsi_nn_TypeGetBits
    (
    const vsi_nn_type_e type
    )
{
    return type_get_bits(type);
} /* vsi_nn_GetTypeBits() */

vsi_bool vsi_nn_QuantCheck
    (
    vsi_nn_tensor_t *input,
    vsi_nn_tensor_t *weight,
    vsi_nn_tensor_t *bias
    )
{
    vsi_bool ret = TRUE;
    vsi_nn_qnt_type_e input_qnt_type, weight_qnt_type;
    vsi_nn_type_e input_dtype, weight_dtype;
    vsi_nn_qnt_type_e qnt_type;

    input_qnt_type = input->attr.dtype.qnt_type;
    input_dtype = input->attr.dtype.vx_type;
    weight_qnt_type = weight->attr.dtype.qnt_type;
    weight_dtype = weight->attr.dtype.vx_type;

    //do not check quant parammeters if types of input/weight/bias is hybrid combinaton
    if( input_dtype != weight_dtype || input_qnt_type != weight_qnt_type ||
        (bias && bias->attr.dtype.qnt_type != input_qnt_type) )
    {
        return ret;
    }

    if(VSI_NN_TYPE_VDATA == weight->attr.dtype.vx_type)
    {
        return ret;
    }
    if(type_is_integer(input_dtype) == FALSE)
    {
        return ret;
    }

    qnt_type = input->attr.dtype.qnt_type;
    switch(qnt_type)
    {
    case VSI_NN_QNT_TYPE_DFP:
        ret = vsi_nn_QuantDFPCheck(input, weight, bias);
        if(ret == FALSE)
        {
            VSILOGE("input_fl[%d] + weight_fl[%d] != bias_fl[%d]",
                input->attr.dtype.fl,
                weight->attr.dtype.fl,
                bias->attr.dtype.fl);
        }
        break;
    case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
    if (weight->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC)
    {
      ret = vsi_nn_QuantAffinePerchannelCheck(input, weight, bias);
      if(ret == FALSE)
      {
        VSILOGE("abs(input_scale * weight_scale - bias_scale) > 1e-5");
      }
    }
    else
    {
      ret = vsi_nn_QuantAffineCheck(input, weight, bias);
      if(ret == FALSE)
      {
        VSILOGE("input_scale[%.12lf] * weight_scale[%.12lf] != bias_scale[%.12lf]",
          input->attr.dtype.scale,
          weight->attr.dtype.scale,
          bias->attr.dtype.scale);
      }
    }
        break;
    default:
        ret = FALSE;
        break;
    }

    return ret;
} /* vsi_nn_QuantCheck() */

vsi_bool vsi_nn_DtypeCompare
    (
    vsi_nn_dtype_t *dtype0,
    vsi_nn_dtype_t *dtype1
    )
{
    if(NULL == dtype0 || NULL == dtype1)
    {
        return FALSE;
    }

    if(dtype0->vx_type != dtype1->vx_type || dtype0->qnt_type != dtype1->qnt_type)
    {
        return FALSE;
    }
    if(dtype0->qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        if(dtype0->fl != dtype1->fl)
        {
            return FALSE;
        }
    }
    else if( dtype0->qnt_type == VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC ||
             dtype0->qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC )
    {
        const float diff = (float)1e-5;
        if(dtype0->zero_point != dtype1->zero_point)
        {
            return FALSE;
        }
        if(vsi_nn_float_compare(dtype0->scale, dtype1->scale, diff) == FALSE)
        {
            return FALSE;
        }
    }

    return TRUE;
} /* vsi_nn_DtypeCompare */

vsi_status vsi_nn_vxConvertTensorToFloat32Data
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr,
    float *f32_data,
    vsi_size_t f32_data_sz
    )
{
    vsi_status status;
    uint8_t *data;
    vsi_size_t elements;
    uint32_t stride;
    vsi_nn_tensor_attr_t tensor_attr, *_attr;

    data = NULL;
    if(NULL == context || NULL == tensor || NULL == f32_data)
    {
        return VSI_FAILURE;
    }
    if(NULL == attr)
    {
        memset(&tensor_attr, 0, sizeof(tensor_attr));
        status = vsi_nn_vxGetTensorAttr(tensor, &tensor_attr);
        TEST_CHECK_STATUS(status, final);
        _attr = &tensor_attr;
    }
    else
    {
        _attr = attr;
    }

    status = VSI_FAILURE;
    elements = vsi_nn_vxGetTensorElementNum(_attr);
    stride = vsi_nn_TypeGetBytes(_attr->dtype.vx_type);
    if(f32_data_sz != elements * sizeof(float))
    {
        VSILOGE("buffer sz %"VSI_SIZE_T_SPECIFIER" != required sz %"VSI_SIZE_T_SPECIFIER"",
            f32_data_sz, elements * sizeof(float));
        return status;
    }
    data = vsi_nn_vxCopyTensorToData(context, tensor, _attr);
    TEST_CHECK_PTR(data, final);

    vsi_nn_DtypeConvertRawDataToFloat32(data,
                                        elements * stride,
                                        (const vsi_nn_dtype_t *)&_attr->dtype,
                                        f32_data,
                                        elements);

    status = VSI_SUCCESS;
final:
    if(data)
    {
        free(data);
        data = NULL;
    }
    return status;
} /* vsi_nn_vxConvertTensorToFloat32Data() */

vsi_status vsi_nn_vxConvertFloat32DataToTensor
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr,
    float *f32_data,
    vsi_size_t f32_data_sz
    )
{
    vsi_status status;
    uint8_t *data;
    vsi_size_t elements;
    uint32_t stride;
    vsi_nn_tensor_attr_t tensor_attr, *_attr;

    data = NULL;
    if(NULL == context || NULL == tensor || NULL == f32_data)
    {
        return VSI_FAILURE;
    }
    if(NULL == attr)
    {
        memset(&tensor_attr, 0, sizeof(tensor_attr));
        status = vsi_nn_vxGetTensorAttr(tensor, &tensor_attr);
        TEST_CHECK_STATUS(status, final);
        _attr = &tensor_attr;
    }
    else
    {
        _attr = attr;
    }

    status = VSI_FAILURE;
    elements = vsi_nn_vxGetTensorElementNum(_attr);
    stride = vsi_nn_GetTypeBytes(_attr->dtype.vx_type);
    if(f32_data_sz != elements * sizeof(float))
    {
        VSILOGE("buffer sz %"VSI_SIZE_T_SPECIFIER" != required sz %"VSI_SIZE_T_SPECIFIER"",
            f32_data_sz, elements * sizeof(float));
        return status;
    }

    data = (uint8_t *)malloc(elements * stride);
    TEST_CHECK_PTR(data, final);
    memset(data, 0, sizeof(uint8_t) * elements * stride);
    vsi_nn_DtypeConvertFloat32ToRawData(f32_data,
                                        elements,
                                        data,
                                        elements * vsi_nn_TypeGetBytes(_attr->dtype.vx_type),
                                        (const vsi_nn_dtype_t *)&_attr->dtype);

    status = vsi_nn_vxCopyDataToTensor(context, tensor, _attr, data);
final:
    if(data)
    {
        free(data);
        data = NULL;
    }
    return status;
} /* vsi_nn_vxConvertFloat32DataToTensor() */
