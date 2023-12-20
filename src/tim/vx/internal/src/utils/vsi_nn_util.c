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
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>

#if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
#include <io.h>
#include <direct.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

typedef struct _vx_status_desc_t
{
    vx_status status;
    const char* desc;
} vx_status_desc_t;

static vx_status_desc_t const vx_status_desc[] =
{
    { VX_STATUS_MIN               /* (-25) */, "The lower bound of status codes in VX. Used for bounds checks only." },
    { VX_ERROR_REFERENCE_NONZERO  /* (-24) */, "An operation did not complete due to a"
                                                " reference count being non-zero." },
    { VX_ERROR_MULTIPLE_WRITERS   /* (-23) */, "The graph has more than one node outputting"
                                                " to the same data object. This is an invalid graph structure." },
    { VX_ERROR_GRAPH_ABANDONED    /* (-22) */, "The graph is stopped due to an error or a callback that abandoned"
                                                " execution." },
    { VX_ERROR_GRAPH_SCHEDULED    /* (-21) */, "The supplied graph already has been scheduled and may be currently"
                                                " executing." },
    { VX_ERROR_INVALID_SCOPE      /* (-20) */, "The supplied parameter is from another scope and cannot be used"
                                                " in the current scope." },
    { VX_ERROR_INVALID_NODE       /* (-19) */, "The supplied node could not be created." },
    { VX_ERROR_INVALID_GRAPH      /* (-18) */, "The supplied graph has invalid connections (cycles)." },
    { VX_ERROR_INVALID_TYPE       /* (-17) */, "The supplied type parameter is incorrect." },
    { VX_ERROR_INVALID_VALUE      /* (-16) */, "The supplied parameter has an incorrect value." },
    { VX_ERROR_INVALID_DIMENSION  /* (-15) */, "The supplied parameter is too big or too small in dimension." },
    { VX_ERROR_INVALID_FORMAT     /* (-14) */, "The supplied parameter is in an invalid format." },
    { VX_ERROR_INVALID_LINK       /* (-13) */, "The link is not possible as specified. The parameters are"
                                                " incompatible." },
    { VX_ERROR_INVALID_REFERENCE  /* (-12) */, "The reference provided is not valid." },
    { VX_ERROR_INVALID_MODULE     /* (-11) */, "The module does not contain the entry point." },
    { VX_ERROR_INVALID_PARAMETERS /* (-10) */, "The supplied parameter information does not match the"
                                                " kernel contract." },
    { VX_ERROR_OPTIMIZED_AWAY     /* (-9)  */, "The object refered to has been optimized out of existence." },
    { VX_ERROR_NO_MEMORY          /* (-8)  */, "An internal or implicit allocation failed. Typically catastrophic."
                                                " After detection, deconstruct the context." },
    { VX_ERROR_NO_RESOURCES       /* (-7)  */, "An internal or implicit resource can not be acquired (not memory)."
                                                " This is typically catastrophic. After detection, deconstruct"
                                                " the context." },
    { VX_ERROR_NOT_COMPATIBLE     /* (-6)  */, "The attempt to link two parameters together failed due"
                                                " to type incompatibilty." },
    { VX_ERROR_NOT_ALLOCATED      /* (-5)  */, "The parameter must be allocated by the system. " },
    { VX_ERROR_NOT_SUFFICIENT     /* (-4)  */, "The given graph has failed verification due to an insufficient"
                                                " number of required parameters, which cannot be automatically"
                                                " created. Typically this indicates required atomic parameters." },
    { VX_ERROR_NOT_SUPPORTED      /* (-3)  */, "The requested set of parameters produce a configuration that cannot"
                                                " be supported. " },
    { VX_ERROR_NOT_IMPLEMENTED    /* (-2)  */, "The requested kernel is missing. " },
    { VX_FAILURE                  /* (-1)  */, "A generic error code, used when no other describes the error." },
    { VX_SUCCESS                  /* (0)   */, "Success" },
};
/* Check whether enum value changed */
_compiler_assert(VX_ERROR_NOT_IMPLEMENTED == -2, VX_STATUS_VALUE_CHANGED);
_compiler_assert(VX_ERROR_INVALID_PARAMETERS == -10, VX_STATUS_VALUE_CHANGED);
_compiler_assert(VX_ERROR_INVALID_GRAPH == -18, VX_STATUS_VALUE_CHANGED);
_compiler_assert(VX_STATUS_MIN == -25, VX_STATUS_VALUE_CHANGED);

static const int16_t vx_status_desc_cnt = _cnt_of_array( vx_status_desc );

char* vsi_nn_strncpy
    (
    char* dest,
    const char* source,
    size_t count
    )
{
    char* ret = NULL;
    #if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
        strncpy_s(dest, count, source, _TRUNCATE);
    #else
        strncpy(dest, source, count);
    #endif
    return ret;
}

char* vsi_nn_strncat
    (
    char* dest,
    const char* source,
    size_t count
    )
{
    char* ret = NULL;
    #if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
        strncat_s(dest, count, source, _TRUNCATE);
        ret = dest;
    #else
        ret = strncat(dest, source, count);
    #endif
    return ret;
}

char* vsi_nn_getenv
    (
    const char * var_name
    )
{
    char* var = NULL;
    #if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
        size_t var_size = 0;
        _dupenv_s(&var, &var_size, var_name);
    #else
        var = getenv(var_name);
    #endif
    return var;
};

FILE* vsi_nn_fopen
    (
    const char * file_name,
    const char * mode
    )
{
    FILE * file = NULL;
    #if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
        fopen_s(&file, file_name, mode);
    #else
        file = fopen(file_name, mode);
    #endif
    return file;
}

static vsi_size_t _compute_stride_rounding
    (
    vsi_size_t out,
    vsi_size_t stride,
    vsi_nn_round_type_e rounding
    )
{
    if( VSI_NN_ROUND_CEIL == rounding )
    {
        out = ( out + stride - 1 ) / stride;
    }
    else
    {
        out = out / stride;
    }
    return out;
}

static vsi_size_t _compute_padding
    (
    vsi_size_t in_size,
    vsi_size_t ksize,
    vsi_size_t stride,
    vsi_size_t dilation_rate,
    vsi_size_t out_size
    )
{
    vsi_size_t effective_ksize;
    vsi_ssize_t padding;
    effective_ksize = (ksize - 1) * dilation_rate + 1;
    padding = (out_size - 1) * stride + effective_ksize - in_size;
    return vsi_nn_max(padding, 0);
} /* _compute_padding() */

int32_t vsi_nn_get_vx_pad_mode
    (
    vsi_nn_pad_mode_e mode
    )
{
    int32_t pad_mode = 0;
    switch (mode) {
        case VSI_NN_PAD_MODE_CONSTANT:
            pad_mode = VX_PAD_CONSTANT;
            break;
        case VSI_NN_PAD_MODE_REPLICATE:
            pad_mode = VX_PAD_REPLICATE;
            break;
        case VSI_NN_PAD_MODE_SYMMETRIC:
            pad_mode = VX_PAD_MIRROR_SYMMETRIC;
            break;
        case VSI_NN_PAD_MODE_REFLECT:
            pad_mode = VX_PAD_MIRROR_REFLECT;
            break;
        default:
            VSILOGE("Wrong pad_mode value");
            break;
    }

    return pad_mode;
}

uint8_t * vsi_nn_LoadBinaryData
    (
    const char * filename,
    vsi_size_t  * sz
    )
{
    uint8_t  * data;
    vsi_size_t   fsize;
    vsi_size_t      cnt;
    FILE      * fp;

    fp = vsi_nn_fopen( filename, "rb" );
    if( NULL == fp )
    {
        return NULL;
    }
    fseek( fp, 0L, SEEK_END );
    fsize = (vsi_size_t)ftell( fp );
    fseek( fp, 0L, SEEK_SET );
    data = (uint8_t *)malloc( fsize );
    cnt = 0;
    if( NULL == data )
    {
        VSILOGE( "Malloc %d memory fail.", fsize );
    }
    else
    {
        while( cnt < fsize )
        {
            cnt += (vsi_size_t)fread( &data[cnt], 1, fsize, fp );
            if( cnt == 0 )
            {
                break;
            }
        }
        VSILOGW( "Read %d bytes from file %s.", (uint32_t)cnt, filename );
    }
    fclose( fp );
    if( NULL != sz )
    {
        *sz = cnt;
    }
    return data;
} /* vsi_nn_LoadBinaryData() */

vsi_size_t vsi_nn_GetStrideSize
    (
    vsi_nn_tensor_attr_t * attr,
    vsi_size_t            * stride
    )
{
    if( NULL == attr || NULL == stride )
    {
        return 0;
    }

    return vsi_nn_GetStrideSizeBySize(attr->size, attr->dim_num, attr->dtype.vx_type, stride);
} /* vsi_nn_GetStrideSize() */

vsi_size_t vsi_nn_GetStrideSizeBySize
    (
    vsi_size_t   * size,
    vsi_size_t     dim_num,
    vsi_nn_type_e type,
    vsi_size_t   * stride
    )
{
    vsi_size_t total_bytes;
    vsi_size_t i;
    vsi_size_t type_bits;

    if( NULL == size || NULL == stride )
    {
        return 0;
    }
    type_bits = vsi_nn_TypeGetBits( type);
    stride[0] = type_bits / BITS_PER_BYTE;
    total_bytes = stride[0];
    if( type_bits < BITS_PER_BYTE && type_bits != 0 )
    {
        total_bytes = 1;
        if( size[0] % (BITS_PER_BYTE / type_bits) == 0 )
        {
             stride[1] = size[0] * type_bits / BITS_PER_BYTE;
        }
        else
        {
             stride[1] = size[0] * type_bits / BITS_PER_BYTE + 1;
        }

        total_bytes *= stride[1];
        for(i = 2; i < dim_num; i++)
        {
            stride[i] = size[i-1] * stride[i-1];
            total_bytes *= size[i];
        }
        total_bytes *= size[1];
    }
    else
    {
        for( i = 1; i < dim_num; i ++ )
        {
            stride[i] = size[i - 1] * stride[i - 1];
            total_bytes *= size[i];
        }
        total_bytes *= size[0];
    }

    for( i = dim_num; i < VSI_NN_MAX_DIM_NUM; i ++ )
    {
        stride[i] = total_bytes;
    }
    return total_bytes;
} /* vsi_nn_GetStrideSizeBySize() */

vsi_size_t vsi_nn_GetTotalBytesBySize
    (
    vsi_size_t   * size,
    vsi_size_t     dim_num,
    vsi_nn_type_e type
    )
{
    return vsi_nn_ShapeProduct( size, dim_num ) * vsi_nn_GetTypeBytes( type );
} /* vsi_nn_GetTotalBytesBySize() */

float vsi_nn_DataAsFloat32
    (
    uint8_t    * data,
    vsi_nn_type_e type
    )
{
    float val;
    uint32_t *p = (uint32_t*)(&val);
    int16_t fp16;

    *p = 0xFFFFFFFF;
    switch( type )
    {
    case VSI_NN_TYPE_BOOL8:
        val = (float)((int8_t*)data)[0];
        break;
    case VSI_NN_TYPE_INT4:

    case VSI_NN_TYPE_INT8:
        val = (float)((int8_t*)data)[0];
        break;
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_FLOAT8_E4M3:
    case VSI_NN_TYPE_FLOAT8_E5M2:
        val = (float)data[0];
        break;
    case VSI_NN_TYPE_INT16:
        val = (float)( (int16_t *)data )[0];
        break;
    case VSI_NN_TYPE_UINT16:
        val = (float)( (uint16_t *)data )[0];
        break;
    case VSI_NN_TYPE_FLOAT16:
        fp16 = ( (int16_t *)data )[0];
        val = vsi_nn_Fp16ToFp32( fp16 );
        break;
    case VSI_NN_TYPE_BFLOAT16:
        fp16 = ( (int16_t *)data )[0];
        val = vsi_nn_BFp16ToFp32( fp16 );
        break;
    case VSI_NN_TYPE_INT32:
        val = (float)( (int32_t *)data )[0];
        break;
    case VSI_NN_TYPE_UINT32:
        val = (float)( (uint32_t *)data )[0];
        break;
    case VSI_NN_TYPE_FLOAT32:
        val = ( (float *)data )[0];
        break;
    case VSI_NN_TYPE_INT64:
    case VSI_NN_TYPE_UINT64:
    case VSI_NN_TYPE_FLOAT64:
    default:
        VSILOGW( "Unsupport type %d", type );
        break;
    }
    return val;
} /* vsi_nn_DataAsFloat32() */

void vsi_nn_UpdateTensorDims
    (
    vsi_nn_tensor_attr_t * attr
    )
{
    uint32_t i;
    uint32_t num;
    if( NULL == attr )
    {
        return;
    }

    num = 0;
    for( i = 0; i < attr->dim_num; i ++ )
    {
        if( 0 == attr->size[i] )
        {
            break;
        }
        num ++;
    }

    if( attr->dim_num > VSI_NN_MAX_DIM_NUM )
    {
        VSILOGW( "Error dim number: %d", attr->dim_num );
        attr->dim_num = num;
    }
    else if( attr->dim_num != num )
    {
        VSILOGW( "Dim number and size mismatch: %d vs calculated = %d ", attr->dim_num, num );
        attr->dim_num = VSI_NN_DIM_AUTO;
    }
} /* vsi_nn_UpdateTensorDims() */

vsi_size_t vsi_nn_ComputeFilterSize
    (
    vsi_size_t   i_size,
    vsi_size_t   ksize,
    uint32_t * pad,
    uint32_t   stride,
    uint32_t   dilation,
    vsi_nn_round_type_e rounding
    )
{
    vsi_size_t out;
    if( 0 == stride )
    {
        if (i_size == ksize) {
            stride = 1;
        } else {
            VSILOGE( "Error stride value: 0." );
            return 0;
        }
    }
    if (dilation > 1)
    {
        ksize = dilation * (ksize - 1) + 1;
    }
    out = i_size + pad[0] + pad[1] - ksize;
    out = _compute_stride_rounding( out, stride, rounding );
    out ++;
    return out;
} /* vsi_nn_ComputeFilterSize() */

vsi_size_t vsi_nn_compute_filter_shape
    (
    vsi_nn_pad_e padding_type,
    vsi_size_t image_size,
    vsi_size_t ksize,
    uint32_t stride,
    uint32_t dilation_rate
    )
{
    vsi_size_t effective_ksize;
    effective_ksize = (ksize - 1) * dilation_rate + 1;
    switch (padding_type)
    {
    case VSI_NN_PAD_SAME:
        return (image_size + stride - 1) / stride;
    case VSI_NN_PAD_VALID:
        return (image_size + stride - effective_ksize) / stride;
    default:
        return 0;
    }
} /* vsi_nn_compute_filter_shape() */

void vsi_nn_compute_padding_per_axis
    (
    vsi_size_t   in_shape,
    vsi_size_t   ksize,
    uint32_t     stride,
    uint32_t     dilation,
    vsi_nn_pad_e pad_type,
    vsi_size_t   out_pad[2]
    )
{
    vsi_size_t out_size;
    vsi_size_t total_pads;
    if(dilation == 0)  dilation = 1;
    out_size = vsi_nn_compute_filter_shape(pad_type, in_shape, ksize, stride, dilation);
    total_pads = _compute_padding(in_shape, ksize, stride, dilation, out_size);

    out_pad[0] = total_pads / 2;
    out_pad[1] = total_pads - out_pad[0];
}

void vsi_nn_compute_padding
    (
    vsi_size_t   * in_shape,
    vsi_size_t   * ksize,
    uint32_t   * stride,
    uint32_t   * dilation,
    vsi_nn_pad_e pad_type,
    vsi_size_t   * out_pad
    )
{
    uint32_t dilation_w, dilation_h;
    if (NULL == in_shape || NULL == ksize
        || NULL == stride || NULL == out_pad)
    {
        return;
    }
    if (pad_type == VSI_NN_PAD_AUTO)
    {
        return;
    }
    if (NULL == dilation || (dilation[0] == 0 && dilation[1] == 0))
    {
        dilation_w = 1;
        dilation_h = 1;
    }
    else
    {
        dilation_w = dilation[0];
        dilation_h = dilation[1];
    }

    vsi_nn_compute_padding_per_axis(in_shape[0], ksize[0], stride[0], dilation_w, pad_type, out_pad);
    vsi_nn_compute_padding_per_axis(in_shape[1], ksize[1], stride[1], dilation_h, pad_type, out_pad + 2);
} /* vsi_nn_compute_padding() */

void vsi_nn_compute_padding_3d
    (
    const vsi_size_t   in_shape[3],
    const vsi_size_t   ksize[3],
    const uint32_t     stride[3],
    const uint32_t     dilation[3],
    const vsi_nn_pad_e pad_type,
    vsi_size_t   out_pad[6]
    )
{
    uint32_t dilation_w, dilation_h, dilation_d;
    if (NULL == in_shape || NULL == ksize
        || NULL == stride || NULL == out_pad)
    {
        return;
    }
    if (pad_type == VSI_NN_PAD_AUTO)
    {
        return;
    }
    if (NULL == dilation || (dilation[0] == 0 && dilation[1] == 0 && dilation[2] == 0))
    {
        dilation_w = 1;
        dilation_h = 1;
        dilation_d = 1;
    }
    else
    {
        dilation_w = dilation[0];
        dilation_h = dilation[1];
        dilation_d = dilation[2];
    }

    vsi_nn_compute_padding_per_axis(in_shape[0], ksize[0], stride[0], dilation_w, pad_type, out_pad);
    vsi_nn_compute_padding_per_axis(in_shape[1], ksize[1], stride[1], dilation_h, pad_type, out_pad + 2);
    vsi_nn_compute_padding_per_axis(in_shape[2], ksize[2], stride[2], dilation_d, pad_type, out_pad + 4);
}

void vsi_nn_ComputePadWithPadType
    (
    vsi_size_t   * in_shape,
    uint32_t     in_dim_num,
    vsi_size_t   * ksize,
    uint32_t   * stride,
    vsi_nn_pad_e pad_type,
    vsi_nn_round_type_e rounding,
    vsi_size_t   * out_pad
    )
{
    VSI_UNREFERENCED(in_dim_num);
    VSI_UNREFERENCED(rounding);
    vsi_nn_compute_padding(in_shape, ksize, stride, NULL, pad_type, out_pad);
} /* vsi_nn_ComputePadWithPadType() */

void vsi_nn_compute_padding_conv1d
(
    vsi_size_t   * in_shape,
    vsi_size_t   * ksize,
    uint32_t   * stride,
    uint32_t   * dilation,
    vsi_nn_pad_e pad_type,
    vsi_size_t   * out_pad
)
{
    vsi_size_t out_h;
    vsi_size_t pad_h;
    uint32_t dilation_h;
    if (NULL == in_shape || NULL == ksize
        || NULL == stride || NULL == out_pad)
    {
        return;
    }
    if (pad_type == VSI_NN_PAD_AUTO)
    {
        return;
    }
    if (NULL == dilation || dilation[0] == 0)
    {
        dilation_h = 1;
    }
    else
    {
        dilation_h = dilation[0];
    }

    out_h = vsi_nn_compute_filter_shape(pad_type, in_shape[0], ksize[0], stride[0], dilation_h);
    pad_h = _compute_padding(in_shape[0], ksize[0], stride[0], dilation_h, out_h);
    out_pad[0] = pad_h / 2;
    out_pad[1] = pad_h - out_pad[0];
} /* vsi_nn_compute_padding_conv1d() */

void vsi_nn_ComputePadWithPadTypeForConv1D
    (
    vsi_size_t   * in_shape,
    uint32_t     in_dim_num,
    vsi_size_t   * ksize,
    uint32_t   * stride,
    vsi_nn_pad_e pad_type,
    vsi_nn_round_type_e rounding,
    vsi_size_t   * out_pad
    )
{
    VSI_UNREFERENCED(in_dim_num);
    VSI_UNREFERENCED(rounding);
    vsi_nn_compute_padding_conv1d(in_shape, ksize, stride, NULL, pad_type, out_pad);
} /* vsi_nn_ComputePadWithPadTypeForConv1D() */

void vsi_nn_InitTensorsId
    (
    vsi_nn_tensor_id_t * ids,
    int                  num
    )
{
    num --;
    while( num >=0 )
    {
        ids[num] = VSI_NN_TENSOR_ID_NA;
        num --;
    }
} /* vsi_nn_InitTensorsId() */

void vsi_nn_GetPadForOvx
    (
    uint32_t * in_pad,
    uint32_t * out_pad
    )
{
    if( NULL == in_pad || NULL == out_pad )
    {
        return;
    }

    /* Workaround for ovx api. */
    out_pad[0] = in_pad[0];
    out_pad[1] = in_pad[2];
    if( out_pad[0] != in_pad[1] )
    {
        out_pad[0] = (uint32_t)( 0 - (int32_t)out_pad[0] );
    }
    if( out_pad[1] != in_pad[3] )
    {
        out_pad[1] = (uint32_t)( 0 - (int32_t)out_pad[1] );
    }
} /* vsi_nn_PadForDriver() */

vsi_bool vsi_nn_CreateTensorGroup
    (
    vsi_nn_graph_t  *  graph,
    vsi_nn_tensor_t *  in_tensor,
    uint32_t          axis,
    vsi_nn_tensor_t ** out_tensors,
    uint32_t          group_number
    )
{
    vsi_bool   ret;
    vsi_size_t sz;
    uint32_t i;
    vsi_size_t start[VSI_NN_MAX_DIM_NUM];
    vsi_size_t end[VSI_NN_MAX_DIM_NUM];
    vsi_nn_tensor_attr_t attr;

    if ( NULL == graph || NULL == in_tensor
        || NULL == out_tensors || 0 == group_number
        || axis >= VSI_NN_MAX_DIM_NUM ||
        0 == in_tensor->attr.size[axis] )
    {
        VSILOGW( "Create tensor group fail." );
        return FALSE;
    }

    if( 0 != ( in_tensor->attr.size[axis] % group_number ) )
    {
        VSILOGW( "Create tensor group fail." );
        return FALSE;
    }

    ret = TRUE;
    sz = in_tensor->attr.size[axis] / group_number;

    memcpy( &attr, &in_tensor->attr, sizeof( attr ) );
    attr.size[axis] = sz;
    memset( start, 0, sizeof(vsi_size_t) * VSI_NN_MAX_DIM_NUM );
    end[0] = in_tensor->attr.size[0];
    end[1] = in_tensor->attr.size[1];
    end[2] = in_tensor->attr.size[2];
    end[3] = in_tensor->attr.size[3];
    end[axis] = 0;
    for( i = 0; i <  group_number; i ++ )
    {
        start[axis] = end[axis];
        end[axis] += sz;
#ifdef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
        if (attr.dtype.qnt_type ==
                VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC ||
            attr.dtype.qnt_type == VSI_NN_QNT_TYPE_PERCHANNEL_SYMMETRIC_FLOAT8)
        {
            attr.dtype.scales = in_tensor->attr.dtype.scales + sz * i;
            attr.dtype.scale_dim = (int32_t)sz;
            attr.dtype.zero_points = in_tensor->attr.dtype.zero_points + sz * i;
            attr.dtype.zero_points_dim = (int32_t)sz;
        }
#endif
        out_tensors[i] = vsi_nn_CreateTensor( graph, &attr );
        if( NULL == out_tensors[i] )
        {
            VSILOGE( "Create tensor %d fail.", i );
            ret = FALSE;
            break;
        }
        if (out_tensors[i]->t)
        {
            vxReleaseTensor(&out_tensors[i]->t);
        }
        out_tensors[i]->t = vsi_nn_CreateViewTensor(graph, start, end, in_tensor);
        if( NULL == out_tensors[i]->t )
        {
            VSILOGE( "Create tensor %d from view fail.", i );
            ret = FALSE;
            break;
        }
    }
    return ret;
} /* vsi_nn_CreateTensorGroup() */

uint32_t vsi_nn_ShapeToString
    (
    vsi_size_t * shape,
    vsi_size_t   dim_num,
    char      * buf,
    uint32_t   buf_sz,
    vsi_bool     for_print
    )
{
#define _PRINT_FMT     (0)
#define _NOT_PRINT_FMT (1)
    vsi_size_t s;
    uint32_t count;
    const char * all_fmt[] = {" %"VSI_SIZE_T_SPECIFIER",", "%"VSI_SIZE_T_SPECIFIER"_" };
    const char * fmt;
    if( NULL == shape || NULL == buf
        || dim_num == 0 || buf_sz == 0 )
    {
        return 0;
    }
    if( FALSE == for_print )
    {
        fmt = all_fmt[_NOT_PRINT_FMT];
    }
    else
    {
        fmt = all_fmt[_PRINT_FMT];
    }
    count = 0;
    for( s = 0; s < dim_num; s++ )
    {
        if( count >= buf_sz )
        {
            break;
        }
        count += snprintf( &buf[count], buf_sz - count,
            fmt, shape[s] );
    }
    buf[count - 1] = 0;
    return count;
} /* vsi_nn_ShapeToString() */

int32_t vsi_nn_Access
    (
    const char *path,
    int32_t mode
    )
{
    if(NULL == path)
    {
        return -1;
    }

#if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
    return _access(path, mode);
#else
    return access(path, mode);
#endif
} /* vsi_nn_Access() */

int32_t vsi_nn_Mkdir
    (
    const char *path,
    int32_t mode
    )
{
    VSI_UNREFERENCED(mode);
    if(NULL == path)
    {
        return -1;
    }

#if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
    return _mkdir(path);
#else
    return mkdir(path, mode);
#endif
} /* vsi_nn_Mkdir() */

vsi_bool vsi_nn_CheckFilePath
    (
    const char *path
    )
{
    if(NULL == path)
    {
        VSILOGE("Please set file path");
        return FALSE;
    }

    if(vsi_nn_Access(path, 0) == 0)
    {
        return TRUE;
    }

    if(vsi_nn_Mkdir(path, 0775) == 0)
    {
        VSILOGI("Create directory %s", path);
        return TRUE;
    }
    else
    {
        VSILOGE("Create directory %s fail", path);
    }

    return FALSE;
} /* vsi_nn_CheckFilePath() */

/*
 * AlignedBuffer is figured as bellow:
 * | margin start at raw_addr | aligned_header | begin_guard  |
 *  data start at align_addr | end_guard |
*/
#define BEGIN_GUARD_SIZE 64
#define END_GUARD_SIZE 64
typedef struct
{
    uint8_t* raw_addr;
    uint8_t begin_guard[BEGIN_GUARD_SIZE];
} aligned_header;

uint8_t * vsi_nn_MallocAlignedBuffer
    (
    vsi_size_t mem_size,
    vsi_size_t align_start_size,
    vsi_size_t align_block_size
    )
{
    vsi_size_t sz;
    uintptr_t temp;
    uint8_t* raw_addr;
    uint8_t* p;
    uint8_t* align_addr;
    aligned_header* header;

    sz = sizeof(aligned_header) + mem_size +
        align_start_size + align_block_size + END_GUARD_SIZE;
    raw_addr = (uint8_t *)malloc( sz * sizeof( uint8_t ) );
    if (raw_addr == NULL)
    {
        return NULL;
    }
    memset(raw_addr, 0, sizeof( uint8_t ) * sz);
    p = raw_addr + sizeof(aligned_header);

    temp = (uintptr_t)(((uintptr_t)p) % align_start_size);
    if (temp == 0)
    {
        align_addr = p;
    }
    else
    {
        align_addr = p + align_start_size - temp;
    }
    header = (aligned_header*)(align_addr - sizeof(aligned_header));
    header->raw_addr = raw_addr;
    return align_addr;
}/* vsi_nn_MallocAlignedBuffer() */

void vsi_nn_FreeAlignedBuffer
    (
    uint8_t* handle
    )
{
    aligned_header* header;
    header = (aligned_header*)(handle - sizeof(aligned_header));
    free(header->raw_addr);
}

vsi_bool vsi_nn_IsBufferAligned
    (
    uint8_t * buf,
    vsi_size_t align_start_size
    )
{
    uintptr_t temp;

    temp = (uintptr_t)(((uintptr_t)buf) % align_start_size);
    if (temp == 0)
    {
        return TRUE;
    }
    return FALSE;
}/* vsi_nn_IsBufferAligned() */

void vsi_nn_FormatToString
    (
    vsi_nn_tensor_t *tensor,
    char *buf,
    vsi_size_t buf_sz
    )
{
    switch(tensor->attr.dtype.vx_type)
    {
    case VSI_NN_TYPE_INT4:vsi_nn_strncpy(buf,  "i4 ",  buf_sz);break;
    case VSI_NN_TYPE_INT8:vsi_nn_strncpy(buf,  "i8 ",  buf_sz);break;
    case VSI_NN_TYPE_INT16:vsi_nn_strncpy(buf, "i16", buf_sz);break;
    case VSI_NN_TYPE_INT32:vsi_nn_strncpy(buf, "i32", buf_sz);break;
    case VSI_NN_TYPE_INT64:vsi_nn_strncpy(buf, "i64", buf_sz);break;
    case VSI_NN_TYPE_UINT4:vsi_nn_strncpy(buf,  "u4 ",  buf_sz);break;
    case VSI_NN_TYPE_UINT8:vsi_nn_strncpy(buf,  "u8 ",  buf_sz);break;
    case VSI_NN_TYPE_UINT16:vsi_nn_strncpy(buf, "u16", buf_sz);break;
    case VSI_NN_TYPE_UINT32:vsi_nn_strncpy(buf, "u32", buf_sz);break;
    case VSI_NN_TYPE_UINT64:vsi_nn_strncpy(buf, "u64", buf_sz);break;
    case VSI_NN_TYPE_FLOAT16:vsi_nn_strncpy(buf, "f16", buf_sz);break;
    case VSI_NN_TYPE_FLOAT32:vsi_nn_strncpy(buf, "f32", buf_sz);break;
    case VSI_NN_TYPE_FLOAT64:vsi_nn_strncpy(buf, "f64", buf_sz);break;
    case VSI_NN_TYPE_BFLOAT16:vsi_nn_strncpy(buf, "bf16", buf_sz);break;
    case VSI_NN_TYPE_BOOL8:vsi_nn_strncpy(buf, "bool8", buf_sz);break;
    default:
        break;
    }
} /* vsi_nn_FormatToString() */

const char* vsi_nn_DescribeStatus
    (
    vsi_status status
    )
{
    static const char* unknown = "unknown";
    int16_t i = 0;

    for( i = 0; i < vx_status_desc_cnt; i++ )
    {
        if(vx_status_desc[i].status == status )
        {
            return vx_status_desc[i].desc;
        }
    }
    return unknown;
} /* vsi_nn_DescribeStatus() */

int32_t vsi_nn_partition
(
    void* data,
    int32_t left,
    int32_t right,
    comp_func func,
    vsi_bool is_recursion,
    uint32_t* indices
)
{
    int32_t key_index;
    int32_t low = left;
    int32_t high = right;
    if (left < right)
    {
        key_index = indices[left];
        while (low < high)
        {
            while (low < high && func(data, key_index, indices[high]))
            {
                high--;
            }
            indices[low] = indices[high];
            while (low < high && func(data, indices[low], key_index))
            {
                low++;
            }
            indices[high] = indices[low];
        }
        indices[low] = key_index;
        if (is_recursion)
        {
            vsi_nn_partition(data, left, low - 1, func, TRUE, indices);
            vsi_nn_partition(data, low + 1, right, func, TRUE, indices);
        }
    }
    return low;
}

void vsi_nn_print_size_array( vsi_size_t* array, size_t size )
{
    size_t i;
    size_t n;
#define _MSG_SIZE   (256)
    char buf[256];
    n = 0;
    for( i = 0; i < size; i ++ )
    {
        n += snprintf( &buf[n], _MSG_SIZE - n, "%"VSI_SIZE_T_SPECIFIER", ", array[i] );
        if( n >= _MSG_SIZE )
        {
            break;
        }
    }
    VSILOGD( "%s", buf );
#undef _MSG_SIZE
} /* vsi_nn_print_size_array() */

vsi_bool vsi_nn_IsEVISFeatureAvaiable
    (
    vsi_nn_context_t context
    )
{
    if ( context->config.evis.ver == VSI_NN_HW_EVIS_1
      || context->config.evis.ver == VSI_NN_HW_EVIS_2
      )
    {
        return TRUE;
    }

    return FALSE;
}

/* compare verision, return 1 greater, 0 equal, -1 less*/
int32_t vsi_nn_compareVersion
    (
    vsi_nn_graph_t * graph,
    uint32_t version_major,
    uint32_t version_minor,
    uint32_t version_patch
    )
{
    uint32_t   graph_version_major = 0;
    uint32_t   graph_version_minor = 0;
    uint32_t   graph_version_patch = 0;

    vsi_nn_GetGraphVersion( graph, &graph_version_major,
        &graph_version_minor, &graph_version_patch );

    if (graph_version_major > version_major)
    {
        return 1;
    }
    else if (graph_version_major < version_major)
    {
        return -1;
    }

    if (graph_version_minor > version_minor)
    {
        return 1;
    }
    else if (graph_version_minor < version_minor)
    {
        return -1;
    }

    if (graph_version_patch > version_patch)
    {
        return 1;
    }
    else if (graph_version_patch < version_patch)
    {
        return -1;
    }

    return 0;
}

float vsi_nn_activation
    (
    float value,
    vsi_nn_activation_e activation
    )
{
    switch(activation)
    {
        case VSI_NN_ACT_NONE:
            return value;
        case VSI_NN_ACT_RELU:
            return value < 0.f ? 0.f : value;
        case VSI_NN_ACT_RELU6:
            return vsi_nn_max(0.f, vsi_nn_min(value, 6.f));
        case VSI_NN_ACT_TANH:
            return (float)tanh(value);
        case VSI_NN_ACT_SIGMOID:
            return (float)(1.0f / (1.0f + exp(-value)));
        case VSI_NN_ACT_HARD_SIGMOID:
            value = value * 0.2f + 0.5f;
            return vsi_nn_max(0.f, vsi_nn_min(value, 1.f));
        default:
            VSILOGE("Unsupported activation: %d\n", activation);
            exit(1);
    }
}

vsi_bool vsi_nn_is_same_data_type(
    vsi_nn_tensor_t * src,
    vsi_nn_tensor_t * dst
    )
{
    return (src->attr.dtype.vx_type == dst->attr.dtype.vx_type);
}

vsi_bool vsi_nn_is_same_quant_type(
    vsi_nn_tensor_t * src,
    vsi_nn_tensor_t * dst
    )
{
    vsi_nn_dtype_t *src_dtype = NULL, *dst_dtype = NULL;

    src_dtype = &src->attr.dtype;
    dst_dtype = &dst->attr.dtype;

    if (src_dtype->qnt_type != dst_dtype->qnt_type)
    {
        return FALSE;
    }

    switch (src_dtype->qnt_type)
    {
        case VSI_NN_QNT_TYPE_DFP:
            if (src_dtype->fl != dst_dtype->fl)
            {
                return FALSE;
            }
            break;
        case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
        case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        case VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8:
        {
            const float diff = (float)1e-5;
            if (src_dtype->zero_point != dst_dtype->zero_point)
            {
                return FALSE;
            }
            if (vsi_nn_float_compare(src_dtype->scale, dst_dtype->scale, diff)
                == FALSE)
            {
                return FALSE;
            }
            break;
        }
        case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC:
        case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_ASYMMETRIC:
        case VSI_NN_QNT_TYPE_PERCHANNEL_SYMMETRIC_FLOAT8:
        {
            const float diff = (float)1e-5;
            int32_t i = 0;
            int32_t scale_cnt0 = src_dtype->scale_dim;
            int32_t scale_cnt1 = dst_dtype->scale_dim;

            if (scale_cnt0 == scale_cnt1)
            {
                const float* src_scale_ptr = src_dtype->scales;
                const float* dst_scale_ptr = dst_dtype->scales;
                for (i = 0; i < scale_cnt0; i++)
                {
                    if (vsi_nn_float_compare(
                            src_scale_ptr[i], dst_scale_ptr[i], diff) == FALSE)
                    {
                        return FALSE;
                    }
                }
            }
            break;
        }
        default:
            break;
    }

    return TRUE;
}

vsi_bool vsi_nn_is_same_type
    (
    vsi_nn_tensor_t * src,
    vsi_nn_tensor_t * dst
    )
{
    return (vsi_nn_is_same_data_type(src, dst) && vsi_nn_is_same_quant_type(src, dst));
}

vsi_bool vsi_nn_is_broadcast_operaton
    (
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            *  output
    )
{
    vsi_size_t out_rank = output->attr.dim_num;
    vsi_size_t i = 0;

    for (i = 0; i < out_rank; i++)
    {
        size_t j = 0;
        vsi_size_t dst_size = output->attr.size[i];

        for (j = 0; j < input_num; j++)
        {
            vsi_size_t src_size = i < inputs[j]->attr.dim_num  ? inputs[j]->attr.size[i] : 1;

            if (dst_size != src_size)
            {
                return TRUE;
            }
        }
    }
    return FALSE;
}

vsi_bool vsi_nn_is_broadcast_axes_operaton
    (
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            *  output,
    int32_t                    *  axis,
    int32_t                       axis_num
    )
{
    vsi_size_t out_rank = output->attr.dim_num;
    vsi_size_t i = 0;

    if (vsi_nn_is_broadcast_operaton(inputs, input_num, output) == FALSE)
    {
        return FALSE;
    }

    for (i = 0; i < out_rank; i++)
    {
        size_t j = 0;
        int32_t k = 0;
        vsi_size_t src0_size = i < inputs[0]->attr.dim_num  ?
                        inputs[0]->attr.size[i] : 1;

        for (k = 0; k < axis_num; k++)
        {
            if (axis[k] == (int32_t)i)
            {
                for (j = 1; j < input_num; j++)
                {
                    vsi_size_t src_size = i < inputs[j]->attr.dim_num  ?
                        inputs[j]->attr.size[i] : 1;

                    if (src0_size == src_size)
                    {
                        return FALSE;
                    }
                }

                break;
            }
        }

        if (axis[k] == (int32_t)i)
        {
            continue;
        }

        for (j = 1; j < input_num; j++)
        {
            vsi_size_t src_size = i < inputs[j]->attr.dim_num  ? inputs[j]->attr.size[i] : 1;

            if (src0_size != src_size)
            {
                return FALSE;
            }
        }
    }
    return TRUE;
}

float vsi_nn_get_tensor_scale
    (
    vsi_nn_tensor_t * tensor
    )
{
    float scale = 1.0f;

    switch (tensor->attr.dtype.qnt_type)
    {
        case VSI_NN_QNT_TYPE_DFP:
        {
            int8_t fl = tensor->attr.dtype.fl;
            if (fl >= 0)
            {
                scale = 1.0f / ( (float) ( (int64_t)1 << fl ));
            }
            else
            {
                scale = (float) ( (int64_t)1 << -fl );
            }
        }
            break;
        case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
        case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        case VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8:
            scale = tensor->attr.dtype.scale;
            break;
    default:
        break;
    }

    return scale;
}

int32_t vsi_nn_get_tensor_zero_point
    (
    vsi_nn_tensor_t * tensor
    )
{
    int32_t zero_point = 0;

    switch (tensor->attr.dtype.qnt_type)
    {
        case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
        case VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8:
            zero_point = 0;
            break;
        case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
            zero_point = tensor->attr.dtype.zero_point;
            break;
    default:
        break;
    }

    return zero_point;
}

void vsi_nn_get_tensor_clamp_min_max
    (
    vsi_nn_tensor_t * input,
    float *clampMin,
    float *clampMax
    )
{
    float zero_point = (float)vsi_nn_get_tensor_zero_point(input);
    vsi_nn_type_e vx_type = input->attr.dtype.vx_type;

    if (vx_type == VSI_NN_TYPE_UINT8)
    {
        *clampMin = - zero_point;
        *clampMax = 255 - zero_point;
    }
    else if (vx_type == VSI_NN_TYPE_INT8)
    {
        if (input->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC)
        {
            *clampMin = -127 - zero_point;
        }
        else
        {
            *clampMin = -128 - zero_point;
        }
        *clampMax = 127 - zero_point;
    }
    else if (vx_type == VSI_NN_TYPE_INT16)
    {
        *clampMin = -32768 - zero_point;
        *clampMax = 32767 - zero_point;
    }
    else if (vx_type == VSI_NN_TYPE_UINT16)
    {
        *clampMin = - zero_point;
        *clampMax = 65535 - zero_point;
    }
    else if (vx_type == VSI_NN_TYPE_FLOAT8_E4M3) {
        *clampMin = -448;
        *clampMax = 448;
    }
    else if (vx_type == VSI_NN_TYPE_FLOAT8_E5M2) {
        *clampMin = -57344;
        *clampMax = 57344;
    }
    else
    {
        uint32_t f32_min = 0xff800000;
        uint32_t f32_max = 0x7f800000;

        *clampMin = *(float*)&f32_min;
        *clampMax = *(float*)&f32_max;
    }
}

vsi_status vsi_nn_Pack4bitData
    (
    vsi_nn_tensor_t * tensor,
    uint8_t   * src,
    uint8_t * dest
    )
{
    vsi_status status;
    uint32_t i = 0, j = 0;
    uint8_t high = 0, low = 0;
    vsi_size_t src_size;

    status = VSI_SUCCESS;
    src_size = vsi_nn_GetElementNum( tensor );
    for( i = 0; i < src_size; i++ )
    {
        if( (i+1) % tensor->attr.size[0] == 0)
        {
            high = 0;
            low = src[i];
        }
        else
        {
            high = src[i+1];
            low = src[i];
            i++;
        }
        dest[j] = (high << 4) | (low & 0xF);
        j++;
    }
    return status;
} /* vsi_nn_Pack4bitData() */

vsi_status vsi_nn_Unpack4bitData
    (
    vsi_nn_tensor_t * tensor,
    uint8_t   * src,
    uint8_t * dest,
    vsi_nn_type_e type
    )
{
    vsi_status status;
    uint32_t i = 0, j = 0;
    uint8_t high = 0, low = 0;
    vsi_size_t stride[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t src_size;

    status = VSI_SUCCESS;
    src_size = vsi_nn_GetStrideSize(&tensor->attr, stride);
    for( i = 0 ; i < src_size; i++)
    {
        high = src[i] >> 4;
        low = src[i] & 0x0F;
        if( type == VSI_NN_TYPE_INT4 )
        {
            if( high > 7)
            {
                high = high | 0xF0;
            }
            if( low > 7)
            {
                low = low | 0xF0;
            }
        }
        if( tensor->attr.size[0] % stride[1] == 0 )
        {
            if( tensor->attr.size[0] == 1 )
            {
                dest[j] = low;
                j++;
            }
            else
            {
                dest[j] = low;
                dest[j+1] = high;
                j += 2;
            }
        }
        else
        {
            if( (i+1) % stride[1] == 0 )
            {
                dest[j] = low;
                j++;
            }
            else
            {
                dest[j] = low;
                dest[j+1] = high;
                j += 2;
            }
        }
    }
    return status;
} /* vsi_nn_Unpack4bitData() */

vsi_bool vsi_nn_is_3d_tensor
    (
    vsi_nn_tensor_t * tensor
    )
{
    if (3 == tensor->attr.dim_num)
    {
        return TRUE;
    }
    else
    {
        return FALSE;
    }
}

vsi_bool vsi_nn_is_stream_process_supported_types
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t** inputs,
    size_t input_num
    )
{
    size_t i = 0;

    if ( graph->ctx->config.support_stream_processor == 0 )
    {
        return FALSE;
    }

    if ( graph->ctx->config.sp_exec_count == 0 )
    {
        return FALSE;
    }

    for (i = 0; i < input_num; i++)
    {
        if (inputs && input_num > 0 && inputs[i] &&
            ( inputs[i]->attr.dtype.vx_type == VSI_NN_TYPE_INT32 ||
              inputs[i]->attr.dtype.vx_type == VSI_NN_TYPE_UINT32))
        {
            return FALSE;
        }
    }

    return TRUE;
}

vsi_bool vsi_nn_is_sp_supported_broadcast
    (
        vsi_nn_graph_t*   graph,
        vsi_nn_tensor_t** inputs,
        uint32_t          input_num,
        vsi_nn_tensor_t*  output
    )
{
typedef enum
{
    VSI_BROADCAST_BITS_NONE          = 0x0,
    VSI_BROADCAST_BITS_ON_AXIS_0     = 0x1,
    VSI_BROADCAST_BITS_ON_AXIS_1     = 0x2,
    VSI_BROADCAST_BITS_ON_AXIS_2     = 0x4,
    VSI_BROADCAST_BITS_ON_AXIS_3     = 0x8,
    VSI_BROADCAST_BITS_ON_AXIS_4     = 0x10,
    VSI_BROADCAST_BITS_ON_AXIS_5     = 0x20,
    VSI_BROADCAST_BITS_ON_AXIS_10    = 0x3,
    VSI_BROADCAST_BITS_ON_AXIS_210   = 0x7,
    VSI_BROADCAST_BITS_ON_AXIS_21    = 0x6,
} vsi_broadcast_bits_status_e;
#define _PACK_ELTWISE_SP_KEY(A_BROADCAST, B_BROADCAST) \
    ( (A_BROADCAST) | (B_BROADCAST << 8))
    int32_t broadcast_bits_0 = VSI_BROADCAST_BITS_NONE;
    int32_t broadcast_bits_1 = VSI_BROADCAST_BITS_NONE;
    uint32_t i = 0;
    uint32_t k = 0;
    uint32_t rank = output->attr.dim_num;
    vsi_bool is_broadcast = FALSE;
    vsi_bool support = TRUE;
    uint32_t key = 0;
    vsi_broadcast_bits_status_e broadcast_bits_status[VSI_NN_MAX_DIM_NUM] = {VSI_BROADCAST_BITS_NONE};

    if (vsi_nn_is_stream_process_supported_types(graph, inputs, input_num) == FALSE)
    {
        return FALSE;
    }

    for ( k = 1; k < input_num; k++ )
    {
        vsi_nn_tensor_t *input0 = inputs[k - 1];
        vsi_nn_tensor_t *input1 = inputs[k];
        uint32_t rank0 = input0->attr.dim_num;
        uint32_t rank1 = input1->attr.dim_num;

        broadcast_bits_status[0] = VSI_BROADCAST_BITS_NONE;
        broadcast_bits_status[1] = VSI_BROADCAST_BITS_NONE;

        for ( i = 0; i < rank; i++ )
        {
            vsi_size_t sz0 = i < rank0 ? input0->attr.size[i] : 1;
            vsi_size_t sz1 = i < rank1 ? input1->attr.size[i] : 1;

            if (sz0 != sz1)
            {
                broadcast_bits_0 |= sz0 == 1 ? (1 << i) : 0;
                broadcast_bits_1 |= sz1 == 1 ? (1 << i) : 0;

                is_broadcast = vx_true_e;
            }
        }

        broadcast_bits_status[0] = broadcast_bits_0;
        broadcast_bits_status[1] = broadcast_bits_1;

        if (broadcast_bits_status[0] == VSI_BROADCAST_BITS_ON_AXIS_1 &&
            broadcast_bits_status[1] == VSI_BROADCAST_BITS_NONE)
        {
            vsi_size_t channel = rank0 > 2 ? input0->attr.size[2] : 1;

            if (channel == 1)
            {
                broadcast_bits_status[0] = VSI_BROADCAST_BITS_ON_AXIS_21;
            }
        }
        else if (broadcast_bits_status[1] == VSI_BROADCAST_BITS_ON_AXIS_1 &&
                 broadcast_bits_status[0] == VSI_BROADCAST_BITS_NONE)
        {
            vx_size channel = rank0 > 2 ? input0->attr.size[2] : 1;

            if (channel == 1)
            {
                broadcast_bits_status[1] = VSI_BROADCAST_BITS_ON_AXIS_21;
            }
        }

        key = _PACK_ELTWISE_SP_KEY(broadcast_bits_status[0], broadcast_bits_status[1]);

        switch ( key )
        {
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_NONE,        VSI_BROADCAST_BITS_NONE):
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_ON_AXIS_2,   VSI_BROADCAST_BITS_NONE):
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_NONE,        VSI_BROADCAST_BITS_ON_AXIS_2):
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_ON_AXIS_21,  VSI_BROADCAST_BITS_NONE):
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_NONE,        VSI_BROADCAST_BITS_ON_AXIS_21):
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_ON_AXIS_210, VSI_BROADCAST_BITS_NONE):
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_NONE,        VSI_BROADCAST_BITS_ON_AXIS_210):
            break;
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_ON_AXIS_0,   VSI_BROADCAST_BITS_NONE):
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_ON_AXIS_10,  VSI_BROADCAST_BITS_NONE):
            support = support && (vsi_nn_TypeGetBits(input0->attr.dtype.vx_type) != 4);
            break;
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_NONE,        VSI_BROADCAST_BITS_ON_AXIS_0):
        case _PACK_ELTWISE_SP_KEY(VSI_BROADCAST_BITS_NONE,        VSI_BROADCAST_BITS_ON_AXIS_10):
            support = support && (vsi_nn_TypeGetBits(input1->attr.dtype.vx_type) != 4);
            break;
        default:
            support = !is_broadcast;
            break;
        }

        if (support == FALSE)
        {
            break;
        }
    }

    return support;
}
