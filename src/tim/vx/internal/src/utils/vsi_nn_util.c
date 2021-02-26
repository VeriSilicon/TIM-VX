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
#include <math.h>
#include <fcntl.h>

#ifdef _WIN32
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

static uint32_t _compute_stride_rounding
    (
    uint32_t out,
    uint32_t stride,
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

static uint32_t _compute_padding
    (
    uint32_t in_size,
    uint32_t ksize,
    uint32_t stride,
    uint32_t dilation_rate,
    uint32_t out_size
    )
{
    uint32_t effective_ksize;
    int32_t padding;
    effective_ksize = (ksize - 1) * dilation_rate + 1;
    padding = (out_size - 1) * stride + effective_ksize - in_size;
    return vsi_nn_max(padding, 0);
} /* _compute_padding() */

uint8_t * vsi_nn_LoadBinaryData
    (
    const char * filename,
    uint32_t  * sz
    )
{
    uint8_t  * data;
    uint32_t   fsize;
    size_t      cnt;
    FILE      * fp;

    fp = fopen( filename, "rb" );
    if( NULL == fp )
    {
        return NULL;
    }
    fseek( fp, 0L, SEEK_END );
    fsize = (uint32_t)ftell( fp );
    fseek( fp, 0L, SEEK_SET );
    data = (uint8_t *)malloc( fsize );
    cnt = 0;
    if( NULL == data )
    {
        VSILOGE( "Malloc %d memory fail.", fsize );
    }
    else
    {
        while( (uint32_t)cnt < fsize )
        {
            cnt += fread( &data[cnt], 1, fsize, fp );
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
        *sz = (uint32_t)cnt;
    }
    return data;
} /* vsi_nn_LoadBinaryData() */

uint32_t vsi_nn_GetStrideSize
    (
    vsi_nn_tensor_attr_t * attr,
    uint32_t            * stride
    )
{

    if( NULL == attr || NULL == stride )
    {
        return 0;
    }

    return vsi_nn_GetStrideSizeBySize(attr->size, attr->dim_num, attr->dtype.vx_type, stride);
} /* vsi_nn_GetStrideSize() */

uint32_t vsi_nn_GetStrideSizeBySize
    (
    uint32_t   * size,
    uint32_t     dim_num,
    vsi_nn_type_e type,
    uint32_t   * stride
    )
{
    uint32_t total_bytes;
    uint32_t i;

    if( NULL == size || NULL == stride )
    {
        return 0;
    }

    stride[0] = vsi_nn_GetTypeBytes( type );
    total_bytes = stride[0];
    for( i = 1; i < dim_num; i ++ )
    {
        stride[i] = size[i - 1] * stride[i - 1];
        total_bytes *= size[i];
    }
    total_bytes *= size[0];
    for( i = dim_num; i < VSI_NN_MAX_DIM_NUM; i ++ )
    {
        stride[i] = total_bytes;
    }
    return total_bytes;
} /* vsi_nn_GetStrideSizeBySize() */

uint32_t vsi_nn_GetTotalBytesBySize
    (
    uint32_t   * size,
    uint32_t     dim_num,
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
    case VSI_NN_TYPE_INT8:
        val = (float)((int8_t*)data)[0];
        break;
    case VSI_NN_TYPE_UINT8:
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


uint32_t vsi_nn_ComputeFilterSize
    (
    uint32_t   i_size,
    uint32_t   ksize,
    uint32_t * pad,
    uint32_t   stride,
    uint32_t   dilation,
    vsi_nn_round_type_e rounding
    )
{
    uint32_t out;
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

uint32_t vsi_nn_compute_filter_shape
    (
    vsi_nn_pad_e padding_type,
    uint32_t image_size,
    uint32_t ksize,
    uint32_t stride,
    uint32_t dilation_rate
    )
{
    uint32_t effective_ksize;
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

void vsi_nn_compute_padding
    (
    uint32_t   * in_shape,
    uint32_t   * ksize,
    uint32_t   * stride,
    uint32_t   * dilation,
    vsi_nn_pad_e pad_type,
    uint32_t   * out_pad
    )
{
    uint32_t out_w, out_h;
    uint32_t pad_w, pad_h;
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

    out_w = vsi_nn_compute_filter_shape(pad_type, in_shape[0], ksize[0], stride[0], dilation_w);
    out_h = vsi_nn_compute_filter_shape(pad_type, in_shape[1], ksize[1], stride[1], dilation_h);
    pad_w = _compute_padding(in_shape[0], ksize[0], stride[0], dilation_w, out_w);
    pad_h = _compute_padding(in_shape[1], ksize[1], stride[1], dilation_h, out_h);
    out_pad[0] = pad_w / 2;
    out_pad[1] = pad_w - out_pad[0];
    out_pad[2] = pad_h / 2;
    out_pad[3] = pad_h - out_pad[2];
} /* vsi_nn_compute_padding() */

void vsi_nn_ComputePadWithPadType
    (
    uint32_t   * in_shape,
    uint32_t     in_dim_num,
    uint32_t   * ksize,
    uint32_t   * stride,
    vsi_nn_pad_e pad_type,
    vsi_nn_round_type_e rounding,
    uint32_t   * out_pad
    )
{
    vsi_nn_compute_padding(in_shape, ksize, stride, NULL, pad_type, out_pad);
} /* vsi_nn_ComputePadWithPadType() */

void vsi_nn_compute_padding_conv1d
(
    uint32_t   * in_shape,
    uint32_t   * ksize,
    uint32_t   * stride,
    uint32_t   * dilation,
    vsi_nn_pad_e pad_type,
    uint32_t   * out_pad
)
{
    uint32_t out_h;
    uint32_t pad_h;
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
    uint32_t   * in_shape,
    uint32_t     in_dim_num,
    uint32_t   * ksize,
    uint32_t   * stride,
    vsi_nn_pad_e pad_type,
    vsi_nn_round_type_e rounding,
    uint32_t   * out_pad
    )
{
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
    uint32_t sz;
    uint32_t i;
    uint32_t start[VSI_NN_MAX_DIM_NUM];
    uint32_t end[VSI_NN_MAX_DIM_NUM];
    vsi_nn_tensor_attr_t attr;

    if( NULL == graph || NULL == in_tensor
        || NULL == out_tensors || 0 == group_number
        || 0 == in_tensor->attr.size[axis] )
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
    memset( start, 0, sizeof( uint32_t ) * VSI_NN_MAX_DIM_NUM );
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
        if ( attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC )
        {
            attr.dtype.scales = in_tensor->attr.dtype.scales + sz * i;
            attr.dtype.scale_dim = sz;
            attr.dtype.zero_points = in_tensor->attr.dtype.zero_points + sz * i;
            attr.dtype.zero_points_dim = sz;
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
    uint32_t * shape,
    uint32_t   dim_num,
    char      * buf,
    uint32_t   buf_sz,
    vsi_bool     for_print
    )
{
#define _PRINT_FMT     (0)
#define _NOT_PRINT_FMT (1)
    uint32_t s;
    uint32_t count;
    const char * all_fmt[] = {" %d,", "%d_" };
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

#ifdef _WIN32
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
    if(NULL == path)
    {
        return -1;
    }

#ifdef _WIN32
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

typedef struct
{
    uint8_t* raw_addr;
} aligned_header;

uint8_t * vsi_nn_MallocAlignedBuffer
    (
    uint32_t mem_size,
    uint32_t align_start_size,
    uint32_t align_block_size
    )
{
    uint32_t sz;
    uintptr_t temp;
    uint8_t* raw_addr;
    uint8_t* p;
    uint8_t* align_addr;
    aligned_header* header;

    sz = sizeof(aligned_header) + mem_size + align_start_size + align_block_size;
    raw_addr = (uint8_t *)malloc( sz * sizeof( uint8_t ) );
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
    uint32_t align_start_size
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
    uint32_t buf_sz
    )
{
    switch(tensor->attr.dtype.vx_type)
    {
    case VSI_NN_TYPE_INT8:strncpy(buf,  "i8 ",  buf_sz);break;
    case VSI_NN_TYPE_INT16:strncpy(buf, "i16", buf_sz);break;
    case VSI_NN_TYPE_INT32:strncpy(buf, "i32", buf_sz);break;
    case VSI_NN_TYPE_INT64:strncpy(buf, "i64", buf_sz);break;
    case VSI_NN_TYPE_UINT8:strncpy(buf,  "u8 ",  buf_sz);break;
    case VSI_NN_TYPE_UINT16:strncpy(buf, "u16", buf_sz);break;
    case VSI_NN_TYPE_UINT32:strncpy(buf, "u32", buf_sz);break;
    case VSI_NN_TYPE_UINT64:strncpy(buf, "u64", buf_sz);break;
    case VSI_NN_TYPE_FLOAT16:strncpy(buf, "f16", buf_sz);break;
    case VSI_NN_TYPE_FLOAT32:strncpy(buf, "f32", buf_sz);break;
    case VSI_NN_TYPE_FLOAT64:strncpy(buf, "f64", buf_sz);break;
    case VSI_NN_TYPE_BFLOAT16:strncpy(buf, "bf16", buf_sz);break;
    case VSI_NN_TYPE_BOOL8:strncpy(buf, "bool8", buf_sz);break;
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


/* Greatest Common Divisor*/
static vsi_bool vsi_nn_GetDataDivisors
    (
    vx_uint32 input_value,
    vx_uint32 *divisors,
    vx_uint32 gcd
    )
{
    vx_uint32 i                 = 0;
#define VSI_NN_MAX_IMAGE_WIDTH  (65536)
    for (i = vsi_nn_min(input_value, VSI_NN_MAX_IMAGE_WIDTH - 1); i > 0; i --)
    {
        if ((i % gcd == 0) && (input_value % i == 0))
        {
            *divisors = i;

            return TRUE;
        }
    }
#undef VSI_NN_MAX_IMAGE_WIDTH
    return FALSE;
}

void vsi_nn_OptimizedEltOPShape
    (
       vsi_nn_tensor_t * input,
       uint32_t          sizes[VSI_NN_MAX_DIM_NUM],
       uint32_t        * num_of_dims
    )
{
    uint32_t element_count     = 0;
    uint32_t i                 = 0;
#define VSI_NN_MAX_IMAGE_WIDTH  (65536)
    element_count = vsi_nn_GetElementNum(input);

    for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        sizes[i] = 1;
    }

    if (element_count < VSI_NN_MAX_IMAGE_WIDTH)
    {
        sizes[0] = element_count;
        *num_of_dims = 2;
    }
    else
    {
        vx_uint32 divisors = 1;
        for (i = 0; i < 2; i++)
        {
            divisors = 1;
            vsi_nn_GetDataDivisors(element_count, &divisors, 1);
            if (1 == divisors)
            {
                divisors = element_count;
            }
            sizes[i] = divisors;
            element_count = element_count / divisors;
        }

        sizes[2] = element_count;
        if (1 == sizes[2])
        {
            *num_of_dims = 2;
        }
        else
        {
            *num_of_dims = 3;
        }
    }
#undef VSI_NN_MAX_IMAGE_WIDTH
}

vsi_bool vsi_nn_OptimizedEltWiseOPShape
    (
    vsi_nn_tensor_t * input0,
    vsi_nn_tensor_t * input1,
    vsi_nn_tensor_t * output,
    uint32_t          sizes0[VSI_NN_MAX_DIM_NUM],
    uint32_t          sizes1[VSI_NN_MAX_DIM_NUM],
    uint32_t          sizes2[VSI_NN_MAX_DIM_NUM],
    uint32_t        * dim_num
    )
{
    vsi_bool  status            = TRUE;
    uint32_t  i                 = 0;
    uint32_t  cnt               = 0;
    uint32_t  dims              = 0;
    uint32_t  element_count0    = 0;
    uint32_t  element_count1    = 0;
    vsi_bool  enable_broadcast  = FALSE;
    vsi_bool  enable_broadcast1 = FALSE;
    uint32_t  broadcast_Bits    = 0;

    element_count0 = vsi_nn_GetElementNum(input0);
    element_count1 = vsi_nn_GetElementNum(input1);

    if (element_count0 == 1 || element_count1 == 1)
    {
        enable_broadcast1 = TRUE;
    }

    /*************step 1:init tensor shape*****************/
    for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        sizes0[i] = 1;
        sizes1[i] = 1;
        sizes2[i] = 1;
    }

    /*************step 2:squeeze tensor shape*****************/
    for (i = 0; i < output->attr.dim_num; i++)
    {
        uint32_t sz0 = input0->attr.dim_num > i ? input0->attr.size[i] : 1;
        uint32_t sz1 = input1->attr.dim_num > i ? input1->attr.size[i] : 1;
        uint32_t sz2 = output->attr.dim_num > i ? output->attr.size[i] : 1;

        if (sz0 == sz1 && sz0 == 1)
        {
            continue;
        }
        else
        {
            sizes0[cnt] = sz0;
            sizes1[cnt] = sz1;
            sizes2[cnt] = sz2;

            cnt ++;
            dims ++;
        }
    }

    for (i = 0; i < dims; i++)
    {
        uint32_t sz0 = sizes0[i];
        uint32_t sz1 = sizes1[i];

        if (sz0 != sz1)
        {
            enable_broadcast = TRUE;
            broadcast_Bits |= (1 << i);
        }
    }

    /*************step 3:reshape tensor shape*****************/
    if (enable_broadcast == FALSE || enable_broadcast1)
    {
        vsi_nn_OptimizedEltOPShape(input0, sizes0, &dims);
        vsi_nn_OptimizedEltOPShape(input1, sizes1, &dims);
        vsi_nn_OptimizedEltOPShape(output, sizes2, &dims);
    }
    else
    {
#define VSI_NN_MAX_IMAGE_WIDTH  (65536)
        switch (broadcast_Bits)
        {
        case VSI_NN_BROAD_CAST_BITS_0:
            {
                vx_uint32 element_count = 1;
                vx_uint32 divisors = 1;

                for (i = 1; i < dims; i++)
                {
                    element_count *= sizes0[i];
                }

                divisors = 1;
                vsi_nn_GetDataDivisors(element_count, &divisors, 1);

                sizes0[1] = divisors;
                sizes1[1] = divisors;
                sizes2[1] = divisors;
                sizes0[2] = element_count / divisors;
                sizes1[2] = element_count / divisors;
                sizes2[2] = element_count / divisors;
                dims = 3;

                break;
            }
        case VSI_NN_BROAD_CAST_BITS_0 | VSI_NN_BROAD_CAST_BITS_1:
        case VSI_NN_BROAD_CAST_BITS_0 | VSI_NN_BROAD_CAST_BITS_1 | VSI_NN_BROAD_CAST_BITS_2:
            {
                vx_uint32 w0 = sizes0[0] * sizes0[1];
                vx_uint32 w1 = sizes1[0] * sizes1[1];
                vx_uint32 w  = sizes2[0] * sizes2[1];
                vx_uint32 h = sizes0[2];

                if (h < VSI_NN_MAX_IMAGE_WIDTH && (w0 == 1 || w1 == 1)
                    && w < VSI_NN_MAX_IMAGE_WIDTH)
                {
                    sizes0[0] = w0;
                    sizes1[0] = w1;
                    sizes2[0] = w;
                    sizes0[1] = sizes0[2];
                    sizes1[1] = sizes1[2];
                    sizes2[1] = sizes2[2];
                    sizes0[2] = 1;
                    sizes1[2] = 1;
                    sizes2[2] = 1;
                }

                break;
            }
        case VSI_NN_BROAD_CAST_BITS_2:
            {
                vx_uint32 w = sizes0[0] * sizes0[1];

                if (w < VSI_NN_MAX_IMAGE_WIDTH)
                {
                    sizes0[0] = w;
                    sizes1[0] = w;
                    sizes2[0] = w;
                    sizes0[1] = sizes0[2];
                    sizes1[1] = sizes1[2];
                    sizes2[1] = sizes2[2];
                    sizes0[2] = 1;
                    sizes1[2] = 1;
                    sizes2[2] = 1;
                }

                break;
            }
        default:
            if (dims == output->attr.dim_num)
                status = FALSE;
            break;
        }
    }

#undef VSI_NN_MAX_IMAGE_WIDTH

    if (status == TRUE)
        *dim_num = vsi_nn_max(dims, 2);

    if (dims > 2 && sizes2[2] != 1)
    {
        status = FALSE;
    }

    return status;
}

void vsi_nn_print_int_array( int32_t* array, size_t size )
{
    size_t i;
    size_t n;
#define _MSG_SIZE   (256)
    char buf[256];
    n = 0;
    for( i = 0; i < size; i ++ )
    {
        n += snprintf( &buf[n], _MSG_SIZE - n, "%d, ", array[i] );
        if( n >= _MSG_SIZE )
        {
            break;
        }
    }
    VSILOGD( "%s", buf );
#undef _MSG_SIZE
} /* vsi_nn_print_int_array() */

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
    vx_bool result = FALSE;

    if (src->attr.dtype.vx_type == dst->attr.dtype.vx_type)
    {
        switch (src->attr.dtype.qnt_type)
        {
        case VSI_NN_QNT_TYPE_NONE:
            result = TRUE;
            break;

        case VSI_NN_QNT_TYPE_DFP:
            if (src->attr.dtype.fl == dst->attr.dtype.fl)
            {
                result = TRUE;
            }
            break;

        case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
            if (src->attr.dtype.scale == dst->attr.dtype.scale &&
                src->attr.dtype.zero_point == dst->attr.dtype.zero_point)
            {
                result = TRUE;
            }
            break;

        case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC:
            {
                int32_t i = 0;
                int32_t scale_cnt0 = src->attr.dtype.scale_dim;
                int32_t scale_cnt1 = dst->attr.dtype.scale_dim;

                if (scale_cnt0 == scale_cnt1)
                {
                    const float *src_scale_ptr = src->attr.dtype.scales;
                    const float *dst_scale_ptr = dst->attr.dtype.scales;
                    for (i = 0; i < scale_cnt0; i++)
                    {
                        if (src_scale_ptr[i] != dst_scale_ptr[i])
                            break;
                    }

                    if (i == scale_cnt0)
                        result = TRUE;
                }
            }
            break;

        default:
            break;
        }
    }

    return result;
}

vsi_bool vsi_nn_is_same_type
    (
    vsi_nn_tensor_t * src,
    vsi_nn_tensor_t * dst
    )
{
    return (vsi_nn_is_same_data_type(src, dst) && vsi_nn_is_same_quant_type(src, dst));
}
