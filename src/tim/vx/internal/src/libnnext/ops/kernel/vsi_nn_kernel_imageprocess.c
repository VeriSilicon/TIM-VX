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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_IMAGEPROCESS)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_IMAGEPROCESS)
#define _VX_KERNEL_NAME         ("vsi_nn_kernel_imageprocess")
#define _VX_KERNEL_FUNC_KERNEL  (vximageprocessKernel)

//static uint32_t layerNum = 0;

static void resize_crop_op
    (
    uint8_t *buffer_ptr0,
    uint8_t *buffer_ptr1,
    vsi_nn_tensor_attr_t *attr0,
    vsi_nn_tensor_attr_t *attr1,
    uint32_t *stride_size0,
    uint32_t *stride_size1,
    int32_t *resize_crop_start
    )
{
    int32_t index[4];
    for (index[3] = 0; index[3] < (int32_t)attr1->size[3]; index[3]++)
    {
        for (index[2] = 0; index[2] < (int32_t)attr1->size[2]; index[2]++)
        {
            for (index[1] = 0; index[1] < (int32_t)attr1->size[1]; index[1]++)
            {
                for (index[0] = 0; index[0] < (int32_t)attr1->size[0]; index[0]++)
                {
                    int32_t index_in = (((index[3] + resize_crop_start[3]) * attr0->size[2]
                    + (index[2] + resize_crop_start[2])) * attr0->size[1]
                    + (index[1] + resize_crop_start[1])) * attr0->size[0]
                    + (index[0] + resize_crop_start[0]);
                    int32_t index_out = (((index[3]) * attr1->size[2]
                    + (index[2])) * attr1->size[1]
                    + (index[1])) * attr1->size[0]
                    + (index[0]);
                    float val;
                    vsi_nn_DtypeToFloat32(&buffer_ptr0[stride_size0[0] * index_in],
                        &val, &attr0->dtype);
                    vsi_nn_Float32ToDtype(val, &buffer_ptr1[stride_size1[0] * index_out],
                        &attr1->dtype);
                }
            }
        }
    }
}

static void reverse_channel_op
    (
    uint8_t *buffer_ptr0,
    uint8_t *buffer_ptr1,
    vsi_nn_tensor_attr_t *attr,
    uint32_t *stride_size
    )
{
    int32_t index[4];
    for (index[3] = 0; index[3] < (int32_t)attr->size[3]; index[3]++)
    {
        for (index[2] = 0; index[2] < 3; index[2]++)
        {
            for (index[1] = 0; index[1] < (int32_t)attr->size[1]; index[1]++)
            {
                for (index[0] = 0; index[0] < (int32_t)attr->size[0]; index[0]++)
                {
                    int32_t index_in = (((index[3]) * attr->size[2]
                    + (2 - index[2])) * attr->size[1]
                    + (index[1])) * attr->size[0]
                    + (index[0]);
                    int32_t index_out = (((index[3]) * attr->size[2]
                    + (index[2])) * attr->size[1]
                    + (index[1])) * attr->size[0]
                    + (index[0]);
                    float val;
                    vsi_nn_DtypeToFloat32(&buffer_ptr0[stride_size[0] * index_in],
                        &val, &attr->dtype);
                    vsi_nn_Float32ToDtype(val, &buffer_ptr1[stride_size[0] * index_out],
                        &attr->dtype);
                }
            }
        }
    }
}

static void mean_pixel_op
    (
    uint8_t *buffer_ptr0,
    uint8_t *buffer_ptr1,
    vsi_nn_tensor_attr_t *attr,
    uint32_t *stride_size,
    float mean_scale,
    float *mean_mean_value
    )
{
    int32_t index[4];
    for (index[3] = 0; index[3] < (int32_t)attr->size[3]; index[3]++)
    {
        for (index[2] = 0; index[2] < (int32_t)attr->size[2]; index[2]++)
        {
            for (index[1] = 0; index[1] < (int32_t)attr->size[1]; index[1]++)
            {
                for (index[0] = 0; index[0] < (int32_t)attr->size[0]; index[0]++)
                {
                    int32_t index_in = (((index[3]) * attr->size[2]
                    + (index[2])) * attr->size[1]
                    + (index[1])) * attr->size[0]
                    + (index[0]);
                    int32_t index_out = (((index[3]) * attr->size[2]
                    + (index[2])) * attr->size[1]
                    + (index[1])) * attr->size[0]
                    + (index[0]);
                    float val;
                    vsi_nn_DtypeToFloat32(&buffer_ptr0[stride_size[0] * index_in],
                        &val, &attr->dtype);
                    val = (val - mean_mean_value[0]) * mean_scale;
                    vsi_nn_Float32ToDtype(val, &buffer_ptr1[stride_size[0] * index_out],
                        &attr->dtype);
                }
            }
        }
    }
}

static void mean_channel_op
    (
    uint8_t *buffer_ptr0,
    uint8_t *buffer_ptr1,
    vsi_nn_tensor_attr_t *attr,
    uint32_t *stride_size,
    float mean_scale,
    float *mean_mean_value
    )
{
    int32_t index[4];
    for (index[3] = 0; index[3] < (int32_t)attr->size[3]; index[3]++)
    {
        for (index[2] = 0; index[2] < (int32_t)attr->size[2]; index[2]++)
        {
            for (index[1] = 0; index[1] < (int32_t)attr->size[1]; index[1]++)
            {
                for (index[0] = 0; index[0] < (int32_t)attr->size[0]; index[0]++)
                {
                    int32_t index_in = (((index[3]) * attr->size[2]
                    + (index[2])) * attr->size[1]
                    + (index[1])) * attr->size[0]
                    + (index[0]);
                    int32_t index_out = (((index[3]) * attr->size[2]
                    + (index[2])) * attr->size[1]
                    + (index[1])) * attr->size[0]
                    + (index[0]);
                    float val;
                    vsi_nn_DtypeToFloat32(&buffer_ptr0[stride_size[0] * index_in],
                        &val, &attr->dtype);
                    val = (val - mean_mean_value[index[2]]) * mean_scale;
                    vsi_nn_Float32ToDtype(val, &buffer_ptr1[stride_size[0] * index_out],
                        &attr->dtype);
                }
            }
        }
    }
}

static vsi_status VX_CALLBACK vximageprocessKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM          (14)
#define TENSOR_NUM_INPUT (1)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VX_SUCCESS;
    int32_t  i;
    vx_context context = NULL;
    vsi_nn_tensor_attr_t attr[TENSOR_NUM];
    uint32_t stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];
    vx_tensor_addressing user_addr[TENSOR_NUM] = {NULL};
    uint8_t *buffer_ptr[TENSOR_NUM] = {NULL};
    vx_tensor tensor[TENSOR_NUM];

    int32_t crop_enable, resize_crop_dim_num, resize_crop_start[4] = {0};
    int32_t mean_type, mean_mean_value_size;
    vx_bool reverse_channel;
    float mean_scale, mean_mean_value[4] = {0};
    uint8_t *temp_ptr[2] = {NULL};
    uint32_t buf_sz;

    //prepare data
    context = vxGetContext((vx_reference)node);

    for( i = 0; i < TENSOR_NUM_INPUT; i ++ )
    {
        tensor[i] = (vx_tensor)paramObj[i];
        buffer_ptr[i] = vsi_nn_ConvertRawTensorToData2(context, tensor[i],
            &(attr[i]), stride_size[i], &(user_addr[i]), VX_READ_ONLY);
    }
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        tensor[i] = (vx_tensor)paramObj[i];
        buffer_ptr[i] = vsi_nn_ConvertRawTensorToData2(context, tensor[i],
            &(attr[i]), stride_size[i], &(user_addr[i]), VX_WRITE_ONLY);
    }

    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(crop_enable),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(resize_crop_dim_num),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    for (i = 0; i < resize_crop_dim_num; i++)
    {
        vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 2 + i], &(resize_crop_start[i]),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    }

    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 6], &(reverse_channel),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 7], &(mean_type),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 8], &(mean_scale),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 9], &(mean_mean_value_size),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    for (i = 0; i < mean_mean_value_size; i++)
    {
        vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 10 + i], &(mean_mean_value[i]),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    }

    //op calc
    buf_sz = vsi_nn_GetTensorSize(attr[1].size, attr[1].dim_num, attr[1].dtype.vx_type);
    temp_ptr[0] = (uint8_t *)malloc( buf_sz );
    temp_ptr[1] = (uint8_t *)malloc( buf_sz );

    if (crop_enable == TRUE)
    {
        resize_crop_op(buffer_ptr[0], temp_ptr[0], &attr[0], &attr[1],
            stride_size[0], stride_size[1], resize_crop_start);
    }

    if (reverse_channel)
    {
        reverse_channel_op(temp_ptr[0], temp_ptr[1], &attr[1],
            stride_size[1]);
    }

    if (mean_type == VSI_NN_IMAGEPROCESS_MEAN_PIXEL)
    {
        mean_pixel_op(temp_ptr[1], buffer_ptr[1], &attr[1],
            stride_size[1], mean_scale, mean_mean_value);
    }
    else if (mean_type == VSI_NN_IMAGEPROCESS_MEAN_CHANNEL)
    {
        mean_channel_op(temp_ptr[1], buffer_ptr[1], &attr[1],
            stride_size[1], mean_scale, mean_mean_value);
    }

    //save data
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        status = vsi_nn_copy_tensor_patch(tensor[i], &attr[i], buffer_ptr[i], VX_WRITE_ONLY);
        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }
    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i]) free(buffer_ptr[i]);
    }

    if (temp_ptr[0]) free(temp_ptr[0]);
    if (temp_ptr[1]) free(temp_ptr[1]);
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

vx_status VX_CALLBACK vxScaletoTensorInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status status            = VX_SUCCESS;
    vx_image  bgrImg            = (vx_image)paramObj[0];
    vx_tensor output            = (vx_tensor)paramObj[1];
    vx_scalar xRatio_s          = (vx_scalar)paramObj[2];
    vx_scalar yRatio_s          = (vx_scalar)paramObj[3];
    vx_uint32 width             = 0;
    vx_uint32 height            = 0;
    vx_int32   xRatio           = 0;
    vx_int32   yRatio           = 0;
    vx_uint32  output_size[DIM_SIZE]   = {1, 1, 1, 1};
    vx_int8    dstFixedPointPos = 0;
    vx_enum    dstFormat;
    vx_float32 outputScale      = 1.0;
    vx_int32   output_ZP        = 0;
    uint32_t   output_dims      = 0;
    vsi_nn_tensor_attr_t attr;
    uint32_t i;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vxQueryImage(bgrImg, VX_IMAGE_WIDTH, &width, sizeof(width));
    vxQueryImage(bgrImg, VX_IMAGE_HEIGHT, &height, sizeof(height));

    vxCopyScalar(xRatio_s, (void*)&xRatio, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar(yRatio_s, (void*)&yRatio, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    status = vsi_nn_vxGetTensorAttr(output, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        return status;
    }

    output_dims  = attr.dim_num;
    dstFormat    = attr.dtype.vx_type;
    for (i = 0; i < output_dims; i++)
    {
        output_size[i] = attr.size[i];
    }
    dstFixedPointPos = attr.dtype.fl;
    output_ZP        = attr.dtype.zero_point;
    outputScale      = attr.dtype.scale;

    if (xRatio == (1 << 15) && yRatio == (1 << 15))
    {
        vx_uint32 uniExtractR_2x8[16] = {
            0x00099999, // TCfg
            0x00044444, // ASelt
            0x09060300, 0x0000000c, // ABin
            0x00099999, // BSelt
            0x06060606, 0x00000006, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniExtractG_2x8[16] = {
            0x00099999, // TCfg
            0x00044444, // ASelt
            0x2a272421, 0x0000002d, // ABin
            0x00099999, // BSelt
            0x06060606, 0x00000006, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniExtractB_2x8[16] = {
            0x00099999, // TCfg
            0x00044444, // ASelt
            0x4b484542, 0x0000004e, // ABin
            0x00099999, // BSelt
            0x06060606, 0x00000006, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        if (dstFormat == VSI_NN_TYPE_FLOAT16 || dstFormat == VSI_NN_TYPE_INT16)
            shaderParam.globalWorkScale[0]  = 8;
        else if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_UINT8)
            shaderParam.globalWorkScale[0]  = 10;

        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((output_size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (output_size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];

        if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_INT16)
        {
            if(dstFixedPointPos > 0)
                outputScale = (vx_float32) ((int64_t)1 << dstFixedPointPos);
            else
            {
                outputScale = 1.0f;
                uniExtractR_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
                uniExtractG_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
                uniExtractB_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
            }
        }
        else if (dstFormat == VSI_NN_TYPE_UINT8)
        {
            vx_float32 outputZP = (vx_float32)output_ZP;

            outputScale = 1.0f / outputScale;

            vxSetNodeUniform(nodObj, "outputZP", 1, &outputZP);
        }

        vxSetNodeUniform(nodObj, "uniExtractR_2x8", 1, uniExtractR_2x8);
        vxSetNodeUniform(nodObj, "uniExtractG_2x8", 1, uniExtractG_2x8);
        vxSetNodeUniform(nodObj, "uniExtractB_2x8", 1, uniExtractB_2x8);
        status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);
    }
    else
    {
        vx_uint32 uniVecShift10[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000400, 0x00000000, 0x00000400, 0x00000000,
            0x00000400, 0x00000000, 0x00000400, 0x00000000 // Constant
        };
        vx_uint32 uniAddRShift[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002405, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniGetTempVal[16] = {
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00230001, 0x00670045, // ABin
            0x05050505, // BSelt
            0x00110000, 0x00330022, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniExtractBytes[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002414, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniUnpackToR[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x09060300, 0x09060300, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniUnpackToG[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x0a070401, 0x0a070401, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniUnpackToB[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x0b080502, 0x0b080502, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniDataMulAlpha_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x01010101, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniDataSubMean_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertIntergetoF32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniExtactInteger_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002300, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkScale[0]  = 4;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((output_size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (output_size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];

        status |= vxSetNodeUniform(nodObj, "uniDataMulAlpha_4x4", 1, uniDataMulAlpha_4x4);
        status |= vxSetNodeUniform(nodObj, "uniDataSubMean_4x4", 1, uniDataSubMean_4x4);
        status |= vxSetNodeUniform(nodObj, "uniUnpackToR", 1, uniUnpackToR);
        status |= vxSetNodeUniform(nodObj, "uniUnpackToG", 1, uniUnpackToG);
        status |= vxSetNodeUniform(nodObj, "uniUnpackToB", 1, uniUnpackToB);
        status |= vxSetNodeUniform(nodObj, "uniVecShift10", 1, uniVecShift10);
        status |= vxSetNodeUniform(nodObj, "uniAddRShift", 1, uniAddRShift);
        status |= vxSetNodeUniform(nodObj, "uniGetTempVal", 1, uniGetTempVal);
        status |= vxSetNodeUniform(nodObj, "uniExtractBytes", 1, uniExtractBytes);

        if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_INT16)
        {
            if(dstFixedPointPos > 0)
                outputScale = (vx_float32) ((int64_t)1 << dstFixedPointPos);
            else
                outputScale = 1.0f / (vx_float32) ((int64_t)1 << -dstFixedPointPos);

            status |= vxSetNodeUniform(nodObj, "uniConvertIntergetoF32_4x4",
                1, uniConvertIntergetoF32_4x4);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);
            status |= vxSetNodeUniform(nodObj, "uniExtactInteger_2x8", 1, uniExtactInteger_2x8);
        }
        else if (dstFormat == VSI_NN_TYPE_UINT8)
        {
            vx_float32 outputZP = (vx_float32)output_ZP;

            outputScale = 1.0f / outputScale;

            status |= vxSetNodeUniform(nodObj, "uniConvertIntergetoF32_4x4",
                1, uniConvertIntergetoF32_4x4);
            status |= vxSetNodeUniform(nodObj, "outputZP", 1, &outputZP);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);
            status |= vxSetNodeUniform(nodObj, "uniExtactInteger_2x8", 1, uniExtactInteger_2x8);
        }
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}

vx_status VX_CALLBACK vxGrayScaletoTensorInitializer(vx_node nodObj, const vx_reference *paramObj, vx_uint32 paraNum)
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status status            = VX_SUCCESS;
    vx_image  inputImg          = (vx_image)paramObj[0];
    vx_scalar xRatio_s          = (vx_scalar)paramObj[2];
    vx_scalar yRatio_s          = (vx_scalar)paramObj[3];
    vx_tensor output            = (vx_tensor)paramObj[1];
    vx_uint32 width             = 0;
    vx_uint32 height            = 0;
    vx_int32   xRatio           = 0;
    vx_int32   yRatio           = 0;
    vx_uint32  output_size[4]   = {1, 1, 1, 1};
    vx_int8    dstFixedPointPos = 0;
    vx_enum    dstFormat;
    vx_float32 outputScale      = 1.0;
    vx_int32   output_ZP        = 0;
    uint32_t   output_dims      = 0;
    vsi_nn_tensor_attr_t attr;
    uint32_t i;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vxQueryImage(inputImg, VX_IMAGE_WIDTH, &width, sizeof(width));
    vxQueryImage(inputImg, VX_IMAGE_HEIGHT, &height, sizeof(height));

    vxCopyScalar(xRatio_s, (void*)&xRatio, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar(yRatio_s, (void*)&yRatio, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    status = vsi_nn_vxGetTensorAttr(output, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        return status;
    }

    output_dims  = attr.dim_num;
    dstFormat    = attr.dtype.vx_type;
    for (i = 0; i < output_dims; i++)
    {
        output_size[i] = attr.size[i];
    }
    dstFixedPointPos = attr.dtype.fl;
    output_ZP        = attr.dtype.zero_point;
    outputScale      = attr.dtype.scale;

    if (xRatio == (1 << 15) && yRatio == (1 << 15))
    {
        vx_uint32 uniDataMeanStddevLo_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };
        vx_uint32 uniDataMeanStddevHi_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };

        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        if (dstFormat == VSI_NN_TYPE_FLOAT16 || dstFormat == VSI_NN_TYPE_INT16)
            shaderParam.globalWorkScale[0]  = 16;
        else if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_UINT8)
            shaderParam.globalWorkScale[0]  = 16;

        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.localWorkSize[0]    = 8;
        shaderParam.localWorkSize[1]    = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((output_size[0] +
            shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
        shaderParam.globalWorkSize[1]   = gcmALIGN((output_size[1] +
            shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);

        if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_INT16)
        {
            if(dstFixedPointPos > 0)
                outputScale = (vx_float32) ((int64_t)1 << dstFixedPointPos);
            else
            {
                outputScale = 1.0f;
                uniDataMeanStddevLo_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
                uniDataMeanStddevHi_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
            }
        }
        else if (dstFormat == VSI_NN_TYPE_UINT8)
        {
            vx_float32 outputZP = (vx_float32)output_ZP;

            outputScale = 1.0f / outputScale;

            vxSetNodeUniform(nodObj, "outputZP", 1, &outputZP);
        }

        vxSetNodeUniform(nodObj, "uniDataMeanStddevLo_2x8", 1, uniDataMeanStddevLo_2x8);
        vxSetNodeUniform(nodObj, "uniDataMeanStddevHi_2x8", 1, uniDataMeanStddevHi_2x8);
        status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);
    }
    else
    {
        vx_uint32 uniVecShift10[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000400, 0x00000000, 0x00000400, 0x00000000,
            0x00000400, 0x00000000, 0x00000400, 0x00000000 // Constant
        };
        vx_uint32 uniAddRShift[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002405, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniGetTempVal[16] = {
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00230001, 0x00670045, // ABin
            0x05050505, // BSelt
            0x00110000, 0x00330022, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniExtractBytes[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002414, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniDataMulAlpha_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x01010101, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniDataSubMean_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertIntergetoF32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniExtactInteger_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002300, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkScale[0]  = 4;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.localWorkSize[0]    = 2;
        shaderParam.localWorkSize[1]    = 4;
        shaderParam.globalWorkSize[0]   = gcmALIGN((output_size[0] +
            shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
        shaderParam.globalWorkSize[1]   = gcmALIGN((output_size[1] +
            shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);

        if (dstFormat == VSI_NN_TYPE_FLOAT16)
        {
            status |= vxSetNodeUniform(nodObj, "uniDataMulAlpha_4x4", 1, uniDataMulAlpha_4x4);
            status |= vxSetNodeUniform(nodObj, "uniDataSubMean_4x4", 1, uniDataSubMean_4x4);
        }

        status |= vxSetNodeUniform(nodObj, "uniVecShift10", 1, uniVecShift10);
        status |= vxSetNodeUniform(nodObj, "uniAddRShift", 1, uniAddRShift);
        status |= vxSetNodeUniform(nodObj, "uniGetTempVal", 1, uniGetTempVal);
        status |= vxSetNodeUniform(nodObj, "uniExtractBytes", 1, uniExtractBytes);

        if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_INT16)
        {
            if(dstFixedPointPos > 0)
                outputScale *= (vx_float32) ((int64_t)1 << dstFixedPointPos);
            else
                outputScale *= 1.0f / (vx_float32) ((int64_t)1 << -dstFixedPointPos);

            status |= vxSetNodeUniform(nodObj, "uniConvertIntergetoF32_4x4",
                1, uniConvertIntergetoF32_4x4);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);
            status |= vxSetNodeUniform(nodObj, "uniExtactInteger_2x8", 1,
                uniExtactInteger_2x8);
        }
        else if (dstFormat == VSI_NN_TYPE_UINT8)
        {
            vx_float32 outputZP = (vx_float32)output_ZP;

            outputScale = 1.0f / outputScale;

            status |= vxSetNodeUniform(nodObj, "uniConvertIntergetoF32_4x4",
                1, uniConvertIntergetoF32_4x4);
            status |= vxSetNodeUniform(nodObj, "outputZP", 1, &outputZP);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);
            status |= vxSetNodeUniform(nodObj, "uniExtactInteger_2x8", 1,
                uniExtactInteger_2x8);
        }
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}

static vx_param_description_t s_params[] =
{
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
};

static vx_param_description_t vxScaletoTensorKernelParam[] =
{
    {VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

static vx_param_description_t vxGrayScaletoTensorKernelParam[] =
{
    {VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#ifdef __cplusplus
extern "C" {
#endif

vx_kernel_description_t _VX_KERNEL_VAR =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    s_params,
    _cnt_of_array( s_params ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxScaletoTensorKernelInfo_fp16 =
{
    VX_KERNEL_ENUM_SCALETOTENSOR,
    VX_KERNEL_NAME_SCALETOTENSOR_FP16,
    NULL,
    vxScaletoTensorKernelParam,
    (sizeof(vxScaletoTensorKernelParam) / sizeof(vxScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxScaletoTensorKernelInfo_int8 =
{
    VX_KERNEL_ENUM_SCALETOTENSOR,
    VX_KERNEL_NAME_SCALETOTENSOR_INT8,
    NULL,
    vxScaletoTensorKernelParam,
    (sizeof(vxScaletoTensorKernelParam) / sizeof(vxScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxScaletoTensorKernelInfo_fp16_copy =
{
    VX_KERNEL_ENUM_SCALETOTENSOR,
    VX_KERNEL_NAME_SCALETOTENSOR_FP16_COPY,
    NULL,
    vxScaletoTensorKernelParam,
    (sizeof(vxScaletoTensorKernelParam) / sizeof(vxScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxScaletoTensorKernelInfo_int8_copy =
{
    VX_KERNEL_ENUM_SCALETOTENSOR,
    VX_KERNEL_NAME_SCALETOTENSOR_INT8_COPY,
    NULL,
    vxScaletoTensorKernelParam,
    (sizeof(vxScaletoTensorKernelParam) / sizeof(vxScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxScaletoTensorKernelInfo_int16 =
{
    VX_KERNEL_ENUM_SCALETOTENSOR,
    VX_KERNEL_NAME_SCALETOTENSOR_INT16,
    NULL,
    vxScaletoTensorKernelParam,
    (sizeof(vxScaletoTensorKernelParam) / sizeof(vxScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxScaletoTensorKernelInfo_int16_copy =
{
    VX_KERNEL_ENUM_SCALETOTENSOR,
    VX_KERNEL_NAME_SCALETOTENSOR_INT16_COPY,
    NULL,
    vxScaletoTensorKernelParam,
    (sizeof(vxScaletoTensorKernelParam) / sizeof(vxScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxScaletoTensorKernelInfo_uint8 =
{
    VX_KERNEL_ENUM_SCALETOTENSOR,
    VX_KERNEL_NAME_SCALETOTENSOR_UINT8,
    NULL,
    vxScaletoTensorKernelParam,
    (sizeof(vxScaletoTensorKernelParam) / sizeof(vxScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxScaletoTensorKernelInfo_uint8_copy =
{
    VX_KERNEL_ENUM_SCALETOTENSOR,
    VX_KERNEL_NAME_SCALETOTENSOR_UINT8_COPY,
    NULL,
    vxScaletoTensorKernelParam,
    (sizeof(vxScaletoTensorKernelParam) / sizeof(vxScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGrayScaletoTensorKernelInfo_fp16 =
{
    VX_KERNEL_ENUM_GRAYSCALETOTENSOR,
    VX_KERNEL_NAME_GRAYSCALETOTENSOR_FP16,
    NULL,
    vxGrayScaletoTensorKernelParam,
    (sizeof(vxGrayScaletoTensorKernelParam) / sizeof(vxGrayScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGrayScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGrayScaletoTensorKernelInfo_int8 =
{
    VX_KERNEL_ENUM_GRAYSCALETOTENSOR,
    VX_KERNEL_NAME_GRAYSCALETOTENSOR_INT8,
    NULL,
    vxGrayScaletoTensorKernelParam,
    (sizeof(vxGrayScaletoTensorKernelParam) / sizeof(vxGrayScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGrayScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGrayScaletoTensorKernelInfo_fp16_copy =
{
    VX_KERNEL_ENUM_GRAYSCALETOTENSOR,
    VX_KERNEL_NAME_GRAYSCALETOTENSOR_FP16_COPY,
    NULL,
    vxGrayScaletoTensorKernelParam,
    (sizeof(vxGrayScaletoTensorKernelParam) / sizeof(vxGrayScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGrayScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGrayScaletoTensorKernelInfo_int8_copy =
{
    VX_KERNEL_ENUM_GRAYSCALETOTENSOR,
    VX_KERNEL_NAME_GRAYSCALETOTENSOR_INT8_COPY,
    NULL,
    vxGrayScaletoTensorKernelParam,
    (sizeof(vxGrayScaletoTensorKernelParam) / sizeof(vxGrayScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGrayScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGrayScaletoTensorKernelInfo_int16 =
{
    VX_KERNEL_ENUM_GRAYSCALETOTENSOR,
    VX_KERNEL_NAME_GRAYSCALETOTENSOR_INT16,
    NULL,
    vxGrayScaletoTensorKernelParam,
    (sizeof(vxGrayScaletoTensorKernelParam) / sizeof(vxGrayScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGrayScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGrayScaletoTensorKernelInfo_int16_copy =
{
    VX_KERNEL_ENUM_GRAYSCALETOTENSOR,
    VX_KERNEL_NAME_GRAYSCALETOTENSOR_INT16_COPY,
    NULL,
    vxGrayScaletoTensorKernelParam,
    (sizeof(vxGrayScaletoTensorKernelParam) / sizeof(vxGrayScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGrayScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGrayScaletoTensorKernelInfo_uint8 =
{
    VX_KERNEL_ENUM_GRAYSCALETOTENSOR,
    VX_KERNEL_NAME_GRAYSCALETOTENSOR_UINT8,
    NULL,
    vxGrayScaletoTensorKernelParam,
    (sizeof(vxGrayScaletoTensorKernelParam) / sizeof(vxGrayScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGrayScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGrayScaletoTensorKernelInfo_uint8_copy =
{
    VX_KERNEL_ENUM_GRAYSCALETOTENSOR,
    VX_KERNEL_NAME_GRAYSCALETOTENSOR_UINT8_COPY,
    NULL,
    vxGrayScaletoTensorKernelParam,
    (sizeof(vxGrayScaletoTensorKernelParam) / sizeof(vxGrayScaletoTensorKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGrayScaletoTensorInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_IMAGEPROCESS_list[] =
{
    &_VX_KERNEL_VAR,
    &vxScaletoTensorKernelInfo_fp16,
    &vxScaletoTensorKernelInfo_int8,
    &vxScaletoTensorKernelInfo_int16,
    &vxScaletoTensorKernelInfo_uint8,
    &vxScaletoTensorKernelInfo_fp16_copy,
    &vxScaletoTensorKernelInfo_int8_copy,
    &vxScaletoTensorKernelInfo_int16_copy,
    &vxScaletoTensorKernelInfo_uint8_copy,
    &vxGrayScaletoTensorKernelInfo_fp16,
    &vxGrayScaletoTensorKernelInfo_int8,
    &vxGrayScaletoTensorKernelInfo_int16,
    &vxGrayScaletoTensorKernelInfo_uint8,
    &vxGrayScaletoTensorKernelInfo_fp16_copy,
    &vxGrayScaletoTensorKernelInfo_int8_copy,
    &vxGrayScaletoTensorKernelInfo_int16_copy,
    &vxGrayScaletoTensorKernelInfo_uint8_copy,
    NULL
};
#ifdef __cplusplus
}
#endif
