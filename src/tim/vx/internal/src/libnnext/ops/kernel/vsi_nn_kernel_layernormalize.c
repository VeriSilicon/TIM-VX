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
#include <math.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

void myLayerNormFunc
    (
    void* src,
    int16_t* scale,
    float*   bias,
    float    eps,
    void* dst,
    uint32_t input_dim,
    uint32_t width,
    uint32_t height,
    uint32_t channel,
    uint32_t batch
    )
{
    uint32_t ch = (input_dim <= 2) ? 1 : channel;
    uint32_t bn = (input_dim <= 3) ? 1 : batch;
    uint32_t b = 0, c = 0, h = 0, w = 0;

    int16_t* imgIn, *imgOut;
    imgIn = (int16_t*)src;
    imgOut = (int16_t*)dst;

    VSILOGI("Hello myLayerNormFunc!\n");
    for (b = 0; b < bn; b++)
    {
        for (c = 0; c < ch; c++)
        {
            for (h = 0; h < height; h++)
            {
                uint32_t len = (h + (c + b*ch)*height) * width;
                float sum = .0f;
                float sumsq = .0f;
                float mean = .0f;
                float vari = .0f;

                for (w = 0; w < width; w++)
                {
                    uint32_t index = len + w;
                    sum += vsi_nn_Fp16toFp32(imgIn[index]);
                }
                mean = sum / width;
                for (w = 0; w < width; w++)
                {
                    uint32_t index = len + w;
                    float data = vsi_nn_Fp16toFp32(imgIn[index]) - mean;
                    sumsq += data * data;
                }
                vari = sumsq / width;
                vari = (float)(1.0 / sqrtf(vari + eps));
                for (w = 0; w < width; w++)
                {
                    uint32_t index = len + w;
                    float data = vsi_nn_Fp16toFp32(imgIn[index]) - mean;
                    float scaleVal = vsi_nn_Fp16toFp32(scale[w]);
                    float biasVal = bias[w];
                    float normVal = data * vari * scaleVal + biasVal;
                    imgOut[index] = vsi_nn_Fp32ToFp16(normVal);
                }
            }
        }
    }
    return;
}
void myLayerNormFunc_u8
    (
    void* src,
    int16_t* scale,
    float*   bias,
    float    eps,
    void* dst,
    uint32_t input_dim,
    uint32_t width,
    uint32_t height,
    uint32_t channel,
    uint32_t batch,
    int32_t inZp,
    int32_t outZp,
    float inScale,
    float outScale
    )
{
    uint32_t ch = (input_dim <= 2) ? 1 : channel;
    uint32_t bn = (input_dim <= 3) ? 1 : batch;
    uint32_t b = 0, c = 0, h = 0, w = 0;

    uint8_t* imgIn, *imgOut;
    imgIn = (uint8_t*)src;
    imgOut = (uint8_t*)dst;

    VSILOGI("Hello myLayerNormFunc!\n");
    for (b = 0; b < bn; b++)
    {
        for (c = 0; c < ch; c++)
        {
            for (h = 0; h < height; h++)
            {
                uint32_t len = (h + (c + b*ch)*height) * width;
                float sum = .0f;
                float sumsq = .0f;
                float mean = .0f;
                float vari = .0f;

                for (w = 0; w < width; w++)
                {
                    uint32_t index = len + w;
                    //sum += vsi_nn_Fp16toFp32(imgIn[index]);
                    sum += vsi_nn_AffineToFp32(imgIn[index], inScale, inZp, VSI_NN_TYPE_UINT8);
                }
                mean = sum / width;
                for (w = 0; w < width; w++)
                {
                    uint32_t index = len + w;
                    //float data = vsi_nn_Fp16toFp32(imgIn[index]) - mean;
                    float data = vsi_nn_AffineToFp32(imgIn[index], inScale, inZp, VSI_NN_TYPE_UINT8) - mean;
                    sumsq += data * data;
                }
                vari = sumsq / width;
                vari = (float)(1.0 / sqrtf(vari + eps));
                for (w = 0; w < width; w++)
                {
                    uint32_t index = len + w;
                    //float data = vsi_nn_Fp16toFp32(imgIn[index]) - mean;
                    float data = vsi_nn_AffineToFp32(imgIn[index], inScale, inZp, VSI_NN_TYPE_UINT8) - mean;
                    float scaleVal = vsi_nn_Fp16toFp32(scale[w]);
                    float biasVal = bias[w];
                    float normVal = data * vari * scaleVal + biasVal;
                    //imgOut[index] = vsi_nn_Fp32ToFp16(normVal);
                    imgOut[index] = (vx_uint8)vsi_nn_Fp32ToAffine(normVal, outScale, outZp, VSI_NN_TYPE_UINT8);
                }
            }
        }
    }
    return;
}
vsi_status VX_CALLBACK vxLayerNormKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 5)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[4] = { NULL };
        vsi_nn_tensor_attr_t attr[4];
        int16_t *input = NULL, *output = NULL, *scale = NULL;
        float *bias = NULL;
        uint32_t input_size[4] = {1, 1, 1, 1}, output_size[4] = {1, 1, 1, 1};
        uint32_t scale_size[4] = {1, 1, 1, 1}, bias_size[4] = {1, 1, 1, 1};
        uint32_t input_stride_size[4]  = {0};
        uint32_t output_stride_size[4] = {0};
        uint32_t scale_stride_size[4]  = {0};
        uint32_t bias_stride_size[4] = {0};
        vx_tensor_addressing input_user_addr = NULL;
        vx_tensor_addressing output_user_addr = NULL;
        vx_tensor_addressing scale_user_addr = NULL;
        vx_tensor_addressing bias_user_addr = NULL;
        vsi_nn_type_e inputFormat = VSI_NN_TYPE_FLOAT16, outputFormat = VSI_NN_TYPE_FLOAT16;
        vsi_nn_type_e scaleFormat = VSI_NN_TYPE_FLOAT16, biasFormat = VSI_NN_TYPE_FLOAT16;
        uint32_t input_dims = 0, output_dims = 0;
        uint32_t scale_dims = 0, bias_dims = 0;
        uint32_t i;
        int32_t in_zp, out_zp;
        float in_scale, out_scale;
        // scalar
        vx_scalar scalar[1] = { NULL };
        float eps = .0f;

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];
        imgObj[2] = (vx_tensor)paramObj[2];
        imgObj[3] = (vx_tensor)paramObj[3];
        scalar[0] = (vx_scalar)paramObj[4];
        memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
        memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
        memset(&attr[2], 0, sizeof(vsi_nn_tensor_attr_t));
        memset(&attr[3], 0, sizeof(vsi_nn_tensor_attr_t));
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            goto OnError;
        }

        status  = vsi_nn_vxGetTensorAttr(imgObj[0], &attr[0]);
        status |= vsi_nn_vxGetTensorAttr(imgObj[1], &attr[1]);
        status |= vsi_nn_vxGetTensorAttr(imgObj[2], &attr[2]);
        status |= vsi_nn_vxGetTensorAttr(imgObj[3], &attr[3]);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
            goto OnError;
        }
        input_dims  = attr[0].dim_num;
        inputFormat = attr[0].dtype.vx_type;
        for (i = 0; i < input_dims; i++)
        {
            input_size[i] = attr[0].size[i];
        }
        in_zp = attr[0].dtype.zero_point;
        in_scale = attr[0].dtype.scale;

        //bias
        bias_dims  = attr[1].dim_num;
        biasFormat = attr[1].dtype.vx_type;
        for (i = 0; i < bias_dims; i++)
        {
            bias_size[i] = attr[1].size[i];
        }
        //scale
        scale_dims  = attr[2].dim_num;
        scaleFormat = attr[2].dtype.vx_type;
        for (i = 0; i < scale_dims; i++)
        {
            scale_size[i] = attr[2].size[i];
        }

        //output
        output_dims  = attr[3].dim_num;
        outputFormat = attr[3].dtype.vx_type;
        for (i = 0; i < output_dims; i++)
        {
            output_size[i] = attr[3].size[i];
        }
        out_zp = attr[3].dtype.zero_point;
        out_scale = attr[3].dtype.scale;

        input_size[2] = (input_dims <= 2)?1:input_size[2];
        input_size[3] = (input_dims <= 3)?1:input_size[3];

        input_stride_size[0]  = vsi_nn_GetTypeBytes(inputFormat);
        output_stride_size[0] = vsi_nn_GetTypeBytes(outputFormat);
        for (i=1; i< input_dims; i++)
        {
            input_stride_size[i]  = input_stride_size[i-1] * input_size[i-1];
            output_stride_size[i] = output_stride_size[i-1] * output_size[i-1];
        }
        input  = (int16_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
        output = (int16_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(int16_t));
        input_user_addr = vxCreateTensorAddressing(context, input_size, input_stride_size, (vx_uint8)input_dims);
        vsi_nn_copy_tensor_patch(imgObj[0], &attr[0], input, VX_READ_ONLY);
        //scale and bias
        scale_stride_size[0]  = vsi_nn_GetTypeBytes(scaleFormat);
        bias_stride_size[0] = vsi_nn_GetTypeBytes(biasFormat);
        for (i=1; i< scale_dims; i++)
        {
            scale_stride_size[i]  = scale_stride_size[i-1] * scale_size[i-1];
            bias_stride_size[i] = bias_stride_size[i-1] * bias_size[i-1];
        }
        scale  = (int16_t*)malloc(scale_size[0]*sizeof(int16_t));
        bias = (float*)malloc(bias_size[0]*sizeof(float));
        bias_user_addr = vxCreateTensorAddressing(context, bias_size, bias_stride_size, (vx_uint8)bias_dims);
        vsi_nn_copy_tensor_patch(imgObj[1], &attr[1], bias, VX_READ_ONLY);
        scale_user_addr = vxCreateTensorAddressing(context, scale_size, scale_stride_size, (vx_uint8)scale_dims);
        vsi_nn_copy_tensor_patch(imgObj[2], &attr[2], scale, VX_READ_ONLY);

        // scalar
        status = vxCopyScalar(scalar[0], &eps, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxCopyScalar failure! at line %d\n", __LINE__);
            goto OnError;
        }
        // Call C Prototype
        if(inputFormat == VSI_NN_TYPE_FLOAT16)
        {
            myLayerNormFunc(input, scale, bias, eps, output, input_dims, input_size[0],
                input_size[1], input_size[2], input_size[3]);
        }
        else
        {
            myLayerNormFunc_u8(input, scale, bias, eps, output, input_dims, input_size[0],
                input_size[1], input_size[2], input_size[3], in_zp, out_zp, in_scale, out_scale);
        }

        //output tensor
        output_user_addr = vxCreateTensorAddressing(context, output_size,
            output_stride_size, (vx_uint8)output_dims);
        vsi_nn_copy_tensor_patch(imgObj[3], &attr[3], output, VX_WRITE_ONLY);

OnError:
        if(input) free(input);
        if(scale) free(scale);
        if(bias) free(bias);
        if(output) free(output);
        if(input_user_addr) vxReleaseTensorAddressing(&input_user_addr);
        if(scale_user_addr) vxReleaseTensorAddressing(&scale_user_addr);
        if(bias_user_addr) vxReleaseTensorAddressing(&bias_user_addr);
        if(output_user_addr) vxReleaseTensorAddressing(&output_user_addr);
    }

    return status;
}
vsi_status VX_CALLBACK vxLayerNormInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status = VX_SUCCESS;
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     input           = (vx_tensor)paramObj[0];
    vx_tensor     scale           = (vx_tensor)paramObj[2];
    vx_tensor     output          = (vx_tensor)paramObj[3];
    uint32_t      input_size[4]   = {1, 1, 1, 1};
    uint32_t      input_dims      = 0;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e scaleDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e outputDataFormat = VSI_NN_TYPE_FLOAT16;
    vx_float32 scaleIn = 0;
    vx_float32 scaleOut = 0;
    vx_float32 reScaleOut_u8 = 0;
    vx_float32 reOutZP = 0.f;
    int32_t output_ZP = 0;
    int32_t input_ZP = 0;
    vx_uint32 iter = 0;
    int32_t sumInZp = 0;
    int32_t tmpZp1 = 0;
    int32_t tmpZp2 = 0;
    vx_float32 e2InScale = 0;
    vsi_nn_tensor_attr_t attr[3];
    uint32_t i;

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[2], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    status |= vsi_nn_vxGetTensorAttr(scale, &attr[2]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        return status;
    }

    input_dims  = attr[0].dim_num;
    inputDataFormat = attr[0].dtype.vx_type;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr[0].size[i];
    }
    input_ZP = attr[0].dtype.zero_point;
    scaleIn = attr[0].dtype.scale;
    outputDataFormat = attr[1].dtype.vx_type;
    output_ZP = attr[1].dtype.zero_point;
    scaleOut = attr[1].dtype.scale;
    scaleDataFormat = attr[2].dtype.vx_type;

    if(outputDataFormat == VSI_NN_TYPE_UINT8)
    {
        reScaleOut_u8 = 1.0f / scaleOut;
        reOutZP = (vx_float32)output_ZP;
    }
    iter = ((input_size[0] + 15) / 16) * 16;
    sumInZp = input_ZP * iter * (-1);
    tmpZp1 = (-2) * input_ZP;
    tmpZp2 = iter * input_ZP * input_ZP;
    e2InScale = scaleIn * scaleIn;

    input_size[2] = (input_dims <= 2)?1:input_size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = input_size[0];
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = 1;
    shaderParam.globalWorkSize[1]   = gcmALIGN((input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1], 4);
    shaderParam.globalWorkSize[2]   = input_size[2];

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    {
        vx_float32 dimRatio = 1.0f / (vx_float32)input_size[0];
        vx_uint32 uniFp16SumSqr_dp8x2[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 UniFP16toFP32Lo4_dp4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniExtractHalf4_dp4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertSecFp16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniSumU8_16x1[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniSqrSum_16x1[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0x55555555, // BSelt
            0x76543210, 0xfedcba98, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniConvert1stUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        };
        vx_uint32 uniConvert2ndUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        };
        vx_uint32 uniConvert3rdUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00090008, 0x000b000a, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        };
        vx_uint32 uniConvert4thUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x000d000c, 0x000f000e, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        };
        vx_uint32 uniConvertInt32toUint8_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 UniPackFP16even_2x8[16] = {
           0x11111111, // TCfg
           0x11110000, // ASelt
           0x06040200, 0x06040200, // ABin
           0x22222222, // BSelt
           0x00000000, 0x00000000, // BBin
           0x00000100, // AccumType, ConstantType, and PostShift
           0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        };
        if (inputDataFormat == VSI_NN_TYPE_UINT8 && outputDataFormat == VSI_NN_TYPE_FLOAT16
            && scaleDataFormat == VSI_NN_TYPE_FLOAT16)
        {
            status  = vxSetNodeUniform(nodObj, "width", 1, &input_size[0]);
            status |= vxSetNodeUniform(nodObj, "dimRatio", 1, &dimRatio);
            status |= vxSetNodeUniform(nodObj, "UniFP16toFP32Lo4_dp4x4", 1, UniFP16toFP32Lo4_dp4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertSecFp16Fp32_4x4", 1, uniConvertSecFp16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniSumU8_16x1", 1, uniSumU8_16x1);
            status |= vxSetNodeUniform(nodObj, "uniSqrSum_16x1", 1, uniSqrSum_16x1);
            status |= vxSetNodeUniform(nodObj, "uniConvert1stUint8SubZpToFp32_4x4", 1, uniConvert1stUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert2ndUint8SubZpToFp32_4x4", 1, uniConvert2ndUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert3rdUint8SubZpToFp32_4x4", 1, uniConvert3rdUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert4thUint8SubZpToFp32_4x4", 1, uniConvert4thUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "inputZP", 1, &input_ZP);
            status |= vxSetNodeUniform(nodObj, "input_scale", 1, &scaleIn);
            status |= vxSetNodeUniform(nodObj, "sumInZp", 1, &sumInZp);
            status |= vxSetNodeUniform(nodObj, "tmpZp1", 1, &tmpZp1);
            status |= vxSetNodeUniform(nodObj, "tmpZp2", 1, &tmpZp2);
            status |= vxSetNodeUniform(nodObj, "e2InScale", 1, &e2InScale);
            status |= vxSetNodeUniform(nodObj, "UniPackFP16even_2x8", 1, UniPackFP16even_2x8);
        }
        else
        {
            status  = vxSetNodeUniform(nodObj, "uniFp16SumSqr_dp8x2", 1, uniFp16SumSqr_dp8x2);
            status |= vxSetNodeUniform(nodObj, "width", 1, &input_size[0]);
            status |= vxSetNodeUniform(nodObj, "dimRatio", 1, &dimRatio);
            status |= vxSetNodeUniform(nodObj, "UniFP16toFP32Lo4_dp4x4", 1, UniFP16toFP32Lo4_dp4x4);
            status |= vxSetNodeUniform(nodObj, "uniExtractHalf4_dp4x4", 1, uniExtractHalf4_dp4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvertSecFp16Fp32_4x4", 1, uniConvertSecFp16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniSumU8_16x1", 1, uniSumU8_16x1);
            status |= vxSetNodeUniform(nodObj, "uniSqrSum_16x1", 1, uniSqrSum_16x1);
            status |= vxSetNodeUniform(nodObj, "uniConvert1stUint8SubZpToFp32_4x4", 1, uniConvert1stUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert2ndUint8SubZpToFp32_4x4", 1, uniConvert2ndUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert3rdUint8SubZpToFp32_4x4", 1, uniConvert3rdUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert4thUint8SubZpToFp32_4x4", 1, uniConvert4thUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "inputZP", 1, &input_ZP);
            status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
            status |= vxSetNodeUniform(nodObj, "input_scale", 1, &scaleIn);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &reScaleOut_u8);
            status |= vxSetNodeUniform(nodObj, "outputZP", 1, &reOutZP);
            status |= vxSetNodeUniform(nodObj, "sumInZp", 1, &sumInZp);
            status |= vxSetNodeUniform(nodObj, "tmpZp1", 1, &tmpZp1);
            status |= vxSetNodeUniform(nodObj, "tmpZp2", 1, &tmpZp2);
            status |= vxSetNodeUniform(nodObj, "e2InScale", 1, &e2InScale);
        }
        if(status < 0)
        {
            VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        }
    }
    return status;
}
static vx_param_description_t vxLayerNormKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxLayerNormKernelInfo =
{
    VX_KERNEL_ENUM_LAYERNORM,
    VX_KERNEL_NAME_LAYERNORM,
    NULL,
    vxLayerNormKernelParam,
    (sizeof(vxLayerNormKernelParam) / sizeof(vxLayerNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxLayerNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxLayerNormKernelInfo_u8 =
{
    VX_KERNEL_ENUM_LAYERNORM,
    VX_KERNEL_NAME_LAYERNORM_UINT8,
    NULL,
    vxLayerNormKernelParam,
    (sizeof(vxLayerNormKernelParam) / sizeof(vxLayerNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxLayerNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxLayerNormKernelInfo_FP16toU8 =
{
    VX_KERNEL_ENUM_LAYERNORM_FP16TOU8,
    VX_KERNEL_NAME_LAYERNORM_FP16TOU8,
    NULL,
    vxLayerNormKernelParam,
    (sizeof(vxLayerNormKernelParam) / sizeof(vxLayerNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxLayerNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxLayerNormKernelInfo_U8toFP16 =
{
    VX_KERNEL_ENUM_LAYERNORM,
    VX_KERNEL_NAME_LAYERNORM_U8TOFP16,
    NULL,
    vxLayerNormKernelParam,
    (sizeof(vxLayerNormKernelParam) / sizeof(vxLayerNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxLayerNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxLayerNormKernelInfo_CPU =
{
    VX_KERNEL_ENUM_LAYERNORM,
    VX_KERNEL_NAME_LAYERNORM,
    vxLayerNormKernel,
    vxLayerNormKernelParam,
    (sizeof(vxLayerNormKernelParam) / sizeof(vxLayerNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_LAYERNORM_list[] =
{
    &vxLayerNormKernelInfo_CPU,
    &vxLayerNormKernelInfo,
    &vxLayerNormKernelInfo_u8,
    &vxLayerNormKernelInfo_FP16toU8,
    &vxLayerNormKernelInfo_U8toFP16,
    NULL
};
#ifdef __cplusplus
}
#endif
