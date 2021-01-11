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
#include "vsi_nn_test.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_SCALE)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_SCALE)
#define _VX_KERNEL_NAME         ("vsi_nn_kernel_scale")
#define _VX_KERNEL_FUNC_KERNEL  (vxScaleKernel)

static vsi_status VX_CALLBACK vxScaleKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if( 6 == paramNum )
    {
        vx_context  context = NULL;
        vx_tensor   input_tensor = NULL;
        vx_tensor   scale_tensor = NULL;
        vx_tensor   bias_tensor = NULL;
        vx_tensor   output_tensor = NULL;
        uint8_t *  input_buffer = NULL;
        uint8_t *  scale_buffer = NULL;
        uint8_t *  bias_buffer = NULL;
        uint8_t *  output_buffer = NULL;
        vx_scalar   axis_scalar = NULL;
        vx_scalar   has_bias_scalar = NULL;
        int         axis = 1;
        float  has_bias = 0;
        uint32_t   input_dims = 0;
        uint32_t   scale_dims = 0;
        uint32_t   bias_dims = 0;
        uint32_t   output_dims = 0;
        vsi_enum     inputFormat = VSI_NN_TYPE_FLOAT16;
        vsi_enum     scaleFormat = VSI_NN_TYPE_FLOAT16;
        vsi_enum     biasFormat = VSI_NN_TYPE_FLOAT32;
        vsi_enum     outputFormat = VSI_NN_TYPE_FLOAT16;
        uint32_t   input_size[4] = {1, 1, 1, 1};
        uint32_t   scale_size[4] = {1, 1, 1, 1};
        uint32_t   bias_size[4] = {1, 1, 1, 1};
        uint32_t   output_size[4] = {1, 1, 1, 1};
        uint32_t   input_stride_size[VSI_NN_MAX_DIM_NUM] = { 0 };
        uint32_t   output_stride_size[VSI_NN_MAX_DIM_NUM] = { 0 };
        vx_tensor_addressing input_user_addr = NULL;
        vx_tensor_addressing scale_user_addr = NULL;
        vx_tensor_addressing bias_user_addr = NULL;
        vx_tensor_addressing output_user_addr = NULL;
        vsi_nn_tensor_attr_t out_attr;

        status = VX_SUCCESS;

        memset(&out_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));

        input_tensor = (vx_tensor)paramObj[0];
        scale_tensor = (vx_tensor)paramObj[1];
        bias_tensor = (vx_tensor)paramObj[2];
        output_tensor = (vx_tensor)paramObj[3];
        axis_scalar = (vx_scalar)paramObj[4];
        has_bias_scalar = (vx_scalar)paramObj[5];

        context = vxGetContext((vx_reference)node);
        if( NULL == context)
        {
            VSILOGE("vxGetContext failure!\n");
            status = VX_FAILURE;
            goto OnError;
        }

        input_buffer = vsi_nn_ConvertRawTensorToData(context, input_tensor,
            &input_dims, &inputFormat, input_size, input_stride_size,
            &input_user_addr, VX_READ_ONLY);
        if( NULL == input_buffer )
        {
            VSILOGE("vsi_nn_ConvertRawTensorToData failure!\n");
            status = VX_ERROR_NO_MEMORY;
            goto OnError;
        }

        scale_buffer = vsi_nn_ConvertRawTensorToData(context, scale_tensor,
            &scale_dims, &scaleFormat, scale_size, input_stride_size,
            &scale_user_addr, VX_READ_ONLY);
        if( NULL == scale_buffer )
        {
            VSILOGE("vsi_nn_ConvertRawTensorToData failure!\n");
            status = VX_ERROR_NO_MEMORY;
            goto OnError;
        }

        bias_buffer = vsi_nn_ConvertRawTensorToData(context, bias_tensor,
            &bias_dims, &biasFormat, bias_size, input_stride_size,
            &bias_user_addr, VX_READ_ONLY);
        if( NULL == bias_buffer )
        {
            VSILOGE("vsi_nn_ConvertRawTensorToData failure!\n");
            status = VX_ERROR_NO_MEMORY;
            goto OnError;
        }

        output_buffer = vsi_nn_ConvertRawTensorToData(context, output_tensor,
            &output_dims, &outputFormat, output_size, output_stride_size,
            &output_user_addr, VX_WRITE_ONLY);
        if( NULL == output_buffer )
        {
            VSILOGE("vsi_nn_ConvertRawTensorToData failure!\n");
            status = VX_ERROR_NO_MEMORY;
            goto OnError;
        }

        status = vsi_nn_vxGetTensorAttr(output_tensor, &out_attr);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
            goto OnError;
        }

        status = vxCopyScalar(axis_scalar, &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if( VX_SUCCESS != status)
        {
            VSILOGE("vxCopyScalar axis failure! status:%d\n", status);
            goto OnError;
        }
        status = vxCopyScalar(has_bias_scalar, &has_bias, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if( VX_SUCCESS != status )
        {
            VSILOGE("vxCopyScalar axis failure! has_bias:%f\n", has_bias);
            goto OnError;
        }

        if( input_dims != output_dims )
        {
            VSILOGE("Invalid parameters, input_dims output_dims mismatch %d:%d\n",
                input_dims, output_dims);
            status = VX_ERROR_INVALID_PARAMETERS;
            goto OnError;
        }
        if( input_size[0] != scale_size[0] || input_size[0] != bias_size[0] )
        {
            VSILOGE("Invalid parameters, input size mismatch %d:%d:%d\n",
                input_size[0], scale_size[0], bias_size[0]);
            status = VX_ERROR_INVALID_PARAMETERS;
            goto OnError;
        }
        {
            uint32_t i = 0;
            uint32_t j = 0;
            uint32_t fixed_num = 1;
            uint32_t changed_num = 1;

            fixed_num = input_size[1] * input_size[2] * input_size[3];
            changed_num = input_size[0];

            for( i = 0; i < fixed_num; i++ )
            {
                int16_t* cur_input_row_ofst = ((int16_t *)input_buffer) + i * changed_num;
                int16_t* cur_scale_row_ofst = ((int16_t *)scale_buffer);
                float* cur_bias_row_ofst = ((float *)bias_buffer);
                int16_t* cur_output_row_ofst = ((int16_t *)output_buffer) + i * changed_num;

                for( j = 0; j < changed_num; j++ )
                {
                    float cur_input_v = vsi_nn_Fp16ToFp32(*(cur_input_row_ofst + j));
                    float cur_scale_v = vsi_nn_Fp16ToFp32(*(cur_scale_row_ofst + j));
                    float cur_bias_v = *(cur_bias_row_ofst + j);

                    float cur_result = cur_input_v * cur_scale_v + cur_bias_v;
                    *(cur_output_row_ofst + j) = vsi_nn_Fp32ToFp16(cur_result);
                }
            }

#if defined(_SAVE_TENSOR)
            {
                static int count = 0;
                char fname[256] = { 0 };
                sprintf(fname, "scale_output_tensor.%d.axis.%d.txt", count, axis);
                vsi_nn_SaveDataToText(fname, output_buffer,
                    vsi_nn_ShapeProduct(output_size, output_dims), VSI_NN_TYPE_FLOAT16, NULL);
                count++;
            }
#endif
        }
        status = vsi_nn_vxCopyDataToTensor(context, output_tensor, &out_attr, output_buffer);
        TEST_CHECK_STATUS(status, OnError);
OnError:
        if( NULL != input_buffer )
        {
            free( input_buffer );
            input_buffer = NULL;
        }
        if( NULL != scale_buffer )
        {
            free( scale_buffer );
            scale_buffer = NULL;
        }
        if( NULL != bias_buffer )
        {
            free( bias_buffer );
            bias_buffer = NULL;
        }
        if( NULL != output_buffer )
        {
            free( output_buffer );
            output_buffer = NULL;
        }

        if (input_user_addr)
        {
            vxReleaseTensorAddressing(&input_user_addr);
        }
        if (scale_user_addr)
        {
            vxReleaseTensorAddressing(&scale_user_addr);
        }
        if (bias_user_addr)
        {
            vxReleaseTensorAddressing(&bias_user_addr);
        }
        if (output_user_addr)
        {
            vxReleaseTensorAddressing(&output_user_addr);
        }

    }

    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t s_params[] =
{
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
};

vsi_status VX_CALLBACK vxScaleInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
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
    uint32_t uniExtractHalf8_2x8[16] = {
        0x11111111, // TCfg
        0x11110000, // ASelt
        0x06040200, 0x06040200, // ABin
        0x22222222, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00002100, // AccumType, ConstantType, and PostShift
        0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
        0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
    };
    uint32_t uniFp16MulFp16ToFp32_Lo_4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00010000, 0x00030002, // ABin
        0x01010101, // BSelt
        0x00010000, 0x00030002, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    uint32_t uniFp16MulFp16ToFp32_Hi_4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00050004, 0x00070006, // ABin
        0x01010101, // BSelt
        0x00050004, 0x00070006, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };

    vsi_status status = VX_SUCCESS;

    vx_tensor input     = (vx_tensor)paramObj[0];
    uint32_t input_size[DIM_SIZE] = {1, 1, 1, 1};
    vx_uint32 i = 0;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    for (i = 0; i < attr.dim_num; i++)
    {
        input_size[i] = attr.size[i];
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];

    vxSetNodeUniform(nodObj, "uniExtractHalf8_2x8", 1, uniExtractHalf8_2x8);
    vxSetNodeUniform(nodObj, "uniFp16MulFp16ToFp32_Lo_4x4", 1, uniFp16MulFp16ToFp32_Lo_4x4);
    vxSetNodeUniform(nodObj, "uniFp16MulFp16ToFp32_Hi_4x4", 1, uniFp16MulFp16ToFp32_Hi_4x4);

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}

static vx_param_description_t vxScaleKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
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

vx_kernel_description_t vxScaleKernelInfo =
{
    VX_KERNEL_ENUM_SCALE,
    VX_KERNEL_NAME_SCALE_FP16,
    NULL,
    vxScaleKernelParam,
    (sizeof(vxScaleKernelParam) / sizeof(vxScaleKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxScaleInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_SCALE_list[] =
{
    &_VX_KERNEL_VAR,
    &vxScaleKernelInfo,
    NULL
};
#ifdef __cplusplus
}
#endif

