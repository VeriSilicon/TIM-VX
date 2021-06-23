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
#include "libnnext/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

void tensorStackConcatFunc
    (
    int16_t* dataIn,
    int16_t* dataIO,
    int32_t  index,
    uint32_t width,
    uint32_t height,
    uint32_t channel,
    uint32_t batch
    )
{
    int32_t stride = width * sizeof(int16_t);
    VSILOGI("Hello tensorStackConcatFunc!\n");
    memcpy(dataIO + index * width, dataIn, stride);
    return;
}
vsi_status VX_CALLBACK vxTensorStackConcatKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 3)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[2] = { NULL };
        vsi_nn_tensor_attr_t attr[2];
        int16_t *input = NULL, *output = NULL;
        uint32_t input_size[4] = {1, 1, 1, 1}, output_size[4] = {1, 1, 1, 1};
        uint32_t input_stride_size[4]  = {1, 1, 1, 1};
        uint32_t output_stride_size[4] = {1, 1, 1, 1};
        vx_tensor_addressing input_user_addr = NULL;
        vx_tensor_addressing output_user_addr = NULL;
        vsi_nn_type_e inputFormat = VSI_NN_TYPE_FLOAT16, outputFormat = VSI_NN_TYPE_FLOAT16;
        uint32_t input_dims = 0, output_dims = 0;
        uint32_t i;
        // scalar
        vx_scalar scalar[1] = { NULL };
        int32_t index = 0;

        status = VX_SUCCESS;
        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];
        scalar[0] = (vx_scalar)paramObj[2];
        memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
        memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            return status;
        }

        status  = vsi_nn_vxGetTensorAttr(imgObj[0], &attr[0]);
        status |= vsi_nn_vxGetTensorAttr(imgObj[1], &attr[1]);
        status |= vsi_nn_vxGetTensorAttr(imgObj[1], &attr[1]);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
            goto final;
        }

        //input
        input_dims  = attr[0].dim_num;
        inputFormat = attr[0].dtype.vx_type;
        for (i = 0; i < input_dims; i++)
        {
            input_size[i] = attr[0].size[i];
        }
        //output
        output_dims  = attr[1].dim_num;
        outputFormat = attr[1].dtype.vx_type;
        for (i = 0; i < output_dims; i++)
        {
            output_size[i] = attr[1].size[i];
        }

        input_size[2] = (input_dims <= 2)?1:input_size[2];
        input_size[3] = (input_dims <= 3)?1:input_size[3];
        input_stride_size[0]  = vsi_nn_GetTypeBytes(inputFormat);
        for (i=1; i< input_dims; i++)
        {
            input_stride_size[i]  = input_stride_size[i-1] * input_size[i-1];
        }
        input  = (int16_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
        input_user_addr = vxCreateTensorAddressing(context, input_size, input_stride_size, (vx_uint8)input_dims);
        vsi_nn_copy_tensor_patch(imgObj[0], &attr[0], input, VX_READ_ONLY);
        output_stride_size[0] = vsi_nn_GetTypeBytes(outputFormat);
        for (i=1; i< output_dims; i++)
        {
            output_stride_size[i] = output_stride_size[i-1] * output_size[i-1];
        }
        output = (int16_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(int16_t));
        output_user_addr = vxCreateTensorAddressing(context, output_size,
            output_stride_size, (vx_uint8)output_dims);

        vsi_nn_copy_tensor_patch(imgObj[1], &attr[1], output, VX_READ_ONLY);
        // scalar
        status = vxCopyScalar(scalar[0], &index, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxCopyScalar failure! at line %d\n", __LINE__);
            goto final;
        }
        // Call C Prototype
        tensorStackConcatFunc(input, output, index, input_size[0],
            input_size[1], input_size[2], input_size[3]);
        //output tensor
        vsi_nn_copy_tensor_patch(imgObj[1], &attr[1], output, VX_WRITE_ONLY);
final:
        if(input) free(input);
        if(output) free(output);
        if(input_user_addr) vxReleaseTensorAddressing(&input_user_addr);
        if(output_user_addr) vxReleaseTensorAddressing(&output_user_addr);
    }
    return status;
}
vsi_status VX_CALLBACK vxTensorStackConcatInitializer
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
    uint32_t      input_size[4]   = {1, 1, 1, 1};
    uint32_t      input_dims      = 0;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_nn_tensor_attr_t attr;
    uint32_t      i;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        return status;
    }

    input_dims      = attr.dim_num;
    inputDataFormat = attr.dtype.vx_type;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr.size[i];
    }
    input_size[2] = (input_dims <= 2)?1:input_size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    if (inputDataFormat == VSI_NN_TYPE_FLOAT16 || inputDataFormat == VSI_NN_TYPE_INT16)
        shaderParam.globalWorkScale[0]  = 16;
    else
        shaderParam.globalWorkScale[0]  = 32;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = (input_size[2] + shaderParam.globalWorkScale[2] - 1)
        / shaderParam.globalWorkScale[2];

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}
static vx_param_description_t vxTensorStackConcatKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxTensorStackConcatKernelInfo =
{
    VX_KERNEL_ENUM_TENSORSTACKCONCAT,
    VX_KERNEL_NAME_TENSORSTACKCONCAT,
    NULL,
    vxTensorStackConcatKernelParam,
    (sizeof(vxTensorStackConcatKernelParam) / sizeof(vxTensorStackConcatKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxTensorStackConcatInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTensorStackConcatKernelInfo8Bits =
{
    VX_KERNEL_ENUM_TENSORSTACKCONCAT8BITS,
    VX_KERNEL_NAME_TENSORSTACKCONCAT8BITS,
    NULL,
    vxTensorStackConcatKernelParam,
    (sizeof(vxTensorStackConcatKernelParam) / sizeof(vxTensorStackConcatKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxTensorStackConcatInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTensorStackConcatKernelInfo_CPU =
{
    VX_KERNEL_ENUM_TENSORSTACKCONCAT,
    VX_KERNEL_NAME_TENSORSTACKCONCAT,
    vxTensorStackConcatKernel,
    vxTensorStackConcatKernelParam,
    (sizeof(vxTensorStackConcatKernelParam) / sizeof(vxTensorStackConcatKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_TENSORSTACKCONCAT_list[] =
{
    &vxTensorStackConcatKernelInfo_CPU,
    &vxTensorStackConcatKernelInfo,
    &vxTensorStackConcatKernelInfo8Bits,
    NULL
};
#ifdef __cplusplus
}
#endif
