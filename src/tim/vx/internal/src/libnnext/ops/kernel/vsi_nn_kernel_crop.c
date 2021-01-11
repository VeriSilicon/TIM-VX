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
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

void myTensorCropFunc
    (
    int8_t *src,
    int8_t *dst
    )
{

    return;
}
vsi_status VX_CALLBACK TensorCropInternalKernel
    (vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 2)
    {

    }

    return status;
}

vsi_status VX_CALLBACK TensorCropInitializer
    (vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status   = VX_SUCCESS;
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in threads
        {0, 0, 0}}; // globalWorkSize: image size in threads

    vx_tensor input             = (vx_tensor)paramObj[0];
    vx_tensor output            = (vx_tensor)paramObj[1];
    uint32_t output_size[4]     = {1, 1, 1, 1};
    vsi_enum dataFormat, dstFormat;
    int8_t  input_fixPointPos   = 0;
    vx_uint32 i  =  0;
    int32_t offset[3];
    size_t size[DIM_SIZE];
    vsi_nn_tensor_attr_t attr[2];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    dataFormat        = attr[0].dtype.vx_type;
    input_fixPointPos = attr[0].dtype.fl;
    dstFormat         = attr[1].dtype.vx_type;
    for (i = 0; i < attr[1].dim_num; i++)
    {
        output_size[i] = attr[1].size[i];
    }

    vxCopyScalar((vx_scalar)paramObj[2], &offset[0], VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[3], &offset[1], VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[4], &offset[2], VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    memset(size, 0, sizeof(size_t) * DIM_SIZE);
    switch(dstFormat)
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_UINT8:
        size[0] = 16;
        size[1] = 4;
        break;
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_UINT16:
    case VSI_NN_TYPE_FLOAT16:
        size[0] = 8;
        size[1] = 4;
        break;
    }

    shaderParam.globalWorkOffset[0] = offset[0];
    shaderParam.globalWorkOffset[1] = offset[1];
    shaderParam.globalWorkOffset[2] = offset[2];
    shaderParam.globalWorkScale[0]  = size[0];
    shaderParam.globalWorkScale[1]  = size[1];
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0] = gcmALIGN((output_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1] = (output_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2] = output_size[2];

    if(dataFormat == VSI_NN_TYPE_INT16 && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 uniConvertInt16toFp16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

#define cropMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define CROP_MAX_POST_SHIFT_BITS     (31)
#define CROP_MAX_MULTIPLIER_NUM      (65535)

        if (input_fixPointPos > 0)
        {
            vx_uint8  postshift      = cropMIN(input_fixPointPos, CROP_MAX_POST_SHIFT_BITS);

            uniConvertInt16toFp16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = cropMIN((int64_t)1 << (-input_fixPointPos), CROP_MAX_MULTIPLIER_NUM);

            for (i = 0; i < 8; i++)
            {
                uniConvertInt16toFp16_2x8[i + 8] = multiplier;
            }
        }
#undef cropMIN
#undef CROP_MAX_POST_SHIFT_BITS
#undef CROP_MAX_MULTIPLIER_NUM

        status |= vxSetNodeUniform(nodObj, "uniConvertInt16toFp16_2x8", 1, uniConvertInt16toFp16_2x8);
    }

    vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}

vx_param_description_t basekernel_tensorCrop_params[] = {
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxTensorCropKernelInt16Info =
{
    VX_KERNEL_ENUM_TENSORCROP_INT16,
    VX_KERNEL_NAME_TENSORCROP_INT16,
    NULL,
    basekernel_tensorCrop_params,
    (sizeof(basekernel_tensorCrop_params) / sizeof(basekernel_tensorCrop_params[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    TensorCropInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTensorCropKernelInt8Info =
{
    VX_KERNEL_ENUM_TENSORCROP_INT8,
    VX_KERNEL_NAME_TENSORCROP_INT8,
    NULL,
    basekernel_tensorCrop_params,
    (sizeof(basekernel_tensorCrop_params) / sizeof(basekernel_tensorCrop_params[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    TensorCropInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTensorCropKernelInt16Fp16Info =
{
    VX_KERNEL_ENUM_TENSORCROP_INT16_FP16,
    VX_KERNEL_NAME_TENSORCROP_INT16_FP16,
    NULL,
    basekernel_tensorCrop_params,
    (sizeof(basekernel_tensorCrop_params) / sizeof(basekernel_tensorCrop_params[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    TensorCropInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_CROP_list[] =
{
    NULL,
    &vxTensorCropKernelInt16Info,
    &vxTensorCropKernelInt8Info,
    &vxTensorCropKernelInt16Fp16Info,
    NULL
};
#ifdef __cplusplus
}
#endif
