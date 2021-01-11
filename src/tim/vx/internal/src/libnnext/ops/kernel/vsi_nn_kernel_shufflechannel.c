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

vsi_status vxShuffleChannelFunc
    (
    vx_context context,
    vx_tensor input,
    vx_tensor output,
    int32_t group_number,
    int32_t axis
    )
{
    vsi_status status = VX_SUCCESS;
    vsi_nn_tensor_attr_t input_attr;
    vsi_nn_tensor_attr_t output_attr;
    uint8_t *in_data = NULL;
    uint8_t *out_data = NULL;
    uint32_t stride_size[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t buf_sz = 0;
    uint32_t group_row = group_number;
    uint32_t chs = 0, group_col = 0;
    uint32_t len = 1, num = 1, feature_map_size = 1;
    uint32_t n = 0, i = 0, j = 0;
    uint32_t type_bytes = 0, len_bytes = 0, fms_bytes = 0;

    status  = vsi_nn_vxGetTensorAttr(input, &input_attr);
    status |= vsi_nn_vxGetTensorAttr(output, &output_attr);
    TEST_CHECK_STATUS(status, final);
    in_data = vsi_nn_vxCopyTensorToData(context, input, &input_attr);
    TEST_CHECK_PTR(in_data, final);
    buf_sz = vsi_nn_GetStrideSize(&output_attr, stride_size);
    out_data = (uint8_t *)malloc( buf_sz );
    TEST_CHECK_PTR(out_data, final);

    chs = input_attr.size[axis];
    group_col = chs / group_row;
    type_bytes = vsi_nn_TypeGetBytes( input_attr.dtype.vx_type );

    for ( i = 0; i < (uint32_t)axis; i++)
    {
        len *= input_attr.size[i];
    }
    for ( i = axis + 1; i < input_attr.dim_num; i++)
    {
        num *= input_attr.size[i];
    }
    for ( i = 0; i <= (uint32_t)axis; i++)
    {
        feature_map_size *= input_attr.size[i];
    }

    /* Shuffle Channel CPU Implement, the shape and dtype of output must same as input */
    len_bytes = len * type_bytes;
    fms_bytes = feature_map_size * type_bytes;
    for ( n = 0; n < num; n++)
    {
        for ( i = 0; i < group_row; i++)
        {
            for ( j = 0; j < group_col; j++)
            {
                uint8_t *in_ptr = in_data + n * fms_bytes + (i * group_col + j) * len_bytes;
                uint8_t *out_ptr = out_data + n * fms_bytes + (j * group_row + i) * len_bytes;

                memcpy(out_ptr, in_ptr, len_bytes);
            }
        }
    }

    /* Copy data to output tensor */
    status = vsi_nn_vxCopyDataToTensor(context, output, &output_attr, out_data);
    TEST_CHECK_STATUS(status, final);
final:
    if (in_data) free(in_data);
    if (out_data) free(out_data);
    return status;
}
vsi_status VX_CALLBACK vxShuffleChannelKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 4)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[2] = { NULL };
        // scalar
        vx_scalar scalar[2] = { NULL };
        int32_t group_number = 0;
        int32_t axis = 0;

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];
        scalar[0] = (vx_scalar)paramObj[2];
        scalar[1] = (vx_scalar)paramObj[3];

        context = vxGetContext((vx_reference)node);
        TEST_CHECK_PTR(context,final);
        // scalar
        status = vxCopyScalar(scalar[0], &group_number, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        TEST_CHECK_STATUS(status, final);
        status = vxCopyScalar(scalar[1], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        TEST_CHECK_STATUS(status, final);

        // Call C Prototype
        status = vxShuffleChannelFunc(context, imgObj[0], imgObj[1], group_number, axis);
        TEST_CHECK_STATUS(status, final);
    }
final:
    return status;
}
vsi_status VX_CALLBACK vxShuffleChannelInitializer
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
    vx_scalar     group_numbers   = (vx_scalar)paramObj[2];
    vx_scalar     axis_s          = (vx_scalar)paramObj[3];
    uint32_t      input_size[4]   = {1, 1, 1, 1};
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    int32_t       group_number    = 0;
    int32_t       axis            = 0;
    int32_t       group_column    = 0;
    float         rgroup_column   = 0.0f;
    uint32_t      chs             = 0;
    vx_uint32     i               = 0;
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
    inputDataFormat = attr.dtype.vx_type;

    status |= vxCopyScalar(group_numbers, &group_number, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar(axis_s, &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(VX_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer failure! \n",__FILE__, __LINE__);
        return status;
    }
    chs = input_size[axis];
    if (chs % group_number)
    {
        VSILOGE("input channel can't be exact divided by group number! at line %d\n", __LINE__);
        return VX_FAILURE;
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    if (axis == 2)
    {
        if (inputDataFormat == VSI_NN_TYPE_FLOAT16 || inputDataFormat == VSI_NN_TYPE_INT16)
            shaderParam.globalWorkScale[0]  = 8;
        else
            shaderParam.globalWorkScale[0]  = 16;
        shaderParam.globalWorkScale[1]  = 4;
        shaderParam.globalWorkScale[2]  = 1;

        shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (input_size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
        shaderParam.globalWorkSize[2]   = input_size[2];
    }
    else if (axis == 1)
    {
        shaderParam.globalWorkScale[0]  = 32;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;

        shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = input_size[1];
        shaderParam.globalWorkSize[2]   = input_size[2];
    }
    else
    {
        VSILOGE("[%s : %d]Initializer failure, not support axis: %d! \n",__FILE__, __LINE__, axis);
        return VX_FAILURE;
    }
    group_column = chs / group_number;
    rgroup_column = 1.0f / group_column;

    status |= vxSetNodeUniform(nodObj, "group_column", 1, &group_column);
    status |= vxSetNodeUniform(nodObj, "rgroup_column", 1, &rgroup_column);
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer failure! \n",__FILE__, __LINE__);
    }
    return status;
}
static vx_param_description_t vxShuffleChannelKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxShuffleChannelKernelInfo =
{
    VX_KERNEL_ENUM_SHUFFLECHANNEL,
    VX_KERNEL_NAME_SHUFFLECHANNEL,
    NULL,
    vxShuffleChannelKernelParam,
    (sizeof(vxShuffleChannelKernelParam) / sizeof(vxShuffleChannelKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxShuffleChannelInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxShuffleChannelKernelInfo8Bits =
{
    VX_KERNEL_ENUM_SHUFFLECHANNEL,
    VX_KERNEL_NAME_SHUFFLECHANNEL8BITS,
    NULL,
    vxShuffleChannelKernelParam,
    (sizeof(vxShuffleChannelKernelParam) / sizeof(vxShuffleChannelKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxShuffleChannelInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxShuffleChannelKernelInfo_CPU =
{
    VX_KERNEL_ENUM_SHUFFLECHANNEL,
    VX_KERNEL_NAME_SHUFFLECHANNEL,
    vxShuffleChannelKernel,
    vxShuffleChannelKernelParam,
    (sizeof(vxShuffleChannelKernelParam) / sizeof(vxShuffleChannelKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxShuffleChannelKernelInfo_16BitsAxis1 =
{
    VX_KERNEL_ENUM_SHUFFLECHANNEL,
    VX_KERNEL_NAME_SHUFFLECHANNEL16BITS_AXIS1,
    NULL,
    vxShuffleChannelKernelParam,
    (sizeof(vxShuffleChannelKernelParam) / sizeof(vxShuffleChannelKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxShuffleChannelInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxShuffleChannelKernelInfo_8BitsAxis1 =
{
    VX_KERNEL_ENUM_SHUFFLECHANNEL,
    VX_KERNEL_NAME_SHUFFLECHANNEL8BITS_AXIS1,
    NULL,
    vxShuffleChannelKernelParam,
    (sizeof(vxShuffleChannelKernelParam) / sizeof(vxShuffleChannelKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxShuffleChannelInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t * vx_kernel_SHUFFLECHANNEL_list[] =
{
    &vxShuffleChannelKernelInfo_CPU,
    &vxShuffleChannelKernelInfo,
    &vxShuffleChannelKernelInfo8Bits,
    &vxShuffleChannelKernelInfo_16BitsAxis1,
    &vxShuffleChannelKernelInfo_8BitsAxis1,
    NULL
};
#ifdef __cplusplus
}
#endif
