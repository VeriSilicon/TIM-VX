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
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_SPACE2DEPTH)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_SPACE2DEPTH)
#define _VX_KERNEL_NAME         ("vsi_nn_kernel_space2depth")
#define _VX_KERNEL_FUNC_KERNEL  (vxSpace2DepthKernel)

static vsi_status VX_CALLBACK vxSpace2DepthKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    /* TODO: */
#define ARG_NUM          (2)
#define TENSOR_NUM_INPUT (1)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VX_SUCCESS;
    uint32_t  i = 0;
    vx_context context = NULL;
    vsi_nn_tensor_attr_t attr[TENSOR_NUM];
    uint32_t stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];
    vx_tensor_addressing user_addr[TENSOR_NUM] = {NULL};
    uint8_t *buffer_ptr[TENSOR_NUM] = {NULL};
    vx_tensor tensor[TENSOR_NUM] = {NULL};

    int32_t block_size_x = 0, block_size_y = 0;
    int32_t output_depth = 0, output_height = 0, output_width = 0;
    int32_t input_batch = 0, input_depth = 0, input_height = 0, input_width = 0;
    int32_t batch = 0, dim = 0;

    for(i = 0; i < TENSOR_NUM; i++)
    {
        memset(&attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }

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

    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(block_size_x),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(block_size_y),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    dim = attr[0].dim_num;
    if(dim < 4)
        attr[0].size[3] = 1;
    //op calc
    //output_batch = attr[1].size[3];
    output_depth = attr[1].size[2];
    output_height = attr[1].size[1];
    output_width = attr[1].size[0];

    input_batch = attr[0].size[3];
    input_depth = attr[0].size[2];
    input_height = attr[0].size[1];
    input_width = attr[0].size[0];

    for (batch = 0; batch < input_batch; ++batch)
    {
        vx_uint32 output_batch_index = batch * output_height * output_width * output_depth;
        vx_uint32 input_batch_index = batch * input_height * input_width * input_depth;
        vx_uint32 in_d;
        for (in_d = 0; in_d < (vx_uint32)input_depth; in_d ++)
        {
            vx_uint32 in_h;
            for (in_h = 0; in_h < (vx_uint32)input_height; ++ in_h)
            {
                vx_uint32 in_w;
                for (in_w = 0; in_w < (vx_uint32)input_width; in_w ++)
                {
                    vx_int32 out_w = in_w / block_size_x;
                    vx_int32 out_h = in_h / block_size_y;
                    //vx_int32 out_d = (in_w  % block_size_x) * input_depth + (in_h % block_size_y) * block_size_x * input_depth + in_d;
                    vx_int32 out_d = (in_w  % block_size_x) + (in_h % block_size_y) * block_size_x + in_d * block_size_x * block_size_y;

                    vx_int32 in_index = in_w + in_h * input_width +in_d * input_height * input_width + input_batch_index;
                    vx_int32 out_index = out_w + out_h * output_width +  out_d * output_width * output_height + output_batch_index;

                    //outputBase[out_index] = inputBase[in_index];
                    float fval;
                    vsi_nn_DtypeToFloat32(&buffer_ptr[0][stride_size[0][0] * in_index],
                        &fval, &attr[0].dtype);
                    vsi_nn_Float32ToDtype(fval, &buffer_ptr[1][stride_size[1][0] * out_index],
                        &attr[1].dtype);
                }
            }
        }
    }

    //save data
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        vsi_nn_copy_tensor_patch(tensor[i], &attr[i], buffer_ptr[i], VX_WRITE_ONLY);
    }
    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
        if (buffer_ptr[i]) free(buffer_ptr[i]);
    }

    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

vsi_status VX_CALLBACK vxSpace2DepthInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_status status = VX_SUCCESS;

    vx_tensor input     = (vx_tensor)paramObj[0];
    uint32_t input_size[4] = {1, 1, 1, 1};
    vx_uint32 input_dimz = 0;
    vx_uint32 input_depth = 0;
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

    input_depth = input_size[2];
    if(input_size[3] > 0)
        input_dimz = input_depth * input_size[3];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 8;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.localWorkSize[2]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    shaderParam.globalWorkSize[2]   = input_dimz;

    {
        vx_uint32 uniExtractEvenFp16Stride2_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniExtractOddFp16Stride2_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00030001, 0x00070005, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        status |= vxSetNodeUniform(nodObj, "uniExtractEvenFp16Stride2_4x4", 1, uniExtractEvenFp16Stride2_4x4);
        status |= vxSetNodeUniform(nodObj, "uniExtractOddFp16Stride2_4x4", 1, uniExtractOddFp16Stride2_4x4);
        //status |= vxSetNodeUniform(nodObj, "input_depth", 1, &input_depth);
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

vx_kernel_description_t vxSpace2DepthKernelInfo_int16_int16 =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_SPACE2DEPTH_INT16_INT16,
    NULL,
    s_params,
    _cnt_of_array( s_params ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxSpace2DepthInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_SPACE2DEPTH_list[] =
{
    NULL,
    &_VX_KERNEL_VAR,
    &vxSpace2DepthKernelInfo_int16_int16,
    NULL
};
#ifdef __cplusplus
}
#endif
