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

#define _VX_KERNEL_VAR          (vx_kernel_RESIZE)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_RESIZE)
#define _VX_KERNEL_NAME         ("vsi_nn_kernel_resize")
#define _VX_KERNEL_FUNC_KERNEL  (vxResizeKernel)

static vsi_status VX_CALLBACK vxResizeKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    /* TODO: */
#define ARG_NUM            (1)
#define TENSOR_NUM_INPUT (1)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VX_SUCCESS;
    vx_context context = NULL;
    vsi_nn_tensor_attr_t attr[TENSOR_NUM];
    uint32_t stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];
    vx_tensor_addressing user_addr[TENSOR_NUM] = {NULL};
    uint8_t *buffer_ptr[TENSOR_NUM] = {NULL};
    vx_tensor tensor[TENSOR_NUM];

    float factor0;
    int32_t factor;
    uint32_t batch, c, h, w;
    uint32_t i, j, k, b;

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

    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(factor0), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    //op calc
    if (factor0 > 1)
    {
        factor = (int32_t)(factor0 + 0.5);
        w = attr[0].size[0];
        h = attr[0].size[1];
        c = attr[0].size[2];
        batch = 1;
        for(b = 0; b < batch; ++b){
            for(k = 0; k < c; ++k){
                for(j = 0; j < h*factor; ++j){
                    for(i = 0; i < w*factor; ++i){
                        int32_t in_index = b*w*h*c + k*w*h + (j/factor)*w + i/factor;
                        int32_t out_index = b*w*h*c*factor*factor + k*w*h*factor*factor +
                            j*w*factor + i;
                        float fval;
                        //out[out_index] = in[in_index];
                        vsi_nn_DtypeToFloat32(&buffer_ptr[0][stride_size[0][0] * in_index],
                            &fval, &attr[0].dtype);
                        vsi_nn_Float32ToDtype(fval, &buffer_ptr[1][stride_size[1][0] * out_index],
                            &attr[1].dtype);
                    }
                }
            }
        }
    }
    else
    {
        factor = (int32_t)(1 / factor0 + 0.5);
        w = attr[1].size[0];
        h = attr[1].size[1];
        c = attr[1].size[2];
        batch = 1;
        for(b = 0; b < batch; ++b){
            for(k = 0; k < c; ++k){
                for(j = 0; j < h; ++j){
                    for(i = 0; i < w; ++i){
                        int32_t in_index = b*w*h*c*factor*factor +
                            k*w*h*factor*factor + j*w*factor*factor + i*factor;
                        int32_t out_index = b*w*h*c + k*w*h + j * w + i;
                        float fval;
                        //out[out_index] = in[in_index];
                        vsi_nn_DtypeToFloat32(&buffer_ptr[0][stride_size[0][0] * in_index], &fval,
                            &attr[0].dtype);
                        vsi_nn_Float32ToDtype(fval, &buffer_ptr[1][stride_size[1][0] * out_index],
                            &attr[1].dtype);
                    }
                }
            }
        }
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
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t s_params[] =
{
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
};

vsi_status VX_CALLBACK vxTensorResizeInitializer
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
    uint32_t uniPackEvenData_2x8[16] = {
        0x33333333, // TCfg
        0x11110000, // ASelt
        0x06040200, 0x06040200, // ABin
        0x00000000, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00003400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vsi_status status = VX_SUCCESS;

    vx_tensor input = (vx_tensor)paramObj[0];
    uint32_t input_size[DIM_SIZE] = {1, 1, 1, 1};
    vsi_nn_tensor_attr_t attr;
    uint32_t i, input_dim;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        return status;
    }
    input_dim  = attr.dim_num;
    for (i = 0; i < input_dim; i++)
    {
        input_size[i] = attr.size[i];
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 16;
    shaderParam.globalWorkScale[1]  = 2;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];

    vxSetNodeUniform(nodObj, "uniPackEvenData_2x8", 1, uniPackEvenData_2x8);

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}

static vx_param_description_t vxTensorResizeKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
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

vx_kernel_description_t vxTensorResize16BitsDownSampleQuarterKernelInfo =
{
    VX_KERNEL_ENUM_RESIZE_16BITS_DOWNSAMPLE_QUARTER,
    VX_KERNEL_NAME_RESIZE_16BITS_DOWNSAMPLE_QUARTER,
    NULL,
    vxTensorResizeKernelParam,
    (sizeof(vxTensorResizeKernelParam) / sizeof(vxTensorResizeKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxTensorResizeInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTensorResize8BitsDownSampleQuarterKernelInfo =
{
    VX_KERNEL_ENUM_RESIZE_8BITS_DOWNSAMPLE_QUARTER,
    VX_KERNEL_NAME_RESIZE_8BITS_DOWNSAMPLE_QUARTER,
    NULL,
    vxTensorResizeKernelParam,
    (sizeof(vxTensorResizeKernelParam) / sizeof(vxTensorResizeKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxTensorResizeInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_RESIZE_list[] =
{
    &_VX_KERNEL_VAR,
    &vxTensorResize16BitsDownSampleQuarterKernelInfo,
    &vxTensorResize8BitsDownSampleQuarterKernelInfo,
    NULL
};
#ifdef __cplusplus
}
#endif
