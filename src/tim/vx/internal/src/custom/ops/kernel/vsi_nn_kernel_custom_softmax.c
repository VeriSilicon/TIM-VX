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
#include <math.h>
#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_test.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_ID           VX_KERNEL_ID(CUSTOM_SOFTMAX)
#define _VX_KERNEL_VAR_CPU      (vx_client_kernel_CUSTOM_SOFTMAX_CPU)
#define _VX_KERNEL_VAR_VX       (vx_client_kernel_CUSTOM_SOFTMAX_VX)
#define _VX_KERNEL_NAME         ("com.vivantecorp.extension.CustomSoftmaxVXC")
#define _VX_KERNEL_FUNC_KERNEL  (vxCustomSoftmaxKernel)

static vsi_status VX_CALLBACK vxCustomSoftmaxKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_SUCCESS;
    vx_tensor input = NULL,output = NULL;
    float *f32_in_buffer = NULL,*f32_out_buffer=NULL;
    vx_context context = NULL;
    vsi_nn_tensor_attr_t in_attr,out_attr;
    uint32_t i,in_elements,out_elements;
    int32_t sf_axis;
    float fMax = 0.0;
    float  fProbSum = 0.0f;

    context = vxGetContext((vx_reference)node);
    input  = (vx_tensor)paramObj[0];
    output = (vx_tensor)paramObj[1];
    vxCopyScalar((vx_scalar)paramObj[2], &(sf_axis),VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    /* Fill input & output attribute data struct */
    status = vsi_nn_vxGetTensorAttr(input, &in_attr);
    TEST_CHECK_STATUS(status, final);
    status = vsi_nn_vxGetTensorAttr(output, &out_attr);
    TEST_CHECK_STATUS(status, final);

    in_elements = vsi_nn_vxGetTensorElementNum(&in_attr);
    out_elements = vsi_nn_vxGetTensorElementNum(&out_attr);

    /* alloc the float32 data buffer */
    f32_in_buffer = (float *)malloc(in_elements * sizeof(float));
    f32_out_buffer= (float *)malloc(out_elements * sizeof(float));
    memset(f32_in_buffer, 0, in_elements * sizeof(float));
    memset(f32_out_buffer, 0, out_elements * sizeof(float));

    /* Copy tensor to buffer, and convert bufer to float32 format */
    status = vsi_nn_vxConvertTensorToFloat32Data(
        context, input, &in_attr, f32_in_buffer, in_elements * sizeof(float));
    TEST_CHECK_STATUS(status, final);

    /* Softmax implement */
    for ( i = 0; i < out_elements; i++)
    {
        fMax = f32_in_buffer[i] > fMax ? f32_in_buffer[i] : fMax;
    }

    for ( i = 0; i < out_elements; i++)
    {
        f32_out_buffer[i] = (float)expf(f32_in_buffer[i] - fMax);
        fProbSum += f32_out_buffer[i];
    }
    for ( i = 0; i < out_elements; i++)
    {
        f32_out_buffer[i] =  f32_out_buffer[i]/ fProbSum;
    }
    status = vsi_nn_vxConvertFloat32DataToTensor(
        context, output, &out_attr, f32_out_buffer, out_elements * sizeof(float));

final:
    if(f32_in_buffer)free(f32_in_buffer);
    if(f32_out_buffer)free(f32_out_buffer);
    return status;
}

static vx_status VX_CALLBACK vxCustomSoftmaxInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    vx_status status = VX_SUCCESS;
    /*TODO: Add initial code for VX program*/
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    int input_size[6] = {1, 1, 1, 1, 1, 1};
    int sf_size;
    uint32_t input_dims;
    uint32_t i;
    vsi_nn_tensor_attr_t input_attr;

    memset(&input_attr, 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr((vx_tensor)paramObj[0], &input_attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    input_dims  = input_attr.dim_num;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = input_attr.size[i];
    }

    sf_size  =  input_size[0];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.localWorkSize[0]    = 1;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.globalWorkSize[0]   =
        gcmALIGN((1 + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   =
        gcmALIGN((1 + shaderParam.globalWorkScale[1] - 1) / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    {
        vx_uint32 Uni4x4_Fp16ToFp32[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };

        vxSetNodeUniform(nodObj, "Uni4x4_Fp16ToFp32", 1, Uni4x4_Fp16ToFp32);
        vxSetNodeUniform(nodObj, "sf_size",  1, &sf_size);
    }
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    if(status < 0)
    {
        VSILOGE("Initializer  failure!");
    }

    return status;
}

static vx_param_description_t s_params[] =
{
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
};

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t _VX_KERNEL_VAR_CPU =
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

vx_kernel_description_t _VX_KERNEL_VAR_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    NULL,
    s_params,
    _cnt_of_array( s_params ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxCustomSoftmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_CUSTOM_SOFTMAX_list[] =
{
    &_VX_KERNEL_VAR_CPU,
    &_VX_KERNEL_VAR_VX,
    NULL
};
#ifdef __cplusplus
}
#endif
