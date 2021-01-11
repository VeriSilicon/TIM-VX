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

#define _VX_KERNEL_VAR          (vx_kernel_FCL2)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_FULLYCONNECTED_AXIS2)
#define _VX_KERNEL_NAME         ("vsi_nn_kernel_fullconnect2")
#define _VX_KERNEL_FUNC_KERNEL  (vxFullconnect2Kernel)

//static uint32_t layerNum = 0;

static vsi_status VX_CALLBACK vxFullconnect2Kernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    /* TODO: */
#define ARG_NUM            (2)
#define TENSOR_NUM_INPUT (3)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VX_SUCCESS;
    uint32_t  i, j, k;
    vx_context context = NULL;
    vsi_nn_tensor_attr_t attr[TENSOR_NUM];
    uint32_t stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];
    vx_tensor_addressing user_addr[TENSOR_NUM] = {NULL};
    uint8_t *buffer_ptr[TENSOR_NUM] = {NULL};
    vx_tensor tensor[TENSOR_NUM];

    //char fileName[256] = {'\0'};
    //uint32_t total_size;
    int32_t axis, weights;
    uint32_t num_fc = 1, num_no_fc = 1;


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

    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(axis),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(weights),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    //op calc
    for(i = 0; i <= (uint32_t)axis; ++i)
    {
        num_fc *= attr[0].size[i];
    }
    for(i = axis + 1; i < attr[0].dim_num; ++i)
    {
        num_no_fc *= attr[0].size[i];
    }

    for(k = 0; k < num_no_fc; ++k)
    {
        for(j = 0; j < (uint32_t)weights; ++j)
        {
            float sum;
            vsi_nn_DtypeToFloat32(&buffer_ptr[2][stride_size[2][0] * j], &sum, &attr[2].dtype);
            for(i = 0; i < num_fc; ++i)
            {
                float x, w;
                vsi_nn_DtypeToFloat32(&buffer_ptr[0][stride_size[0][0] * (i + num_fc * k)],
                    &x, &attr[0].dtype);
                vsi_nn_DtypeToFloat32(&buffer_ptr[1][stride_size[1][0] * (i + num_fc * j)],
                    &w, &attr[1].dtype);
                sum += w * x;
            }
            vsi_nn_Float32ToDtype(sum, &buffer_ptr[3][stride_size[3][0] * (j + weights * k)],
                &attr[3].dtype);
        }
    }

#if 0
    print_index = 3;
    total_size = vsi_nn_ShapeProduct(size[print_index], dim_num[print_index]);
    if (dim_num[print_index] == 3)
    {
        snprintf(fileName, VSI_NN_MAX_PATH, "%s_%d_%d_%d_%d.txt", _VX_KERNEL_NAME, layerNum,
            size[print_index][0], size[print_index][1], size[print_index][2]);
    }
    else
    {
        snprintf(fileName, VSI_NN_MAX_PATH, "%s_%d_%d_%d_%d_%d.txt", _VX_KERNEL_NAME, layerNum,
            size[print_index][0], size[print_index][1], size[print_index][2], size[print_index][3]);
    }
    vsi_nn_SaveDataToText(fileName, buffer_ptr[print_index], total_size,
        data_format[print_index], NULL);
    layerNum++;
#endif
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
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
};

void myFullyConnected_Axis2Func
    (
    int8_t *src,
    int8_t *dst
    )
{

    return;
}
vsi_status VX_CALLBACK vxFullyConnected_Axis2Kernel
    (
    vx_node node,
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

vsi_status VX_CALLBACK vxFullyConnected_Axis2Initializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status   = VX_SUCCESS;

    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in threads
        {0, 0, 0}}; // globalWorkSize: image size in threads

    uint32_t       input_size[DIM_SIZE] = {1, 1, 1, 1};
    uint32_t       output_size[DIM_SIZE] = {1, 1, 1, 1};

    uint32_t uniMulAcc_16x1[16] = {
        0x00005555, // TCfg
        0x00000000, // ASelt
        0x76543210, 0x00000000, // ABin
        0x00005555, // BSelt
        0x76543210, 0x00000000, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    uint32_t loopNum = 0;
    vsi_nn_tensor_attr_t attr[2];
    uint32_t i;
    uint32_t input_dims      = 0;
    uint32_t output_dims     = 0;

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr((vx_tensor)paramObj[1], &attr[0]);
    status |= vsi_nn_vxGetTensorAttr((vx_tensor)paramObj[3], &attr[1]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        return status;
    }
    input_dims  = attr[0].dim_num;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr[0].size[i];
    }
    output_dims  = attr[1].dim_num;
    for (i = 0; i < output_dims; i++)
    {
        output_size[i] = attr[1].size[i];
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0] = 1;
    shaderParam.globalWorkScale[1] = 1;
    shaderParam.globalWorkSize[0] = gcmALIGN((output_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1] = (output_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];

    vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    vxSetNodeUniform(nodObj, "uniMulAcc_16x1", 1, uniMulAcc_16x1);

    loopNum = gcmALIGN(input_size[0], 32);
    vxSetNodeUniform(nodObj, "loopNum", 1, &loopNum);
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}

static vx_param_description_t vxFullyConnected_Axis2KernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
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

vx_kernel_description_t vxFullyConnected_Axis2KernelInfo =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    vxFullyConnected_Axis2Kernel,
    vxFullyConnected_Axis2KernelParam,
    (sizeof(vxFullyConnected_Axis2KernelParam) / sizeof(vxFullyConnected_Axis2KernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxFullyConnected_Axis2Initializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_FCL2_list[] =
{
    &_VX_KERNEL_VAR,
    &vxFullyConnected_Axis2KernelInfo,
    NULL
};
#ifdef __cplusplus
}
#endif
