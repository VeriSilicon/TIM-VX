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

#define _VX_KERNEL_VAR          (vx_kernel_SPATIAL_TRANSFORMER)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_SPATIAL_TRANSFORMER)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_SPATIAL_TRANSFORMER)
#define _VX_KERNEL_FUNC_KERNEL  (vxSpatial_transformerKernel)


static vsi_status VX_CALLBACK vxSpatial_transformerKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    /*To do cpu implementation*/
    vsi_status status = VX_SUCCESS;

    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t s_params[] =
{
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
};

vx_status VX_CALLBACK vxTransform_GemmInputValidator(vx_node node, vx_uint32 index)
{
    return VX_SUCCESS;
}

vx_status VX_CALLBACK vxTransform_GemmOutputValidator(vx_node node, vx_uint32 index, vx_meta_format metaObj)
{
    return VX_SUCCESS;
}

vx_status VX_CALLBACK vxValidator(vx_node node, const vx_reference parameters[],
                                    vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_uint32 index = 0;
    for(index = 0; index < num; index++)
    {
        if(index < 2)
        {
            status |= vxTransform_GemmInputValidator(node,index);
        }
        else
        {
            status |= vxTransform_GemmOutputValidator(node,index,metas[index]);
        }
    }
    return status;
}

static vx_param_description_t vxTransform_GemmKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

vx_status VX_CALLBACK vxTransform_GemmInitializer(vx_node nodObj, const vx_reference *paramObj, vx_uint32 paraNum)
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define gcmMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status    status             = VX_SUCCESS;
    vx_tensor    input0             = (vx_tensor)paramObj[0];
    vx_tensor    input1             = (vx_tensor)paramObj[1];
    vx_tensor    output             = (vx_tensor)paramObj[2];
    vx_enum      src0Format         = VSI_NN_TYPE_FLOAT16;
    vx_enum      src1Format         = VSI_NN_TYPE_FLOAT16;
    vx_enum      dstFormat          = VSI_NN_TYPE_FLOAT16;
    vx_uint32    coord_size[4]      = {1, 1, 1, 1};
    vx_uint32    i     = 0;
    vsi_nn_tensor_attr_t attr[3];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[2], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input0, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(input1, &attr[1]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[2]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    src0Format       = attr[0].dtype.vx_type;
    src1Format       = attr[1].dtype.vx_type;
    for (i = 0; i < attr[1].dim_num; i++)
    {
        coord_size[i] = attr[1].size[i];
    }
    dstFormat        = attr[2].dtype.vx_type;

    if (src0Format == VSI_NN_TYPE_FLOAT16 && src1Format == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        shaderParam.globalWorkScale[0]  = 12;
        shaderParam.globalWorkScale[1]  = 1;
    }

    shaderParam.globalWorkSize[0]   =
                gcmALIGN((coord_size[0] + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   =
                (coord_size[1] + shaderParam.globalWorkScale[1] - 1) / shaderParam.globalWorkScale[1];
    {
        vx_uint32 uniGemm3x3_4x4[16] = {
            0x15151515, // TCfg
            0x00000000, // ASelt
            0x02100210, 0x05430543, // ABin
            0x15151515, // BSelt
            0x05430210, 0x05430210, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vxSetNodeUniform(nodObj, "uniGemm3x3_4x4", 1, uniGemm3x3_4x4);
    }
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                    &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}

static vx_param_description_t vxTransform_setupThresKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

vx_status VX_CALLBACK vxTransform_setupThresInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define gcmMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status    status             = VX_SUCCESS;
    vx_scalar    thresFlag_s        = (vx_scalar)paramObj[2];
    vx_enum      src0Format         = VSI_NN_TYPE_FLOAT16;
    vx_enum      src1Format         = VSI_NN_TYPE_FLOAT16;

    vx_int32     thresFlag          = 0;
    vx_uint32    extract_packed[4]  = {0};

    vxCopyScalar(thresFlag_s, &thresFlag, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if(status < 0)
        VSILOGE("error-%s,%d\n",__FILE__,__LINE__);

    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.localWorkSize[0]    = 1;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.globalWorkSize[0]   = 1;
    shaderParam.globalWorkSize[1]   = 1;

    if (src0Format == src1Format && src0Format == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 i = 0;
        vx_uint32 j = 0;
        for (i = 0; i < 4; i++)
        {
            if (thresFlag & (1 << i))
            {
                extract_packed[0] |= ((i << 4) << (i * 8));
            }
            else
            {
                extract_packed[0] |= (((j << 4) + 128) << (i * 8));
                j ++;
            }
        }

        for (i = 4; i < 6; i++)
        {
            if (thresFlag & (1 << i))
            {
                extract_packed[1] |= ((i << 4) << (i * 8 - 32));
            }
            else
            {
                extract_packed[1] |= (((j << 4) + 128) << (i * 8 - 32));
                j ++;
            }
        }

        extract_packed[2] = extract_packed[3] = 0x10101010;
    }

    vxSetNodeUniform(nodObj, "extract_packed", 1, extract_packed);

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}


static vx_param_description_t vxTransform_InterPKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};


vx_status VX_CALLBACK vxTransform_InterPInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define gcmMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status    status             = VX_SUCCESS;
    vx_tensor    input0             = (vx_tensor)paramObj[0];
    vx_tensor    input1             = (vx_tensor)paramObj[1];
    vx_tensor    output             = (vx_tensor)paramObj[2];
    vx_enum      src0Format         = VSI_NN_TYPE_FLOAT16;
    vx_enum      src1Format         = VSI_NN_TYPE_FLOAT16;
    vx_enum      dstFormat          = VSI_NN_TYPE_FLOAT16;
    vx_uint32    coord_size[4]      = {1, 1, 1, 1};
    vx_uint32    input_size[4]      = {1, 1, 1, 1};
    vx_uint32    output_size[4]     = {1, 1, 1, 1};
    vx_uint32    i     = 0;
    vsi_nn_tensor_attr_t attr[3];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[2], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input0, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(input1, &attr[1]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[2]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    for (i = 0; i < attr[0].dim_num; i++)
    {
        input_size[i] = attr[0].size[i];
    }
    src0Format       = attr[0].dtype.vx_type;
    src1Format       = attr[1].dtype.vx_type;
    for (i = 0; i < attr[1].dim_num; i++)
    {
        coord_size[i] = attr[1].size[i];
    }
    dstFormat        = attr[2].dtype.vx_type;
    for (i = 0; i < attr[2].dim_num; i++)
    {
        output_size[i] = attr[2].size[i];
    }

    if ((src0Format == VSI_NN_TYPE_FLOAT16 && src1Format == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_FLOAT16)
     || (src0Format == VSI_NN_TYPE_INT16 && src1Format == VSI_NN_TYPE_INT16 && dstFormat == VSI_NN_TYPE_INT16))
    {
        shaderParam.globalWorkScale[0]  = 2;
        shaderParam.globalWorkScale[1]  = 1;
    }

    shaderParam.globalWorkSize[0]   =
                gcmALIGN((coord_size[0] + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   =
                (coord_size[1] + shaderParam.globalWorkScale[1] - 1) / shaderParam.globalWorkScale[1];
    {
        vx_int32 packedWH2[2]   = {input_size[0], input_size[1]};
        vx_int32 packedWH       = (input_size[1] << 16) | (input_size[0] & 0xFFFF);
        vx_uint32 uniGetDXY_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00100001, 0x00010010, // ABin
            0x09090909, // BSelt
            0x00010000, 0x00000001, // BBin
            0x00000101, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x00000000, 0x3c000000, 0x00000000,
            0x3c000000, 0x00000000, 0x3c000000, 0x00000000 // Constant
        };
        vx_uint32 uniConvertF16toF32_4x4[16] = {
            0x01010101, // TCfg
            0x01010000, // ASelt
            0x00010000, 0x00010000, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        vxSetNodeUniform(nodObj, "uniGetDXY_4x4", 1, uniGetDXY_4x4);
        vxSetNodeUniform(nodObj, "uniConvertF16toF32_4x4", 1, uniConvertF16toF32_4x4);

        //packedWH2[0]   = input_size[0];
        //packedWH2[1]   = input_size[1];
        //packedWH       = (input_size[1] << 16) | (input_size[0] & 0xFFFF);
        vxSetNodeUniform(nodObj, "packedWH2", 1, packedWH2);
        vxSetNodeUniform(nodObj, "packedWH", 1, &packedWH);
    }
    if (output_size[2] > 1)
    {
        vxSetNodeUniform(nodObj, "depth", 1, &output_size[2]);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxSpatial_transformer_CPU =
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

vx_kernel_description_t vxTransform_GemmKernelInfo_F16toF16 =
{
    VX_KERNEL_ENUM_SPATIAL_TRANSFORMER,
    VX_KERNEL_NAME_SPATIAL_TRANSFORMER,
    NULL,
    vxTransform_GemmKernelParam,
    (sizeof(vxTransform_GemmKernelParam) / sizeof(vxTransform_GemmKernelParam[0])),
    vxValidator,
    NULL,
    NULL,
    vxTransform_GemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTransform_setupThresKernelInfo_F16toF16 =
{
    VX_KERNEL_ENUM_SPATIAL_TRANSFORMER,
    VX_KERNEL_NAME_TRANSFORM_SETUP_THRES_F16TOF16,
    NULL,
    vxTransform_setupThresKernelParam,
    (sizeof(vxTransform_setupThresKernelParam) / sizeof(vxTransform_setupThresKernelParam[0])),
    vxValidator,
    NULL,
    NULL,
    vxTransform_setupThresInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTransform_InterPKernelInfo_F16toF16_2D =
{
    VX_KERNEL_ENUM_SPATIAL_TRANSFORMER,
    VX_KERNEL_NAME_TRANSFORM_INTERP_F16TOF16_2D,
    NULL,
    vxTransform_InterPKernelParam,
    (sizeof(vxTransform_InterPKernelParam) / sizeof(vxTransform_InterPKernelParam[0])),
    vxValidator,
    NULL,
    NULL,
    vxTransform_InterPInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTransform_InterPKernelInfo_F16toF16 =
{
    VX_KERNEL_ENUM_SPATIAL_TRANSFORMER,
    VX_KERNEL_NAME_TRANSFORM_INTERP_F16TOF16,
    NULL,
    vxTransform_InterPKernelParam,
    (sizeof(vxTransform_InterPKernelParam) / sizeof(vxTransform_InterPKernelParam[0])),
    vxValidator,
    NULL,
    NULL,
    vxTransform_InterPInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_SPATIAL_TRANSFORMER_list[] =
{
    &vxSpatial_transformer_CPU,
    &vxTransform_setupThresKernelInfo_F16toF16,
    &vxTransform_GemmKernelInfo_F16toF16,
    &vxTransform_InterPKernelInfo_F16toF16_2D,
    &vxTransform_InterPKernelInfo_F16toF16,
    NULL
};
#ifdef __cplusplus
}
#endif
