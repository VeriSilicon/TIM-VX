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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

#define VX_KERNEL_NAME_PRE_PROCESS_GRAY_SCALE_U8TOU8      CVIVANTE_NAMESPACE("evis.pre_process_gray_scale_U8toU8")
#define VX_KERNEL_NAME_PRE_PROCESS_GRAY_SCALE_U8TOI8      CVIVANTE_NAMESPACE("evis.pre_process_gray_scale_U8toI8")
#define VX_KERNEL_NAME_PRE_PROCESS_GRAY_SCALE_U8TOI16     CVIVANTE_NAMESPACE("evis.pre_process_gray_scale_U8toI16")
#define VX_KERNEL_NAME_PRE_PROCESS_GRAY_SCALE_U8TOF16     CVIVANTE_NAMESPACE("evis.pre_process_gray_scale_U8toF16")
#define VX_KERNEL_NAME_PRE_PROCESS_GRAY_COPY_U8TOU8       CVIVANTE_NAMESPACE("evis.pre_process_gray_copy_U8toU8")
#define VX_KERNEL_NAME_PRE_PROCESS_GRAY_COPY_U8TOI8       CVIVANTE_NAMESPACE("evis.pre_process_gray_copy_U8toI8")
#define VX_KERNEL_NAME_PRE_PROCESS_GRAY_COPY_U8TOI16      CVIVANTE_NAMESPACE("evis.pre_process_gray_copy_U8toI16")
#define VX_KERNEL_NAME_PRE_PROCESS_GRAY_COPY_U8TOF16      CVIVANTE_NAMESPACE("evis.pre_process_gray_copy_U8toF16")

#define KERNEL_SOURCE_1    "pre_process_gray",
#define KERNEL_SOURCE_2    "pre_process_gray_copy"

typedef enum
{
    COPY = 0,
    SCALE
} vsi_nn_gray_convert_type_e;

#define HASH_PRE_PROCESS_GRAY_KEY(_input0_type, _output_type, _convert_type, _image_2d) \
    ((_input0_type << 24) | (_output_type << 16) | (_convert_type << 8) | (_image_2d))

#define TENSOR_PRE_PROCESS_GRAY_KERNELS(IN0_TYPE, OUT_TYPE, CONVERT_TYPE, SOURCE) \
    { HASH_PRE_PROCESS_GRAY_KEY(IN0_TYPE, OUT_TYPE, CONVERT_TYPE, 0), \
        VX_KERNEL_NAME_PRE_PROCESS_GRAY_##CONVERT_TYPE##_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } pre_process_gray_map[] =
{
    TENSOR_PRE_PROCESS_GRAY_KERNELS(U8, U8,  SCALE,        KERNEL_SOURCE_1)
    TENSOR_PRE_PROCESS_GRAY_KERNELS(U8, I8,  SCALE,        KERNEL_SOURCE_1)
    TENSOR_PRE_PROCESS_GRAY_KERNELS(U8, I16, SCALE,        KERNEL_SOURCE_1)
    TENSOR_PRE_PROCESS_GRAY_KERNELS(U8, F16, SCALE,        KERNEL_SOURCE_1)
    TENSOR_PRE_PROCESS_GRAY_KERNELS(U8, U8,  COPY,         KERNEL_SOURCE_2)
    TENSOR_PRE_PROCESS_GRAY_KERNELS(U8, I8,  COPY,         KERNEL_SOURCE_2)
    TENSOR_PRE_PROCESS_GRAY_KERNELS(U8, I16, COPY,         KERNEL_SOURCE_2)
    TENSOR_PRE_PROCESS_GRAY_KERNELS(U8, F16, COPY,         KERNEL_SOURCE_2)
};

static vx_param_description_t vxPreProcessGrayKernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _EVIS_PRE_PROCESS_GRAY_PARAM_NUM          _cnt_of_array(vxPreProcessGrayKernel_param_def)

DEF_KERNEL_INITIALIZER(_pre_process_gray_copy_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    float       dstZP      = 0;
    float       outputScale   = 1;
    uint32_t    width      = 0;
    uint32_t    height     = 0;

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_int_array_t * out_shape = NULL;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );

    out_shape  = attr[0]->shape;
    dstZP      = (float)attr[0]->asymm.zero_point;
    outputScale   = attr[0]->asymm.scale;
    width      = out_shape->data[0];
    height     = out_shape->data[1];

    if( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[0]->dfp.fl > 0)
        {
            outputScale = (float)((int64_t)1 << attr[0]->dfp.fl);
        }
        else
        {
            outputScale = (1.0f / (float)((int64_t)1 << -attr[0]->dfp.fl));
        }
        dstZP = 0.0f;
    }
    else if(attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        outputScale = 1.0f/outputScale;
    }
    else if( attr[0]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        outputScale = 1;
        dstZP = 0.0f;
    }

    shaderParam.global_scale[0]  = 16;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = height;
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniDataMeanStddevLo_2x8 = {{
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniDataMeanStddevHi_2x8 = {{
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param(node, "uniDataMeanStddevLo_2x8", &uniDataMeanStddevLo_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniDataMeanStddevHi_2x8", &uniDataMeanStddevHi_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &outputScale);
        status |= vsi_nn_kernel_gpu_add_param(node, "outputZP", &dstZP);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _pre_process_gray_copy_initializer() */

DEF_KERNEL_INITIALIZER(_pre_process_gray_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    float       dstZP      = 0;
    float       outputScale   = 1;
    uint32_t    width      = 0;
    uint32_t    height     = 0;

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_int_array_t * out_shape = NULL;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );

    out_shape  = attr[0]->shape;
    dstZP      = (float)attr[0]->asymm.zero_point;
    outputScale   = attr[0]->asymm.scale;
    width      = out_shape->data[0];
    height     = out_shape->data[1];

    if( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[0]->dfp.fl > 0)
        {
            outputScale = (float)((int64_t)1 << attr[0]->dfp.fl);
        }
        else
        {
            outputScale = (1.0f / (float)((int64_t)1 << -attr[0]->dfp.fl));
        }
        dstZP = 0.0f;
    }
    else if(attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        outputScale = 1.0f/outputScale;
    }
    else if( attr[0]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        outputScale = 1;
        dstZP = 0.0f;
    }

    shaderParam.global_scale[0]  = 4;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = height;
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniVecShift10 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniAddRShift = {{
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002405, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGetTempVal = {{
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00230001, 0x00670045, // ABin
            0x05050505, // BSelt
            0x00110000, 0x00330022, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractBytes = {{
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002414, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniDataMulAlpha_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x01010101, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniDataSubMean_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertIntergetoF32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtactInteger_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002300, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param(node, "uniVecShift10", &uniVecShift10);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniAddRShift", &uniAddRShift);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniGetTempVal", &uniGetTempVal);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractBytes", &uniExtractBytes);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertIntergetoF32_4x4", &uniConvertIntergetoF32_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniExtactInteger_2x8", &uniExtactInteger_2x8);

        status |= vsi_nn_kernel_gpu_add_param(node, "outputZP", &dstZP);
        status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &outputScale);

        if(attr[0]->dtype == F16)
        {
            status |= vsi_nn_kernel_gpu_add_param(node, "uniDataMulAlpha_4x4", &uniDataMulAlpha_4x4);
            status |= vsi_nn_kernel_gpu_add_param(node, "uniDataSubMean_4x4", &uniDataSubMean_4x4);
        }
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _pre_process_gray_copy_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    const vsi_nn_kernel_param_t * params
    )
{
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    vsi_nn_gray_convert_type_e convert_type = SCALE;
    vsi_status status = VSI_FAILURE;
    uint32_t key = 0;
    int i = 0;
    vsi_bool enable_copy  = vsi_nn_kernel_param_get_int32( params, "enable_copy" );

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if(enable_copy)
    {
        convert_type = COPY;
    }
    else
    {
        convert_type = SCALE;
    }

    key = HASH_PRE_PROCESS_GRAY_KEY( input0_dtype, output_dtype, convert_type, 0 );

    for( i = 0; i < _cnt_of_array(pre_process_gray_map); i ++ )
    {
        if( pre_process_gray_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(pre_process_gray_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  pre_process_gray_map[i].function_name );
        kernel->info.parameters = vxPreProcessGrayKernel_param_def;
        kernel->info.numParams = _cnt_of_array( vxPreProcessGrayKernel_param_def );

        if(enable_copy)
        {
            kernel->info.initialize = _pre_process_gray_copy_initializer;
        }
        else
        {
            kernel->info.initialize = _pre_process_gray_initializer;
        }
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                pre_process_gray_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                pre_process_gray_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t tmp_params[_EVIS_PRE_PROCESS_GRAY_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( inputs, outputs, kernel, params );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            uint32_t index = 2;
            int32_t scale_x  = vsi_nn_kernel_param_get_int32( params, "scale_x" );
            int32_t scale_y  = vsi_nn_kernel_param_get_int32( params, "scale_y" );
            int32_t left     = vsi_nn_kernel_param_get_int32( params, "left" );
            int32_t top      = vsi_nn_kernel_param_get_int32( params, "top" );
            float mean       = vsi_nn_kernel_param_get_float32( params, "mean" );
            float scale      = vsi_nn_kernel_param_get_float32( params, "scale" );

            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( tmp_params, _EVIS_PRE_PROCESS_GRAY_PARAM_NUM,
                inputs, 1, outputs, 1 );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_x );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_y );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &left );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &top );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &mean );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &scale );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _EVIS_PRE_PROCESS_GRAY_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &tmp_params[2] );
            vsi_nn_kernel_scalar_release( &tmp_params[3] );
            vsi_nn_kernel_scalar_release( &tmp_params[4] );
            vsi_nn_kernel_scalar_release( &tmp_params[5] );
            vsi_nn_kernel_scalar_release( &tmp_params[6] );
            vsi_nn_kernel_scalar_release( &tmp_params[7] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( pre_process_gray, _setup )

