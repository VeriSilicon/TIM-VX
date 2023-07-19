/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

#define SOURCE_AXIS0_0     "layer_normalization_0"
#define SOURCE_AXIS0_1     "layer_normalization_1"
#define SOURCE_AXIS0_2     "layer_normalization_2"
#define SOURCE_AXIS0_3     "layer_normalization_3"
#define SOURCE_AXIS01      "layer_normalization_axis01"

#define HASH_LAYERNORM_SH_KERNEL_NAME(SRC0_TYPE, SCALE_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layer_norm_axis0_"#SRC0_TYPE"_"#SCALE_TYPE"to"#DST_TYPE)

#define HASH_LAYERNORM_SH_KERNEL_2D_NAME(SRC0_TYPE, SCALE_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layer_norm_axis0_"#SRC0_TYPE"_"#SCALE_TYPE"to"#DST_TYPE"_2D")

#define HASH_LAYERNORM_SH_KERNEL_SCALE_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layer_norm_"#SRC0_TYPE"F32to"#DST_TYPE)

#define HASH_LAYERNORM_SH_KERNEL_SCALE_2D_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layer_norm_"#SRC0_TYPE"F32to"#DST_TYPE"_2D")

// normalization
#define HASH_LAYERNORM_KEY(_input0_type, _input2_type, _output_type, _reshape_flag) \
    ((_input0_type << 24) | (_input2_type << 16) | (_output_type << 8) | _reshape_flag)

#define LAYERNORM_KERNELS_3D(IN0_TYPE, SCALE_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, SCALE_TYPE, OUT_TYPE, 0), \
        HASH_LAYERNORM_SH_KERNEL_NAME(IN0_TYPE, SCALE_TYPE, OUT_TYPE), \
        SOURCE },

#define LAYERNORM_KERNELS_2D(IN0_TYPE, SCALE_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, SCALE_TYPE, OUT_TYPE, 1), \
        HASH_LAYERNORM_SH_KERNEL_2D_NAME(IN0_TYPE, SCALE_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_LAYERNORM_SCALE_KERNELS(IN0_TYPE, SCALE_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, F32, OUT_TYPE, 0), \
        HASH_LAYERNORM_SH_KERNEL_SCALE_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_LAYERNORM_SCALE_KERNELS_2D(IN0_TYPE, SCALE_TYPE, OUT_TYPE, SOURCE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, F32, OUT_TYPE, 1), \
        HASH_LAYERNORM_SH_KERNEL_SCALE_2D_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

// layer norm on aix 0 and 1

#define HASH_LN_AXIS01_SUMS_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layernorm_axis01_sums_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_LN_AXIS01_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.layernorm_axis01_"#SRC0_TYPE"_"#SRC1_TYPE"to"#DST_TYPE)

#define LN_AXIS01_SUMS_KERNELS(IN0_TYPE, OUT_TYPE) \
    {   HASH_LAYERNORM_KEY(IN0_TYPE, U4, OUT_TYPE, 0), \
        HASH_LN_AXIS01_SUMS_SH_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE_AXIS01 },

#define LAYERNORM_AXIS01_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE) \
    { HASH_LAYERNORM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
      HASH_LN_AXIS01_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
      SOURCE_AXIS01 },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _layernorm_kernel_map[] =
{
    // Register kernel here
    LAYERNORM_KERNELS_3D( U8,  F16, U8,  SOURCE_AXIS0_0 )
    LAYERNORM_KERNELS_2D( U8,  F16, U8,  SOURCE_AXIS0_0 )
    LAYERNORM_KERNELS_3D( U8,  F16, F16, SOURCE_AXIS0_0 )
    LAYERNORM_KERNELS_2D( U8,  F16, F16, SOURCE_AXIS0_0 )
    LAYERNORM_KERNELS_3D( I8,  F16, I8,  SOURCE_AXIS0_0 )
    LAYERNORM_KERNELS_2D( I8,  F16, I8,  SOURCE_AXIS0_0 )
    LAYERNORM_KERNELS_3D( I8,  F16, F16, SOURCE_AXIS0_0 )
    LAYERNORM_KERNELS_2D( I8,  F16, F16, SOURCE_AXIS0_0 )

    LAYERNORM_KERNELS_3D( F16, F16, F16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( F16, F16, F16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_3D( I16, F16, I16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( I16, F16, I16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_3D( F16, F16, I16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( F16, F16, I16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_3D( F16, F16, I8,  SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( F16, F16, I8,  SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_3D( F16, F16, U8,  SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( F16, F16, U8,  SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_3D( I16, F16, F16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( I16, F16, F16, SOURCE_AXIS0_1 )

    LAYERNORM_KERNELS_3D( F16, F32, F16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( F16, F32, F16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_3D( I16, F32, I16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( I16, F32, I16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_3D( F16, F32, I16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( F16, F32, I16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_3D( F16, F32, I8,  SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( F16, F32, I8,  SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_3D( F16, F32, U8,  SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( F16, F32, U8,  SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_3D( I16, F32, F16, SOURCE_AXIS0_1 )
    LAYERNORM_KERNELS_2D( I16, F32, F16, SOURCE_AXIS0_1 )

    LAYERNORM_KERNELS_3D( U8,  F32, U8,  SOURCE_AXIS0_2 )
    LAYERNORM_KERNELS_2D( U8,  F32, U8,  SOURCE_AXIS0_2 )
    LAYERNORM_KERNELS_3D( U8,  F32, F16, SOURCE_AXIS0_2 )
    LAYERNORM_KERNELS_2D( U8,  F32, F16, SOURCE_AXIS0_2 )
    LAYERNORM_KERNELS_3D( I8,  F32, I8,  SOURCE_AXIS0_2 )
    LAYERNORM_KERNELS_2D( I8,  F32, I8,  SOURCE_AXIS0_2 )
    LAYERNORM_KERNELS_3D( I8,  F32, F16, SOURCE_AXIS0_2 )
    LAYERNORM_KERNELS_2D( I8,  F32, F16, SOURCE_AXIS0_2 )

    TENSOR_LAYERNORM_SCALE_KERNELS( BF16,  F32, BF16, SOURCE_AXIS0_3 )
    TENSOR_LAYERNORM_SCALE_KERNELS_2D( BF16,  F32, BF16, SOURCE_AXIS0_3 )
};

static const _kernel_map_type _layernorm_axis01_kernel_map[] =
{
    // Register kernel here
    LN_AXIS01_SUMS_KERNELS( I8,  F32 )
    LN_AXIS01_SUMS_KERNELS( U8,  F32 )
    LN_AXIS01_SUMS_KERNELS( F16, F32 )
    LN_AXIS01_SUMS_KERNELS( I16, F32 )

    LAYERNORM_AXIS01_KERNELS( U8,  F16, U8 )
    LAYERNORM_AXIS01_KERNELS( U8,  F16, F16 )
    LAYERNORM_AXIS01_KERNELS( I8,  F16, I8 )
    LAYERNORM_AXIS01_KERNELS( I8,  F16, F16 )
    LAYERNORM_AXIS01_KERNELS( F16, F16, F16 )
    LAYERNORM_AXIS01_KERNELS( F16, F16, I16 )
    LAYERNORM_AXIS01_KERNELS( F16, F16, I8 )
    LAYERNORM_AXIS01_KERNELS( F16, F16, U8 )
    LAYERNORM_AXIS01_KERNELS( I16, F16, I16 )
    LAYERNORM_AXIS01_KERNELS( I16, F16, F16 )

    LAYERNORM_AXIS01_KERNELS( U8,  F32, U8 )
    LAYERNORM_AXIS01_KERNELS( U8,  F32, F16 )
    LAYERNORM_AXIS01_KERNELS( I8,  F32, I8 )
    LAYERNORM_AXIS01_KERNELS( I8,  F32, F16 )
    LAYERNORM_AXIS01_KERNELS( F16, F32, F16 )
    LAYERNORM_AXIS01_KERNELS( F16, F32, I16 )
    LAYERNORM_AXIS01_KERNELS( F16, F32, I8 )
    LAYERNORM_AXIS01_KERNELS( F16, F32, U8 )
    LAYERNORM_AXIS01_KERNELS( I16, F32, I16 )
    LAYERNORM_AXIS01_KERNELS( I16, F32, F16 )

};

/*
 * Kernel params
 */

static vx_param_description_t _layernorm_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _layernorm_axis01_sums_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _layernorm_axis01_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

#define _LAYERNORM_PARAM_NUM    _cnt_of_array( _layernorm_kernel_param_def )
#define _LAYERNORM_SUMS_PARAM_NUM       _cnt_of_array( _layernorm_axis01_sums_param_def )
#define _LAYERNORM_AXIS01_PARAM_NUM     _cnt_of_array( _layernorm_axis01_kernel_param_def )

/*
 * Kernel initializer
 */

DEF_KERNEL_INITIALIZER(_layernorm_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t* attr[3] = {NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    float output_scale = 1;
    float output_zp = 0;
    float inv_multiplier = 0;
    int32_t height = 0, width = 0, chn = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    input_shape  = attr[0]->shape;

    output_scale = 1.0f / attr[2]->scale;
    output_zp = (float)attr[2]->zero_point;

    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->data[1]);
    chn = (int32_t)((input_shape->size <= 2) ? 1 : input_shape->data[2]);

    inv_multiplier = 1.0f / (float)width;

    shaderParam.global_scale[0]  = width;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = 1;
    shaderParam.global_size[1]   = height;
    shaderParam.global_size[2]   = chn;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniDataToFP32_0_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniDataToFP32_1_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniDataToFP32_2_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00090008, 0x000b000a, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniDataToFP32_3_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x000d000c, 0x000f000e, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractInteger_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSumX_16x1 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001,
            0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSumX2_16x1 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0x55555555, // BSelt
            0x76543210, 0xfedcba98, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSum_X_X2_8x2 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x0000aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01050004, 0x03070206, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniExtractOddData_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        uint32_t pack_key      = 0;
#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (IN1_TYPE << 16) | (OUT_TYPE << 8))

        pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype, attr[2]->dtype );

        status  = vsi_nn_kernel_gpu_add_param(node, "width", &width);
        status |= vsi_nn_kernel_gpu_add_param(node, "inv_multiplier", &inv_multiplier);
        CHECK_STATUS_FAIL_GOTO(status, OnError );

        switch( pack_key )
        {
            case _PACK_SELECT_KEY( U8, F16, F16 ):
            case _PACK_SELECT_KEY( U8, F32, F16 ):
            case _PACK_SELECT_KEY( I8, F16, F16 ):
            case _PACK_SELECT_KEY( I8, F32, F16 ):
            case _PACK_SELECT_KEY( U8, F16, U8 ):
            case _PACK_SELECT_KEY( U8, F32, U8 ):
            case _PACK_SELECT_KEY( I8, F16, I8 ):
            case _PACK_SELECT_KEY( I8, F32, I8 ):
                {
                    if (attr[2]->dtype == F16)
                    {
                        status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8",
                            &uniExtractHalf8_2x8);
                    }
                    else
                    {
                        status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8",
                            &uniExtractInteger_2x8);
                        status |= vsi_nn_kernel_gpu_add_param(node, "output_scale", &output_scale);
                        status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                    }
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniSumX_16x1", &uniSumX_16x1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniSumX2_16x1", &uniSumX2_16x1);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_0_4x4",
                        &uniDataToFP32_0_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_1_4x4",
                        &uniDataToFP32_1_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_2_4x4",
                        &uniDataToFP32_2_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_3_4x4",
                        &uniDataToFP32_3_4x4);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( I16, F16, F16 ):
            case _PACK_SELECT_KEY( I16, F32, F16 ):
            case _PACK_SELECT_KEY( F16, F16, F16 ):
            case _PACK_SELECT_KEY( F16, F32, F16 ):
            case _PACK_SELECT_KEY( I16, F16, I16 ):
            case _PACK_SELECT_KEY( I16, F32, I16 ):
            case _PACK_SELECT_KEY( F16, F16, I16 ):
            case _PACK_SELECT_KEY( F16, F32, I16 ):
            case _PACK_SELECT_KEY( F16, F16, U8 ):
            case _PACK_SELECT_KEY( F16, F32, U8 ):
            case _PACK_SELECT_KEY( F16, F16, I8 ):
            case _PACK_SELECT_KEY( F16, F32, I8 ):
                {
                    if (attr[2]->dtype == F16)
                    {
                        status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8",
                            &uniExtractHalf8_2x8);
                    }
                    else
                    {
                        status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8",
                            &uniExtractInteger_2x8);
                    }
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniSum_X_X2_8x2", &uniSum_X_X2_8x2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_0_4x4",
                        &uniDataToFP32_0_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_1_4x4",
                        &uniDataToFP32_1_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_scale", &output_scale);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( BF16, F32, BF16 ):
                {
                    status  = vsi_nn_kernel_gpu_add_param( node,
                                "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                                "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                                "uniExtractOddData_2x8", &uniExtractOddData_2x8 );
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            default:
                VSI_ASSERT( FALSE );
                return VSI_FAILURE;
        }
#undef _PACK_SELECT_KEY
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }

    return status;
}

DEF_KERNEL_INITIALIZER(_layernorm_axis01_sums_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t* attr[2] = {NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    int32_t width = 0;
    int32_t height = 0;
    int32_t chn = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );

    input_shape  = attr[0]->shape;

    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->data[1]);
    chn = (int32_t)(attr[1]->shape->data[1]);

    shaderParam.global_scale[0]  = 1;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.local_size[0]  = 16;
    shaderParam.local_size[1]  = 1;
    shaderParam.local_size[2]  = 1;

    if (attr[0]->dtype == I8 || attr[0]->dtype == U8)
    {
        shaderParam.global_size[0]   = (width + 255) / 256 * 16;
    }
    else if (attr[0]->dtype == I16 || attr[0]->dtype == F16)
    {
        shaderParam.global_size[0]   = (width + 127) / 128 * 16;
    }
    shaderParam.global_size[1]   = chn;
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    if (attr[0]->dtype == U8 || attr[0]->dtype == I8)
    {
        gpu_dp_inst_t uniSumX_16x1 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSumX2_16x1 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0x55555555, // BSelt
            0x76543210, 0xfedcba98, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status  = vsi_nn_kernel_gpu_add_param(node, "uniSumX_16x1", &uniSumX_16x1);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniSumX2_16x1", &uniSumX2_16x1);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }
    else if (attr[0]->dtype == I16 || attr[0]->dtype == F16)
    {
        gpu_dp_inst_t uniSum_X_X2_8x2 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x0000aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status  = vsi_nn_kernel_gpu_add_param(node, "uniSum_X_X2_8x2", &uniSum_X_X2_8x2);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

    status = vsi_nn_kernel_gpu_add_param(node, "width", &width);
    status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }

    return status;
}

DEF_KERNEL_INITIALIZER(_layernorm_axis01_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t* attr[3] = {NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    float output_scale = 1.0f;
    float output_zp = 0;
    float inv_multiplier = 0;
    vx_uint32 group_num = 0;
    vx_int32 height = 0, width = 0, chn = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[4] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    input_shape  = attr[0]->shape;
    output_scale = 1.0f / attr[2]->scale;
    output_zp = (float)attr[2]->zero_point;

    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->data[1]);
    chn = (int32_t)(attr[1]->shape->data[1]);

    inv_multiplier = (float)(1.0 / (width * height));

    group_num = (width + 255) / 256;
    if (attr[0]->dtype == I16 || attr[0]->dtype == F16)
    {
        group_num = (width + 127) / 128;
    }

    shaderParam.global_scale[0]  = 8;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
                                        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = (chn + shaderParam.global_scale[1] - 1)
        / shaderParam.global_scale[1];
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniDataToFP32_0_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniDataToFP32_1_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractInteger_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };


        status  = vsi_nn_kernel_gpu_add_param(node, "height", &height);
        status |= vsi_nn_kernel_gpu_add_param(node, "inv_multiplier", &inv_multiplier);
        status |= vsi_nn_kernel_gpu_add_param(node, "group_num", &group_num);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_0_4x4", &uniDataToFP32_0_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_1_4x4", &uniDataToFP32_1_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
        status |= vsi_nn_kernel_gpu_add_param(node, "output_scale", &output_scale);
        if (attr[2]->dtype == F16)
        {
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniExtract8Data_2x8", &uniExtractHalf8_2x8);
        }
        else
        {
            status |= vsi_nn_kernel_gpu_add_param(
                node, "uniExtract8Data_2x8", &uniExtractInteger_2x8);
        }
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }

    return status;
}

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    int32_t is_img2d_input
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input2_dtype = F16;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input2_dtype = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_LAYERNORM_KEY( input0_dtype, input2_dtype, output_dtype, is_img2d_input );

    for( i = 0; i < _cnt_of_array(_layernorm_kernel_map); i ++ )
    {
        if ( _layernorm_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_layernorm_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _layernorm_kernel_map[i].function_name );
        kernel->info.parameters = _layernorm_kernel_param_def;
        kernel->info.numParams = _LAYERNORM_PARAM_NUM;
        kernel->info.initialize = _layernorm_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _layernorm_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _layernorm_kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

static vsi_status _query_kernel_axis01
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel_sums,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input2_dtype = F16;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input2_dtype = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_LAYERNORM_KEY( input0_dtype, U4, F32, 0 );

    for( i = 0; i < _cnt_of_array(_layernorm_axis01_kernel_map); i ++ )
    {
        if ( _layernorm_axis01_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_layernorm_axis01_kernel_map) )
    {
        snprintf( kernel_sums->info.name, VX_MAX_KERNEL_NAME, "%s",  _layernorm_axis01_kernel_map[i].function_name );
        kernel_sums->info.parameters = _layernorm_axis01_sums_param_def;
        kernel_sums->info.numParams = _LAYERNORM_SUMS_PARAM_NUM;
        kernel_sums->info.initialize = _layernorm_axis01_sums_initializer;

        vsi_nn_kernel_add_source( kernel_sums, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _layernorm_axis01_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_sums, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _layernorm_axis01_kernel_map[i].source_name );
    }

    key = HASH_LAYERNORM_KEY( input0_dtype, input2_dtype, output_dtype, 0 );

    for ( i = 0; i < _cnt_of_array(_layernorm_axis01_kernel_map); i ++ )
    {
        if ( _layernorm_axis01_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_layernorm_axis01_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _layernorm_axis01_kernel_map[i].function_name );
        kernel->info.parameters = _layernorm_axis01_kernel_param_def;
        kernel->info.numParams = _LAYERNORM_AXIS01_PARAM_NUM;
        kernel->info.initialize = _layernorm_axis01_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _layernorm_axis01_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _layernorm_axis01_kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel_axis01() */

static vsi_nn_kernel_node_t _setup_axis01
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
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL, rs_gamma = NULL, rs_beta = NULL;
    vsi_nn_kernel_node_param_t sums_node_params[_LAYERNORM_SUMS_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t node_params[_LAYERNORM_AXIS01_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t sums_node = NULL;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    float input_scale = vsi_nn_get_tensor_scale(inputs[0]);
    vsi_nn_kernel_t * kernel_sums = NULL;
    vsi_nn_tensor_t * tensor_sums = NULL;
    float eps  = vsi_nn_kernel_param_get_float32( params, "eps" ) /
                (input_scale * input_scale);
    int32_t axis[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t axis_num  = 1;
    int32_t new_axis[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t new_shape[2][VSI_NN_MAX_DIM_NUM] = {{ 1, 1, 1, 1 }};
    uint32_t axis_size = 0;
    uint32_t rank_in = 0, rank_para = 0;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    status = vsi_nn_kernel_optimize_tensor_shape(
        inputs[0]->attr.size, inputs[0]->attr.dim_num,
        axis, axis_num, new_shape[0], &rank_in, new_axis, &axis_size);
    if ( status == FALSE || axis_size > 2)
    {
        return NULL;
    }

    status = vsi_nn_kernel_optimize_tensor_shape(
        inputs[1]->attr.size, inputs[1]->attr.dim_num,
        axis, axis_num, new_shape[1], &rank_para, new_axis, &axis_size);
    if ( status == FALSE || axis_size > 2)
    {
        return NULL;
    }

    rs_input = vsi_nn_kernel_tensor_reshape(inputs[0]->t, new_shape[0], rank_in);

    rs_beta = vsi_nn_kernel_tensor_reshape(inputs[1]->t, new_shape[1], rank_para);

    rs_gamma = vsi_nn_kernel_tensor_reshape(inputs[2]->t, new_shape[1], rank_para);

    rs_output = vsi_nn_kernel_tensor_reshape(outputs[0]->t, new_shape[0], rank_in);

    kernel_sums = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
    CHECK_PTR_FAIL_GOTO( kernel_sums, "Create kernel fail.", final );
    // Assign unique_id
    kernel_sums->unique_id = kernel->unique_id;

    memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = ((new_shape[0][0] + 255) / 256) * 4;

    if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT16
        || inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16)
    {
        attr.size[0] = ((new_shape[0][0] + 127) / 128) * 4;
    }
    attr.size[1] = new_shape[0][2];
    attr.size[2] = 1;
    attr.size[3] = new_shape[0][3];
    attr.dim_num = rank_in;
    tensor_sums = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( tensor_sums, "Create tensor fail.", final );

    status = _query_kernel_axis01(inputs, outputs, kernel_sums, kernel);
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }

    /*
    ** sum(x) and sumsq(x*x)
    */
    sums_node = vsi_nn_kernel_create_node(graph, kernel_sums);
    CHECK_PTR_FAIL_GOTO( sums_node, "Create kernel fail.", final );
    if (sums_node)
    {
        sums_node_params[0] = rs_input;
        sums_node_params[1] = (vsi_nn_kernel_node_param_t)tensor_sums->t;

        status = vsi_nn_kernel_node_pass_param(
            sums_node, sums_node_params, _LAYERNORM_SUMS_PARAM_NUM);
        CHECK_STATUS(status);
        {
            // Set default border mode.
            vx_border_t border;
            border.mode = VX_BORDER_CONSTANT;
            border.constant_value.U16 = 0;
            status = vxSetNodeAttribute(
                (vx_node)sums_node, VX_NODE_BORDER, &border, sizeof(border));
            CHECK_STATUS(status);
        }
    }

    node = vsi_nn_kernel_create_node( graph, kernel );
    CHECK_PTR_FAIL_GOTO( node, "Create kernel fail.", final );
    if (node)
    {
        uint32_t index = 0;
        node_params[index++] = rs_input;
        node_params[index++] = rs_beta;
        node_params[index++] = rs_gamma;
        node_params[index++] = (vsi_nn_kernel_node_param_t)tensor_sums->t;
        node_params[index++] = rs_output;
        node_params[index++] = vsi_nn_kernel_scalar_create(graph, F32, &eps);

        status = vsi_nn_kernel_node_pass_param(
            node, node_params, _LAYERNORM_AXIS01_PARAM_NUM);
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release(&node_params[5]);
    }


final:
    if (rs_beta)
    {
        vsi_nn_kernel_tensor_release( &rs_beta );
    }
    if (rs_gamma)
    {
        vsi_nn_kernel_tensor_release( &rs_gamma );
    }
    if (rs_input)
    {
        vsi_nn_kernel_tensor_release( &rs_input );
    }
    if (rs_output)
    {
        vsi_nn_kernel_tensor_release( &rs_output );
    }
    if ( kernel_sums )
    {
        vsi_nn_kernel_release( &kernel_sums );
    }
    if ( tensor_sums )
    {
        vsi_nn_ReleaseTensor( &tensor_sums );
    }
    if (sums_node) {vsi_nn_kernel_node_release( &sums_node );}

    return node;
}

static vsi_nn_kernel_node_t _setup_axis0
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
    vsi_nn_kernel_node_param_t node_params[_LAYERNORM_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    float input_scale = vsi_nn_get_tensor_scale(inputs[0]);
    vsi_nn_tensor_t* rs_tensors[4] = { NULL };
    float eps  = vsi_nn_kernel_param_get_float32( params, "eps" ) /
                (input_scale * input_scale);
    int32_t axis[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t axis_num  = 1;
    int32_t new_axis[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t new_shape[2][VSI_NN_MAX_DIM_NUM] = {{ 1, 1, 1, 1 }};
    uint32_t axis_size = 0;
    uint32_t rank_in = 0;
    int32_t is_img2d_input = 0;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    status = vsi_nn_kernel_optimize_tensor_shape(
        inputs[0]->attr.size, inputs[0]->attr.dim_num,
        axis, axis_num, new_shape[0], &rank_in, new_axis, &axis_size);
    if ( status == FALSE)
    {
        return NULL;
    }

    is_img2d_input = rank_in < 3 || (new_shape[0][2] == 1);

    status = _query_kernel( inputs, outputs, kernel, is_img2d_input);
    if (VSI_SUCCESS != status)
    {
        goto final;
    }

    new_shape[1][0] = new_shape[0][0];
    new_shape[1][1] = 1;
    rs_tensors[0] = vsi_nn_reshape_tensor(graph, inputs[0], new_shape[0], rank_in);
    rs_tensors[1] = vsi_nn_reshape_tensor(graph, inputs[1], new_shape[1], 2);
    rs_tensors[2] = vsi_nn_reshape_tensor(graph, inputs[2], new_shape[1], 2);
    rs_tensors[3] = vsi_nn_reshape_tensor(graph, outputs[0], new_shape[0], rank_in);

    // Nomalization
    node = vsi_nn_kernel_create_node( graph, kernel );
    if (node)
    {
       vsi_nn_kernel_node_pack_io(node_params, _LAYERNORM_PARAM_NUM,
           rs_tensors, 3, &rs_tensors[3], 1);
        node_params[4] = vsi_nn_kernel_scalar_create( graph, F32, &eps );

        status  = vsi_nn_kernel_node_pass_param( node, node_params,
            _LAYERNORM_PARAM_NUM );
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release( &node_params[4] );
        {
            // Set default border mode.
            vx_border_t border;
            border.mode = VX_BORDER_CONSTANT;
            border.constant_value.U16 = 0;
            status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
            CHECK_STATUS(status);
        }
    }

    /* Pass parameters to node. */
final:
    vsi_safe_release_tensor(rs_tensors[0]);
    vsi_safe_release_tensor(rs_tensors[1]);
    vsi_safe_release_tensor(rs_tensors[2]);
    vsi_safe_release_tensor(rs_tensors[3]);

    return node;
} /* _setup() */

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
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t *input_size = inputs[0]->attr.size;

    if (input_size[0] >= GPU_TENSOR_MAX_WIDTH)
    {
       node = _setup_axis01(graph, inputs, input_num, outputs, output_num, params, kernel);
    }
    else
    {
        node = _setup_axis0(graph, inputs, input_num, outputs, output_num, params, kernel);
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( layer_norm, _setup )
