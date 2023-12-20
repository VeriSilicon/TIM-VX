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
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

#define HASH_LOG_SOFTMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, _image_2d) \
    ((AXIS << 20) | (IN_DTYPE << 12) | (OUT_DTYPE << 4) | (_image_2d))

 #define HASH_LOG_SOFTMAX_KERNEL_SOURCE_NAME(_suffix) \
    "log_softmax_axis"#_suffix

#define HASH_LOG_SOFTMAX_KERNELS( AXIS, IN_DTYPE, OUT_DTYPE, _suffix) \
        { HASH_LOG_SOFTMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 0), \
        CVIVANTE_NAMESPACE("evis.log_softmax_axis"#AXIS"_"#IN_DTYPE"to"#OUT_DTYPE), \
        HASH_LOG_SOFTMAX_KERNEL_SOURCE_NAME(_suffix) },

#define HASH_LOG_SOFTMAX_KERNELS_2D( AXIS, IN_DTYPE, OUT_DTYPE, _suffix ) \
        { HASH_LOG_SOFTMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 1), \
        CVIVANTE_NAMESPACE("evis.log_softmax_axis"#AXIS"_"#IN_DTYPE"to"#OUT_DTYPE"_2D"), \
        HASH_LOG_SOFTMAX_KERNEL_SOURCE_NAME(_suffix) },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } _log_softmax_evis_kernel_map[] =
{
    HASH_LOG_SOFTMAX_KERNELS(0, F16,  F16,  0)
    HASH_LOG_SOFTMAX_KERNELS(0, F16,  I16,  0)
    HASH_LOG_SOFTMAX_KERNELS(0, F16,  U8,   0)
    HASH_LOG_SOFTMAX_KERNELS(0, F16,  I8,   0)
    HASH_LOG_SOFTMAX_KERNELS(0, I16,  I16,  0)
    HASH_LOG_SOFTMAX_KERNELS(0, I16,  F16,  0)
    HASH_LOG_SOFTMAX_KERNELS(0, BF16, BF16, 0_BF16)
    HASH_LOG_SOFTMAX_KERNELS(0, BF16, F32,  0_BF16)
    HASH_LOG_SOFTMAX_KERNELS(0, BF16, F16,  0_BF16)
    HASH_LOG_SOFTMAX_KERNELS(0, U8,   U8,   0)
    HASH_LOG_SOFTMAX_KERNELS(0, U8,   F16,  0)
    HASH_LOG_SOFTMAX_KERNELS(0, I8,   I8,   0)
    HASH_LOG_SOFTMAX_KERNELS(0, I8,   F16,  0)
    HASH_LOG_SOFTMAX_KERNELS(1, F16,  F16,  1)
    HASH_LOG_SOFTMAX_KERNELS(1, F16,  I16,  1)
    HASH_LOG_SOFTMAX_KERNELS(1, F16,  U8,   1)
    HASH_LOG_SOFTMAX_KERNELS(1, F16,  I8,   1)
    HASH_LOG_SOFTMAX_KERNELS(1, I16,  I16,  1)
    HASH_LOG_SOFTMAX_KERNELS(1, I16,  F16,  1)
    HASH_LOG_SOFTMAX_KERNELS(1, BF16, BF16, 1_BF16)
    HASH_LOG_SOFTMAX_KERNELS(1, BF16, F32,  1_BF16)
    HASH_LOG_SOFTMAX_KERNELS(1, BF16, F16,  1_BF16)
    HASH_LOG_SOFTMAX_KERNELS(1, U8,   U8,   1)
    HASH_LOG_SOFTMAX_KERNELS(1, U8,   F16,  1)
    HASH_LOG_SOFTMAX_KERNELS(1, I8,   I8,   1)
    HASH_LOG_SOFTMAX_KERNELS(1, I8,   F16,  1)
    HASH_LOG_SOFTMAX_KERNELS(2, F16,  F16,  2)
    HASH_LOG_SOFTMAX_KERNELS(2, F16,  I16,  2)
    HASH_LOG_SOFTMAX_KERNELS(2, F16,  U8,   2)
    HASH_LOG_SOFTMAX_KERNELS(2, F16,  I8,   2)
    HASH_LOG_SOFTMAX_KERNELS(2, I16,  I16,  2)
    HASH_LOG_SOFTMAX_KERNELS(2, I16,  F16,  2)
    HASH_LOG_SOFTMAX_KERNELS(2, BF16, BF16, 2)
    HASH_LOG_SOFTMAX_KERNELS(2, U8,   U8,   2)
    HASH_LOG_SOFTMAX_KERNELS(2, U8,   F16,  2)
    HASH_LOG_SOFTMAX_KERNELS(2, I8,   I8,   2)
    HASH_LOG_SOFTMAX_KERNELS(2, I8,   F16,  2)

    HASH_LOG_SOFTMAX_KERNELS_2D(0, F16,  F16,  0)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, F16,  I16,  0)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, F16,  U8,   0)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, F16,  I8,   0)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, I16,  I16,  0)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, I16,  F16,  0)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, BF16, BF16, 0_BF16)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, BF16, F32,  0_BF16)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, BF16, F16,  0_BF16)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, U8,   U8,   0)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, U8,   F16,  0)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, I8,   I8,   0)
    HASH_LOG_SOFTMAX_KERNELS_2D(0, I8,   F16,  0)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, F16,  F16,  1)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, F16,  I16,  1)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, F16,  U8,   1)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, F16,  I8,   1)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, I16,  I16,  1)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, I16,  F16,  1)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, BF16, BF16, 1_BF16)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, BF16, F32,  1_BF16)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, BF16, F16,  1_BF16)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, U8,   U8,   1)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, U8,   F16,  1)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, I8,   I8,   1)
    HASH_LOG_SOFTMAX_KERNELS_2D(1, I8,   F16,  1)

};

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _EVIS_PARAM_NUM          _cnt_of_array(kernel_param_def)

#define SCALAR_INPUT_AXIS          (2)
#define SCALAR_INPUT_BETA          (3)

DEF_KERNEL_INITIALIZER(_log_softmax_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    int32_t     axis                        = 0;
    float       beta                        = 0;
    float      input_scale                  = 0;
    float      output_scale                 = 0;
    int32_t    outputZP                     = 0;
    uint32_t   inputWidth                   = 0;
    uint32_t   inputWidthRemain4            = 0;
    vsi_nn_kernel_tensor_attr_t * attr[2]   = { NULL, NULL };
    vsi_size_array_t * output_shape          = NULL;
    float   logE                            = (float)(log10(exp(1.0f)) / log10(2.0f));
    float   rlogE                           = (float)(log10(2.0f) / log10(exp(1.0f)));
    float   scaleLogE                       = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[3], &beta);
    CHECK_STATUS_FAIL_GOTO(status, final );

    scaleLogE = logE * beta;

    output_shape  = attr[1]->shape;

    gpu_param.dim = 2;
    switch (axis)
    {
        case 0:
            gpu_param.global_scale[0] = 1;
            gpu_param.global_scale[1] = 1;
            gpu_param.global_size[0]  = output_shape->data[1];
            gpu_param.global_size[1]  = output_shape->size > 2 ? output_shape->data[2] : 1;
        break;
        case 1:
            gpu_param.global_scale[0] = 8;
            gpu_param.global_scale[1] = 1;
            gpu_param.global_size[0]  =
            gpu_align_p2((output_shape->data[0] + gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
            gpu_param.global_size[1]  = output_shape->size > 2 ? output_shape->data[2] : 1;
        break;
        case 2:
            gpu_param.global_scale[0] = 8;
            gpu_param.global_scale[1] = 1;
            gpu_param.global_size[0]  =
            gpu_align_p2((output_shape->data[0] + gpu_param.global_scale[0] - 1) / gpu_param.global_scale[0], 4);
            gpu_param.global_size[1]  = output_shape->data[1];
        break;
        default:
        break;
    }

    {
        gpu_dp_inst_t uniGetSubData0to3_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniGetSubData4to7_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniPackMaxData_2x8 = {{
            0x00000111, // TCfg
            0x00000000, // ASelt
            0x00050300, 0x00000000, // ABin
            0x00000222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00004400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000000,
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
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractHalf4_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGetSubLoData_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGetSubHiData_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00550044, 0x00770066, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractOddData_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        switch( axis )
        {
            case 0:
            {
                inputWidth        = (uint32_t)(output_shape->data[axis] / 4 * 4);
                inputWidthRemain4 = (uint32_t)(output_shape->data[axis] % 4);

                status = vsi_nn_kernel_gpu_add_param( node,
                        "inputWidth", &inputWidth );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "inputWidthRemain4", &inputWidthRemain4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniPackMaxData_2x8", &uniPackMaxData_2x8 );
                if (attr[0]->dtype == BF16)
                {
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniExtractHalf4_4x4", &uniExtractHalf4_4x4 );
                }
                else
                {
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniGetSubData0to3_4x4", &uniGetSubData0to3_4x4 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniGetSubData4to7_4x4", &uniGetSubData4to7_4x4 );
                }
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            break;
        case 1:
        case 2:
            {
                if (attr[0]->dtype == BF16)
                {
                    status = vsi_nn_kernel_gpu_add_param( node,
                            "uniExtractHalf8_2x8", &uniExtractHalf8_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniExtractOddData_2x8", &uniExtractOddData_2x8 );
                }
                else
                {
                    status = vsi_nn_kernel_gpu_add_param( node,
                            "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniGetSubLoData_4x4", &uniGetSubLoData_4x4 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniGetSubHiData_4x4", &uniGetSubHiData_4x4 );
                }
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            break;
        default:
            break;
        }
    }

    if( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr[1]->dfp.fl;

        if (fl > 0)
        {
            output_scale = (float)((int64_t)1 << fl);
        }
        else
        {
            output_scale = (float)1.0f / (float) ((int64_t)1 << -fl);
        }

        status = vsi_nn_kernel_gpu_add_param( node,
            "outputScale", &output_scale );
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        float output_offset_asymmetric = 0;
        outputZP = attr[1]->asymm.zero_point;
        output_scale = 1.0f / (float)(attr[1]->asymm.scale);
        output_offset_asymmetric = (float)outputZP;

        status = vsi_nn_kernel_gpu_add_param( node,
            "outputScale", &output_scale );
        status |= vsi_nn_kernel_gpu_add_param( node,
            "output_offset_asymmetric", &output_offset_asymmetric );
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else
    {
        output_scale = 1;
        outputZP     = 0;
    }

    if( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr[0]->dfp.fl;
        if (fl > 0)
        {
            input_scale = 1.0f / (float) ((int64_t)1 << fl);
        }
        else
        {
            input_scale = (float)((int64_t)1 << -fl);
        }
    }
    else if( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        input_scale = attr[0]->asymm.scale;
    }
    else
    {
        input_scale = 1.0f;
    }

    scaleLogE = scaleLogE * input_scale;
    beta = beta * input_scale;

    status = vsi_nn_kernel_gpu_add_param( node,
        "rlogE", &rlogE );
    status |= vsi_nn_kernel_gpu_add_param( node,
        "betaValue", &beta );
    status |= vsi_nn_kernel_gpu_add_param( node,
        "scaleLogE", &scaleLogE );
    status |= vsi_nn_kernel_gpu_add_param( node,
        "axisSize", &output_shape->data[axis] );

    status |= vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
    }

    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
    }

    return status;
} /* _log_softmax_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    int32_t axis,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    size_t i;

    input_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    key = HASH_LOG_SOFTMAX_HASH_KEY( axis, input_dtype, output_dtype, image_2d );

    for( i = 0; i < _cnt_of_array(_log_softmax_evis_kernel_map); i ++ )
    {
        if( _log_softmax_evis_kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(_log_softmax_evis_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _log_softmax_evis_kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _log_softmax_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _log_softmax_evis_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _log_softmax_evis_kernel_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_EVIS_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    int32_t axis = 0;
    float beta = 1.0f;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    axis = vsi_nn_kernel_param_get_int32(params, "axis");
    beta = vsi_nn_kernel_param_get_float32(params, "beta");

    if( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     || axis > 2)
    {
        return NULL;
    }

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( inputs, outputs, axis, image_2d, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( node_params, _EVIS_PARAM_NUM,
                    inputs, 1, outputs, 1 );
            node_params[SCALAR_INPUT_AXIS] = vsi_nn_kernel_scalar_create(
                    graph, I32, &axis );
            node_params[SCALAR_INPUT_BETA] = vsi_nn_kernel_scalar_create(
                    graph, F32, &beta );

            status = vsi_nn_kernel_node_pass_param( node, node_params, _EVIS_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_AXIS] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_BETA] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( log_softmax, _setup )

