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

#define VX_KERNEL_NAME_POW_F16F16TOF16                     CVIVANTE_NAMESPACE("evis.pow_F16F16toF16")
#define VX_KERNEL_NAME_POW_F16F16TOF16_2D                  CVIVANTE_NAMESPACE("evis.pow_F16F16toF16_2D")
#define VX_KERNEL_NAME_POW_F16F16TOU8                      CVIVANTE_NAMESPACE("evis.pow_F16F16toU8")
#define VX_KERNEL_NAME_POW_F16F16TOU8_2D                   CVIVANTE_NAMESPACE("evis.pow_F16F16toU8_2D")
#define VX_KERNEL_NAME_POW_F16F16TOI8                      CVIVANTE_NAMESPACE("evis.pow_F16F16toI8")
#define VX_KERNEL_NAME_POW_F16F16TOI8_2D                   CVIVANTE_NAMESPACE("evis.pow_F16F16toI8_2D")
#define VX_KERNEL_NAME_POW_F16F16TOI16                     CVIVANTE_NAMESPACE("evis.pow_F16F16toI16")
#define VX_KERNEL_NAME_POW_F16F16TOI16_2D                  CVIVANTE_NAMESPACE("evis.pow_F16F16toI16_2D")
#define VX_KERNEL_NAME_POW_F16U8TOF16                      CVIVANTE_NAMESPACE("evis.pow_F16U8toF16")
#define VX_KERNEL_NAME_POW_F16U8TOF16_2D                   CVIVANTE_NAMESPACE("evis.pow_F16U8toF16_2D")
#define VX_KERNEL_NAME_POW_F16I8TOF16                      CVIVANTE_NAMESPACE("evis.pow_F16I8toF16")
#define VX_KERNEL_NAME_POW_F16I8TOF16_2D                   CVIVANTE_NAMESPACE("evis.pow_F16I8toF16_2D")
#define VX_KERNEL_NAME_POW_F16I16TOF16                     CVIVANTE_NAMESPACE("evis.pow_F16I16toF16")
#define VX_KERNEL_NAME_POW_F16I16TOF16_2D                  CVIVANTE_NAMESPACE("evis.pow_F16I16toF16_2D")
#define VX_KERNEL_NAME_POW_F16U8TOU8                       CVIVANTE_NAMESPACE("evis.pow_F16U8toU8")
#define VX_KERNEL_NAME_POW_F16U8TOU8_2D                    CVIVANTE_NAMESPACE("evis.pow_F16U8toU8_2D")
#define VX_KERNEL_NAME_POW_F16I8TOI8                       CVIVANTE_NAMESPACE("evis.pow_F16I8toI8")
#define VX_KERNEL_NAME_POW_F16I8TOI8_2D                    CVIVANTE_NAMESPACE("evis.pow_F16I8toI8_2D")
#define VX_KERNEL_NAME_POW_F16I16TOI16                     CVIVANTE_NAMESPACE("evis.pow_F16I16toI16")
#define VX_KERNEL_NAME_POW_F16I16TOI16_2D                  CVIVANTE_NAMESPACE("evis.pow_F16I16toI16_2D")
#define VX_KERNEL_NAME_POW_U8F16TOF16                      CVIVANTE_NAMESPACE("evis.pow_U8F16toF16")
#define VX_KERNEL_NAME_POW_U8F16TOF16_2D                   CVIVANTE_NAMESPACE("evis.pow_U8F16toF16_2D")
#define VX_KERNEL_NAME_POW_I8F16TOF16                      CVIVANTE_NAMESPACE("evis.pow_I8F16toF16")
#define VX_KERNEL_NAME_POW_I8F16TOF16_2D                   CVIVANTE_NAMESPACE("evis.pow_I8F16toF16_2D")
#define VX_KERNEL_NAME_POW_I16F16TOF16                     CVIVANTE_NAMESPACE("evis.pow_I16F16toF16")
#define VX_KERNEL_NAME_POW_I16F16TOF16_2D                  CVIVANTE_NAMESPACE("evis.pow_I16F16toF16_2D")
#define VX_KERNEL_NAME_POW_U8F16TOU8                       CVIVANTE_NAMESPACE("evis.pow_U8F16toU8")
#define VX_KERNEL_NAME_POW_U8F16TOU8_2D                    CVIVANTE_NAMESPACE("evis.pow_U8F16toU8_2D")
#define VX_KERNEL_NAME_POW_I8F16TOI8                       CVIVANTE_NAMESPACE("evis.pow_I8F16toI8")
#define VX_KERNEL_NAME_POW_I8F16TOI8_2D                    CVIVANTE_NAMESPACE("evis.pow_I8F16toI8_2D")
#define VX_KERNEL_NAME_POW_I16F16TOI16                     CVIVANTE_NAMESPACE("evis.pow_I16F16toI16")
#define VX_KERNEL_NAME_POW_I16F16TOI16_2D                  CVIVANTE_NAMESPACE("evis.pow_I16F16toI16_2D")
#define VX_KERNEL_NAME_POW_U8U8TOU8                        CVIVANTE_NAMESPACE("evis.pow_U8U8toU8")
#define VX_KERNEL_NAME_POW_U8U8TOU8_2D                     CVIVANTE_NAMESPACE("evis.pow_U8U8toU8_2D")
#define VX_KERNEL_NAME_POW_I8I8TOI8                        CVIVANTE_NAMESPACE("evis.pow_I8I8toI8")
#define VX_KERNEL_NAME_POW_I8I8TOI8_2D                     CVIVANTE_NAMESPACE("evis.pow_I8I8toI8_2D")
#define VX_KERNEL_NAME_POW_I16I16TOI16                     CVIVANTE_NAMESPACE("evis.pow_I16I16toI16")
#define VX_KERNEL_NAME_POW_I16I16TOI16_2D                  CVIVANTE_NAMESPACE("evis.pow_I16I16toI16_2D")
#define VX_KERNEL_NAME_POW_BF16BF16TOBF16                  CVIVANTE_NAMESPACE("evis.pow_BF16BF16toBF16")
#define VX_KERNEL_NAME_POW_BF16BF16TOBF16_2D               CVIVANTE_NAMESPACE("evis.pow_BF16BF16toBF16_2D")
#define VX_KERNEL_NAME_POW_U8U8TOF16                       CVIVANTE_NAMESPACE("evis.pow_U8U8toF16")
#define VX_KERNEL_NAME_POW_U8U8TOF16_2D                    CVIVANTE_NAMESPACE("evis.pow_U8U8toF16_2D")

#define KERNEL_SOURCE_1    "pow_fp16",
#define KERNEL_SOURCE_2    "pow_fp16_i8",
#define KERNEL_SOURCE_3    "pow_fp16_i16",
#define KERNEL_SOURCE_4    "pow_u8",
#define KERNEL_SOURCE_5    "pow_i8",
#define KERNEL_SOURCE_6    "pow_i16"


#define HASH_POW_KEY(_input0_type, _input1_type, _output_type, _image_2d) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_image_2d))

#define TENSOR_POW_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_POW_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
        VX_KERNEL_NAME_POW_##IN0_TYPE##IN1_TYPE##TO##OUT_TYPE, \
        SOURCE },

#define TENSOR_POW_KERNELS_2D(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_POW_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1), \
        VX_KERNEL_NAME_POW_##IN0_TYPE##IN1_TYPE##TO##OUT_TYPE##_2D, \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } pow_map[] =
{
    TENSOR_POW_KERNELS(F16, F16, F16,       KERNEL_SOURCE_1)
    TENSOR_POW_KERNELS(F16, F16, U8,        KERNEL_SOURCE_1)
    TENSOR_POW_KERNELS(F16, U8, F16,        KERNEL_SOURCE_1)
    TENSOR_POW_KERNELS(F16, U8, U8,         KERNEL_SOURCE_1)

    TENSOR_POW_KERNELS(F16, F16, I8,        KERNEL_SOURCE_2)
    TENSOR_POW_KERNELS(F16, I8, F16,        KERNEL_SOURCE_2)
    TENSOR_POW_KERNELS(F16, I8, I8,         KERNEL_SOURCE_2)

    TENSOR_POW_KERNELS(F16, F16, I16,       KERNEL_SOURCE_3)
    TENSOR_POW_KERNELS(F16, I16, F16,       KERNEL_SOURCE_3)
    TENSOR_POW_KERNELS(F16, I16, I16,       KERNEL_SOURCE_3)

    TENSOR_POW_KERNELS(U8, F16, F16,        KERNEL_SOURCE_4)
    TENSOR_POW_KERNELS(U8, F16, U8,         KERNEL_SOURCE_4)
    TENSOR_POW_KERNELS(U8, U8, U8,          KERNEL_SOURCE_4)
    TENSOR_POW_KERNELS(U8, U8, F16,         KERNEL_SOURCE_4)

    TENSOR_POW_KERNELS(I8, F16, F16,        KERNEL_SOURCE_5)
    TENSOR_POW_KERNELS(I8, F16, I8,         KERNEL_SOURCE_5)
    TENSOR_POW_KERNELS(I8, I8, I8,          KERNEL_SOURCE_5)

    TENSOR_POW_KERNELS(I16, F16, F16,       KERNEL_SOURCE_6)
    TENSOR_POW_KERNELS(I16, F16, I16,       KERNEL_SOURCE_6)
    TENSOR_POW_KERNELS(I16, I16, I16,       KERNEL_SOURCE_6)
    TENSOR_POW_KERNELS(BF16, BF16, BF16,    KERNEL_SOURCE_3)

    TENSOR_POW_KERNELS_2D(F16, F16, F16,    KERNEL_SOURCE_1)
    TENSOR_POW_KERNELS_2D(F16, F16, U8,     KERNEL_SOURCE_1)
    TENSOR_POW_KERNELS_2D(F16, U8, F16,     KERNEL_SOURCE_1)
    TENSOR_POW_KERNELS_2D(F16, U8, U8,      KERNEL_SOURCE_1)

    TENSOR_POW_KERNELS_2D(F16, F16, I8,     KERNEL_SOURCE_2)
    TENSOR_POW_KERNELS_2D(F16, I8, F16,     KERNEL_SOURCE_2)
    TENSOR_POW_KERNELS_2D(F16, I8, I8,      KERNEL_SOURCE_2)

    TENSOR_POW_KERNELS_2D(F16, F16, I16,    KERNEL_SOURCE_3)
    TENSOR_POW_KERNELS_2D(F16, I16, F16,    KERNEL_SOURCE_3)
    TENSOR_POW_KERNELS_2D(F16, I16, I16,    KERNEL_SOURCE_3)

    TENSOR_POW_KERNELS_2D(U8, F16, F16,     KERNEL_SOURCE_4)
    TENSOR_POW_KERNELS_2D(U8, F16, U8,      KERNEL_SOURCE_4)
    TENSOR_POW_KERNELS_2D(U8, U8, U8,       KERNEL_SOURCE_4)
    TENSOR_POW_KERNELS_2D(U8, U8, F16,      KERNEL_SOURCE_4)

    TENSOR_POW_KERNELS_2D(I8, F16, F16,     KERNEL_SOURCE_5)
    TENSOR_POW_KERNELS_2D(I8, F16, I8,      KERNEL_SOURCE_5)
    TENSOR_POW_KERNELS_2D(I8, I8, I8,       KERNEL_SOURCE_5)

    TENSOR_POW_KERNELS_2D(I16, F16, F16,    KERNEL_SOURCE_6)
    TENSOR_POW_KERNELS_2D(I16, F16, I16,    KERNEL_SOURCE_6)
    TENSOR_POW_KERNELS_2D(I16, I16, I16,    KERNEL_SOURCE_6)
    TENSOR_POW_KERNELS_2D(BF16, BF16, BF16, KERNEL_SOURCE_3)
};

static vx_param_description_t vxPowKernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};
#define _EVIS_POW_PARAM_NUM          _cnt_of_array(vxPowKernel_param_def)

DEF_KERNEL_INITIALIZER(_pow_initializer)
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

    int8_t      in0_fl     = 0;
    int32_t     src0ZP     = 0;
    float       src0Scale  = 1.0f;
    int8_t      in1_fl     = 0;
    int32_t     src1ZP     = 0;
    float       src1Scale  = 1.0f;
    int8_t      out_fl     = 0;
    float       dstZP      = 0;
    float       dstScale   = 1.0f;

    int8_t     postshift0  = 0;
    int8_t     postshift1  = 0;
    float      outScale_fl = 1;

    uint16_t M0            = 0;
    uint16_t M1            = 0;

    uint32_t    zAx        = 1;
    uint32_t pack_key      = 0;
    // dim number ???
    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    vsi_int_array_t * out_shape = NULL;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    out_shape   = attr[2]->shape;

    if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        in0_fl = (int8_t)attr[0]->dfp.fl;
        postshift0 = in0_fl - 0;
    }
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM
        || attr[0]->quant == VSI_NN_KERNEL_QUANT_SYMM )
    {
        src0ZP     = attr[0]->asymm.zero_point;
        src0Scale  = attr[0]->asymm.scale;

        vsi_nn_GetFP32MultiAndPostShift(src0Scale / 1.0f, &M0, &postshift0);
    }

    if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        in1_fl = (int8_t)attr[1]->dfp.fl;
        postshift1 = in1_fl - 0;
    }
    else if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM
        || attr[1]->quant == VSI_NN_KERNEL_QUANT_SYMM)
    {
        src1ZP     = attr[1]->asymm.zero_point;
        src1Scale  = attr[1]->asymm.scale;

        vsi_nn_GetFP32MultiAndPostShift(src1Scale / 1.0f, &M1, &postshift1);
    }

    if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        out_fl = (int8_t)attr[2]->dfp.fl;
        if (out_fl > 0)
        {
            outScale_fl = (vx_float32)((int64_t)1 << out_fl);
        }
        else
        {
            outScale_fl = (1.0f / (vx_float32)((int64_t)1 << -out_fl));
        }
    }
    else if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_ASYMM
        || attr[2]->quant == VSI_NN_KERNEL_QUANT_SYMM )
    {
        dstZP     = (float)attr[2]->asymm.zero_point;
        dstScale  = 1.0f / attr[2]->asymm.scale;
    }

    if ( out_shape->size < 3 )
    {
        zAx = 1;
    }
    else
    {
        zAx = out_shape->data[2];
    }

#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (IN1_TYPE << 8) | ( OUT_TYPE << 16))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype,
            attr[1]->dtype, attr[2]->dtype );

    shaderParam.global_scale[0]  = 8;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((out_shape->data[0] + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = gpu_align_p2((out_shape->data[1] + shaderParam.global_scale[1] - 1)
        / shaderParam.global_scale[1], 2);
    shaderParam.global_size[2]   = gpu_align_p2((zAx + shaderParam.global_scale[2] - 1)
        / shaderParam.global_scale[2], 1);

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniConvertFstDataToFp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertSecDataToFp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertFstDataToFp32_4x4_2 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertSecDataToFp32_4x4_2 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
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

        gpu_dp_inst_t uniConvertUint8SubZpToFp32_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertSecUint8SubZpToFp32_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvertUint8SubZpToFp32_4x4_2 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvertSecUint8SubZpToFp32_4x4_2 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvertHalfToFp16_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01050004, 0x03070206, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractOddData_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertHalftoFp16_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        uint32_t multiplierA = (M0 << 16) | M0;
        uint32_t multiplierB = (M1 << 16) | M1;
        int32_t i = 8;

        uniConvertUint8SubZpToFp32_4x4.data[7] |= (postshift0 & 0x1F);
        uniConvertSecUint8SubZpToFp32_4x4.data[7] |= (postshift0 & 0x1F);
        uniConvertUint8SubZpToFp32_4x4_2.data[7] |= (postshift1 & 0x1F);
        uniConvertSecUint8SubZpToFp32_4x4_2.data[7] |= (postshift1 & 0x1F);
        for ( i = 8; i < 16; i += 2 )
        {
            uniConvertUint8SubZpToFp32_4x4.data[i] = multiplierA;
            uniConvertSecUint8SubZpToFp32_4x4.data[i] = multiplierA;
            uniConvertUint8SubZpToFp32_4x4_2.data[i] = multiplierB;
            uniConvertSecUint8SubZpToFp32_4x4_2.data[i] = multiplierB;
        }

        if ( attr[0]->dtype == I8 || attr[0]->dtype == I16 )
        {
            gpu_dp_inst_update_postshfit( &uniConvertFstDataToFp32_4x4, postshift0 );
            gpu_dp_inst_update_postshfit( &uniConvertSecDataToFp32_4x4, postshift0 );
        }

        if ( attr[1]->dtype == I8 || attr[1]->dtype == I16 )
        {
            gpu_dp_inst_update_postshfit( &uniConvertFstDataToFp32_4x4_2, postshift1 );
            gpu_dp_inst_update_postshfit( &uniConvertSecDataToFp32_4x4_2, postshift1 );
        }

        switch( pack_key )
        {
            case _PACK_SELECT_KEY( F16, F16, I8 ):
            case _PACK_SELECT_KEY( F16, I8, F16 ):
            case _PACK_SELECT_KEY( F16, I8, I8 ):
            case _PACK_SELECT_KEY( F16, F16, I16 ):
            case _PACK_SELECT_KEY( F16, I16, F16 ):
            case _PACK_SELECT_KEY( F16, I16, I16 ):
            case _PACK_SELECT_KEY( I8, F16, F16 ):
            case _PACK_SELECT_KEY( I8, F16, I8 ):
            case _PACK_SELECT_KEY( I8, I8, I8 ):
            case _PACK_SELECT_KEY( I16, F16, F16 ):
            case _PACK_SELECT_KEY( I16, F16, I16 ):
            case _PACK_SELECT_KEY( I16, I16, I16 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniConvertFstDataToFp32_4x4",
                        &uniConvertFstDataToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecDataToFp32_4x4",
                        &uniConvertSecDataToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertFstDataToFp32_4x4_2",
                        &uniConvertFstDataToFp32_4x4_2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecDataToFp32_4x4_2",
                        &uniConvertSecDataToFp32_4x4_2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "outScale_fl", &outScale_fl);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( U8, F16, F16 ):
            case _PACK_SELECT_KEY( U8, F16, U8 ):
            case _PACK_SELECT_KEY( U8, U8, U8 ):
            case _PACK_SELECT_KEY( U8, U8, F16 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniConvertUint8SubZpToFp32_4x4",
                        &uniConvertUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecUint8SubZpToFp32_4x4",
                        &uniConvertSecUint8SubZpToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertFstDataToFp32_4x4_2",
                        &uniConvertFstDataToFp32_4x4_2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecDataToFp32_4x4_2",
                        &uniConvertSecDataToFp32_4x4_2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertUint8SubZpToFp32_4x4_2",
                        &uniConvertUint8SubZpToFp32_4x4_2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecUint8SubZpToFp32_4x4_2",
                        &uniConvertSecUint8SubZpToFp32_4x4_2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalftoFp16_2x8",
                        &uniConvertHalftoFp16_2x8);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_ZP0", &src0ZP);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_ZP1", &src1ZP);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_ZP", &dstZP);
                    status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &dstScale);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( F16, F16, F16 ):
            case _PACK_SELECT_KEY( F16, F16, U8 ):
            case _PACK_SELECT_KEY( F16, U8, F16 ):
            case _PACK_SELECT_KEY( F16, U8, U8 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniConvertFstDataToFp32_4x4",
                        &uniConvertFstDataToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecDataToFp32_4x4",
                        &uniConvertSecDataToFp32_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertUint8SubZpToFp32_4x4_2",
                        &uniConvertUint8SubZpToFp32_4x4_2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertSecUint8SubZpToFp32_4x4_2",
                        &uniConvertSecUint8SubZpToFp32_4x4_2);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertHalfToFp16_2x8",
                        &uniConvertHalfToFp16_2x8);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_ZP1", &src1ZP);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_ZP", &dstZP);
                    status |= vsi_nn_kernel_gpu_add_param(node, "outputScale", &dstScale);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( BF16, BF16, BF16 ):
                {
                    status = vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part0_2x8",
                        &uniConvBF16toF32_Part0_2x8);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part1_2x8",
                        &uniConvBF16toF32_Part1_2x8);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractOddData_2x8",
                        &uniExtractOddData_2x8);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
            default:
                break;
        }
#undef _PACK_SELECT_KEY
        status = vsi_nn_kernel_gpu_add_param(node, "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if ( attr[0] )
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if ( attr[1] )
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if ( attr[2] )
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }
    return status;
} /* _pow_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_bool image_2d,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input0_dtype;
    vsi_nn_kernel_dtype_e input1_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    key = HASH_POW_KEY( input0_dtype, input1_dtype, output_dtype, image_2d );

    for ( i = 0; i < _cnt_of_array(pow_map); i ++ )
    {
        if ( pow_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(pow_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  pow_map[i].function_name );
        kernel->info.parameters = vxPowKernel_param_def;
        kernel->info.numParams = _cnt_of_array( vxPowKernel_param_def );
        kernel->info.initialize = _pow_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                pow_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                pow_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_EVIS_POW_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;

    if ( !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (outputs[0]->attr.dim_num == 2);
    status = _query_kernel( inputs, outputs, image_2d, kernel );
    if ( VSI_SUCCESS == status )
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( tmp_params, _EVIS_POW_PARAM_NUM,
                inputs, 2, outputs, 1 );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _EVIS_POW_PARAM_NUM );

        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( pow, _setup )

