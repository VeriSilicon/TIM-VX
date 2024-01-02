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
#include "libnnext/vx_lib_nnext.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "matrixmul_f16u8_f16"
#define KERNEL_SOURCE_2    "matrixmul_f16"
#define KERNEL_SOURCE_3    "matrixmul_transB_f16"
#define KERNEL_SOURCE_4    "matrixmul_transB_f16_mix"
#define KERNEL_SOURCE_5    "matrixmul_transB_u8_mix"
#define KERNEL_SOURCE_6    "matrixmul_u8f16_u8"
#define KERNEL_SOURCE_7    "matrixmul_transA"
#define KERNEL_SOURCE_8    "matrixmul_u8f16_f16"
#define KERNEL_SOURCE_9    "matrixmul_u8"
#define KERNEL_SOURCE_10   "matrixmul_f16u8_u8"
#define KERNEL_SOURCE_11   "matrixmul_f16f16_u8"
#define KERNEL_SOURCE_12   "matrixmul_u8u8_f16"
#define KERNEL_SOURCE_13   "matrixmul_i16"
#define KERNEL_SOURCE_14   "matrixmul_f16i16_i16"
#define KERNEL_SOURCE_15   "matrixmul_bf16"
#define KERNEL_SOURCE_16   "matrixmul_u8i16_i16"
#define KERNEL_SOURCE_17   "matrixmul_merge"
#define KERNEL_SOURCE_18   "matrixmul_cross"
#define KERNEL_SOURCE_19   "matrixmul_cross_i16"

#define HASH_MATRIX_MUL_KEY(_type0, _type1, _type2, _trans_a, _trans_b, _cross) \
    ((_type0 << 24) | (_type1 << 16) | (_type2 << 8) | (_trans_a << 4) | (_trans_b << 2) | (_cross))

#define HASH_MATRIX_MUL_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.gemm_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE)

#define HASH_MATRIX_MUL_TRANSB_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.gemm_transb_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE)

#define HASH_MATRIX_MUL_TRANSA_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.gemm_transa_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE)

#define HASH_MATRIX_MUL_SH_KERNEL_CROSS_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.gemm_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE"_cross")

#define HASH_MATRIX_MUL_SH_KERNEL_MERGE_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.gemm_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE"_merge")

#define TENSOR_MATRIX_MUL_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MATRIX_MUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0, 0, 0), \
        HASH_MATRIX_MUL_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MATRIX_MUL_TRANSB_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MATRIX_MUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0, 1, 0), \
        HASH_MATRIX_MUL_TRANSB_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MATRIX_MUL_TRANSA_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MATRIX_MUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1, 0, 0), \
        HASH_MATRIX_MUL_TRANSA_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MATRIX_MUL_CROSS_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MATRIX_MUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0, 0, 1), \
        HASH_MATRIX_MUL_SH_KERNEL_CROSS_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MATRIX_MUL_MERGE_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MATRIX_MUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0, 0, 2), \
        HASH_MATRIX_MUL_SH_KERNEL_MERGE_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },


static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } matrix_mul_map[] =
{
    TENSOR_MATRIX_MUL_KERNELS(U8,  U8,  U8,       KERNEL_SOURCE_9)
    TENSOR_MATRIX_MUL_KERNELS(I8,  I8,  I8,       KERNEL_SOURCE_9)
    TENSOR_MATRIX_MUL_KERNELS(I16, I16, I16,      KERNEL_SOURCE_13)
    TENSOR_MATRIX_MUL_KERNELS(F16, U8,  F16,      KERNEL_SOURCE_1)
    TENSOR_MATRIX_MUL_KERNELS(F16, I8,  F16,      KERNEL_SOURCE_1)
    TENSOR_MATRIX_MUL_KERNELS(F16, I16, F16,      KERNEL_SOURCE_1)
    TENSOR_MATRIX_MUL_KERNELS(F16, U8,  U8,       KERNEL_SOURCE_10)
    TENSOR_MATRIX_MUL_KERNELS(F16, I8,  I8,       KERNEL_SOURCE_10)
    TENSOR_MATRIX_MUL_KERNELS(F16, I16, I16,      KERNEL_SOURCE_14)
    TENSOR_MATRIX_MUL_KERNELS(U8,  F16, U8,       KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(I8,  F16, I8,       KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(I16, F16, I16,      KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(U8,  U8,  F16,      KERNEL_SOURCE_12)
    TENSOR_MATRIX_MUL_KERNELS(I8,  I8,  F16,      KERNEL_SOURCE_12)
    TENSOR_MATRIX_MUL_KERNELS(I16, I16, F16,      KERNEL_SOURCE_12)
    TENSOR_MATRIX_MUL_KERNELS(U8,  F16, F16,      KERNEL_SOURCE_8)
    TENSOR_MATRIX_MUL_KERNELS(I8,  F16, F16,      KERNEL_SOURCE_8)
    TENSOR_MATRIX_MUL_KERNELS(I16, F16, F16,      KERNEL_SOURCE_8)
    TENSOR_MATRIX_MUL_KERNELS(F16, F16, F16,      KERNEL_SOURCE_2)
    TENSOR_MATRIX_MUL_KERNELS(BF16,BF16,BF16,     KERNEL_SOURCE_15)
    TENSOR_MATRIX_MUL_KERNELS(F16, F16, U8,       KERNEL_SOURCE_11)
    TENSOR_MATRIX_MUL_KERNELS(F16, F16, I8,       KERNEL_SOURCE_11)
    TENSOR_MATRIX_MUL_KERNELS(F16, F16, I16,      KERNEL_SOURCE_11)
    TENSOR_MATRIX_MUL_KERNELS(F32, F32, F32,      KERNEL_SOURCE_2)
    TENSOR_MATRIX_MUL_KERNELS(U8,  I16, I16,      KERNEL_SOURCE_16)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(F16, F16, F16,    KERNEL_SOURCE_3)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(F16, U8,  F16,    KERNEL_SOURCE_4)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(F16, U8,  U8,     KERNEL_SOURCE_4)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(U8,  U8,  F16,    KERNEL_SOURCE_5)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(U8,  U8,  U8,     KERNEL_SOURCE_5)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(I16, I16, I16,    KERNEL_SOURCE_13)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(BF16,BF16,BF16,   KERNEL_SOURCE_15)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(U8,  I16, I16,    KERNEL_SOURCE_16)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(U8,  U8,  U8,     KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(I8,  I8,  I8,     KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(I16, I16, I16,    KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(U8,  F16, U8,     KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(I8,  F16, I8,     KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(I16, F16, I16,    KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(F16, F16, F16,    KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(BF16,BF16,BF16,   KERNEL_SOURCE_15)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(U8,  I16, I16,    KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_MERGE_KERNELS(U8,  U8,  U8,      KERNEL_SOURCE_17)
    TENSOR_MATRIX_MUL_MERGE_KERNELS(I8,  I8,  I8,      KERNEL_SOURCE_17)
    TENSOR_MATRIX_MUL_MERGE_KERNELS(I16, I16, I16,     KERNEL_SOURCE_19)
    TENSOR_MATRIX_MUL_MERGE_KERNELS(F16, F16, F16,     KERNEL_SOURCE_17)
    TENSOR_MATRIX_MUL_CROSS_KERNELS(U8,  U8,  U8,      KERNEL_SOURCE_18)
    TENSOR_MATRIX_MUL_CROSS_KERNELS(I8,  I8,  I8,      KERNEL_SOURCE_18)
    TENSOR_MATRIX_MUL_CROSS_KERNELS(I16, I16, I16,     KERNEL_SOURCE_19)
    TENSOR_MATRIX_MUL_CROSS_KERNELS(F16, F16, F16,     KERNEL_SOURCE_18)
};

/*
 * Kernel params
 */
static vx_param_description_t _matrix_mul_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _matrix_mul_kernel_cross_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _MATRIX_MUL_PARAM_NUM  _cnt_of_array( _matrix_mul_kernel_param_def )
#define _MATRIX_MUL_CROSS_PARAM_NUM  _cnt_of_array( _matrix_mul_kernel_cross_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_matrix_mul_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
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

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    int32_t       transA = 0;
    int32_t       transB = 0;
    int32_t       width  = 0;
    int32_t       height = 0;
    vsi_size_t    chn    = 0;
    int32_t       a_depth = 0;
    int32_t       b_depth = 0;
    vsi_size_t    outer   = 0;

    int32_t     src0ZP     = 0;
    float       src0Scale  = 0;
    int32_t     src1ZP     = 0;
    float       src1Scale  = 0;
    float       dstZP      = 0;
    float       dstScale   = 0;
    uint16_t M0            = 0;
    uint16_t M1            = 0;
    int32_t  postShift0    = 0;
    int32_t  postShift1    = 0;

    uint32_t pack_key = 0;
    int32_t  ac2zero  = 0;
    int32_t  bc2zero  = 0;

    float    mulKIn0In1Zp  = 0;
    float    inOutScale    = 0;
    int32_t  K             = 0;

    uint32_t  evis2 = 0;
    vx_context  ctx        = vxGetContext((vx_reference)node);
    vx_hardware_caps_params_t   hw_param;

    VSI_UNREFERENCED(param_size);
    memset(&hw_param, 0, sizeof(vx_hardware_caps_params_t));
    status = vxQueryHardwareCaps(ctx, &hw_param, sizeof(vx_hardware_caps_params_t));
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    if (hw_param.evis2 == TRUE)
    {
        evis2 = 1;
    }

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &transA);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &transB);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &K);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    src0ZP     = attr[0]->zero_point;
    src0Scale  = attr[0]->scale;
    src1ZP     = attr[1]->zero_point;
    src1Scale  = attr[1]->scale;
    dstZP      = (float)attr[2]->zero_point;
    dstScale   = attr[2]->scale;

    gpu_quantize_multiplier_16bit(src0Scale / 1.0f, &M0, &postShift0);
    gpu_quantize_multiplier_16bit(src1Scale / 1.0f, &M1, &postShift1);

    mulKIn0In1Zp = (float)((int)(K + 3) / 4 * 4 * src1ZP * src0ZP);
    inOutScale =  src0Scale * src1Scale / dstScale;

    a_depth = (int32_t)(attr[0]->shape->size > 2 ? attr[0]->shape->data[2] : 1);
    b_depth = (int32_t)(attr[1]->shape->size > 2 ? attr[1]->shape->data[2] : 1);

    if (b_depth == 1)
    {
        bc2zero = 1;
    }
    if (a_depth == 1)
    {
        ac2zero = 1;
    }

    width = (int32_t)(attr[2]->shape->data[0]);
    height = (int32_t)(attr[2]->shape->data[1]);
    chn = (attr[2]->shape->size > 2 ? attr[2]->shape->data[2] : 1);

    if (((attr[0]->shape->size == 4 && attr[1]->shape->size == 3) ||
        (attr[0]->shape->size == 3 && attr[1]->shape->size == 4))
        && attr[0]->shape->data[2] > 1 && attr[1]->shape->data[2] > 1
        && chn != attr[0]->shape->data[2] * attr[1]->shape->data[2])
    {
        vsi_size_t iter = attr[0]->shape->data[2] * attr[1]->shape->data[2] / chn;
        if (attr[0]->shape->size == 4)
        {
            ac2zero = 1;
            bc2zero = 0;
            chn = attr[1]->shape->data[2];
            outer = attr[0]->shape->data[2] / iter;
        }
        else
        {
            ac2zero = 0;
            bc2zero = 1;
            chn = attr[0]->shape->data[2];
            outer = attr[1]->shape->data[2] / iter;
        }
    }
    else if (attr[0]->shape->size == 4 && attr[1]->shape->size == 3
        && attr[0]->shape->data[2] != 1 && attr[1]->shape->data[2] != 1)
    {
        ac2zero = 1;
        bc2zero = 0;
        chn = attr[1]->shape->data[2];
        outer = attr[0]->shape->data[2];
    }
    else if (attr[1]->shape->size == 4 && attr[0]->shape->size == 3
        && attr[0]->shape->data[2] != 1 && attr[1]->shape->data[2] != 1)
    {
        ac2zero = 0;
        bc2zero = 1;
        chn = attr[0]->shape->data[2];
        outer = attr[1]->shape->data[2];
    }

    gpu_param.global_scale[0]  = 4;
    gpu_param.global_scale[1]  = 4;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((width + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = gpu_align_p2((height + gpu_param.global_scale[1] - 1)
                                        / gpu_param.global_scale[1], 4);
    gpu_param.global_size[2]   = (size_t)chn;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, OUT_TYPE, TRANSA, TRANSB, EVIS2)    \
        ((IN0_TYPE << 24) | (IN1_TYPE << 16) | (OUT_TYPE << 8) | (TRANSA << 4) | (TRANSB << 2) | (EVIS2))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype, attr[2]->dtype, transA, transB, evis2);
    {
        gpu_dp_inst_t uniU8SubZptoFp16_dp2x8 = {{
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniFp16MulFp16AddtoFp32_dp8x2 = {{
            0x00005555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x00000000, // ABin
            0x00005555, // BSelt
            0x76543210, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
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
        gpu_dp_inst_t uniConvertUint8SubZpToFp32B_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvert1stFp16ToFp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniGemmFp16toFp32Row0Lo_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00100010, 0x00100010, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmFp16toFp32Row0Hi_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00320032, 0x00320032, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmFp16toFp32Row1Lo_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00540054, 0x00540054, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmFp16toFp32Row1Hi_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00760076, 0x00760076, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniGemmU8F16toF32Lo_4x4b = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0x51514040, 0x73736262, // ABin
            0x00000000, // BSelt
            0x32103210, 0x32103210, // BBin
            0x00000000, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmU8F16toF32Hi_4x4b = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0x51514040, 0x73736262, // ABin
            0x00000000, // BSelt
            0x76547654, 0x76547654, // BBin
            0x00000000, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmF16I16toF32Lo_4x4b = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0x51514040, 0x73736262, // ABin
            0x00000000, // BSelt
            0x32103210, 0x32103210, // BBin
            0x00000000, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmF16I16toF32Hi_4x4b = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0x51514040, 0x73736262, // ABin
            0x00000000, // BSelt
            0x76547654, 0x76547654, // BBin
            0x00000000, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniGemm1stU8F16toF32Lo_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00100010, 0x00100010, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemm2ndU8F16toF32Lo_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00320032, 0x00320032, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemm1stU8F16toF32Hi_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00540054, 0x00540054, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemm2ndU8F16toF32Hi_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00760076, 0x00760076, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmFp16MulZptoFp32_4x4 = {{
            0xaaaaaaaa, // TCfg
            0x50505050, // ASelt
            0x51514040, 0x73736262, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00020001, 0x00010001, 0x00020001, 0x00010001, 0x00020001 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniGemmU8U8toFp32Block4_4x4 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x32103210, 0x32103210, // ABin
            0x55555555, // BSelt
            0xd951c840, 0xfb73ea62, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmU8U8MulZptoFp32_8x4 = {{
            0xaaaaaaaa, 0xaaaaaaaa, // TCfg
            0xf02a0600, 0x2a8620e0, 0x0640e8f2, 0x60f0f42b, 0xf8f62b86, // BinSelect
            0x00000700, // AccumType, ConstantType, and PostShift
            0x03020302, 0x03020302, 0x03020302, 0x03020302, 0x03020302, 0x03020302, 0x03020302, 0x03020302 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniGemmF16U8toF32_4x4 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x32103210, 0x32103210, // ABin
            0x55555555, // BSelt
            0xd951c840, 0xfb73ea62, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmF16U8toF32Hi_4x4 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76547654, 0x76547654, // ABin
            0x55555555, // BSelt
            0xd951c840, 0xfb73ea62, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmFp16U8MulZptoFp32_4x4 = {{
            0xaaaaaaaa, // TCfg
            0x55550000, // ASelt
            0x76543210, 0x76543210, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00020002, 0x00020002, 0x00030003, 0x00030003, 0x00040004, 0x00040004 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmFp16I16MulZptoFp32_4x4 = {{
            0xaaaaaaaa, // TCfg
            0x55550000, // ASelt
            0x76543210, 0x76543210, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00020002, 0x00020002, 0x00020002, 0x00020002, 0x00020002, 0x00020002, 0x00020002, 0x00020002 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmF16I16toF32A_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00100010, 0x00100010, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmF16I16toF32B_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00320032, 0x00320032, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmF16I16toF32C_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00540054, 0x00540054, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmF16I16toF32D_4x4 = {{
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00760076, 0x00760076, // ABin
            0x05050505, // BSelt
            0x00510040, 0x00730062, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
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

        gpu_dp_inst_t uniI16MulI16SumtoI32_16x1 = {{
            0xaaaa5555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0xaaaa5555, // BSelt
            0x76543210, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00020001, 0x00040003, 0x00060005, 0x00080007 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniI16MulI16SumtoI32B_16x1 = {{
            0x0002aaab, // TCfg
            0x00015554, // ASelt
            0x65432100, 0x00000007, // ABin
            0x0002aaa8, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002300, // AccumType, ConstantType, and PostShift
            0x00010000, 0x00030002, 0x00050004, 0x00070006,
            0x00000008, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        float scaleIn0divOut = src0Scale / dstScale;
        float scaleIn1divOut = src1Scale / dstScale;
        float inScaleMul = src0Scale * src1Scale;
        float reScaleOut = 1 / dstScale;
        float inScaledivOut = inScaleMul / dstScale;
        float inout_beta = src0ZP * src1ZP * 8 * inScaledivOut + dstZP;
        uint32_t multiplierA = (M0 << 16) | M0;
        uint32_t multiplierB = (M1 << 16) | M1;
        uint32_t multiplierZpA = (src0ZP << 16) | src0ZP;
        uint32_t multiplierZpB = (src1ZP << 16) | src1ZP;
        uint32_t multiplierU8ZpAB = (src0ZP << 24) | (src1ZP << 16) | (src0ZP << 8) | (src1ZP);
        int32_t i = 8;
        uniConvertUint8SubZpToFp32_4x4.data[7] |= (postShift0 & 0x1F);
        uniConvertUint8SubZpToFp32B_4x4.data[7] |= (postShift1 & 0x1F);
        for( i = 8; i < 16; i += 2)
        {
            uniConvertUint8SubZpToFp32_4x4.data[i] = multiplierA;
            uniConvertUint8SubZpToFp32B_4x4.data[i] = multiplierB;
        }
        for( i = 8; i < 16; i++)
        {
            uniGemmFp16MulZptoFp32_4x4.data[i] = multiplierZpA;
            uniGemmU8U8MulZptoFp32_8x4.data[i] = multiplierU8ZpAB;
            uniGemmFp16U8MulZptoFp32_4x4.data[i] = multiplierZpB;
            uniGemmFp16I16MulZptoFp32_4x4.data[i] = multiplierZpB;
        }
        for( i = 8; i < 12; i++)
        {
            uniI16MulI16SumtoI32B_16x1.data[i] = multiplierZpA;
        }
        for( i = 12; i < 16; i++)
        {
            uniI16MulI16SumtoI32_16x1.data[i] = multiplierZpB;
        }

        if (outer)
        {
            status = vsi_nn_kernel_gpu_add_param( node, "outer", &outer );
            CHECK_STATUS_FAIL_GOTO(status, OnError );
        }

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( U8, U8, F16, 0, 1, 0 ):
        case _PACK_SELECT_KEY( U8, U8, U8,  0, 1, 0 ):
        case _PACK_SELECT_KEY( U8, U8, F16, 0, 1, 1 ):
        case _PACK_SELECT_KEY( U8, U8, U8,  0, 1, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniU8SubZptoFp16_dp2x8", &uniU8SubZptoFp16_dp2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniFp16MulFp16AddtoFp32_dp8x2", &uniFp16MulFp16AddtoFp32_dp8x2 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input0_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src1ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "inScaleMul", &inScaleMul );
                status |= vsi_nn_kernel_gpu_add_param( node, "inScaledivOut", &inScaledivOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, U8, F16, 0, 1, 0 ):
        case _PACK_SELECT_KEY( F16, U8, U8,  0, 1, 0 ):
        case _PACK_SELECT_KEY( F16, U8, F16, 0, 1, 1 ):
        case _PACK_SELECT_KEY( F16, U8, U8,  0, 1, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniU8SubZptoFp16_dp2x8", &uniU8SubZptoFp16_dp2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniFp16MulFp16AddtoFp32_dp8x2", &uniFp16MulFp16AddtoFp32_dp8x2 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1Scale", &src1Scale );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src1ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "scaleIn2divOut", &scaleIn1divOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, F16, 0, 1, 0 ):
        case _PACK_SELECT_KEY( F16, F16, F16, 0, 1, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniFp16MulFp16AddtoFp32_dp8x2", &uniFp16MulFp16AddtoFp32_dp8x2 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, U8,  0, 0, 0 ):
        case _PACK_SELECT_KEY( F16, F16, I8,  0, 0, 0 ):
        case _PACK_SELECT_KEY( F16, F16, I16, 0, 0, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16toFp32Row0Lo_4x4", &uniGemmFp16toFp32Row0Lo_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16toFp32Row0Hi_4x4", &uniGemmFp16toFp32Row0Hi_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16toFp32Row1Lo_4x4", &uniGemmFp16toFp32Row1Lo_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16toFp32Row1Hi_4x4", &uniGemmFp16toFp32Row1Hi_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "outputScale", &reScaleOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, U8,  0, 0, 1 ):
        case _PACK_SELECT_KEY( F16, F16, I8,  0, 0, 1 ):
        case _PACK_SELECT_KEY( F16, F16, I16, 0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8F16toF32Lo_4x4b", &uniGemmU8F16toF32Lo_4x4b );
                status |= vsi_nn_kernel_gpu_add_param( node, "outputScale", &reScaleOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  U8,  U8,  0, 0, 0 ):
        case _PACK_SELECT_KEY( I8,  I8,  I8,  0, 0, 0 ):
        case _PACK_SELECT_KEY( U8,  U8,  U8,  0, 0, 1 ):
        case _PACK_SELECT_KEY( I8,  I8,  I8,  0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8U8toFp32Block4_4x4", &uniGemmU8U8toFp32Block4_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8U8MulZptoFp32_8x4", &uniGemmU8U8MulZptoFp32_8x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "inOutScale", &inOutScale );
                status |= vsi_nn_kernel_gpu_add_param( node, "mulKIn0In1Zp", &mulKIn0In1Zp );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( I16, I16, I16, 0, 0, 0 ):
        case _PACK_SELECT_KEY( I16, I16, I16, 0, 0, 1 ):
        case _PACK_SELECT_KEY( I16, I16, I16, 0, 1, 0 ):
        case _PACK_SELECT_KEY( I16, I16, I16, 0, 1, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32_4x4", &uniConvertUint8SubZpToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32B_4x4", &uniConvertUint8SubZpToFp32B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input0_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src1ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "outputScale", &reScaleOut );
                if (outer == 0)
                {
                    status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniI16MulI16SumtoI32_16x1", &uniI16MulI16SumtoI32_16x1 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniI16MulI16SumtoI32B_16x1", &uniI16MulI16SumtoI32B_16x1 );
                    status |= vsi_nn_kernel_gpu_add_param( node, "inout_scale", &inScaledivOut );
                    status |= vsi_nn_kernel_gpu_add_param( node, "inout_beta", &inout_beta );
                }
            }
            break;
        case _PACK_SELECT_KEY( F16, U8,  F16, 0, 0, 0 ):
        case _PACK_SELECT_KEY( F16, I8,  F16, 0, 0, 0 ):
        case _PACK_SELECT_KEY( F16, U8,  F16, 0, 0, 1 ):
        case _PACK_SELECT_KEY( F16, I8,  F16, 0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmF16U8toF32_4x4", &uniGemmF16U8toF32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmF16U8toF32Hi_4x4", &uniGemmF16U8toF32Hi_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16U8MulZptoFp32_4x4", &uniGemmFp16U8MulZptoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1Scale", &src1Scale );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, I16, F16, 0, 0, 0 ):
        case _PACK_SELECT_KEY( F16, I16, F16, 0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32B_4x4", &uniConvertUint8SubZpToFp32B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvert1stFp16ToFp32_4x4", &uniConvert1stFp16ToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src1ZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, I16, I16, 0, 0, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmF16I16toF32A_4x4", &uniGemmF16I16toF32A_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmF16I16toF32B_4x4", &uniGemmF16I16toF32B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmF16I16toF32C_4x4", &uniGemmF16I16toF32C_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmF16I16toF32D_4x4", &uniGemmF16I16toF32D_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16I16MulZptoFp32_4x4", &uniGemmFp16I16MulZptoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "in1outScale", &scaleIn1divOut );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, I16, I16, 0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmF16I16toF32Lo_4x4b", &uniGemmF16I16toF32Lo_4x4b );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmF16I16toF32Hi_4x4b", &uniGemmF16I16toF32Hi_4x4b );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16I16MulZptoFp32_4x4", &uniGemmFp16I16MulZptoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "in1outScale", &scaleIn1divOut );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, U8,  U8,  0, 0, 0 ):
        case _PACK_SELECT_KEY( F16, U8,  U8,  0, 0, 1 ):
        case _PACK_SELECT_KEY( F16, I8,  I8,  0, 0, 0 ):
        case _PACK_SELECT_KEY( F16, I8,  I8,  0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmF16U8toF32_4x4", &uniGemmF16U8toF32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmF16U8toF32Hi_4x4", &uniGemmF16U8toF32Hi_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16U8MulZptoFp32_4x4", &uniGemmFp16U8MulZptoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "in1outScale", &scaleIn1divOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  U8,  U8,  1, 0, 0 ):
        case _PACK_SELECT_KEY( I8,  I8,  I8,  1, 0, 0 ):
        case _PACK_SELECT_KEY( I16, I16, I16, 1, 0, 0 ):
        case _PACK_SELECT_KEY( F16, F16, F16, 1, 0, 0 ):
        case _PACK_SELECT_KEY( U8,  F16,  U8, 1, 0, 0 ):
        case _PACK_SELECT_KEY( I8,  F16,  I8, 1, 0, 0 ):
        case _PACK_SELECT_KEY( I16, F16, I16, 1, 0, 0 ):
        case _PACK_SELECT_KEY( U8,  I16, I16, 1, 0, 0 ):
        case _PACK_SELECT_KEY( U8,  U8,  U8,  1, 0, 1 ):
        case _PACK_SELECT_KEY( I8,  I8,  I8,  1, 0, 1 ):
        case _PACK_SELECT_KEY( I16, I16, I16, 1, 0, 1 ):
        case _PACK_SELECT_KEY( F16, F16, F16, 1, 0, 1 ):
        case _PACK_SELECT_KEY( U8,  F16,  U8, 1, 0, 1 ):
        case _PACK_SELECT_KEY( I8,  F16,  I8, 1, 0, 1 ):
        case _PACK_SELECT_KEY( I16, F16, I16, 1, 0, 1 ):
        case _PACK_SELECT_KEY( U8,  I16, I16, 1, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32_4x4", &uniConvertUint8SubZpToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32B_4x4", &uniConvertUint8SubZpToFp32B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvert1stFp16ToFp32_4x4", &uniConvert1stFp16ToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input0_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src1ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "outputScale", &reScaleOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  F16, U8,  0, 0, 0 ):
        case _PACK_SELECT_KEY( I8,  F16, I8,  0, 0, 0 ):
        case _PACK_SELECT_KEY( I16, F16, I16, 0, 0, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32_4x4", &uniConvertUint8SubZpToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvert1stFp16ToFp32_4x4", &uniConvert1stFp16ToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input0_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "outputScale", &reScaleOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  F16, U8,  0, 0, 1 ):
        case _PACK_SELECT_KEY( I8,  F16, I8,  0, 0, 1 ):
        case _PACK_SELECT_KEY( I16, F16, I16, 0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16MulZptoFp32_4x4", &uniGemmFp16MulZptoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8F16toF32Lo_4x4b", &uniGemmU8F16toF32Lo_4x4b );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8F16toF32Hi_4x4b", &uniGemmU8F16toF32Hi_4x4b );
                status |= vsi_nn_kernel_gpu_add_param( node, "in0outScale", &scaleIn0divOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  U8, F16,  0, 0, 0 ):
        case _PACK_SELECT_KEY( I8,  I8, F16,  0, 0, 0 ):
        case _PACK_SELECT_KEY( U8,  U8, F16,  0, 0, 1 ):
        case _PACK_SELECT_KEY( I8,  I8, F16,  0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8U8toFp32Block4_4x4", &uniGemmU8U8toFp32Block4_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8U8MulZptoFp32_8x4", &uniGemmU8U8MulZptoFp32_8x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input01Scale", &inScaleMul );
                status |= vsi_nn_kernel_gpu_add_param( node, "mulKIn0In1Zp", &mulKIn0In1Zp );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( I16, I16, F16, 0, 0, 0 ):
        case _PACK_SELECT_KEY( I16, I16, F16, 0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32_4x4", &uniConvertUint8SubZpToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32B_4x4", &uniConvertUint8SubZpToFp32B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input0_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src1ZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  F16, F16,  0, 0, 0 ):
        case _PACK_SELECT_KEY( I8,  F16, F16,  0, 0, 0 ):
        case _PACK_SELECT_KEY( I16, F16, F16,  0, 0, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniGemm1stU8F16toF32Lo_4x4", &uniGemm1stU8F16toF32Lo_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemm2ndU8F16toF32Lo_4x4", &uniGemm2ndU8F16toF32Lo_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemm1stU8F16toF32Hi_4x4", &uniGemm1stU8F16toF32Hi_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemm2ndU8F16toF32Hi_4x4", &uniGemm2ndU8F16toF32Hi_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16MulZptoFp32_4x4", &uniGemmFp16MulZptoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input0Scale", &src0Scale );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  F16, F16,  0, 0, 1 ):
        case _PACK_SELECT_KEY( I8,  F16, F16,  0, 0, 1 ):
        case _PACK_SELECT_KEY( I16, F16, F16,  0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8F16toF32Lo_4x4b", &uniGemmU8F16toF32Lo_4x4b );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8F16toF32Hi_4x4b", &uniGemmU8F16toF32Hi_4x4b );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16MulZptoFp32_4x4", &uniGemmFp16MulZptoFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input0Scale", &src0Scale );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, F16, 0, 0, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16toFp32Row0Lo_4x4", &uniGemmFp16toFp32Row0Lo_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16toFp32Row0Hi_4x4", &uniGemmFp16toFp32Row0Hi_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16toFp32Row1Lo_4x4", &uniGemmFp16toFp32Row1Lo_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmFp16toFp32Row1Hi_4x4", &uniGemmFp16toFp32Row1Hi_4x4 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( BF16, BF16, BF16, 0, 0, 0 ):
        case _PACK_SELECT_KEY( BF16, BF16, BF16, 0, 0, 1 ):
        case _PACK_SELECT_KEY( BF16, BF16, BF16, 0, 1, 0 ):
        case _PACK_SELECT_KEY( BF16, BF16, BF16, 0, 1, 1 ):
        case _PACK_SELECT_KEY( BF16, BF16, BF16, 1, 0, 0 ):
        case _PACK_SELECT_KEY( BF16, BF16, BF16, 1, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniExtractOddData_2x8", &uniExtractOddData_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, F16, 0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8F16toF32Lo_4x4b", &uniGemmU8F16toF32Lo_4x4b );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F32, F32, F32, 0, 0, 0 ):
        case _PACK_SELECT_KEY( F32, F32, F32, 0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8, I16, I16, 0, 0, 0 ):
        case _PACK_SELECT_KEY( U8, I16, I16, 0, 0, 1 ):
        case _PACK_SELECT_KEY( U8, I16, I16, 0, 1, 0 ):
        case _PACK_SELECT_KEY( U8, I16, I16, 0, 1, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertUint8SubZpToFp32_4x4", &uniConvertUint8SubZpToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertUint8SubZpToFp32B_4x4", &uniConvertUint8SubZpToFp32B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input0_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src1ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "outputScale", &reScaleOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
        status = vsi_nn_kernel_gpu_add_param( node, "ac2zero", &ac2zero );
        status |= vsi_nn_kernel_gpu_add_param( node, "bc2zero", &bc2zero );
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }
#undef _PACK_SELECT_KEY

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
} /* _matrix_mul_initializer() */

DEF_KERNEL_INITIALIZER(_matrix_mul_cross_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
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

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    int32_t       transA = 0;
    int32_t       transB = 0;
    int32_t       width  = 0;
    int32_t       height = 0;
    int32_t       axis_size = 0;

    int32_t     src0ZP     = 0;
    float       src0Scale  = 0;
    int32_t     src1ZP     = 0;
    float       src1Scale  = 0;
    float       dstZP      = 0;
    float       dstScale   = 0;

    uint32_t pack_key = 0;

    float    mulKIn0In1Zp  = 0;
    float    inOutScale    = 0;
    int32_t  K             = 0;

    uint32_t  evis2 = 0;
    vx_context  ctx        = vxGetContext((vx_reference)node);
    vx_hardware_caps_params_t   hw_param;

    VSI_UNREFERENCED(param_size);
    memset(&hw_param, 0, sizeof(vx_hardware_caps_params_t));
    status = vxQueryHardwareCaps(ctx, &hw_param, sizeof(vx_hardware_caps_params_t));
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    if (hw_param.evis2 == TRUE)
    {
        evis2 = 1;
    }

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &transA);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &transB);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &K);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[10], &axis_size);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    src0ZP     = attr[0]->zero_point;
    src0Scale  = attr[0]->scale;
    src1ZP     = attr[1]->zero_point;
    src1Scale  = attr[1]->scale;
    dstZP      = (float)attr[2]->zero_point;
    dstScale   = attr[2]->scale;

    mulKIn0In1Zp = (float)((int)(K + 3) / 4 * 4 * src1ZP * src0ZP);
    inOutScale =  src0Scale * src1Scale / dstScale;

    width = (int32_t)(attr[2]->shape->data[0]);
    height = (int32_t)(attr[2]->shape->data[1]);

    gpu_param.global_scale[0]  = 4;
    gpu_param.global_scale[1]  = 4;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((width + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = gpu_align_p2((height + gpu_param.global_scale[1] - 1)
                                        / gpu_param.global_scale[1], 4);
    gpu_param.global_size[2]   = (size_t)axis_size;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, OUT_TYPE, TRANSA, TRANSB, EVIS2)    \
        ((IN0_TYPE << 24) | (IN1_TYPE << 16) | (OUT_TYPE << 8) | (TRANSA << 4) | (TRANSB << 2) | (EVIS2))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype, attr[2]->dtype, transA, transB, evis2);
    {
        uint16_t M0            = 0;
        uint16_t M1            = 0;
        int32_t  postShift0    = 0;
        int32_t  postShift1    = 0;
        uint32_t multiplierA   = 0;
        uint32_t multiplierB   = 0;
        gpu_dp_inst_t uniGemmU8U8MulZptoFp32_8x4 = {{
            0xaaaaaaaa, 0xaaaaaaaa, // TCfg
            0xf02a0600, 0x2a8620e0, 0x0640e8f2, 0x60f0f42b, 0xf8f62b86, // BinSelect
            0x00000700, // AccumType, ConstantType, and PostShift
            0x03020302, 0x03020302, 0x03020302, 0x03020302,
            0x03020302, 0x03020302, 0x03020302, 0x03020302 // Constant
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
        gpu_dp_inst_t uniGemmU8U8toFp32Block4_4x4 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x32103210, 0x32103210, // ABin
            0x55555555, // BSelt
            0xd951c840, 0xfb73ea62, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniGemmU8F16toF32Lo_4x4b = {{
            0x55555555, // TCfg
            0x50505050, // ASelt
            0x51514040, 0x73736262, // ABin
            0x00000000, // BSelt
            0x32103210, 0x32103210, // BBin
            0x00000000, // AccumType, ConstantType, and PostShift
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
        gpu_dp_inst_t uniConvertUint8SubZpToFp32B_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        float reScaleOut = 1 / dstScale;
        uint32_t multiplierU8ZpAB = (src0ZP << 24) | (src1ZP << 16) | (src0ZP << 8) | (src1ZP);
        int32_t i = 8;
        gpu_quantize_multiplier_16bit(src0Scale / 1.0f, &M0, &postShift0);
        gpu_quantize_multiplier_16bit(src1Scale / 1.0f, &M1, &postShift1);

        multiplierA = (M0 << 16) | M0;
        multiplierB = (M1 << 16) | M1;

        uniConvertUint8SubZpToFp32_4x4.data[7] |= (postShift0 & 0x1F);
        uniConvertUint8SubZpToFp32B_4x4.data[7] |= (postShift1 & 0x1F);
        for( i = 8; i < 16; i += 2)
        {
            uniConvertUint8SubZpToFp32_4x4.data[i] = multiplierA;
            uniConvertUint8SubZpToFp32B_4x4.data[i] = multiplierB;
        }
        for( i = 8; i < 16; i++)
        {
            uniGemmU8U8MulZptoFp32_8x4.data[i] = multiplierU8ZpAB;
        }

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( U8,  U8,  U8,  0, 0, 1 ):
        case _PACK_SELECT_KEY( I8,  I8,  I8,  0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8U8toFp32Block4_4x4", &uniGemmU8U8toFp32Block4_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8U8MulZptoFp32_8x4", &uniGemmU8U8MulZptoFp32_8x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "inOutScale", &inOutScale );
                status |= vsi_nn_kernel_gpu_add_param( node, "mulKIn0In1Zp", &mulKIn0In1Zp );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( I16, I16, I16, 0, 0, 0 ):
        case _PACK_SELECT_KEY( I16, I16, I16, 0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32_4x4", &uniConvertUint8SubZpToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32B_4x4", &uniConvertUint8SubZpToFp32B_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input0_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src1ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "outputScale", &reScaleOut );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, F16, 0, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniGemmU8F16toF32Lo_4x4b", &uniGemmU8F16toF32Lo_4x4b );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
    }
#undef _PACK_SELECT_KEY

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
} /* _matrix_mul_cross_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    int32_t transa,
    int32_t transb,
    int32_t cross
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input1_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_MATRIX_MUL_KEY( input0_dtype, input1_dtype, output_dtype, transa, transb, cross);

    for( i = 0; i < _cnt_of_array(matrix_mul_map); i ++ )
    {
        if ( matrix_mul_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(matrix_mul_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  matrix_mul_map[i].function_name );
        if (cross == 1)
        {
            kernel->info.parameters = _matrix_mul_kernel_cross_param_def;
            kernel->info.numParams = _cnt_of_array( _matrix_mul_kernel_cross_param_def );
            kernel->info.initialize = _matrix_mul_cross_initializer;
        }
        else
        {
            kernel->info.parameters = _matrix_mul_kernel_param_def;
            kernel->info.numParams = _cnt_of_array( _matrix_mul_kernel_param_def );
            kernel->info.initialize = _matrix_mul_initializer;
        }

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                matrix_mul_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                matrix_mul_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_MATRIX_MUL_CROSS_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL;
    vsi_nn_tensor_t*  tmp_inputs[2]  = {NULL};
    vsi_nn_tensor_t*  tmp_outputs[1] = {NULL};
    int32_t transposeA  = vsi_nn_kernel_param_get_int32( params, "transposeA" );
    int32_t transposeB  = vsi_nn_kernel_param_get_int32( params, "transposeB" );
    int32_t adjointA  = vsi_nn_kernel_param_get_int32( params, "adjointA" );
    int32_t adjointB  = vsi_nn_kernel_param_get_int32( params, "adjointB" );
    vsi_size_t shapes[3][VSI_NN_MAX_DIM_NUM] = {{0}};
    uint32_t new_rank[3] = {0};
    vsi_size_t M = inputs[0]->attr.size[1];
    vsi_size_t K = inputs[0]->attr.size[0];
    vsi_size_t N = inputs[1]->attr.size[0];
    vsi_size_t depthA = 1, depthB = 1;

    uint32_t cross_flg = 0;
    uint32_t size_axis_in_out[3] = {0};
    uint32_t stride_axis_in_out[9] = {0};

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    if ((inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32
        && inputs[1]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32
        && outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32)
        &&(M % 4 != 0 || K % 4 != 0 || N %4 != 0))
    {
        return NULL;
    }

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = vsi_nn_kernel_optimize_matrixmul_broadcast_shape(
                                       inputs[0]->attr.size,
                                       inputs[1]->attr.size,
                                       outputs[0]->attr.size,
                                       inputs[0]->attr.dim_num,
                                       inputs[1]->attr.dim_num,
                                       outputs[0]->attr.dim_num,
                                       shapes[0], shapes[1], shapes[2], new_rank,
                                       &cross_flg, size_axis_in_out, stride_axis_in_out);
    if (status)
    {
        tmp_inputs[0] = vsi_nn_reshape_tensor(graph, inputs[0], shapes[0], new_rank[0]);
        tmp_inputs[1] = vsi_nn_reshape_tensor(graph, inputs[1], shapes[1], new_rank[1]);
        tmp_outputs[0] = vsi_nn_reshape_tensor(graph, outputs[0], shapes[2], new_rank[2]);

        M = tmp_inputs[0]->attr.size[1];
        K = tmp_inputs[0]->attr.size[0];
        N = tmp_inputs[1]->attr.size[0];
    }
    else
    {
        VSILOGE("illegal inputs shape");
        status = VSI_FAILURE;
        goto final;
    }

    if (transposeA)
    {
        K = tmp_inputs[0]->attr.size[1];
        M = tmp_inputs[0]->attr.size[0];
    }
    else if (transposeB)
    {
        N = tmp_inputs[1]->attr.size[1];
    }

    depthA = tmp_inputs[0]->attr.dim_num > 2 ? tmp_inputs[0]->attr.size[2] : 1;
    depthB = tmp_inputs[1]->attr.dim_num > 2 ? tmp_inputs[1]->attr.size[2] : 1;

    if (M == 1 && depthB == 1 && depthA > 1)
    {
        vsi_size_t  shape[VSI_NN_MAX_DIM_NUM] = {0};
        shape[0] = tmp_inputs[0]->attr.size[0];
        shape[1] = tmp_inputs[0]->attr.size[2];
        shape[2] = 1;
        shape[3] = tmp_inputs[0]->attr.dim_num > 3 ? tmp_inputs[0]->attr.size[3] : 1;
        rs_input = vsi_nn_kernel_tensor_reshape( tmp_inputs[0]->t, shape, 4 );

        shape[0] = tmp_outputs[0]->attr.size[0];
        shape[1] = tmp_outputs[0]->attr.size[2];
        shape[2] = 1;
        shape[3] = tmp_outputs[0]->attr.dim_num > 3 ? tmp_outputs[0]->attr.size[3] : 1;
        rs_output = vsi_nn_kernel_tensor_reshape( tmp_outputs[0]->t, shape, 4 );
    }

    status = _query_kernel( tmp_inputs, tmp_outputs, kernel, transposeA, transposeB, cross_flg );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 3;
            size_t param_num = cross_flg == 1 ? _MATRIX_MUL_CROSS_PARAM_NUM : _MATRIX_MUL_PARAM_NUM;
            /* Pass parameters to node. */
            if (rs_input)
            {
                tmp_params[0] = rs_input;
                tmp_params[1] = (vsi_nn_kernel_node_param_t)(tmp_inputs[1]->t);
                tmp_params[2] = rs_output;
            }
            else
            {
                vsi_nn_kernel_node_pack_io( tmp_params, param_num,
                        tmp_inputs, 2, tmp_outputs, 1 );
            }
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &transposeA );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &transposeB );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &adjointA );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &adjointB );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &M );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &K );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &N );
            if (cross_flg == 1)
            {
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &size_axis_in_out[0] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &size_axis_in_out[1] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &size_axis_in_out[2] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_axis_in_out[0] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_axis_in_out[1] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_axis_in_out[2] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_axis_in_out[3] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_axis_in_out[4] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_axis_in_out[5] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_axis_in_out[6] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_axis_in_out[7] );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_axis_in_out[8] );
            }
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, param_num );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &tmp_params[3] );
            vsi_nn_kernel_scalar_release( &tmp_params[4] );
            vsi_nn_kernel_scalar_release( &tmp_params[5] );
            vsi_nn_kernel_scalar_release( &tmp_params[6] );
            vsi_nn_kernel_scalar_release( &tmp_params[7] );
            vsi_nn_kernel_scalar_release( &tmp_params[8] );
            vsi_nn_kernel_scalar_release( &tmp_params[9] );
            if (cross_flg == 1)
            {
                vsi_nn_kernel_scalar_release( &tmp_params[10] );
                vsi_nn_kernel_scalar_release( &tmp_params[11] );
                vsi_nn_kernel_scalar_release( &tmp_params[12] );
                vsi_nn_kernel_scalar_release( &tmp_params[13] );
                vsi_nn_kernel_scalar_release( &tmp_params[14] );
                vsi_nn_kernel_scalar_release( &tmp_params[15] );
                vsi_nn_kernel_scalar_release( &tmp_params[16] );
                vsi_nn_kernel_scalar_release( &tmp_params[17] );
                vsi_nn_kernel_scalar_release( &tmp_params[18] );
                vsi_nn_kernel_scalar_release( &tmp_params[19] );
                vsi_nn_kernel_scalar_release( &tmp_params[20] );
                vsi_nn_kernel_scalar_release( &tmp_params[21] );
            }
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                border.constant_value.U8 = 0;
                border.constant_value.S16 = 0;
                border.constant_value.U16 = 0;
                border.constant_value.S32 = 0;
                border.constant_value.U32 = 0;
                if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
                {
                    border.constant_value.U8 = (uint8_t)vsi_nn_get_tensor_zero_point(inputs[0]);
                }
                if (K % 4 == 0 && N % 4 == 0)
                {
                    border.mode = VX_BORDER_REPLICATE;
                }
                status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
                CHECK_STATUS(status);
            }
        }
    }
final:
    vsi_safe_release_tensor( tmp_inputs[0] );
    vsi_safe_release_tensor( tmp_inputs[1] );
    vsi_safe_release_tensor( tmp_outputs[0] );
    if (rs_input)
    {
        vsi_nn_kernel_tensor_release( &rs_input );
    }
    if (rs_output)
    {
        vsi_nn_kernel_tensor_release( &rs_output );
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( matrixmul, _setup )
