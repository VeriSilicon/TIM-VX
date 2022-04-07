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

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "scatter_nd_update"
#define KERNEL_SOURCE_2    "scatter_nd_update_big"
#define KERNEL_SOURCE_3    "scatter_nd_update_atom"

#define HASH_SCATTER_ND_UPDATE_KEY(_input0_type, _input2_type, _output_type, _pre_op, _large_type) \
    ((_input0_type << 24) | (_input2_type << 16) | (_output_type << 8) | (_pre_op << 4) | (_large_type))

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_NAME(SRC0_TYPE, SRC2_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_"#SRC0_TYPE#SRC2_TYPE"to"#DST_TYPE)

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_BIG_NAME(SRC0_TYPE, SRC2_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_"#SRC0_TYPE#SRC2_TYPE"to"#DST_TYPE"_big")

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_PRE_NAME(SRC0_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_"#SRC0_TYPE"_pre")

 #define HASH_SCATTER_ND_UPDATE_SH_KERNEL_RESET_NAME() \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_reset")

#define TENSOR_SCATTER_ND_UPDATE_KERNELS(IN0_TYPE, IN1_TYPE, IN2_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(IN0_TYPE, IN2_TYPE, OUT_TYPE, 0, 0), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_NAME(IN0_TYPE, IN2_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_UPDATE_BIG_KERNELS(IN0_TYPE, IN1_TYPE, IN2_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(IN0_TYPE, IN2_TYPE, OUT_TYPE, 0, 1), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_BIG_NAME(IN0_TYPE, IN2_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_UPDATE_PRE_KERNELS(IN0_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(IN0_TYPE, I32, I32, 1, 1), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_PRE_NAME(IN0_TYPE), \
        SOURCE },

 #define TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(I32, I32, I32, 2, 1), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_RESET_NAME(), \
        SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type scatter_nd_update_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_KERNELS(I8,   I32, I8,   I8,     KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_UPDATE_KERNELS(U8,   I32, U8,   U8,     KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_UPDATE_KERNELS(I16,  I32, I16,  I16,    KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_UPDATE_KERNELS(F16,  I32, F16,  F16,    KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_UPDATE_KERNELS(BF16, I32, BF16, BF16,   KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_UPDATE_KERNELS(U8,   I32, U8,   F16,    KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_UPDATE_KERNELS(I8,   I32, I8,   F16,    KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_UPDATE_KERNELS(I16,  I32, I16,  F16,    KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_UPDATE_KERNELS(F16,  I32, F16,  U8,     KERNEL_SOURCE_1)
    TENSOR_SCATTER_ND_UPDATE_BIG_KERNELS(F16, I32, F16, F16,  KERNEL_SOURCE_2)
    TENSOR_SCATTER_ND_UPDATE_BIG_KERNELS(F16, I32, F16, U8,   KERNEL_SOURCE_2)
};

static const _kernel_map_type scatter_nd_update_reset_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(KERNEL_SOURCE_3)
};

static const _kernel_map_type scatter_nd_update_pre_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_PRE_KERNELS(U8,     KERNEL_SOURCE_3)
    TENSOR_SCATTER_ND_UPDATE_PRE_KERNELS(I8,     KERNEL_SOURCE_3)
    TENSOR_SCATTER_ND_UPDATE_PRE_KERNELS(I16,    KERNEL_SOURCE_3)
};

static const _kernel_map_type scatter_nd_update_post_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_BIG_KERNELS(U8,  I32, U8,  F16,  KERNEL_SOURCE_3)
    TENSOR_SCATTER_ND_UPDATE_BIG_KERNELS(I8,  I32, I8,  F16,  KERNEL_SOURCE_3)
    TENSOR_SCATTER_ND_UPDATE_BIG_KERNELS(I16, I32, I16, F16,  KERNEL_SOURCE_3)
    TENSOR_SCATTER_ND_UPDATE_BIG_KERNELS(I8,  I32, I8,  I8,   KERNEL_SOURCE_3)
    TENSOR_SCATTER_ND_UPDATE_BIG_KERNELS(U8,  I32, U8,  U8,   KERNEL_SOURCE_3)
    TENSOR_SCATTER_ND_UPDATE_BIG_KERNELS(I16, I32, I16, I16,  KERNEL_SOURCE_3)
};

/*
 * Kernel params
 */
static vx_param_description_t _scatter_nd_update_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _scatter_nd_update_reset_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _scatter_nd_update_pre_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    //{VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _scatter_nd_update_post_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

#define _SCATTER_ND_UPDATE_PARAM_NUM  _cnt_of_array( _scatter_nd_update_kernel_param_def )
#define _SCATTER_ND_UPDATE_PRE_PARAM_NUM  _cnt_of_array( _scatter_nd_update_pre_kernel_param_def )
#define _SCATTER_ND_UPDATE_POST_PARAM_NUM  _cnt_of_array( _scatter_nd_update_post_kernel_param_def )
#define _SCATTER_ND_UPDATE_RESET_PARAM_NUM  _cnt_of_array( _scatter_nd_update_reset_kernel_param_def )

static vsi_status get_scatter_nd_update_tensor_reshape_size
    (
    vsi_nn_tensor_t ** inputs,
    vsi_size_t sizes[VSI_NN_MAX_DIM_NUM],
    uint32_t block_size,
    uint32_t coordDim,
    vsi_size_t* width,
    vsi_size_t* area,
    vsi_size_t* vol,
    int32_t* newDim,
    int32_t* isBig
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t dims_num = inputs[0]->attr.dim_num;
    vsi_size_t *input_size = inputs[0]->attr.size;
    uint32_t i = 0;
    vsi_size_t elementCnt = 1;

    if (coordDim != 0 && (width == NULL || area == NULL))
    {
        return status;
    }

#define VSI_NN_MAX_IMAGE_WIDTH  (65536)

    newDim[0] = 0;
    for(i = 0; i < dims_num; ++i)
    {
        elementCnt *= input_size[i];
    }

    for(i = 0; i < VSI_NN_MAX_DIM_NUM; ++i)
    {
        sizes[i] = 1;
    }

    sizes[0] = block_size;
    sizes[1] = elementCnt / block_size;
    newDim[0] = 2;

    if ((elementCnt / block_size) >= VSI_NN_MAX_IMAGE_WIDTH)
    {
        isBig[0] |= 1;
    }

    if (coordDim == 1) // index shape
    {
        *width = 0;
        *area = 0;
    }
    else if (coordDim == 2)
    {
        *width = input_size[dims_num - 2];
        *area = 0;
    }
    else if (coordDim == 3)
    {
        *width = input_size[dims_num - 3];
        *area = input_size[dims_num - 3] * input_size[dims_num - 2];
    }
    else if (coordDim == 4)
    {
        *width = input_size[dims_num - 4];
        *area = input_size[dims_num - 4] * input_size[dims_num - 3];
        *vol = input_size[dims_num - 4] * input_size[dims_num - 3] * input_size[dims_num - 2];
    }
    else if (coordDim == 5)
    {
        *width = input_size[dims_num - 5];
        *area = input_size[dims_num - 5] * input_size[dims_num - 4];
        *vol = input_size[dims_num - 5] * input_size[dims_num - 4] * input_size[dims_num - 3];
    }
#undef VSI_NN_MAX_IMAGE_WIDTH

    return VSI_SUCCESS;
} /* _get_EltOP_tensor_reshape_size */

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_scatter_nd_update_initializer)
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

    vsi_nn_kernel_tensor_attr_t * attr[4] = { NULL };
    int32_t     block_size = 1;
    int32_t     height     = 1;
    int32_t     index_num  = 1;
    int32_t     width = 0, area = 0, vol = 0;
    int32_t     coord_dim  = 0;
    int32_t     offsetX = 0, offsetY = 0, offsetZ = 0, offsetW = 0, offset_idx = 0;
    int32_t     src0ZP     = 0;
    float       src0Scale  = 1;
    int32_t     src2ZP     = 0;
    float       src2Scale  = 1;
    int32_t     dstZP      = 0;
    float       dstScale   = 1;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );
    attr[3] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &width);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &area);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &vol);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &coord_dim);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    block_size = (int32_t)(attr[3]->shape->data[0]);
    height     = (int32_t)(attr[3]->shape->data[1]);
    index_num  = (int32_t)(attr[1]->shape->data[1]);

    if (attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        src0ZP     = attr[0]->asymm.zero_point;
        src0Scale  = attr[0]->asymm.scale;
    }
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[0]->dfp.fl > 0)
        {
            src0Scale = (1.0f / ((float) ((int64_t)1 << attr[0]->dfp.fl)));
        }
        else
        {
            src0Scale = ((float) ((int64_t)1 << -attr[0]->dfp.fl));
        }
    }

    if (attr[2]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        src2ZP     = attr[2]->asymm.zero_point;
        src2Scale  = attr[2]->asymm.scale;
    }
    else if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[2]->dfp.fl > 0)
        {
            src2Scale = (1.0f / ((float) ((int64_t)1 << attr[2]->dfp.fl)));
        }
        else
        {
            src2Scale = ((float) ((int64_t)1 << -attr[2]->dfp.fl));
        }
    }

    if (attr[3]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        dstZP      = attr[3]->asymm.zero_point;
        dstScale   = attr[3]->asymm.scale;
    }
    else if ( attr[3]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[3]->dfp.fl > 0)
        {
            dstScale = (float)((int64_t)1 << attr[3]->dfp.fl);
        }
        else
        {
            dstScale = (1.0f / (float)((int64_t)1 << -attr[3]->dfp.fl));
        }
        dstScale = 1.0f/dstScale;
    }

    if (coord_dim == 5)
    {
        offset_idx = 1;
    }
    if (coord_dim == 4 || coord_dim == 5)
    {
        offsetX = vol;
        offsetY = area;
        offsetZ = width;
        offsetW = 1;
    }
    else if (coord_dim == 3)
    {
        offsetX = area;
        offsetY = width;
        offsetZ = 1;
        offsetW = 0;
    }
    else if (coord_dim == 2)
    {
        offsetX = width;
        offsetY = 1;
        offsetZ = 0;
        offsetW = 0;
    }
    else if (coord_dim == 1)
    {
        offsetX = 1;
        offsetY = 0;
        offsetZ = 0;
        offsetW = 0;
    }

    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((block_size + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        uint16_t M0                = 0;
        uint16_t M1                = 0;
        int32_t  postShift0        = 0;
        int32_t  postShift1        = 0;
        uint32_t multAndoutZP0[2]  = {0};
        uint32_t multAndoutZP1[2]  = {0};
        gpu_dp_inst_t uniAccumulateSum_2x8 = {{
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x33221100, 0x77665544, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniU8MulAndPostShift_0_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniU8MulAndPostShift_1_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
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

        gpu_quantize_multiplier_16bit( (double)src0Scale / dstScale, &M0, &postShift0);
        gpu_quantize_multiplier_16bit( (double)src2Scale / dstScale, &M1, &postShift1);
        multAndoutZP0[0] = (uint32_t)(M0);
        multAndoutZP0[1] = (uint32_t)((dstZP << postShift0) - src0ZP * M0);
        multAndoutZP1[0] = (uint32_t)(M1);
        multAndoutZP1[1] = (uint32_t)((dstZP << postShift1) - src2ZP * M1);
        gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_0_Lo_2x8, postShift0 );
        gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_1_Lo_2x8, postShift1 );

        status = vsi_nn_kernel_gpu_add_param( node,
                    "uniAccumulateSum_2x8", &uniAccumulateSum_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniU8MulAndPostShift_0_Lo_2x8", &uniU8MulAndPostShift_0_Lo_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniU8MulAndPostShift_1_Lo_2x8", &uniU8MulAndPostShift_1_Lo_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
        status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP1", &multAndoutZP1 );
        status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniExtractOddData_2x8", &uniExtractOddData_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node, "index_num", &index_num );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetX", &offsetX );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetY", &offsetY );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetZ", &offsetZ );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetW", &offsetW );
        status |= vsi_nn_kernel_gpu_add_param( node, "offset_idx", &offset_idx );
        CHECK_STATUS_FAIL_GOTO(status, OnError);
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
    if (attr[3])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[3] );
        attr[3] = NULL;
    }
    return status;
} /* _scatter_nd_update_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_big_initializer)
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

    vsi_nn_kernel_tensor_attr_t * attr[4] = { NULL };
    int32_t     block_size = 1;
    int32_t     height     = 1;
    int32_t     index_num  = 1;
    int32_t     width = 0, area = 0, vol = 0;
    int32_t     coord_dim  = 0;
    int32_t     offsetX = 0, offsetY = 0, offsetZ = 0, offsetW = 0, offset_idx = 0;
    int32_t     src0ZP     = 0;
    float       src0Scale  = 1;
    int32_t     src2ZP     = 0;
    float       src2Scale  = 1;
    int32_t     dstZP      = 0;
    float       dstScale   = 1;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );
    attr[3] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &width);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &area);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &vol);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &coord_dim);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    block_size = (int32_t)(attr[3]->shape->data[0]);
    height     = (int32_t)(attr[3]->shape->data[1]);
    index_num  = (int32_t)(attr[1]->shape->data[1]);

    if (attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        src0ZP     = attr[0]->asymm.zero_point;
        src0Scale  = attr[0]->asymm.scale;
    }
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[0]->dfp.fl > 0)
        {
            src0Scale = (1.0f / ((float) ((int64_t)1 << attr[0]->dfp.fl)));
        }
        else
        {
            src0Scale = ((float) ((int64_t)1 << -attr[0]->dfp.fl));
        }
    }

    if (attr[2]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        src2ZP     = attr[2]->asymm.zero_point;
        src2Scale  = attr[2]->asymm.scale;
    }
    else if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[2]->dfp.fl > 0)
        {
            src2Scale = (1.0f / ((float) ((int64_t)1 << attr[2]->dfp.fl)));
        }
        else
        {
            src2Scale = ((float) ((int64_t)1 << -attr[2]->dfp.fl));
        }
    }

    if (attr[3]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        dstZP      = attr[3]->asymm.zero_point;
        dstScale   = attr[3]->asymm.scale;
    }
    else if ( attr[3]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[3]->dfp.fl > 0)
        {
            dstScale = (float)((int64_t)1 << attr[3]->dfp.fl);
        }
        else
        {
            dstScale = (1.0f / (float)((int64_t)1 << -attr[3]->dfp.fl));
        }
        dstScale = 1.0f / dstScale;
    }

    if (coord_dim == 5)
    {
        offset_idx = 1;
    }
    if (coord_dim == 4 || coord_dim == 5)
    {
        offsetX = vol;
        offsetY = area;
        offsetZ = width;
        offsetW = 1;
    }
    else if (coord_dim == 3)
    {
        offsetX = area;
        offsetY = width;
        offsetZ = 1;
    }
    else if (coord_dim == 2)
    {
        offsetX = width;
        offsetY = 1;
        offsetZ = 0;
    }
    else if (coord_dim == 1)
    {
        offsetX = 1;
        offsetY = 0;
        offsetZ = 0;
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = block_size;
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        uint16_t M0                = 0;
        uint16_t M1                = 0;
        int32_t  postShift0        = 0;
        int32_t  postShift1        = 0;
        uint32_t multAndoutZP0[2]  = {0};
        uint32_t multAndoutZP1[2]  = {0};
        gpu_dp_inst_t uniAccumulateSum_2x8 = {{
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x33221100, 0x77665544, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniU8MulAndPostShift_0_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniU8MulAndPostShift_1_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_quantize_multiplier_16bit( (double)src0Scale / dstScale, &M0, &postShift0);
        gpu_quantize_multiplier_16bit( (double)src2Scale / dstScale, &M1, &postShift1);
        multAndoutZP0[0] = (uint32_t)(M0);
        multAndoutZP0[1] = (uint32_t)((dstZP << postShift0) - src0ZP * M0);
        multAndoutZP1[0] = (uint32_t)(M1);
        multAndoutZP1[1] = (uint32_t)((dstZP << postShift1) - src2ZP * M1);
        gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_0_Lo_2x8, postShift0 );
        gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_1_Lo_2x8, postShift1 );

        status = vsi_nn_kernel_gpu_add_param( node,
                    "uniAccumulateSum_2x8", &uniAccumulateSum_2x8 );
        if (attr[3]->quant != VSI_NN_KERNEL_QUANT_NONE)
        {
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniU8MulAndPostShift_0_Lo_2x8", &uniU8MulAndPostShift_0_Lo_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniU8MulAndPostShift_1_Lo_2x8", &uniU8MulAndPostShift_1_Lo_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
            status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP1", &multAndoutZP1 );
        }
        status |= vsi_nn_kernel_gpu_add_param( node, "index_num", &index_num );
        status |= vsi_nn_kernel_gpu_add_param( node, "update_width", &block_size );
        status |= vsi_nn_kernel_gpu_add_param( node, "output_width", &block_size );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetX", &offsetX );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetY", &offsetY );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetZ", &offsetZ );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetW", &offsetW );
        status |= vsi_nn_kernel_gpu_add_param( node, "offset_idx", &offset_idx );
        CHECK_STATUS_FAIL_GOTO(status, OnError);
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
    if (attr[3])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[3] );
        attr[3] = NULL;
    }
    return status;
} /* _scatter_nd_update_big_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_pre_initializer)
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
    int32_t     block_size = 1;
    int32_t     update_width = 1;
    int32_t     index_num  = 1;
    int32_t     width = 0, area = 0, vol = 0;
    int32_t     coord_dim  = 0;
    int32_t     offsetX = 0, offsetY = 0, offsetZ = 0, offsetW = 0, offset_idx = 0;
    int32_t     src0ZP     = 0;
    float       src0Scale  = 1;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &width);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &area);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &vol);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &coord_dim);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    block_size   = (int32_t)(attr[2]->shape->data[0]);
    update_width = (int32_t)(attr[1]->shape->data[0]);
    index_num    = (int32_t)(attr[0]->shape->data[1]);

    if (attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        src0ZP     = attr[1]->asymm.zero_point;
        src0Scale  = attr[1]->asymm.scale;
    }
    else if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[1]->dfp.fl > 0)
        {
            src0Scale = (1.0f / ((float) ((int64_t)1 << attr[1]->dfp.fl)));
        }
        else
        {
            src0Scale = ((float) ((int64_t)1 << -attr[1]->dfp.fl));
        }
    }

    if (coord_dim == 5)
    {
        offset_idx = 1;
    }
    if (coord_dim == 4 || coord_dim == 5)
    {
        offsetX = vol;
        offsetY = area;
        offsetZ = width;
        offsetW = 1;
    }
    else if (coord_dim == 3)
    {
        offsetX = area;
        offsetY = width;
        offsetZ = 1;
    }
    else if (coord_dim == 2)
    {
        offsetX = width;
        offsetY = 1;
        offsetZ = 0;
    }
    else if (coord_dim == 1)
    {
        offsetX = 1;
        offsetY = 0;
        offsetZ = 0;
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = block_size;
    gpu_param.global_size[1]   = index_num;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniConvert1stUint8SubZpToFp32_4x4 = {{
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000,
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvert1stUint8SubZpToFp32_4x4", &uniConvert1stUint8SubZpToFp32_4x4 );
        status |= vsi_nn_kernel_gpu_add_param( node, "update_width", &update_width );
        status |= vsi_nn_kernel_gpu_add_param( node, "output_width", &block_size );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetX", &offsetX );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetY", &offsetY );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetZ", &offsetZ );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetW", &offsetW );
        status |= vsi_nn_kernel_gpu_add_param( node, "offset_idx", &offset_idx );
        status |= vsi_nn_kernel_gpu_add_param( node, "input_zp", &src0ZP );
        status |= vsi_nn_kernel_gpu_add_param( node, "input_scale", &src0Scale );
        CHECK_STATUS_FAIL_GOTO(status, OnError);
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
} /* _scatter_nd_update_pre_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_post_initializer)
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
    int32_t     block_size = 1;
    int32_t     height     = 1;
    int32_t     width = 0, area = 0, vol = 0;
    int32_t     coord_dim  = 0;
    int32_t     offsetX = 0, offsetY = 0, offsetZ = 0, offsetW = 0, offset_idx = 0;
    int32_t     src0ZP     = 0;
    float       src0Scale  = 1;
    float       src2Scale  = 1;
    int32_t     dstZP      = 0;
    float       dstScale   = 1;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );  // ref
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[4] );  // update
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[5] );  // output
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &width);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &area);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &vol);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[9], &coord_dim);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    block_size = (int32_t)(attr[2]->shape->data[0]);
    height     = (int32_t)(attr[2]->shape->data[1]);

    if (attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        src0ZP     = attr[0]->asymm.zero_point;
        src0Scale  = attr[0]->asymm.scale;
    }
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[0]->dfp.fl > 0)
        {
            src0Scale = (1.0f / ((float) ((int64_t)1 << attr[0]->dfp.fl)));
        }
        else
        {
            src0Scale = ((float) ((int64_t)1 << -attr[0]->dfp.fl));
        }
    }

    if (attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        src2Scale  = attr[1]->asymm.scale;
    }
    else if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[1]->dfp.fl > 0)
        {
            src2Scale = (1.0f / ((float) ((int64_t)1 << attr[1]->dfp.fl)));
        }
        else
        {
            src2Scale = ((float) ((int64_t)1 << -attr[1]->dfp.fl));
        }
    }

    if (attr[2]->quant == VSI_NN_KERNEL_QUANT_ASYMM)
    {
        dstZP      = attr[2]->asymm.zero_point;
        dstScale   = attr[2]->asymm.scale;
    }
    else if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[2]->dfp.fl > 0)
        {
            dstScale = (float)((int64_t)1 << attr[2]->dfp.fl);
        }
        else
        {
            dstScale = (1.0f / (float)((int64_t)1 << -attr[2]->dfp.fl));
        }
        dstScale = 1.0f / dstScale;
        dstZP = 0;
    }

    if (coord_dim == 5)
    {
        offset_idx = 1;
    }
    if (coord_dim == 4 || coord_dim == 5)
    {
        offsetX = vol;
        offsetY = area;
        offsetZ = width;
        offsetW = 1;
    }
    else if (coord_dim == 3)
    {
        offsetX = area;
        offsetY = width;
        offsetZ = 1;
    }
    else if (coord_dim == 2)
    {
        offsetX = width;
        offsetY = 1;
        offsetZ = 0;
    }
    else if (coord_dim == 1)
    {
        offsetX = 1;
        offsetY = 0;
        offsetZ = 0;
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = block_size;
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        uint16_t M0                = 0;
        int32_t  postShift0        = 0;
        uint32_t multAndoutZP0[2]  = {0};
        gpu_dp_inst_t uniU8MulAndPostShift_0_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
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
        float output_zp = (float)dstZP;
        float scaleInOut = src2Scale / dstScale;

        gpu_quantize_multiplier_16bit( (double)src0Scale / dstScale, &M0, &postShift0);
        multAndoutZP0[0] = (uint32_t)(M0);
        multAndoutZP0[1] = (uint32_t)((dstZP << postShift0) - src0ZP * M0);
        gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_0_Lo_2x8, postShift0 );

        status = vsi_nn_kernel_gpu_add_param( node,
                    "uniU8MulAndPostShift_0_Lo_2x8", &uniU8MulAndPostShift_0_Lo_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
        status |= vsi_nn_kernel_gpu_add_param( node, "output_width", &block_size );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetX", &offsetX );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetY", &offsetY );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetZ", &offsetZ );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetW", &offsetW );
        status |= vsi_nn_kernel_gpu_add_param( node, "offset_idx", &offset_idx );
        status |= vsi_nn_kernel_gpu_add_param( node, "input_scale", &src2Scale );
        status |= vsi_nn_kernel_gpu_add_param( node, "output_zp", &output_zp );
        status |= vsi_nn_kernel_gpu_add_param( node, "scaleInOut", &scaleInOut );
        status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
        CHECK_STATUS_FAIL_GOTO(status, OnError);
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
} /* _scatter_nd_update_post_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_reset_initializer)
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

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    int32_t     block_size = 1;
    int32_t     width = 0;
    int32_t     height = 0;
    int32_t     count_width = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );

    block_size   = (int32_t)(attr[0]->shape->data[0]);
    height = (int32_t)(attr[0]->shape->data[1]);
    width = (int32_t)(block_size * height);
    count_width = (int32_t)((height + 3) / 4);

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = (width + 3) / 4;
    gpu_param.global_size[1]   = 1;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    status = vsi_nn_kernel_gpu_add_param( node, "count_width", &count_width );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _scatter_nd_update_reset_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    int32_t coord_dim,
    int32_t isBig
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input2_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input2_dtype = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_SCATTER_ND_UPDATE_KEY( input0_dtype, input2_dtype, output_dtype, 0, isBig );

    for( i = 0; i < _cnt_of_array(scatter_nd_update_map); i ++ )
    {
        if ( scatter_nd_update_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(scatter_nd_update_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  scatter_nd_update_map[i].function_name );
        kernel->info.parameters = _scatter_nd_update_kernel_param_def;
        kernel->info.numParams = _cnt_of_array( _scatter_nd_update_kernel_param_def );
        if (isBig)
        {
            kernel->info.initialize = _scatter_nd_update_big_initializer;
        }
        else
        {
            kernel->info.initialize = _scatter_nd_update_initializer;
        }

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

static vsi_status _query_kernel_large
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel_reset,
    vsi_nn_kernel_t* kernel_pre,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input2_dtype = F16;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input2_dtype = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_SCATTER_ND_UPDATE_KEY( input0_dtype, I32, I32, 1, 1 );

    for( i = 0; i < _cnt_of_array(scatter_nd_update_pre_map); i ++ )
    {
        if ( scatter_nd_update_pre_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(scatter_nd_update_pre_map) )
    {
        snprintf( kernel_pre->info.name, VX_MAX_KERNEL_NAME, "%s",  scatter_nd_update_pre_map[i].function_name );
        kernel_pre->info.parameters = _scatter_nd_update_pre_kernel_param_def;
        kernel_pre->info.numParams = _SCATTER_ND_UPDATE_PRE_PARAM_NUM;
        kernel_pre->info.initialize = _scatter_nd_update_pre_initializer;

        vsi_nn_kernel_add_source( kernel_pre, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_pre_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_pre, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_pre_map[i].source_name );
    }
    else
    {
        status = VSI_FAILURE;
    }


    key = HASH_SCATTER_ND_UPDATE_KEY( input0_dtype, input2_dtype, output_dtype, 0, 1 );

    for( i = 0; i < _cnt_of_array(scatter_nd_update_post_map); i ++ )
    {
        if ( scatter_nd_update_post_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(scatter_nd_update_post_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  scatter_nd_update_post_map[i].function_name );
        kernel->info.parameters = _scatter_nd_update_post_kernel_param_def;
        kernel->info.numParams = _SCATTER_ND_UPDATE_POST_PARAM_NUM;
        kernel->info.initialize = _scatter_nd_update_post_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_post_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_post_map[i].source_name );
    }
    else
    {
        status |= VSI_FAILURE;
    }

    key = HASH_SCATTER_ND_UPDATE_KEY( I32, I32, I32, 2, 1 );

    for( i = 0; i < _cnt_of_array(scatter_nd_update_reset_map); i ++ )
    {
        if ( scatter_nd_update_reset_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(scatter_nd_update_reset_map) )
    {
        snprintf( kernel_reset->info.name, VX_MAX_KERNEL_NAME, "%s",  scatter_nd_update_reset_map[i].function_name );
        kernel_reset->info.parameters = _scatter_nd_update_reset_kernel_param_def;
        kernel_reset->info.numParams = _SCATTER_ND_UPDATE_RESET_PARAM_NUM;
        kernel_reset->info.initialize = _scatter_nd_update_reset_initializer;

        vsi_nn_kernel_add_source( kernel_reset, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_reset_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_reset, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_reset_map[i].source_name );
    }
    else
    {
        status |= VSI_FAILURE;
    }
    return status;
} /* _query_kernel_large() */

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
    vsi_nn_kernel_node_param_t tmp_params[_SCATTER_ND_UPDATE_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t  shapes[3][VSI_NN_MAX_DIM_NUM] = {{0}};
    int32_t block_size  = vsi_nn_kernel_param_get_int32( params, "block_size" );
    int32_t coord_dim   = vsi_nn_kernel_param_get_int32( params, "coord_dim" );
    int32_t rs_in_dim = 0, rs_idx_dim = 0, rs_out_dim = 0;
    vsi_size_t width = 0, area = 0, vol = 0;
    int32_t big_flg = 0;
    vsi_nn_kernel_dtype_e update_dtype = vsi_nn_kernel_map_dtype(inputs[2]->attr.dtype.vx_type);
    int32_t i = 0;

    status = get_scatter_nd_update_tensor_reshape_size(&inputs[1], shapes[0], coord_dim, 0,
                                                    NULL, NULL, NULL, &rs_idx_dim, &big_flg);
    status |= get_scatter_nd_update_tensor_reshape_size(&inputs[2], shapes[1], block_size, 0,
                                                    NULL, NULL, NULL, &rs_in_dim, &big_flg);
    status |= get_scatter_nd_update_tensor_reshape_size(&outputs[0], shapes[2], block_size, coord_dim,
                                                    &width, &area, &vol, &rs_out_dim, &big_flg);
    if (status != VSI_SUCCESS)
    {
        return NULL;
    }

    if ((update_dtype == U8 || update_dtype == I8 || update_dtype == I16))
    {
        vsi_nn_tensor_attr_t attr;
        vsi_nn_kernel_node_t tmp_node = NULL;
        vsi_nn_kernel_node_t reset_node = NULL;
        vsi_nn_kernel_node_param_t pre_params[_SCATTER_ND_UPDATE_PRE_PARAM_NUM] = { NULL };
        vsi_nn_kernel_node_param_t node_params[_SCATTER_ND_UPDATE_POST_PARAM_NUM] = { NULL };
        vsi_nn_kernel_node_param_t reset_params[_SCATTER_ND_UPDATE_RESET_PARAM_NUM] = { NULL };
        vsi_nn_kernel_t * ikernels[2] = { NULL };
        vsi_nn_tensor_t * tensors[3] = { NULL };

        ikernels[0] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        ikernels[0]->unique_id = kernel->unique_id;
        ikernels[1] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        ikernels[1]->unique_id = kernel->unique_id;

        memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.is_const = FALSE;
        attr.vtl = TRUE;

        for(i = 0; i < rs_out_dim; i++)
        {
            attr.size[i] = shapes[2][i];
        }
        attr.dim_num = rs_out_dim;

        tensors[0] = vsi_nn_CreateTensor( graph, &attr );
        attr.size[0] = 1;
        tensors[1] = vsi_nn_CreateTensor( graph, &attr );
        attr.size[1] = 1;
        tensors[2] = vsi_nn_CreateTensor( graph, &attr );

        status = _query_kernel_large( inputs, outputs, ikernels[0], ikernels[1], kernel);
        if ( VSI_SUCCESS == status)
        {
            // reset count
            reset_node = vsi_nn_kernel_create_node( graph, ikernels[0] );
            if (reset_node)
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                reset_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[0], rs_idx_dim );
                reset_params[index++] = (vsi_nn_kernel_node_param_t)tensors[0]->t;
                reset_params[index++] = (vsi_nn_kernel_node_param_t)tensors[1]->t;
                status = vsi_nn_kernel_node_pass_param( reset_node, reset_params, _SCATTER_ND_UPDATE_RESET_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &reset_params[0] );
            }

            // pre-process
            tmp_node = vsi_nn_kernel_create_node( graph, ikernels[1] );
            if (tmp_node)
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                pre_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[0], rs_idx_dim );
                pre_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[2]->t,  shapes[1], rs_in_dim );
                pre_params[index++] = (vsi_nn_kernel_node_param_t)tensors[0]->t;
                pre_params[index++] = (vsi_nn_kernel_node_param_t)tensors[1]->t;
                pre_params[index++] = (vsi_nn_kernel_node_param_t)tensors[2]->t;
                pre_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
                pre_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &area );
                pre_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &vol );
                pre_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &coord_dim );
                status = vsi_nn_kernel_node_pass_param( tmp_node, pre_params, _SCATTER_ND_UPDATE_PRE_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &pre_params[0] );
                vsi_nn_kernel_tensor_release( &pre_params[1] );
                vsi_nn_kernel_scalar_release( &pre_params[5] );
                vsi_nn_kernel_scalar_release( &pre_params[6] );
                vsi_nn_kernel_scalar_release( &pre_params[7] );
                vsi_nn_kernel_scalar_release( &pre_params[8] );
            }

            node = vsi_nn_kernel_create_node( graph, kernel );
            if ( node )
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                node_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[0]->t,  shapes[2], rs_out_dim );
                node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[0]->t;
                node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[1]->t;
                node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[2]->t;
                node_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[2]->t,  shapes[1], rs_in_dim );
                node_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[2], rs_out_dim );
                node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
                node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &area );
                node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &vol );
                node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &coord_dim );
                status = vsi_nn_kernel_node_pass_param( node, node_params, _SCATTER_ND_UPDATE_POST_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &node_params[0] );
                vsi_nn_kernel_tensor_release( &node_params[4] );
                vsi_nn_kernel_tensor_release( &node_params[5] );
                vsi_nn_kernel_scalar_release( &node_params[6] );
                vsi_nn_kernel_scalar_release( &node_params[7] );
                vsi_nn_kernel_scalar_release( &node_params[8] );
                vsi_nn_kernel_scalar_release( &node_params[9] );
            }
        }

        if ( ikernels[0] )
        {
            vsi_nn_kernel_release( &ikernels[0] );
        }
        if ( ikernels[1] )
        {
            vsi_nn_kernel_release( &ikernels[1] );
        }
        if ( tensors[0] )
        {
            vsi_nn_ReleaseTensor( &tensors[0] );
        }
        if ( tensors[1] )
        {
            vsi_nn_ReleaseTensor( &tensors[1] );
        }
        if ( tensors[2] )
        {
            vsi_nn_ReleaseTensor( &tensors[2] );
        }
        if (reset_node) {vsi_nn_kernel_node_release( &reset_node );}
        if (tmp_node) {vsi_nn_kernel_node_release( &tmp_node );}
    }
    else
    {
        status = _query_kernel( inputs, outputs, kernel, coord_dim, big_flg);
        if ( VSI_SUCCESS == status)
        {
            node = vsi_nn_kernel_create_node( graph, kernel );
            if ( node )
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                tmp_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[0]->t,  shapes[2], rs_out_dim );
                tmp_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[0], rs_idx_dim );
                tmp_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[2]->t,  shapes[1], rs_in_dim );
                //tmp_params[index++] = (vsi_nn_kernel_node_param_t)inputs[2]->t;
                tmp_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[2], rs_out_dim );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &area );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &vol );
                tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &coord_dim );
                status = vsi_nn_kernel_node_pass_param( node, tmp_params, _SCATTER_ND_UPDATE_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &tmp_params[0] );
                vsi_nn_kernel_tensor_release( &tmp_params[1] );
                vsi_nn_kernel_tensor_release( &tmp_params[2] );
                vsi_nn_kernel_tensor_release( &tmp_params[3] );
                vsi_nn_kernel_scalar_release( &tmp_params[4] );
                vsi_nn_kernel_scalar_release( &tmp_params[5] );
                vsi_nn_kernel_scalar_release( &tmp_params[6] );
                vsi_nn_kernel_scalar_release( &tmp_params[7] );
            }
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( scatter_nd_update, _setup )
