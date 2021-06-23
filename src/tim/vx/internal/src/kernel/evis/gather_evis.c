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

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define VX_KERNEL_NAME_GATHER_U8TOU8       CVIVANTE_NAMESPACE("evis.gather_U8toU8")
#define VX_KERNEL_NAME_GATHER_I8TOI8       CVIVANTE_NAMESPACE("evis.gather_I8toI8")
#define VX_KERNEL_NAME_GATHER_I16TOI16     CVIVANTE_NAMESPACE("evis.gather_I16toI16")
#define VX_KERNEL_NAME_GATHER_F16TOF16     CVIVANTE_NAMESPACE("evis.gather_F16toF16")
#define VX_KERNEL_NAME_GATHER_I8TOF16      CVIVANTE_NAMESPACE("evis.gather_I8toF16")
#define VX_KERNEL_NAME_GATHER_I16TOF16     CVIVANTE_NAMESPACE("evis.gather_I16toF16")
#define VX_KERNEL_NAME_GATHER_F16TOI8      CVIVANTE_NAMESPACE("evis.gather_F16toI8")
#define VX_KERNEL_NAME_GATHER_F16TOI16     CVIVANTE_NAMESPACE("evis.gather_F16toI16")
#define VX_KERNEL_NAME_GATHER_U8TOF16      CVIVANTE_NAMESPACE("evis.gather_U8toF16")
#define VX_KERNEL_NAME_GATHER_F16TOU8      CVIVANTE_NAMESPACE("evis.gather_F16toU8")

#define VX_KERNEL_NAME_GATHER_AXIS0_U8TOU8    CVIVANTE_NAMESPACE("evis.gather_U8toU8_axis0")
#define VX_KERNEL_NAME_GATHER_AXIS0_I8TOI8    CVIVANTE_NAMESPACE("evis.gather_I8toI8_axis0")
#define VX_KERNEL_NAME_GATHER_AXIS0_I16TOI16  CVIVANTE_NAMESPACE("evis.gather_I16toI16_axis0")
#define VX_KERNEL_NAME_GATHER_AXIS0_F16TOF16  CVIVANTE_NAMESPACE("evis.gather_F16toF16_axis0")
#define VX_KERNEL_NAME_GATHER_AXIS0_I8TOF16   CVIVANTE_NAMESPACE("evis.gather_I8toF16_axis0")
#define VX_KERNEL_NAME_GATHER_AXIS0_I16TOF16  CVIVANTE_NAMESPACE("evis.gather_I16toF16_axis0")
#define VX_KERNEL_NAME_GATHER_AXIS0_F16TOI8   CVIVANTE_NAMESPACE("evis.gather_F16toI8_axis0")
#define VX_KERNEL_NAME_GATHER_AXIS0_F16TOI16  CVIVANTE_NAMESPACE("evis.gather_F16toI16_axis0")
#define VX_KERNEL_NAME_GATHER_AXIS0_U8TOF16   CVIVANTE_NAMESPACE("evis.gather_U8toF16_axis0")
#define VX_KERNEL_NAME_GATHER_AXIS0_F16TOU8   CVIVANTE_NAMESPACE("evis.gather_F16toU8_axis0")

#define VX_KERNEL_NAME_GATHER_ARRAY_U8TOU8    CVIVANTE_NAMESPACE("evis.gather_U8toU8_array")
#define VX_KERNEL_NAME_GATHER_ARRAY_I8TOI8    CVIVANTE_NAMESPACE("evis.gather_I8toI8_array")
#define VX_KERNEL_NAME_GATHER_ARRAY_I16TOI16  CVIVANTE_NAMESPACE("evis.gather_I16toI16_array")
#define VX_KERNEL_NAME_GATHER_ARRAY_F16TOF16  CVIVANTE_NAMESPACE("evis.gather_F16toF16_array")

#define VX_KERNEL_NAME_GATHER_AXIS0_ARRAY_U8TOU8    CVIVANTE_NAMESPACE("evis.gather_U8toU8_axis0_array")
#define VX_KERNEL_NAME_GATHER_AXIS0_ARRAY_I8TOI8    CVIVANTE_NAMESPACE("evis.gather_I8toI8_axis0_array")
#define VX_KERNEL_NAME_GATHER_AXIS0_ARRAY_I16TOI16  CVIVANTE_NAMESPACE("evis.gather_I16toI16_axis0_array")
#define VX_KERNEL_NAME_GATHER_AXIS0_ARRAY_F16TOF16  CVIVANTE_NAMESPACE("evis.gather_F16toF16_axis0_array")

#define KERNEL_SOURCE_1    "gather"
#define KERNEL_SOURCE_2    "gather_mix"
#define KERNEL_SOURCE_3    "gather_array"

// Add kernel hashtable here
#define HASH_GATHER_KEY(_input0_type, _input1_type, _output_type, _is_axis0, _is_max) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_is_axis0 << 4) | (_is_max))

#define TENSOR_GATHER_KERNELS(IN0_TYPE, IN1TYPE, OUT_TYPE, SOURCE) \
    { HASH_GATHER_KEY(IN0_TYPE, IN1TYPE, OUT_TYPE, 0, 0), \
        VX_KERNEL_NAME_GATHER_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

#define TENSOR_GATHER_AXIS0_KERNELS(IN0_TYPE, IN1TYPE, OUT_TYPE, SOURCE) \
    { HASH_GATHER_KEY(IN0_TYPE, IN1TYPE, OUT_TYPE, 1, 0), \
        VX_KERNEL_NAME_GATHER_AXIS0_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

#define TENSOR_GATHER_ARRAY_KERNELS(IN0_TYPE, IN1TYPE, OUT_TYPE, SOURCE) \
    { HASH_GATHER_KEY(IN0_TYPE, IN1TYPE, OUT_TYPE, 0, 1), \
        VX_KERNEL_NAME_GATHER_ARRAY_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

#define TENSOR_GATHER_AXIS0_ARRAY_KERNELS(IN0_TYPE, IN1TYPE, OUT_TYPE, SOURCE) \
    { HASH_GATHER_KEY(IN0_TYPE, IN1TYPE, OUT_TYPE, 1, 1), \
        VX_KERNEL_NAME_GATHER_AXIS0_ARRAY_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } gather_map[] =
{
    TENSOR_GATHER_KERNELS(U8, I32,  U8,          KERNEL_SOURCE_1)
    TENSOR_GATHER_KERNELS(I8, I32,  I8,          KERNEL_SOURCE_1)
    TENSOR_GATHER_KERNELS(I16, I32, I16,         KERNEL_SOURCE_1)
    TENSOR_GATHER_KERNELS(F16, I32, F16,         KERNEL_SOURCE_1)
    TENSOR_GATHER_KERNELS(I8, I32,  F16,         KERNEL_SOURCE_2)
    TENSOR_GATHER_KERNELS(I16, I32, F16,         KERNEL_SOURCE_2)
    TENSOR_GATHER_KERNELS(F16, I32, I8,          KERNEL_SOURCE_2)
    TENSOR_GATHER_KERNELS(F16, I32, I16,         KERNEL_SOURCE_2)
    TENSOR_GATHER_KERNELS(U8, I32,  F16,         KERNEL_SOURCE_2)
    TENSOR_GATHER_KERNELS(F16, I32, U8,          KERNEL_SOURCE_2)
    TENSOR_GATHER_AXIS0_KERNELS(U8, I32,  U8,    KERNEL_SOURCE_1)
    TENSOR_GATHER_AXIS0_KERNELS(I8, I32,  I8,    KERNEL_SOURCE_1)
    TENSOR_GATHER_AXIS0_KERNELS(I16, I32, I16,   KERNEL_SOURCE_1)
    TENSOR_GATHER_AXIS0_KERNELS(F16, I32, F16,   KERNEL_SOURCE_1)
    TENSOR_GATHER_AXIS0_KERNELS(I8, I32,  F16,   KERNEL_SOURCE_2)
    TENSOR_GATHER_AXIS0_KERNELS(I16, I32, F16,   KERNEL_SOURCE_2)
    TENSOR_GATHER_AXIS0_KERNELS(F16, I32, I8,    KERNEL_SOURCE_2)
    TENSOR_GATHER_AXIS0_KERNELS(F16, I32, I16,   KERNEL_SOURCE_2)
    TENSOR_GATHER_AXIS0_KERNELS(U8, I32,  F16,   KERNEL_SOURCE_2)
    TENSOR_GATHER_AXIS0_KERNELS(F16, I32, U8,    KERNEL_SOURCE_2)
    TENSOR_GATHER_ARRAY_KERNELS(U8, I32,  U8,    KERNEL_SOURCE_3)
    TENSOR_GATHER_ARRAY_KERNELS(I8, I32,  I8,    KERNEL_SOURCE_3)
    TENSOR_GATHER_ARRAY_KERNELS(I16, I32, I16,   KERNEL_SOURCE_3)
    TENSOR_GATHER_ARRAY_KERNELS(F16, I32, F16,   KERNEL_SOURCE_3)
    TENSOR_GATHER_AXIS0_ARRAY_KERNELS(U8, I32,  U8,    KERNEL_SOURCE_3)
    TENSOR_GATHER_AXIS0_ARRAY_KERNELS(I8, I32,  I8,    KERNEL_SOURCE_3)
    TENSOR_GATHER_AXIS0_ARRAY_KERNELS(I16, I32, I16,   KERNEL_SOURCE_3)
    TENSOR_GATHER_AXIS0_ARRAY_KERNELS(F16, I32, F16,   KERNEL_SOURCE_3)
};

/*
 * Kernel params
 */
static vx_param_description_t _gather_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GATHER_PARAM_NUM  _cnt_of_array( _gather_kernel_param_def )

static vsi_status get_gather_tensor_reshape_size
    (
    vsi_nn_tensor_t ** inputs,
    int32_t sizes[VSI_NN_MAX_DIM_NUM],
    uint32_t block_size,
    uint32_t idxFlg,
    int32_t* arrayFlg
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t dims_num = inputs[0]->attr.dim_num;
    uint32_t *input_size = inputs[0]->attr.size;
    uint32_t i = 0;
    uint32_t elementCnt = 1;
#define VSI_NN_MAX_IMAGE_WIDTH  (65536)

    for(i = 0; i < dims_num; ++i)
    {
        elementCnt *= input_size[i];
    }

    for(i = 0; i < VSI_NN_MAX_DIM_NUM; ++i)
    {
        sizes[i] = 1;
    }

    if (idxFlg && elementCnt < VSI_NN_MAX_IMAGE_WIDTH)
    {
        sizes[0] = elementCnt;
        sizes[1] = 1;
        status = VSI_SUCCESS;
    }
    else
    {
        sizes[0] = block_size;
        sizes[1] = elementCnt / block_size;
        if ((elementCnt / block_size) > VSI_NN_MAX_IMAGE_WIDTH)
        {
            arrayFlg[0] = 1;
        }
        status = VSI_SUCCESS;
    }
#undef VSI_NN_MAX_IMAGE_WIDTH

    return status;
} /* _get_EltOP_tensor_reshape_size */

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_gather_initializer)
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

    int32_t       block_size  = 0;
    int32_t       block_num   = 0;
    int32_t       indices_num = 1;
    uint32_t      input_dims1 = 0;
    vx_uint32     i           = 0;
    vsi_nn_kernel_tensor_attr_t* attr[3] = {NULL, NULL};
    vsi_int_array_t * input1_shape = NULL;
    int32_t     src0ZP     = 0;
    float       src0Scale  = 0;
    int32_t     dstZP      = 0;
    float       dstScale   = 0;

    uint32_t pack_key = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &block_size);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &block_num);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    src0ZP     = attr[0]->asymm.zero_point;
    src0Scale  = attr[0]->asymm.scale;
    dstZP      = attr[2]->asymm.zero_point;
    dstScale   = attr[2]->asymm.scale;
    if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
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
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        src0Scale = 1;
    }

    if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[2]->dfp.fl > 0)
        {
            dstScale = (float)((int64_t)1 << attr[2]->dfp.fl);
        }
        else
        {
            dstScale = (1.0f / (float)((int64_t)1 << -attr[2]->dfp.fl));
        }
        dstScale = 1.0f/dstScale;
    }
    else if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        dstScale = 1;
    }

    input1_shape  = attr[1]->shape;
    input_dims1   = (uint32_t)input1_shape->size;
    for (i = 0; i < input_dims1; i++)
    {
        indices_num *= input1_shape->data[i];
    }

    shaderParam.global_scale[0]  = 16;
    if (attr[0]->dtype == I16 || attr[0]->dtype == F16)
    {
        shaderParam.global_scale[0]  = 8;
    }
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((block_size + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = indices_num;
    shaderParam.global_size[2]   = block_num;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE)    \
        (IN0_TYPE | (OUT_TYPE << 8))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[2]->dtype);

    {
        uint16_t M0               = 0;
        int32_t  postShift        = 0;
        uint32_t multAndoutZP0[2] = {0};
        uint32_t multAndoutZP1[2] = {0};
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
        gpu_dp_inst_t uniConvertFp16toU8_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( U8, F16):
        case _PACK_SELECT_KEY( I8, F16):
        case _PACK_SELECT_KEY( I16, F16):
            {
                gpu_quantize_multiplier_16bit( (double)src0Scale / dstScale, &M0, &postShift);
                multAndoutZP0[0] = (uint32_t)(M0);
                multAndoutZP0[1] = (uint32_t)((dstZP << postShift) - src0ZP * M0);

                gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_0_Lo_2x8, postShift );
                status = vsi_nn_kernel_gpu_add_param( node,
                    "uniU8MulAndPostShift_0_Lo_2x8", &uniU8MulAndPostShift_0_Lo_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, U8):
        case _PACK_SELECT_KEY( F16, I8):
        case _PACK_SELECT_KEY( F16, I16):
            {
                int32_t  postShift0       = 0;
                gpu_quantize_multiplier_16bit( (double)src0Scale / dstScale, &M0, &postShift0);

                multAndoutZP1[0] = (uint32_t)(M0);
                multAndoutZP1[1] = (uint32_t)((dstZP << postShift0) - src0ZP * M0);

                gpu_dp_inst_update_postshfit( &uniConvertFp16toU8_2x8, postShift0 );
                status = vsi_nn_kernel_gpu_add_param( node, "multAndoutZP1", &multAndoutZP1 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertFp16toU8_2x8", &uniConvertFp16toU8_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
    }
#undef _PACK_SELECT_KEY

    status = vsi_nn_kernel_gpu_add_param(node, "indices_num", &indices_num);
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
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }

    return status;
}

DEF_KERNEL_INITIALIZER(_gather_axis0_initializer)
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

    int32_t       block_num   = 0;
    int32_t       indices_num = 1;
    uint32_t      input_dims1 = 0;
    vx_uint32     i           = 0;
    vsi_nn_kernel_tensor_attr_t* attr[3] = {NULL, NULL};
    vsi_int_array_t * input1_shape = NULL;
    int32_t     src0ZP     = 0;
    float       src0Scale  = 0;
    int32_t     dstZP      = 0;
    float       dstScale   = 0;

    uint32_t pack_key = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &block_num);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    src0ZP     = attr[0]->asymm.zero_point;
    src0Scale  = attr[0]->asymm.scale;
    dstZP      = attr[2]->asymm.zero_point;
    dstScale   = attr[2]->asymm.scale;
    if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
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
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        src0Scale = 1;
    }

    if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[2]->dfp.fl > 0)
        {
            dstScale = (float)((int64_t)1 << attr[2]->dfp.fl);
        }
        else
        {
            dstScale = (1.0f / (float)((int64_t)1 << -attr[2]->dfp.fl));
        }
        dstScale = 1.0f/dstScale;
    }
    else if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        dstScale = 1;
    }

    input1_shape  = attr[1]->shape;
    input_dims1   = (uint32_t)input1_shape->size;
    for (i = 0; i < input_dims1; i++)
    {
        indices_num *= input1_shape->data[i];
    }

    shaderParam.global_scale[0]  = 4;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((indices_num + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = block_num;
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE)    \
        (IN0_TYPE | (OUT_TYPE << 8))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[2]->dtype);

    {
        uint16_t M0               = 0;
        int32_t  postShift        = 0;
        uint32_t multAndoutZP0[2] = {0};
        uint32_t multAndoutZP1[2] = {0};
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
        gpu_dp_inst_t uniConvertFp16toU8_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniExtraCopyDpKeepinEvis_2x8 = {{
            0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( U8, F16):
        case _PACK_SELECT_KEY( I8, F16):
        case _PACK_SELECT_KEY( I16, F16):
            {
                gpu_quantize_multiplier_16bit( (double)src0Scale / dstScale, &M0, &postShift);
                multAndoutZP0[0] = (uint32_t)(M0);
                multAndoutZP0[1] = (uint32_t)((dstZP << postShift) - src0ZP * M0);

                gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_0_Lo_2x8, postShift );
                status = vsi_nn_kernel_gpu_add_param( node,
                    "uniU8MulAndPostShift_0_Lo_2x8", &uniU8MulAndPostShift_0_Lo_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, U8):
        case _PACK_SELECT_KEY( F16, I8):
        case _PACK_SELECT_KEY( F16, I16):
            {
                int32_t  postShift0       = 0;
                gpu_quantize_multiplier_16bit( (double)src0Scale / dstScale, &M0, &postShift0);

                multAndoutZP1[0] = (uint32_t)(M0);
                multAndoutZP1[1] = (uint32_t)((dstZP << postShift0) - src0ZP * M0);

                gpu_dp_inst_update_postshfit( &uniConvertFp16toU8_2x8, postShift0 );
                status = vsi_nn_kernel_gpu_add_param( node, "multAndoutZP1", &multAndoutZP1 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertFp16toU8_2x8", &uniConvertFp16toU8_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( I16,  I16):
        case _PACK_SELECT_KEY( I8,   I8):
        case _PACK_SELECT_KEY( U8,   U8):
        case _PACK_SELECT_KEY( F16,  F16):
        case _PACK_SELECT_KEY( BF16, BF16):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                    "uniExtraCopyDpKeepinEvis_2x8", &uniExtraCopyDpKeepinEvis_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
    }
#undef _PACK_SELECT_KEY

    status = vsi_nn_kernel_gpu_add_param(node, "indices_num", &indices_num);
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
    const vsi_nn_kernel_param_t * params,
    int32_t axis,
    int32_t is_array
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (input0_dtype == BF16)
    {
        input0_dtype = F16;
    }
    if (output_dtype == BF16)
    {
        output_dtype = F16;
    }

    key = HASH_GATHER_KEY( input0_dtype, I32, output_dtype, axis, is_array);

    for( i = 0; i < _cnt_of_array(gather_map); i ++ )
    {
        if ( gather_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(gather_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  gather_map[i].function_name );
        kernel->info.parameters = _gather_kernel_param_def;
        kernel->info.numParams = _cnt_of_array( _gather_kernel_param_def );
        if (axis)
        {
            kernel->info.initialize = _gather_axis0_initializer;
        }
        else
        {
            kernel->info.initialize = _gather_initializer;
        }

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                gather_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                gather_map[i].source_name );
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
#define VSI_NN_MAX_BLOCK_SIZE  (65536)
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t tmp_params[_GATHER_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    int32_t  shapes[3][VSI_NN_MAX_DIM_NUM] = {{0}};
    int32_t block_size  = vsi_nn_kernel_param_get_int32( params, "block_size" );
    int32_t block_num   = vsi_nn_kernel_param_get_int32( params, "block_num" );
    int32_t axis_num    = vsi_nn_kernel_param_get_int32( params, "axis_num" );
    int32_t axis        = vsi_nn_kernel_param_get_int32( params, "axis" );
    int32_t axis0_flg   = 0;
    int32_t is_array    = block_size > VSI_NN_MAX_BLOCK_SIZE ? 1 : 0;

    if (axis == 0)
    {
        status = get_gather_tensor_reshape_size(&inputs[0], shapes[0], axis_num, 0, &is_array);
        status |= get_gather_tensor_reshape_size(&inputs[1], shapes[1], 1, 1, &is_array);
        status |= get_gather_tensor_reshape_size(&outputs[0], shapes[2], shapes[1][0], 0, &is_array);
        axis0_flg = 1;
    }
    else
    {
        status = get_gather_tensor_reshape_size(&inputs[0], shapes[0], block_size, 0, &is_array);
        status |= get_gather_tensor_reshape_size(&inputs[1], shapes[1], 1, 1, &is_array);
        status |= get_gather_tensor_reshape_size(&outputs[0], shapes[2], block_size, 0, &is_array);
        axis0_flg = 0;
    }
#undef VSI_NN_MAX_BLOCK_SIZE
    if (status != VSI_SUCCESS)
    {
        return NULL;
    }

    if ( !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( inputs, outputs, kernel, params, axis0_flg, is_array);
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 0;
#define RESHAPE_DIM 2
            /* Pass parameters to node. */
            tmp_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[0]->t,  shapes[0], RESHAPE_DIM );
            tmp_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[1], RESHAPE_DIM );
            tmp_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[2], RESHAPE_DIM );
#undef RESHAPE_DIM
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &block_size );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &block_num );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis_num );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _GATHER_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_tensor_release( &tmp_params[0] );
            vsi_nn_kernel_tensor_release( &tmp_params[1] );
            vsi_nn_kernel_tensor_release( &tmp_params[2] );
            vsi_nn_kernel_scalar_release( &tmp_params[3] );
            vsi_nn_kernel_scalar_release( &tmp_params[4] );
            vsi_nn_kernel_scalar_release( &tmp_params[5] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( gather, _setup )

