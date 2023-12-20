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
#include "utils/vsi_nn_dtype_util.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */

typedef enum
{
    INTERNAL_KERNEL_SUMS,
    INTERNAL_KERNEL_MEANS,
    INTERNAL_KERNEL_NORM,
} _internal_kernel_e;

#define KERNEL_SOURCE_0    "group_normalization_0"
#define KERNEL_SOURCE_1    "group_normalization_1"
#define KERNEL_SOURCE_2    "group_normalization_2"

#define HASH_GROUPNORM_SUMS_SH_KERNEL_NAME(SRC0_TYPE) \
    CVIVANTE_NAMESPACE("evis.group_norm_sums_"#SRC0_TYPE)

#define HASH_GROUPNORM_SUMS_SH_KERNEL_2D_NAME(SRC0_TYPE) \
    CVIVANTE_NAMESPACE("evis.group_norm_sums_"#SRC0_TYPE"_2D")

#define HASH_GROUPNORM_MEANS_SH_KERNEL_NAME \
    CVIVANTE_NAMESPACE("evis.group_norm_means")

#define HASH_GROUPNORM_SCALE_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.group_norm_"#SRC0_TYPE"_"#SRC1_TYPE"to"#DST_TYPE)

#define HASH_GROUPNORM_SCALE_SH_KERNEL_2D_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.group_norm_"#SRC0_TYPE"_"#SRC1_TYPE"to"#DST_TYPE"_2D")

// Add kernel hashtable here
// Sum Sqr
#define HASH_GROUPNORM_SUMS_KEY(_input0_type, _output_type, _reshape_flag) \
    ((_input0_type << 24) | (_output_type << 16) | (_reshape_flag << 8))

#define TENSOR_GROUPNORM_SUMS_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_GROUPNORM_SUMS_KEY(IN0_TYPE, OUT_TYPE, 0), \
        HASH_GROUPNORM_SUMS_SH_KERNEL_NAME(IN0_TYPE), \
        SOURCE },

#define TENSOR_GROUPNORM_SUMS_KERNELS_2D(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_GROUPNORM_SUMS_KEY(IN0_TYPE, OUT_TYPE, 1), \
        HASH_GROUPNORM_SUMS_SH_KERNEL_2D_NAME(IN0_TYPE), \
        SOURCE },

#define HASH_GROUPNORM_MEANS_KEY(_input0_type, _output_type) \
    ((_input0_type << 24) | (_output_type << 16))

#define TENSOR_GROUPNORM_MEAN_VARI_KERNELS(SOURCE) \
    { HASH_GROUPNORM_MEANS_KEY(F32, F32), \
        HASH_GROUPNORM_MEANS_SH_KERNEL_NAME, \
        SOURCE },

// normalization
#define HASH_GROUPNORM_KEY(_input0_type, _input1_type, _output_type, _reshape_flag) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_reshape_flag << 4))

#define TENSOR_GROUPNORM_SCALE_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_GROUPNORM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
        HASH_GROUPNORM_SCALE_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_GROUPNORM_SCALE_KERNELS_2D(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_GROUPNORM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1), \
        HASH_GROUPNORM_SCALE_SH_KERNEL_2D_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _groupnorm_sums_kernel_map[] =
{
    // Register kernel here
    TENSOR_GROUPNORM_SUMS_KERNELS(I8, F32, KERNEL_SOURCE_0)
    TENSOR_GROUPNORM_SUMS_KERNELS_2D( I8, F32, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SUMS_KERNELS( U8, F32, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SUMS_KERNELS_2D( U8, F32, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SUMS_KERNELS( I16, F32, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SUMS_KERNELS_2D( I16, F32, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SUMS_KERNELS( F16, F32, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SUMS_KERNELS_2D( F16, F32, KERNEL_SOURCE_2 )
};

static const _kernel_map_type _groupnorm_mean_vari_kernel_map[] =
{
    // Register kernel here
    TENSOR_GROUPNORM_MEAN_VARI_KERNELS( KERNEL_SOURCE_0 )
};

static const _kernel_map_type _groupnorm_kernel_map[] =
{
    // Register kernel here
    TENSOR_GROUPNORM_SCALE_KERNELS( I8, F16, I8, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( I8, F16, I8, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SCALE_KERNELS( I8, F16, F16, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( I8, F16, F16, KERNEL_SOURCE_0 )

    TENSOR_GROUPNORM_SCALE_KERNELS( U8, F16, U8, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( U8, F16, U8, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SCALE_KERNELS( U8, F16, F16, KERNEL_SOURCE_1 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( U8, F16, F16, KERNEL_SOURCE_1 )

    TENSOR_GROUPNORM_SCALE_KERNELS( I16, F16, I16, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( I16, F16, I16, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS( I16, F16, F16, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( I16, F16, F16, KERNEL_SOURCE_2 )

    TENSOR_GROUPNORM_SCALE_KERNELS( F16, F16, F16, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( F16, F16, F16, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS( F16, F16, U8, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( F16, F16, U8, KERNEL_SOURCE_2 )

    TENSOR_GROUPNORM_SCALE_KERNELS( U8, F32, U8, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( U8, F32, U8, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SCALE_KERNELS( U8, F32, F16, KERNEL_SOURCE_1 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( U8, F32, F16, KERNEL_SOURCE_1 )

    TENSOR_GROUPNORM_SCALE_KERNELS( I8, F32, I8, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( I8, F32, I8, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SCALE_KERNELS( I8, F32, F16, KERNEL_SOURCE_0 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( I8, F32, F16, KERNEL_SOURCE_0 )

    TENSOR_GROUPNORM_SCALE_KERNELS( I16, F32, I16, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( I16, F32, I16, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS( I16, F32, F16, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( I16, F32, F16, KERNEL_SOURCE_2 )

    TENSOR_GROUPNORM_SCALE_KERNELS( F16, F32, U8, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( F16, F32, U8, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS( F16, F32, F16, KERNEL_SOURCE_2 )
    TENSOR_GROUPNORM_SCALE_KERNELS_2D( F16, F32, F16, KERNEL_SOURCE_2 )
};

/*
 * Kernel params
 */
static vx_param_description_t _groupnorm_sums_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GROUPNORM_SUMS_PARAM_NUM  _cnt_of_array( _groupnorm_sums_kernel_param_def )

static vx_param_description_t _groupnorm_means_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GROUPNORM_MEANS_PARAM_NUM  _cnt_of_array( _groupnorm_means_kernel_param_def )

static vx_param_description_t _groupnorm_kernel_param_def[] =
{
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
#define _GROUPNORM_PARAM_NUM  _cnt_of_array( _groupnorm_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_groupnorm_sums_initializer)
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
    int32_t is2D = 0;
    int32_t width = 0;
    int32_t height = 0;
    int32_t chn = 0;
    float input_scale = 1;
    float input_scale2 = 1;
    float input_zp = 1;
    float sum_x_tail = 1;
    float sum_x2_tail0 = 1;
    float sum_x2_tail1 = 1;
    float work_item_pixels = 1;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &is2D);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    input_shape = attr[0]->shape;
    input_scale = attr[0]->scale;
    input_scale2 = input_scale * input_scale;
    input_zp    = (float)attr[0]->zero_point;
    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->data[1]);
    chn = (int32_t)(attr[1]->shape->data[1]);
    if (is2D)
    {
        height = 1;
    }

    work_item_pixels = (float)height * 16;

    sum_x_tail = -work_item_pixels * input_zp * input_scale;
    sum_x2_tail0 = work_item_pixels * input_zp * input_zp * input_scale2;
    sum_x2_tail1 = -2 * input_zp * input_scale2;

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
        status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &input_scale);
        status |= vsi_nn_kernel_gpu_add_param(node, "input_scale2", &input_scale2);
        status |= vsi_nn_kernel_gpu_add_param(node, "sum_x_tail", &sum_x_tail);
        status |= vsi_nn_kernel_gpu_add_param(node, "sum_x2_tail0", &sum_x2_tail0);
        status |= vsi_nn_kernel_gpu_add_param(node, "sum_x2_tail1", &sum_x2_tail1);
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
        status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &input_scale);
        status |= vsi_nn_kernel_gpu_add_param(node, "input_scale2", &input_scale2);
        status |= vsi_nn_kernel_gpu_add_param(node, "sum_x_tail", &sum_x_tail);
        status |= vsi_nn_kernel_gpu_add_param(node, "sum_x2_tail0", &sum_x2_tail0);
        status |= vsi_nn_kernel_gpu_add_param(node, "sum_x2_tail1", &sum_x2_tail1);
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

DEF_KERNEL_INITIALIZER(_groupnorm_means_initializer)
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

    vsi_nn_kernel_tensor_attr_t* attr[1] = {NULL};
    int32_t chn = 0;
    int32_t group_stride = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );

    chn = (int32_t)(attr[0]->shape->data[1]);
    group_stride = (int32_t)(attr[0]->shape->data[0]);

    shaderParam.global_scale[0]  = 4;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.local_size[0]    = 16;
    shaderParam.local_size[1]    = 1;
    shaderParam.local_size[2]    = 1;
    shaderParam.global_size[0]   = 16;
    shaderParam.global_size[1]   = chn;
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniResetFp32_4x4 = {{
            0x09090909, // TCfg
                0x00000000, // ASelt
                0x00110000, 0x00330022, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000700, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00000000, 0x00010001, 0x00000000,
                0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status  = vsi_nn_kernel_gpu_add_param(node, "uniResetFp32_4x4", &uniResetFp32_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "group_stride", &group_stride);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }

    return status;
}

DEF_KERNEL_INITIALIZER(_groupnorm_initializer)
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

    vsi_nn_kernel_tensor_attr_t* attr[4] = {NULL, NULL, NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    float input_scale = 1;
    float input_zp = 0;
    float output_scale = 1.0f;
    float output_zp = 0;
    int32_t height = 0, width = 0, chn = 0;
    int32_t is2D = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[4] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );
    attr[3] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &is2D);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    input_shape  = attr[0]->shape;
    input_scale  = attr[0]->scale;
    input_zp = (float)attr[0]->zero_point;
    output_scale = 1.0f / attr[2]->scale;
    output_zp = (float)attr[2]->zero_point;

    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->data[1]);
    chn = (int32_t)(attr[1]->shape->data[1]);
    if (is2D)
    {
        height = 1;
    }

    shaderParam.global_scale[0]  = 16;
    if (attr[0]->dtype == I16 || attr[0]->dtype == F16)
    {
        shaderParam.global_scale[0]  = 8;
    }

    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
                                        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = (height + shaderParam.global_scale[1] - 1)
        / shaderParam.global_scale[1];
    shaderParam.global_size[2]   = chn;
    if (is2D)
    {
        shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
            / shaderParam.global_scale[0], 4);
        shaderParam.global_size[1]   = (chn + shaderParam.global_scale[1] - 1)
            / shaderParam.global_scale[1];
        shaderParam.global_size[2]   = 1;
    }

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

        uint32_t pack_key      = 0;
#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (OUT_TYPE << 8))

        pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[2]->dtype );

        switch( pack_key )
        {
            case _PACK_SELECT_KEY( I8, I8 ):
            case _PACK_SELECT_KEY( U8, U8 ):
            case _PACK_SELECT_KEY( U8, F16 ):
            case _PACK_SELECT_KEY( I8, F16 ):
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
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &input_scale);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_zp", &input_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_0_4x4",
                        &uniDataToFP32_0_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_1_4x4",
                        &uniDataToFP32_1_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_2_4x4",
                        &uniDataToFP32_2_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_3_4x4",
                        &uniDataToFP32_3_4x4);
                    if (attr[2]->dtype != F16)
                    {
                        status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                        status |= vsi_nn_kernel_gpu_add_param(node, "output_scale", &output_scale);
                    }
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( I16, I16 ):
            case _PACK_SELECT_KEY( I16, F16 ):
            case _PACK_SELECT_KEY( F16, F16 ):
            case _PACK_SELECT_KEY( F16, I16 ):
            case _PACK_SELECT_KEY( F16, U8 ):
            case _PACK_SELECT_KEY( F16, I8 ):
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
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_scale", &input_scale);
                    status |= vsi_nn_kernel_gpu_add_param(node, "input_zp", &input_zp);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_0_4x4",
                        &uniDataToFP32_0_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_1_4x4",
                        &uniDataToFP32_1_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_scale", &output_scale);

                    status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
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
    if (attr[3])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[3] );
        attr[3] = NULL;
    }

    return status;
}

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    const uint32_t hashkey,
    _internal_kernel_e kernel_id
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vx_kernel_initialize_f  initializer = NULL;
    vx_param_description_t * param_def = NULL;
    const _kernel_map_type* kernel_map;
    size_t kernel_map_size = 0;
    size_t param_size = 0;
    uint32_t i = 0;

    switch( kernel_id )
    {
        case INTERNAL_KERNEL_SUMS:
            initializer = _groupnorm_sums_initializer;
            kernel_map = _groupnorm_sums_kernel_map;
            kernel_map_size = _cnt_of_array( _groupnorm_sums_kernel_map );
            param_def = _groupnorm_sums_kernel_param_def;
            param_size = _GROUPNORM_SUMS_PARAM_NUM;
            break;
        case INTERNAL_KERNEL_MEANS:
            initializer = _groupnorm_means_initializer;
            kernel_map = _groupnorm_mean_vari_kernel_map;
            kernel_map_size = _cnt_of_array( _groupnorm_mean_vari_kernel_map );
            param_def = _groupnorm_means_kernel_param_def;
            param_size = _GROUPNORM_MEANS_PARAM_NUM;
            break;
        case INTERNAL_KERNEL_NORM:
            initializer = _groupnorm_initializer;
            kernel_map = _groupnorm_kernel_map;
            kernel_map_size = _cnt_of_array( _groupnorm_kernel_map );
            param_def = _groupnorm_kernel_param_def;
            param_size = _GROUPNORM_PARAM_NUM;
            break;
        default:
            VSI_ASSERT( FALSE );
            return VSI_FAILURE;
    }

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == hashkey )
        {
            break;
        }
    }
    if ( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_size;
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                kernel_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
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
#define INTERNAL_KERNEL_SIZE    (2)
#define SUM_SQR_INDEX           (0)
#define MEAN_VARI_INDEX         (1)
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t sums_node_params[_GROUPNORM_SUMS_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t means_node_params[_GROUPNORM_MEANS_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t node_params[_GROUPNORM_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t tmp_node = NULL, tmp_node1 = NULL;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_dtype_e in0_dtype = U8;
    vsi_nn_kernel_dtype_e in2_dtype = F16;
    vsi_nn_kernel_dtype_e out_dtype = U8;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_kernel_t * ikernels[INTERNAL_KERNEL_SIZE] = { NULL };
    vsi_nn_tensor_t * tensors[INTERNAL_KERNEL_SIZE] = { NULL };
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL;
    vsi_size_t new_shape[VSI_NN_MAX_DIM_NUM] = { 1, 1, 1, 1 };
    int32_t is2D_flg = 0;
    uint32_t hashkeys[INTERNAL_KERNEL_SIZE] = { 0 };
    uint32_t hashkey = 0;
    int32_t i = 0;
    float rSpaceOrg = 1.0f / (inputs[0]->attr.size[0] * inputs[0]->attr.size[1]);
    float eps = vsi_nn_kernel_param_get_float32(params, "eps");
    int32_t group_num = vsi_nn_kernel_param_get_int32( params, "group_num" );
    vsi_size_t group_size = inputs[0]->attr.size[2] / group_num;
    float group_ratio = 1.0f / (inputs[0]->attr.size[0] * inputs[0]->attr.size[1] * group_size);

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    // Check if gpu can support the size
    if ( !vsi_nn_kernel_gpu_check_shape(
        outputs[0]->attr.size, outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = vsi_nn_kernel_optimize_group_norm_shape( (const vsi_size_t*)inputs[0]->attr.size,
        inputs[0]->attr.dim_num, group_num, 0, new_shape);
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }
    is2D_flg = (new_shape[2] == 1) && ((int32_t)new_shape[1] == group_num);
    rs_input = vsi_nn_kernel_tensor_reshape(inputs[0]->t, new_shape, 4);
    rs_output = vsi_nn_kernel_tensor_reshape(outputs[0]->t, new_shape, 4);

    for ( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        ikernels[i] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        // Assign unique_id
        ikernels[i]->unique_id = kernel->unique_id;
    }

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in2_dtype = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    hashkeys[SUM_SQR_INDEX]= HASH_GROUPNORM_SUMS_KEY( in0_dtype, F32, is2D_flg );
    hashkeys[MEAN_VARI_INDEX]= HASH_GROUPNORM_MEANS_KEY( F32, F32 );
    hashkey = HASH_GROUPNORM_KEY( in0_dtype, in2_dtype, out_dtype, is2D_flg );

    status = _query_kernel( ikernels[SUM_SQR_INDEX], hashkeys[SUM_SQR_INDEX], INTERNAL_KERNEL_SUMS );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }
    status = _query_kernel( ikernels[MEAN_VARI_INDEX], hashkeys[MEAN_VARI_INDEX], INTERNAL_KERNEL_MEANS );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }
    status = _query_kernel( kernel, hashkey, INTERNAL_KERNEL_NORM );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }

    memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = ((new_shape[0] + 255) / 256) * 4;
    if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT16
        || inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16)
    {
        attr.size[0] = ((new_shape[0] + 127) / 128) * 4;
    }
    attr.size[1] = group_num;
    attr.size[2] = 1;
    attr.size[3] = inputs[0]->attr.dim_num > 3 ? inputs[0]->attr.size[3] : 1;
    attr.dim_num = 4;
    tensors[SUM_SQR_INDEX] = vsi_nn_CreateTensor( graph, &attr );

    attr.size[0] = 4;
    tensors[MEAN_VARI_INDEX] = vsi_nn_CreateTensor( graph, &attr );

    // Sum Sqr
    tmp_node = vsi_nn_kernel_create_node( graph, ikernels[SUM_SQR_INDEX] );
    if (tmp_node)
    {
        uint32_t index = 0;
        sums_node_params[index++] = rs_input;
        sums_node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[SUM_SQR_INDEX]->t;
        sums_node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
        sums_node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &is2D_flg );

        status  = vsi_nn_kernel_node_pass_param( tmp_node, sums_node_params,
            _GROUPNORM_SUMS_PARAM_NUM );
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release( &sums_node_params[2] );
        vsi_nn_kernel_scalar_release( &sums_node_params[3] );
        {
            // Set default border mode.
            vx_border_t border;
            border.mode = VX_BORDER_CONSTANT;
            vsi_nn_Float32ToDtype(0, (uint8_t*)&border.constant_value.U32, &inputs[0]->attr.dtype);
            status = vxSetNodeAttribute( (vx_node)tmp_node, VX_NODE_BORDER, &border, sizeof(border) );
            CHECK_STATUS(status);
        }
    }

    // mean vari
    tmp_node1 = vsi_nn_kernel_create_node( graph, ikernels[MEAN_VARI_INDEX] );
    if (tmp_node1)
    {
        uint32_t index = 0;
        means_node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[SUM_SQR_INDEX]->t;
        means_node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[MEAN_VARI_INDEX]->t;
        means_node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
        means_node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &group_ratio );

        status  = vsi_nn_kernel_node_pass_param( tmp_node1, means_node_params,
            _GROUPNORM_MEANS_PARAM_NUM );
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release( &means_node_params[2] );
        vsi_nn_kernel_scalar_release( &means_node_params[3] );
        {
            // Set default border mode.
            vx_border_t border;
            border.mode = VX_BORDER_CONSTANT;
            border.constant_value.U16 = 0;
            status = vxSetNodeAttribute( (vx_node)tmp_node1, VX_NODE_BORDER, &border, sizeof(border) );
            CHECK_STATUS(status);
        }
    }

    // Nomalization
    node = vsi_nn_kernel_create_node( graph, kernel );
    if (node)
    {
        uint32_t index = 0;
        float  pStride = 0;
        if (!is2D_flg)
        {
            pStride = (float)inputs[1]->attr.size[0] / (float)new_shape[1];
            rSpaceOrg = pStride < 1.0f ? 0.0f : 1.0f / (new_shape[0] / pStride);
        }
        node_params[index++] = rs_input;
        node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[1]->t;
        node_params[index++] = (vsi_nn_kernel_node_param_t)inputs[2]->t;
        node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[MEAN_VARI_INDEX]->t;
        node_params[index++] = rs_output;
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &is2D_flg );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &rSpaceOrg );
        node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &pStride );

        status  = vsi_nn_kernel_node_pass_param( node, node_params,
            _GROUPNORM_PARAM_NUM );
        CHECK_STATUS(status);
        vsi_nn_kernel_scalar_release( &node_params[5] );
        vsi_nn_kernel_scalar_release( &node_params[6] );
        vsi_nn_kernel_scalar_release( &node_params[7] );
        vsi_nn_kernel_scalar_release( &node_params[8] );
    }

    /* Pass parameters to node. */
final:
    if (rs_input)
    {
        vsi_nn_kernel_tensor_release( &rs_input );
    }
    if (rs_output)
    {
        vsi_nn_kernel_tensor_release( &rs_output );
    }
    for( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        if ( ikernels[i] )
        {
            vsi_nn_kernel_release( &ikernels[i] );
        }
        if ( tensors[i] )
        {
            vsi_nn_ReleaseTensor( &tensors[i] );
        }
    }
    if (tmp_node) {vsi_nn_kernel_node_release( &tmp_node );}
    if (tmp_node1) {vsi_nn_kernel_node_release( &tmp_node1 );}
#undef INTERNAL_KERNEL_SIZE
#undef SUM_SQR_INDEX
#undef MEAN_VARI_INDEX
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( group_norm, _setup )
