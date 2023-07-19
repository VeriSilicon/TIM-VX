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

#define KERNEL_SOURCE_0    "minimum_0",
#define KERNEL_SOURCE_1    "minimum_1",

#define HASH_MINIMUM_KEY(_input0_type, _input1_type, _output_type, _image_2d) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_image_2d))

#define HASH_MINIMUM_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.minimum_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE)

#define HASH_MINIMUM_SH_KERNEL_2D_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.minimum_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_MIN_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MINIMUM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
        HASH_MINIMUM_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MIN_KERNELS_2D(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MINIMUM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1), \
        HASH_MINIMUM_SH_KERNEL_2D_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MIN_KERNELS_HALF(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MINIMUM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0), \
        HASH_MINIMUM_SH_KERNEL_NAME(F16, F16, F16), \
        SOURCE },

#define TENSOR_MIN_KERNELS_2D_HALF(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MINIMUM_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1), \
        HASH_MINIMUM_SH_KERNEL_2D_NAME(F16, F16, F16), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } kernel_map[] =
{
    TENSOR_MIN_KERNELS_HALF(F16, F16, F16,       KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS_HALF(BF16, BF16, BF16,    KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS(I8,  I8, I8,         KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS(U8,  U8, U8,         KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS(U8,  U8, I16,        KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS(I16, I16, I16,       KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS(I16, I16, U8,        KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS(F16, F16, I8,        KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS(F16, F16, U8,        KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS(F16, F16, I16,       KERNEL_SOURCE_0)

    TENSOR_MIN_KERNELS(I8,  I8,  F16,       KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS(I8,  F16, I8,        KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS(I8,  F16, F16,       KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS(U8,  U8,  F16,       KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS(U8,  F16, U8,        KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS(I16, I16, F16,       KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS(U8,  F16, F16,       KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS(I16, F16, I16,       KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS(I16, F16, F16,       KERNEL_SOURCE_1)

    TENSOR_MIN_KERNELS_2D_HALF(F16, F16, F16,    KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS_2D_HALF(BF16, BF16, BF16, KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS_2D(I8,  I8, I8,      KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS_2D(U8,  U8, U8,      KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS_2D(U8,  U8,  I16,    KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS_2D(I16, I16, I16,    KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS_2D(I16, I16, U8,     KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS_2D(F16, F16, I8,     KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS_2D(F16, F16, U8,     KERNEL_SOURCE_0)
    TENSOR_MIN_KERNELS_2D(F16, F16, I16,    KERNEL_SOURCE_0)

    TENSOR_MIN_KERNELS_2D(I8,  I8,  F16,    KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS_2D(I8,  F16, I8,     KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS_2D(I8,  F16, F16,    KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS_2D(U8,  U8,  F16,    KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS_2D(U8,  F16, U8,     KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS_2D(U8,  F16, F16,    KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS_2D(I16, I16, F16,    KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS_2D(I16, F16, I16,    KERNEL_SOURCE_1)
    TENSOR_MIN_KERNELS_2D(I16, F16, F16,    KERNEL_SOURCE_1)
};

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};
#define _EVIS_PARAM_NUM          _cnt_of_array(kernel_param_def)

DEF_KERNEL_INITIALIZER(_minimum_initializer)
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
    int32_t input0_zp    = 0;
    float   input0_scale = 1.0f;
    int32_t input1_zp    = 0;
    float   input1_scale = 1.0f;
    int32_t output_zp    = 0;
    float   output_scale = 1.0f;

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    vsi_size_array_t * out_shape = NULL;
    uint32_t pack_key;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    out_shape  = attr[2]->shape;

    if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr[0]->dfp.fl;
        if (fl > 0)
        {
            input0_scale = 1.0f / (float) ((int64_t)1 << fl);
        }
        else
        {
            input0_scale = (float)((int64_t)1 << -fl);
        }
    }
    else if ( attr[0]->quant == VSI_NN_KERNEL_QUANT_ASYMM
        || attr[0]->quant == VSI_NN_KERNEL_QUANT_SYMM)
    {
        input0_zp     = attr[0]->asymm.zero_point;
        input0_scale  = attr[0]->asymm.scale;
    }

    if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr[1]->dfp.fl;
        if (fl > 0)
        {
            input1_scale = 1.0f / (float) ((int64_t)1 << fl);
        }
        else
        {
            input1_scale = (float)((int64_t)1 << -fl);
        }
    }
    else if ( attr[1]->quant == VSI_NN_KERNEL_QUANT_ASYMM
        || attr[1]->quant == VSI_NN_KERNEL_QUANT_SYMM)
    {
        input1_zp     = attr[1]->asymm.zero_point;
        input1_scale  = attr[1]->asymm.scale;
    }

    if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = attr[2]->dfp.fl;
        if (fl > 0)
        {
            output_scale = (float) ((int64_t)1 << fl);
        }
        else
        {
            output_scale = 1.0f / (float)((int64_t)1 << -fl);
        }
    }
    else if ( attr[2]->quant == VSI_NN_KERNEL_QUANT_ASYMM
        || attr[2]->quant == VSI_NN_KERNEL_QUANT_SYMM)
    {
        output_zp     = attr[2]->asymm.zero_point;
        output_scale  = 1.0f / attr[2]->asymm.scale;
    }

#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (IN1_TYPE << 8) | ( OUT_TYPE << 16))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype,
            attr[1]->dtype, attr[2]->dtype );

    if ( ( attr[0]->dtype == I8 && attr[1]->dtype == I8 && attr[2]->dtype == I8  ) ||
         ( attr[0]->dtype == U8 && attr[1]->dtype == U8 && attr[2]->dtype == U8  ) )
    {
        gpu_param.global_scale[0] = 16;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;
    }
    else
    {
        gpu_param.global_scale[0] = 8;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;
    }

    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    switch( pack_key )
    {
    case _PACK_SELECT_KEY( I8,  I8,  I8  ):
    case _PACK_SELECT_KEY( U8,  U8,  U8  ):
        {
            uint16_t M0               = 0;
            uint16_t M1               = 0;
            int32_t  postShift0       = 0;
            int32_t  postShift1       = 0;
            uint32_t multAndoutZP0[2] = {0};
            uint32_t multAndoutZP1[2] = {0};

            gpu_dp_inst_t uniU8MulAndPostShift_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniU8MulAndPostShift_Hi_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x1b1a1918, 0x1f1e1d1c, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            gpu_quantize_multiplier_16bit( (double)input0_scale * output_scale, &M0, &postShift0);
            gpu_quantize_multiplier_16bit( (double)input1_scale * output_scale, &M1, &postShift1);

            multAndoutZP0[0] = (uint32_t)(M0);
            multAndoutZP0[1] = (uint32_t)((output_zp << postShift0) - input0_zp * M0);
            multAndoutZP1[0] = (uint32_t)(M1);
            multAndoutZP1[1] = (uint32_t)((output_zp << postShift1) - input1_zp * M1);

            gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_Lo_2x8, postShift0 );
            gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_Hi_2x8, postShift0 );

            status  = vsi_nn_kernel_gpu_add_param( node, "multAndoutZP1", &multAndoutZP1 );
            status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniU8MulAndPostShift0_Lo_2x8",  &uniU8MulAndPostShift_Lo_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniU8MulAndPostShift0_Hi_2x8",  &uniU8MulAndPostShift_Hi_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, final );

            gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_Lo_2x8, postShift1 );
            gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_Hi_2x8, postShift1 );
            status  = vsi_nn_kernel_gpu_add_param( node,
                "uniU8MulAndPostShift1_Lo_2x8",  &uniU8MulAndPostShift_Lo_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniU8MulAndPostShift1_Hi_2x8",  &uniU8MulAndPostShift_Hi_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    case _PACK_SELECT_KEY( I8,  I8,  F16 ):
    case _PACK_SELECT_KEY( I8,  F16, I8  ):
    case _PACK_SELECT_KEY( I8,  F16, F16 ):
    case _PACK_SELECT_KEY( F16, F16, I8  ):
    case _PACK_SELECT_KEY( U8,  U8,  I16 ):
    case _PACK_SELECT_KEY( U8,  U8,  F16 ):
    case _PACK_SELECT_KEY( U8,  F16, F16 ):
    case _PACK_SELECT_KEY( U8,  F16, U8  ):
    case _PACK_SELECT_KEY( F16, F16, U8  ):
    case _PACK_SELECT_KEY( I16, I16, F16 ):
    case _PACK_SELECT_KEY( I16, F16, F16 ):
    case _PACK_SELECT_KEY( I16, I16, U8  ):
    case _PACK_SELECT_KEY( I16, I16, I16 ):
    case _PACK_SELECT_KEY( I16, F16, I16 ):
    case _PACK_SELECT_KEY( F16, F16, I16 ):
        {
            uint16_t M0               = 0;
            uint16_t M1               = 0;
            int32_t  postShift0       = 0;
            int32_t  postShift1       = 0;
            uint32_t multAndoutZP0[2] = {0};
            uint32_t multAndoutZP1[2] = {0};
            gpu_dp_inst_t uniU8MulAndPostShift_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            gpu_quantize_multiplier_16bit( (double)input0_scale * output_scale, &M0, &postShift0);
            gpu_quantize_multiplier_16bit( (double)input1_scale * output_scale, &M1, &postShift1);

            multAndoutZP0[0] = (uint32_t)(M0);
            multAndoutZP0[1] = (uint32_t)((output_zp << postShift0) - input0_zp * M0);
            multAndoutZP1[0] = (uint32_t)(M1);
            multAndoutZP1[1] = (uint32_t)((output_zp << postShift1) - input1_zp * M1);

            gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_Lo_2x8, postShift0 );
            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniU8MulAndPostShift0_Lo_2x8", &uniU8MulAndPostShift_Lo_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
            if ( attr[1]->dtype != F16 || attr[2]->dtype != F16)
            {
                gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_Lo_2x8, postShift1 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniU8MulAndPostShift1_Lo_2x8", &uniU8MulAndPostShift_Lo_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP1", &multAndoutZP1 );
            }
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    default:
        break;
    }

#undef _PACK_SELECT_KEY

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
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
} /* _minmum_initializer() */

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
    uint32_t key;
    size_t i;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    key = HASH_MINIMUM_KEY( input0_dtype, input1_dtype, output_dtype, image_2d );

    for ( i = 0; i < _cnt_of_array(kernel_map); i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _minimum_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                kernel_map[i].source_name );
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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t tmp_params[_EVIS_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* tmp_inputs[2] = { NULL };
    vsi_nn_type_e dtype1 = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e dtype2 = inputs[1]->attr.dtype.vx_type;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(params);

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    // Reorder tensor
    if ( dtype1 != dtype2 && dtype1 == VSI_NN_TYPE_FLOAT16 )
    {
        int32_t order[2] = {1, 0};
        vsi_nn_reorder_tensor( inputs, order, 2, tmp_inputs );
    }
    else
    {
        memmove( tmp_inputs, inputs, sizeof(vsi_nn_tensor_t*) * 2 );
    }

    image_2d = (outputs[0]->attr.dim_num == 2);
    status = _query_kernel( tmp_inputs, outputs, image_2d, kernel );
    if ( VSI_SUCCESS == status )
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( tmp_params, _EVIS_PARAM_NUM,
                    tmp_inputs, 2, outputs, 1 );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _EVIS_PARAM_NUM );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( minimum, _setup )
