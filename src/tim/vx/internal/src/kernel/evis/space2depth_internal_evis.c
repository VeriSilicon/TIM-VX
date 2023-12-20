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
#define VX_KERNEL_NAME_SPACE2DEPTH_INTERNAL_U8TOU8        CVIVANTE_NAMESPACE("evis.space2depth_internal_U8toU8")
#define VX_KERNEL_NAME_SPACE2DEPTH_INTERNAL_U8TOU8_X2Y1   CVIVANTE_NAMESPACE("evis.space2depth_internal_U8toU8_X2Y1")
#define VX_KERNEL_NAME_SPACE2DEPTH_INTERNAL_I8TOI8        CVIVANTE_NAMESPACE("evis.space2depth_internal_I8toI8")
#define VX_KERNEL_NAME_SPACE2DEPTH_INTERNAL_I8TOI8_X2Y1   CVIVANTE_NAMESPACE("evis.space2depth_internal_I8toI8_X2Y1")
#define VX_KERNEL_NAME_SPACE2DEPTH_INTERNAL_I16TOI16      CVIVANTE_NAMESPACE("evis.space2depth_internal_I16toI16")
#define VX_KERNEL_NAME_SPACE2DEPTH_INTERNAL_I16TOI16_X2Y1 CVIVANTE_NAMESPACE("evis.space2depth_internal_I16toI16_X2Y1")
#define VX_KERNEL_NAME_SPACE2DEPTH_INTERNAL_F16TOF16      CVIVANTE_NAMESPACE("evis.space2depth_internal_F16toF16")
#define VX_KERNEL_NAME_SPACE2DEPTH_INTERNAL_F16TOF16_X2Y1 CVIVANTE_NAMESPACE("evis.space2depth_internal_F16toF16_X2Y1")

#define KERNEL_SOURCE_1    "space2depth_internal"

// Add kernel hashtable here
#define HASH_SPACE2DEPTH_INTERNAL_KEY(_input0_type, _output_type, _opt_stride) \
    ((_input0_type << 24) | (_output_type << 16) | (_opt_stride << 8))

#define TENSOR_SPACE2DEPTH_INTERNAL_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SPACE2DEPTH_INTERNAL_KEY(IN0_TYPE, OUT_TYPE, 0), \
        VX_KERNEL_NAME_SPACE2DEPTH_INTERNAL_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

#define TENSOR_SPACE2DEPTH_INTERNAL_OPT_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SPACE2DEPTH_INTERNAL_KEY(IN0_TYPE, OUT_TYPE, 1), \
        VX_KERNEL_NAME_SPACE2DEPTH_INTERNAL_##IN0_TYPE##TO##OUT_TYPE##_X2Y1, \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } space2depth_internal_map[] =
{
    TENSOR_SPACE2DEPTH_INTERNAL_KERNELS(U8,  U8,        KERNEL_SOURCE_1)
    TENSOR_SPACE2DEPTH_INTERNAL_KERNELS(I8,  I8,        KERNEL_SOURCE_1)
    TENSOR_SPACE2DEPTH_INTERNAL_KERNELS(I16, I16,       KERNEL_SOURCE_1)
    TENSOR_SPACE2DEPTH_INTERNAL_KERNELS(F16, F16,       KERNEL_SOURCE_1)
    TENSOR_SPACE2DEPTH_INTERNAL_OPT_KERNELS(U8,  U8,    KERNEL_SOURCE_1)
    TENSOR_SPACE2DEPTH_INTERNAL_OPT_KERNELS(I8,  I8,    KERNEL_SOURCE_1)
    TENSOR_SPACE2DEPTH_INTERNAL_OPT_KERNELS(I16, I16,   KERNEL_SOURCE_1)
    TENSOR_SPACE2DEPTH_INTERNAL_OPT_KERNELS(F16, F16,   KERNEL_SOURCE_1)
};

/*
 * Kernel params
 */
static vx_param_description_t _space2depth_internal_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _SPACE2DEPTH_INTERNAL_PARAM_NUM  _cnt_of_array( _space2depth_internal_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_space2depth_internal_initializer)
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

    uint32_t    input_dims = 0;
    vsi_nn_kernel_tensor_attr_t* attr[2] = {NULL, NULL};
    int32_t     input_width  = 0;
    int32_t     input_height = 0;
    int32_t     input_depth  = 0;
    int32_t     stride_x   = 0;
    int32_t     stride_y   = 0;
    int32_t     opt_flg    = 0;

    uint32_t pack_key = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &stride_x);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &stride_y);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    input_dims = (uint32_t)attr[0]->shape->size;
    input_width = (int32_t)(attr[0]->shape->data[0]);
    input_height = (int32_t)(attr[0]->shape->data[1]);
    input_depth = (int32_t)(input_dims > 2 ? attr[0]->shape->data[2] : 1);

    shaderParam.global_scale[0]  = 1;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    if (stride_x == 2 && stride_y == 1)
    {
        shaderParam.global_scale[0]  = 16;
        if (attr[0]->dtype == F16 || attr[0]->dtype == I16)
        {
            shaderParam.global_scale[0]  = 8;
        }
        opt_flg = 1;
    }
    shaderParam.global_size[0]   = gpu_align_p2((input_width + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = input_height;
    shaderParam.global_size[2]   = input_depth;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE, OPT_FLG )    \
        (IN0_TYPE | (OUT_TYPE << 8) | (OPT_FLG << 16))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype, opt_flg);

    {
        gpu_dp_inst_t uniExtractEvenUint8Stride2_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x06040200, 0x0e0c0a08, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000700, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractOddUint8Stride2_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x07050301, 0x0f0d0b09, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000700, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniExtractEvenFp16Stride2_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractOddFp16Stride2_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00030001, 0x00070005, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param(node, "input_depth", &input_depth);
        CHECK_STATUS_FAIL_GOTO(status, OnError );

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( U8,  U8,  0 ):
        case _PACK_SELECT_KEY( I8,  I8,  0 ):
        case _PACK_SELECT_KEY( I16, I16, 0 ):
        case _PACK_SELECT_KEY( F16, F16, 0 ):
            break;
        case _PACK_SELECT_KEY( U8,  U8,  1 ):
        case _PACK_SELECT_KEY( I8,  I8,  1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                            "uniExtractEvenUint8Stride2_2x8", &uniExtractEvenUint8Stride2_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniExtractOddUint8Stride2_2x8", &uniExtractOddUint8Stride2_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( I16, I16, 1 ):
        case _PACK_SELECT_KEY( F16, F16, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                            "uniExtractEvenFp16Stride2_4x4", &uniExtractEvenFp16Stride2_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                            "uniExtractOddFp16Stride2_4x4", &uniExtractOddFp16Stride2_4x4 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
    }
#undef _PACK_SELECT_KEY

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

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    const vsi_nn_kernel_param_t * params,
    int32_t opt_flg
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    size_t i = 0;

    VSI_UNREFERENCED(params);

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_SPACE2DEPTH_INTERNAL_KEY( input0_dtype, output_dtype, opt_flg );

    for( i = 0; i < _cnt_of_array(space2depth_internal_map); i ++ )
    {
        if ( space2depth_internal_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(space2depth_internal_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  space2depth_internal_map[i].function_name );
        kernel->info.parameters = _space2depth_internal_kernel_param_def;
        kernel->info.numParams = _SPACE2DEPTH_INTERNAL_PARAM_NUM;
        kernel->info.initialize = _space2depth_internal_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                space2depth_internal_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                space2depth_internal_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_SPACE2DEPTH_INTERNAL_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    int32_t block_size_x  = vsi_nn_kernel_param_get_int32( params, "block_size_x" );
    int32_t block_size_y  = vsi_nn_kernel_param_get_int32( params, "block_size_y" );
    int32_t opt_flg = (block_size_x == 2 && block_size_y == 1) ? 1 : 0;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( inputs, outputs, kernel, params, opt_flg );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            vsi_nn_kernel_node_pack_io( tmp_params, _SPACE2DEPTH_INTERNAL_PARAM_NUM, inputs, 1, outputs, 1 );
            tmp_params[2] = vsi_nn_kernel_scalar_create( graph, I32, &block_size_x );
            tmp_params[3] = vsi_nn_kernel_scalar_create( graph, I32, &block_size_y );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _SPACE2DEPTH_INTERNAL_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &tmp_params[2] );
            vsi_nn_kernel_scalar_release( &tmp_params[3] );
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                border.constant_value.U8 = 0;
                border.constant_value.U16 = 0;
                if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
                {
                    border.constant_value.U8 = (uint8_t)vsi_nn_get_tensor_zero_point(inputs[0]);
                }
                status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
                CHECK_STATUS(status);
            }
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( space2depth_internal, _setup )
