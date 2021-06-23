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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define HASH_TILE_KEY(_input_type, _output_type, _image_2d, _is_size1, _remainder) \
    ((_input_type << 19) | (_output_type << 5) | (_image_2d << 4) | (_is_size1 << 3) | (_remainder))

#define KERNEL_SOURCE    "tile",
#define KERNEL_SOURCE1   "tile_mix",

#define STR(a) #a

 #define TENSOR_TILE_KEY_DIM0_IS1_2D(SRC_TYPE, OUT_TYPE) \
    {   HASH_TILE_KEY(SRC_TYPE, OUT_TYPE, 1, 1, 1), \
        CVIVANTE_NAMESPACE("evis.tile_1toN_"#SRC_TYPE"to"#OUT_TYPE"_2D"), \
        KERNEL_SOURCE },

#define HASH_TILE_SH_KERNEL_NAME(SRC_TYPE, DST_TYPE, REMAINDER) \
    CVIVANTE_NAMESPACE("evis.tile_remain"STR(REMAINDER)"_"#SRC_TYPE"to"#DST_TYPE)

#define TENSOR_TILE_KERNELS(SRC_TYPE, OUT_TYPE, ISSIZE1, REMAINDER) \
    {   HASH_TILE_KEY(SRC_TYPE, OUT_TYPE, 0, ISSIZE1, REMAINDER), \
        HASH_TILE_SH_KERNEL_NAME(SRC_TYPE, OUT_TYPE, REMAINDER), \
        KERNEL_SOURCE1 },

#define HASH_TILE_SH_KERNEL_2D_NAME(SRC_TYPE, DST_TYPE, REMAINDER) \
    CVIVANTE_NAMESPACE("evis.tile_remain"STR(REMAINDER)"_"#SRC_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_TILE_KERNELS_2D(SRC_TYPE, OUT_TYPE, ISSIZE1, REMAINDER) \
    {   HASH_TILE_KEY(SRC_TYPE, OUT_TYPE, 1, ISSIZE1, REMAINDER), \
        HASH_TILE_SH_KERNEL_2D_NAME(SRC_TYPE, OUT_TYPE, REMAINDER), \
        KERNEL_SOURCE1 },

#define TENSOR_TILE_8BITS_KERNELS(SRC_TYPE, OUT_TYPE, REMAINDER) \
    {   HASH_TILE_KEY(SRC_TYPE, OUT_TYPE, 0, 0, REMAINDER), \
        HASH_TILE_SH_KERNEL_NAME(U8, U8, REMAINDER), \
        KERNEL_SOURCE },

#define TENSOR_TILE_16BITS_KERNELS(SRC_TYPE, OUT_TYPE, REMAINDER) \
    {   HASH_TILE_KEY(SRC_TYPE, OUT_TYPE, 0, 0, REMAINDER), \
        HASH_TILE_SH_KERNEL_NAME(I16, I16, REMAINDER), \
        KERNEL_SOURCE },

 #define TENSOR_TILE_8BITS_2D_KERNELS(SRC_TYPE, OUT_TYPE, REMAINDER) \
    {   HASH_TILE_KEY(SRC_TYPE, OUT_TYPE, 1, 0, REMAINDER), \
        HASH_TILE_SH_KERNEL_2D_NAME(U8, U8, REMAINDER), \
        KERNEL_SOURCE },

 #define TENSOR_TILE_16BITS_2D_KERNELS(SRC_TYPE, OUT_TYPE, REMAINDER) \
    {   HASH_TILE_KEY(SRC_TYPE, OUT_TYPE, 1, 0, REMAINDER), \
        HASH_TILE_SH_KERNEL_2D_NAME(I16, I16, REMAINDER), \
        KERNEL_SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } _tile_evis_kernel_map[] =
{
    TENSOR_TILE_8BITS_KERNELS( I8,   I8, 0)
    TENSOR_TILE_8BITS_KERNELS( I8,   I8, 1)
    TENSOR_TILE_8BITS_KERNELS( I8,   I8, 2)
    TENSOR_TILE_8BITS_KERNELS( I8,   I8, 3)
    TENSOR_TILE_8BITS_KERNELS( I8,   I8, 4)
    TENSOR_TILE_8BITS_KERNELS( I8,   I8, 5)
    TENSOR_TILE_8BITS_KERNELS( I8,   I8, 6)
    TENSOR_TILE_8BITS_KERNELS( I8,   I8, 7)
    TENSOR_TILE_8BITS_KERNELS( U8,   U8, 0)
    TENSOR_TILE_8BITS_KERNELS( U8,   U8, 1)
    TENSOR_TILE_8BITS_KERNELS( U8,   U8, 2)
    TENSOR_TILE_8BITS_KERNELS( U8,   U8, 3)
    TENSOR_TILE_8BITS_KERNELS( U8,   U8, 4)
    TENSOR_TILE_8BITS_KERNELS( U8,   U8, 5)
    TENSOR_TILE_8BITS_KERNELS( U8,   U8, 6)
    TENSOR_TILE_8BITS_KERNELS( U8,   U8, 7)

    TENSOR_TILE_16BITS_KERNELS( I16,  I16, 0)
    TENSOR_TILE_16BITS_KERNELS( I16,  I16, 1)
    TENSOR_TILE_16BITS_KERNELS( I16,  I16, 2)
    TENSOR_TILE_16BITS_KERNELS( I16,  I16, 3)
    TENSOR_TILE_16BITS_KERNELS( I16,  I16, 4)
    TENSOR_TILE_16BITS_KERNELS( I16,  I16, 5)
    TENSOR_TILE_16BITS_KERNELS( I16,  I16, 6)
    TENSOR_TILE_16BITS_KERNELS( I16,  I16, 7)
    TENSOR_TILE_16BITS_KERNELS( F16,  F16, 0)
    TENSOR_TILE_16BITS_KERNELS( F16,  F16, 1)
    TENSOR_TILE_16BITS_KERNELS( F16,  F16, 2)
    TENSOR_TILE_16BITS_KERNELS( F16,  F16, 3)
    TENSOR_TILE_16BITS_KERNELS( F16,  F16, 4)
    TENSOR_TILE_16BITS_KERNELS( F16,  F16, 5)
    TENSOR_TILE_16BITS_KERNELS( F16,  F16, 6)
    TENSOR_TILE_16BITS_KERNELS( F16,  F16, 7)
    TENSOR_TILE_16BITS_KERNELS( BF16, BF16, 0)
    TENSOR_TILE_16BITS_KERNELS( BF16, BF16, 1)
    TENSOR_TILE_16BITS_KERNELS( BF16, BF16, 2)
    TENSOR_TILE_16BITS_KERNELS( BF16, BF16, 3)
    TENSOR_TILE_16BITS_KERNELS( BF16, BF16, 4)
    TENSOR_TILE_16BITS_KERNELS( BF16, BF16, 5)
    TENSOR_TILE_16BITS_KERNELS( BF16, BF16, 6)
    TENSOR_TILE_16BITS_KERNELS( BF16, BF16, 7)

    TENSOR_TILE_8BITS_2D_KERNELS( I8,   I8, 0)
    TENSOR_TILE_8BITS_2D_KERNELS( I8,   I8, 1)
    TENSOR_TILE_8BITS_2D_KERNELS( I8,   I8, 2)
    TENSOR_TILE_8BITS_2D_KERNELS( I8,   I8, 3)
    TENSOR_TILE_8BITS_2D_KERNELS( I8,   I8, 4)
    TENSOR_TILE_8BITS_2D_KERNELS( I8,   I8, 5)
    TENSOR_TILE_8BITS_2D_KERNELS( I8,   I8, 6)
    TENSOR_TILE_8BITS_2D_KERNELS( I8,   I8, 7)
    TENSOR_TILE_8BITS_2D_KERNELS( U8,   U8, 0)
    TENSOR_TILE_8BITS_2D_KERNELS( U8,   U8, 1)
    TENSOR_TILE_8BITS_2D_KERNELS( U8,   U8, 2)
    TENSOR_TILE_8BITS_2D_KERNELS( U8,   U8, 3)
    TENSOR_TILE_8BITS_2D_KERNELS( U8,   U8, 4)
    TENSOR_TILE_8BITS_2D_KERNELS( U8,   U8, 5)
    TENSOR_TILE_8BITS_2D_KERNELS( U8,   U8, 6)
    TENSOR_TILE_8BITS_2D_KERNELS( U8,   U8, 7)

    TENSOR_TILE_16BITS_2D_KERNELS( I16,  I16, 0)
    TENSOR_TILE_16BITS_2D_KERNELS( I16,  I16, 1)
    TENSOR_TILE_16BITS_2D_KERNELS( I16,  I16, 2)
    TENSOR_TILE_16BITS_2D_KERNELS( I16,  I16, 3)
    TENSOR_TILE_16BITS_2D_KERNELS( I16,  I16, 4)
    TENSOR_TILE_16BITS_2D_KERNELS( I16,  I16, 5)
    TENSOR_TILE_16BITS_2D_KERNELS( I16,  I16, 6)
    TENSOR_TILE_16BITS_2D_KERNELS( I16,  I16, 7)
    TENSOR_TILE_16BITS_2D_KERNELS( F16,  F16, 0)
    TENSOR_TILE_16BITS_2D_KERNELS( F16,  F16, 1)
    TENSOR_TILE_16BITS_2D_KERNELS( F16,  F16, 2)
    TENSOR_TILE_16BITS_2D_KERNELS( F16,  F16, 3)
    TENSOR_TILE_16BITS_2D_KERNELS( F16,  F16, 4)
    TENSOR_TILE_16BITS_2D_KERNELS( F16,  F16, 5)
    TENSOR_TILE_16BITS_2D_KERNELS( F16,  F16, 6)
    TENSOR_TILE_16BITS_2D_KERNELS( F16,  F16, 7)
    TENSOR_TILE_16BITS_2D_KERNELS( BF16, BF16, 0)
    TENSOR_TILE_16BITS_2D_KERNELS( BF16, BF16, 1)
    TENSOR_TILE_16BITS_2D_KERNELS( BF16, BF16, 2)
    TENSOR_TILE_16BITS_2D_KERNELS( BF16, BF16, 3)
    TENSOR_TILE_16BITS_2D_KERNELS( BF16, BF16, 4)
    TENSOR_TILE_16BITS_2D_KERNELS( BF16, BF16, 5)
    TENSOR_TILE_16BITS_2D_KERNELS( BF16, BF16, 6)
    TENSOR_TILE_16BITS_2D_KERNELS( BF16, BF16, 7)

    TENSOR_TILE_KERNELS( U8,   F16, 0, 0)
    TENSOR_TILE_KERNELS( U8,   F16, 0, 1)
    TENSOR_TILE_KERNELS( U8,   F16, 0, 2)
    TENSOR_TILE_KERNELS( U8,   F16, 0, 3)
    TENSOR_TILE_KERNELS( U8,   F16, 0, 4)
    TENSOR_TILE_KERNELS( U8,   F16, 0, 5)
    TENSOR_TILE_KERNELS( U8,   F16, 0, 6)
    TENSOR_TILE_KERNELS( U8,   F16, 0, 7)

    TENSOR_TILE_KERNELS( U8,   F16, 1, 0)
    TENSOR_TILE_KERNELS( U8,   F16, 1, 1)
    TENSOR_TILE_KERNELS( U8,   F16, 1, 2)
    TENSOR_TILE_KERNELS( U8,   F16, 1, 3)
    TENSOR_TILE_KERNELS( U8,   F16, 1, 4)
    TENSOR_TILE_KERNELS( U8,   F16, 1, 5)
    TENSOR_TILE_KERNELS( U8,   F16, 1, 6)
    TENSOR_TILE_KERNELS( U8,   F16, 1, 7)

    TENSOR_TILE_KERNELS_2D( U8,   F16, 0, 0)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 0, 1)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 0, 2)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 0, 3)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 0, 4)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 0, 5)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 0, 6)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 0, 7)

    TENSOR_TILE_KERNELS_2D( U8,   F16, 1, 0)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 1, 1)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 1, 2)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 1, 3)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 1, 4)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 1, 5)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 1, 6)
    TENSOR_TILE_KERNELS_2D( U8,   F16, 1, 7)

    TENSOR_TILE_KEY_DIM0_IS1_2D(U8,  U8)
    TENSOR_TILE_KEY_DIM0_IS1_2D(I16, I16)
};

/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _EVIS_PARAM_NUM          _cnt_of_array(kernel_param_def)

#define SCALAR_INPUT_BATCH_IN       (2)
#define SCALAR_INPUT_DEPTH_IN       (3)
#define SCALAR_INPUT_DEPTH_OUT      (4)
#define SCALAR_INPUT_MULTIPLES_0    (5)
#define SCALAR_INPUT_MULTIPLES_1    (6)
#define SCALAR_INPUT_MULTIPLES_2    (7)
#define SCALAR_INPUT_MULTIPLES_3    (8)
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_tile_initializer)
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
    vsi_nn_kernel_tensor_attr_t * attr[2]   = { NULL, NULL };
    vsi_int_array_t * in_shape             = NULL;
    uint32_t pack_key;
    int32_t  lastWorkItem = 0;
    float     scaleIn         = 1.0f;
    float     scaleOut        = 1.0f;
    int32_t   output_ZP       = 0;
    int32_t   input_ZP        = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    in_shape  = attr[0]->shape;

    if (VSI_NN_KERNEL_QUANT_DFP == attr[0]->quant)
    {
        if (attr[0]->dfp.fl > 0)
        {
            scaleIn = (1.0f / ((float) ((int64_t)1 << attr[0]->dfp.fl)));
        }
        else
        {
            scaleIn = ((float) ((int64_t)1 << -attr[0]->dfp.fl));
        }
        input_ZP = 0;
    }
    else if (VSI_NN_KERNEL_QUANT_ASYMM == attr[0]->quant)
    {
        input_ZP         = attr[0]->asymm.zero_point;
        scaleIn          = attr[0]->asymm.scale;
    }

    if (VSI_NN_KERNEL_QUANT_DFP == attr[1]->quant)
    {
        if (attr[1]->dfp.fl > 0)
        {
            scaleOut = (float)((int64_t)1 << attr[1]->dfp.fl);
        }
        else
        {
            scaleOut = (1.0f / (float)((int64_t)1 << -attr[1]->dfp.fl));
        }
        output_ZP = 0;
    }
    else if (VSI_NN_KERNEL_QUANT_ASYMM == attr[1]->quant)
    {
        output_ZP        = attr[1]->asymm.zero_point;
        scaleOut         = attr[1]->asymm.scale;
    }

#define _PACK_SELECT_KEY( IN_TYPE, OUT_TYPE )    \
        (( IN_TYPE << 16) | ( OUT_TYPE))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype );

    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0] = (in_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0];
    gpu_param.global_size[1] = in_shape->data[1];
    gpu_param.global_size[2] = in_shape->size > 2 ? in_shape->data[2] : 1;

    lastWorkItem = ((int32_t)gpu_param.global_size[0] - 1) * ((int32_t)gpu_param.global_scale[0]);

    {
        float     uint8Scale = scaleIn / scaleOut;
        uint16_t  M0                   = 0;
        int32_t   postShift            = 0;
        uint32_t  multAndoutZP[2]      = {0};

        gpu_dp_inst_t uniU8MulAndPostShift_Lo_2x8 = {{
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_quantize_multiplier_16bit(uint8Scale, &M0, &postShift);
        multAndoutZP[0] = (uint32_t)(M0);
        multAndoutZP[1] = (uint32_t)((output_ZP << postShift) - input_ZP * M0);

        uniU8MulAndPostShift_Lo_2x8.data[7] |= (postShift & 0x1F);

        switch( pack_key )
        {
        case _PACK_SELECT_KEY(U8, U8 ):
        case _PACK_SELECT_KEY(I8, I8 ):
        case _PACK_SELECT_KEY(I16, I16 ):
        case _PACK_SELECT_KEY(F16, F16 ):
        case _PACK_SELECT_KEY(BF16, BF16 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                    "lastWorkItem", &lastWorkItem );
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            break;
        case _PACK_SELECT_KEY(U8, F16 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                    "lastWorkItem", &lastWorkItem );
                status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniU8MulAndPostShift_Lo_2x8", &uniU8MulAndPostShift_Lo_2x8);
                status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP", multAndoutZP);
                CHECK_STATUS_FAIL_GOTO(status, final );
            }
            break;
        default:
            break;
        }
    }

#undef _PACK_SELECT_KEY

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(attr[0]);
    SAFE_FREE_TENSOR_ATTR(attr[1]);
#undef SAFE_FREE_TENSOR_ATTR

    return status;
} /* _tile_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_bool image_2d,
    vx_uint32 remainder,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_nn_kernel_dtype_e input_dtype;
    vsi_nn_kernel_dtype_e output_dtype;
    vsi_status status = VSI_FAILURE;
    uint32_t key;
    int32_t i = 0;
    int32_t dim0_size1 = inputs[0]->attr.size[0] == 1 ? 1 : 0;

    input_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (input_dtype == output_dtype && image_2d == TRUE && dim0_size1)
    {
        input_dtype = input_dtype == I8 ? U8 : input_dtype;
        input_dtype = input_dtype == F16 ? I16 : input_dtype;
        input_dtype = input_dtype == BF16 ? I16 : input_dtype;
        output_dtype = input_dtype;
        key = HASH_TILE_KEY(input_dtype, output_dtype, 1, 1, 1);
    }
    else
    {
        key = HASH_TILE_KEY( input_dtype, output_dtype, image_2d, dim0_size1, remainder);
    }

    for( i = 0; i < _cnt_of_array(_tile_evis_kernel_map); i ++ )
    {
        if( _tile_evis_kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(_tile_evis_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _tile_evis_kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _tile_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 1,
                _tile_evis_kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _tile_evis_kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

static vsi_bool _is_supported_axis(int32_t* multiples, uint32_t multiples_num)
{
    uint32_t i = 0;

    if ( multiples_num < 4)
    {
        return TRUE;
    }
    else if ( multiples_num > 4)
    {
        return FALSE;
    }

    for ( i = 3;  i < multiples_num;  i++)
    {
        if (multiples[i] > 1)
        {
            return FALSE;
        }
    }

    return TRUE;
}

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
    vx_uint32 remainder = inputs[0]->attr.size[0] % 8;
    int32_t shapes[3][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    uint32_t i = 0;
    uint32_t new_rank = 0;
    vsi_bool ret = FALSE;
    uint32_t dim = inputs[0]->attr.dim_num;
    int32_t multiples[VSI_NN_MAX_DIM_NUM] = { 0 };

    for ( i = 0;  i < dim;  i++)
    {
        multiples[i] = outputs[0]->attr.size[i] / inputs[0]->attr.size[i];
    }

    ret = vsi_nn_kernel_optimize_tile_shape(
            (int32_t *)inputs[0]->attr.size, inputs[0]->attr.dim_num,
            (int32_t *)multiples, inputs[0]->attr.dim_num,
            (int32_t *)outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], shapes[1], shapes[2], &new_rank );

    if (ret)
    {
        if ( _is_supported_axis(shapes[1], new_rank) == FALSE)
        {
            return NULL;
        }

        reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
            inputs[0], (uint32_t*)shapes[0], new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
            outputs[0], (uint32_t*)shapes[2], new_rank );
    }
    else
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)reshape_tensors[1]->attr.size,
                outputs[0]->attr.dim_num ))
    {
        goto final;
    }

    remainder = reshape_tensors[0]->attr.size[0] % 8;
    image_2d = (reshape_tensors[0]->attr.dim_num == 2 || reshape_tensors[0]->attr.size[2] == 1);
    status = _query_kernel( &reshape_tensors[0], &reshape_tensors[1], image_2d, remainder, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Pass parameters to node. */
            uint32_t depthIn = new_rank > 2 ? reshape_tensors[0]->attr.size[2] : 1;
            uint32_t depthOut = new_rank > 2 ? reshape_tensors[1]->attr.size[2] : 1;
            uint32_t batchIn = new_rank > 3 ? reshape_tensors[0]->attr.size[3] : 1;


            vsi_nn_kernel_node_pack_io( node_params, _EVIS_PARAM_NUM,
                    &reshape_tensors[0], 1, &reshape_tensors[1], 1 );

            node_params[SCALAR_INPUT_BATCH_IN] = vsi_nn_kernel_scalar_create(
                    graph, I32, &batchIn );
            node_params[SCALAR_INPUT_DEPTH_IN] = vsi_nn_kernel_scalar_create(
                    graph, I32, &depthIn );
            node_params[SCALAR_INPUT_DEPTH_OUT] = vsi_nn_kernel_scalar_create(
                    graph, I32, &depthOut );
            node_params[SCALAR_INPUT_MULTIPLES_0] = vsi_nn_kernel_scalar_create(
                    graph, I32, &shapes[1][0] );
            node_params[SCALAR_INPUT_MULTIPLES_1] = vsi_nn_kernel_scalar_create(
                    graph, I32, &shapes[1][1] );
            node_params[SCALAR_INPUT_MULTIPLES_2] = vsi_nn_kernel_scalar_create(
                    graph, I32, &shapes[1][2] );
            node_params[SCALAR_INPUT_MULTIPLES_3] = vsi_nn_kernel_scalar_create(
                    graph, I32, &shapes[1][3] );

            status = vsi_nn_kernel_node_pass_param( node, node_params, _EVIS_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );

            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_BATCH_IN] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_DEPTH_IN] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_DEPTH_OUT] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_MULTIPLES_0] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_MULTIPLES_1] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_MULTIPLES_2] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_MULTIPLES_3] );
        }
    }

final:
    if (reshape_tensors[0] != inputs[0])
    {
        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
    }

    if (reshape_tensors[1] != outputs[0])
    {
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
    }

    return node;
} /* _setup() */

REGISTER_BACKEND_EVIS( tile, _setup )

__END_DECLS
