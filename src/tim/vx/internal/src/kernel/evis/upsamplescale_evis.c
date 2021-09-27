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
#include "vsi_nn_error.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    UP_ORG = 0,
    UP_K2,
} _internal_upscale_e;

#define _UPSAMPLESCALE_KERNEL_SOURCE      "upsamplescale"
#define _UPSAMPLESCALE_KERNEL_K2_SOURCE   "upsamplescale_k2"
#define _UPSAMPLESCALE_KERNEL_NAME        CVIVANTE_NAMESPACE("evis.upsamplescale")

#define STR(a) #a
// Add kernel hashtable here
#define UPSAMPLESCALE_HASH_KEY( IN_DTYPE, OUT_DTYPE, FLAG ) \
        (( IN_DTYPE ) | ( OUT_DTYPE << 8) | ( FLAG << 16))

#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
        { UPSAMPLESCALE_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP_ORG ), \
          CVIVANTE_NAMESPACE("evis.upsamplescale_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)), \
          _UPSAMPLESCALE_KERNEL_SOURCE }

#define PACK_KERNEL_MAP_K2( IN_DTYPE, OUT_DTYPE ) \
        { UPSAMPLESCALE_HASH_KEY( IN_DTYPE, OUT_DTYPE, UP_K2 ), \
          CVIVANTE_NAMESPACE("evis.upsamplescale_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_K2"), \
          _UPSAMPLESCALE_KERNEL_K2_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _upsamplescale_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F16, F16 ),
    PACK_KERNEL_MAP( F16, I16 ),
    PACK_KERNEL_MAP( F16, I8 ),
    PACK_KERNEL_MAP( F16, U8 ),
    PACK_KERNEL_MAP( I16, I16 ),
    PACK_KERNEL_MAP( I16, F16 ),
    PACK_KERNEL_MAP( I8,  I8 ),
    PACK_KERNEL_MAP( I8,  F16 ),
    PACK_KERNEL_MAP( U8,  U8 ),
    PACK_KERNEL_MAP( U8,  F16 ),

    PACK_KERNEL_MAP_K2( F16, F16 ),
    PACK_KERNEL_MAP_K2( F16, I16 ),
    PACK_KERNEL_MAP_K2( F16, I8 ),
    PACK_KERNEL_MAP_K2( F16, U8 ),
    PACK_KERNEL_MAP_K2( I16, I16 ),
    PACK_KERNEL_MAP_K2( I16, F16 ),
    PACK_KERNEL_MAP_K2( I8,  I8 ),
    PACK_KERNEL_MAP_K2( I8,  F16 ),
    PACK_KERNEL_MAP_K2( U8,  U8 ),
    PACK_KERNEL_MAP_K2( U8,  F16 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _upsamplescale_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _UPSAMPLESCALE_PARAM_NUM  _cnt_of_array( _upsamplescale_kernel_param_def )
#define SCALAR_STRIDE_VALUE          (2)
#define SCALAR_SCALE_VALUE           (3)
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_upsamplescale_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
#define _PACK_UPSCALE_KEY( IN_TYPE, OUT_TYPE, FLAG )    \
        ( IN_TYPE  | ( OUT_TYPE << 16) | (FLAG << 24) )

    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_nn_kernel_tensor_attr_t * output_attr   = NULL;
    vsi_nn_kernel_tensor_attr_t * input_attr    = NULL;
    vsi_size_array_t * in_shape                  = NULL;
    vsi_nn_kernel_dtype_e        input_dtype    = F16;
    vsi_nn_kernel_dtype_e        output_dtype   = F16;
    int32_t   stride          = 0;
    float     scale           = 0;
    float     scaleIn         = 1.0f;
    float     scaleOut        = 1.0f;
    int32_t   output_ZP       = 0;
    int32_t   input_ZP        = 0;
    int32_t   srcFixPointPos  = 0;
    int32_t   dstFixPointPos  = 0;
    uint32_t  pack_key        = 0;
    _internal_upscale_e flag  = UP_ORG;

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    in_shape  = input_attr->shape;
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[SCALAR_STRIDE_VALUE], &(stride));
    vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_SCALE_VALUE], &(scale));
    input_dtype  = input_attr->dtype;
    output_dtype = output_attr->dtype;

    if (VSI_NN_KERNEL_QUANT_DFP == input_attr->quant)
    {
        srcFixPointPos   = input_attr->dfp.fl;
        if (srcFixPointPos >=0 )
            scaleIn = 1.0f / (float) ((int64_t)1 << srcFixPointPos);
        else
            scaleIn = (float) ((int64_t)1 << -srcFixPointPos);
    }
    else if (VSI_NN_KERNEL_QUANT_ASYMM == input_attr->quant)
    {
        input_ZP         = input_attr->asymm.zero_point;
        scaleIn          = input_attr->asymm.scale;
    }

    if (VSI_NN_KERNEL_QUANT_DFP == output_attr->quant)
    {
        dstFixPointPos   = output_attr->dfp.fl;
        if (dstFixPointPos >=0 )
            scaleOut = 1.0f / (float) ((int64_t)1 << dstFixPointPos);
        else
            scaleOut = (float) ((int64_t)1 << -dstFixPointPos);
    }
    else if (VSI_NN_KERNEL_QUANT_ASYMM == output_attr->quant)
    {
        output_ZP        = output_attr->asymm.zero_point;
        scaleOut         = output_attr->asymm.scale;
    }

    if (stride == 2 && scale >= 0)
    {
        flag = UP_K2;
    }

    if ( flag == UP_K2 )
    {
        gpu_param.global_scale[0] = 8;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;
    }
    else
    {
        gpu_param.global_scale[0] = 1;
        gpu_param.global_scale[1] = 1;
        gpu_param.global_scale[2] = 1;
    }

    gpu_param.global_size[0] = gpu_align_p2(
            (in_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (in_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = in_shape->size > 2 ? in_shape->data[2] : 1;

    pack_key = _PACK_UPSCALE_KEY( input_dtype, output_dtype, flag );

    switch( pack_key )
    {
        case _PACK_UPSCALE_KEY( F16, F16, UP_K2 ):
        case _PACK_UPSCALE_KEY( F16, I16, UP_K2 ):
        case _PACK_UPSCALE_KEY( F16, I8,  UP_K2 ):
        case _PACK_UPSCALE_KEY( F16, U8,  UP_K2 ):
        case _PACK_UPSCALE_KEY( I16, F16, UP_K2 ):
        case _PACK_UPSCALE_KEY( I16, I16, UP_K2 ):
        case _PACK_UPSCALE_KEY( I8,  F16, UP_K2 ):
        case _PACK_UPSCALE_KEY( I8,  I8,  UP_K2 ):
        case _PACK_UPSCALE_KEY( U8,  F16, UP_K2 ):
        case _PACK_UPSCALE_KEY( U8,  U8,  UP_K2 ):
        {
            uint16_t multiplier         = 0;
            int32_t  postShift          = 0;
            uint32_t multAndoutZP[2]    = {0};
            gpu_dp_inst_t uniUpSampleScale2X_lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x11111010, 0x13131212, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};
            gpu_dp_inst_t uniUpSampleScale2X_hi_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x15151414, 0x17171616, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            gpu_quantize_multiplier_16bit(scaleIn * scale / scaleOut, &multiplier, &postShift);
            multAndoutZP[0] = (uint32_t)(multiplier);
            multAndoutZP[1] = (uint32_t)((output_ZP << postShift) - input_ZP * multiplier);

            uniUpSampleScale2X_lo_2x8.data[7] |= (postShift & 0x1F);
            uniUpSampleScale2X_hi_2x8.data[7] |= (postShift & 0x1F);

            status  = vsi_nn_kernel_gpu_add_param( node, "uniUpScale2X_lo_2x8", &uniUpSampleScale2X_lo_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "uniUpScale2X_hi_2x8", &uniUpSampleScale2X_hi_2x8);
            status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP", multAndoutZP);
        }
        break;
        case _PACK_UPSCALE_KEY( F16, F16, UP_ORG ):
        case _PACK_UPSCALE_KEY( F16, I16, UP_ORG ):
        case _PACK_UPSCALE_KEY( F16, I8,  UP_ORG ):
        case _PACK_UPSCALE_KEY( F16, U8,  UP_ORG ):
        case _PACK_UPSCALE_KEY( I16, F16, UP_ORG ):
        case _PACK_UPSCALE_KEY( I16, I16, UP_ORG ):
        case _PACK_UPSCALE_KEY( I8,  F16, UP_ORG ):
        case _PACK_UPSCALE_KEY( I8,  I8,  UP_ORG ):
        case _PACK_UPSCALE_KEY( U8,  F16, UP_ORG ):
        case _PACK_UPSCALE_KEY( U8,  U8,  UP_ORG ):
        {
            float output_scale = scaleIn * scale / scaleOut;
            float tail = output_ZP - input_ZP * output_scale;
            gpu_dp_inst_t uniConvertDatatoF32_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16};

            status  = vsi_nn_kernel_gpu_add_param( node, "uniConvertDatatoF32_4x4", &uniConvertDatatoF32_4x4);
            status |= vsi_nn_kernel_gpu_add_param( node, "output_scale", &output_scale);
            status |= vsi_nn_kernel_gpu_add_param( node, "tail", &tail);
        }
        break;
        default:
            break;
    }

#undef _PACK_UPSCALE_KEY
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (input_attr)
    {
        vsi_nn_kernel_tensor_attr_release( &input_attr );
        input_attr = NULL;
    }
    if (output_attr)
    {
        vsi_nn_kernel_tensor_attr_release( &output_attr );
        output_attr = NULL;
    }

    return status;
} /* _upsamplescale_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t stride,
    float   scale
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _upsamplescale_kernel_map;
    vx_param_description_t * param_def  = _upsamplescale_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _upsamplescale_kernel_param_def );
    vx_kernel_initialize_f  initializer = _upsamplescale_initializer;
    _internal_upscale_e flag = (stride == 2 && scale >= 0 ) ? UP_K2 : UP_ORG;

    uint32_t key = 0;
    int i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = UPSAMPLESCALE_HASH_KEY( in_dtype, out_dtype, flag );

    for( i = 0; i < _cnt_of_array( _upsamplescale_kernel_map ); i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array( _upsamplescale_kernel_map ) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_def_size;
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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_UPSAMPLESCALE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t stride = vsi_nn_kernel_param_get_int32( params, "stride" );
    float   scale  = vsi_nn_kernel_param_get_float32( params, "scale" );

    status = _query_kernel( kernel, inputs, outputs, stride, scale );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _UPSAMPLESCALE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_STRIDE_VALUE] = vsi_nn_kernel_scalar_create( graph, I32, &stride );
            node_params[SCALAR_SCALE_VALUE] = vsi_nn_kernel_scalar_create( graph, F32, &scale );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _UPSAMPLESCALE_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_STRIDE_VALUE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCALE_VALUE] );
            VSI_ASSERT( status == VSI_SUCCESS );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( upsamplescale, _setup )

