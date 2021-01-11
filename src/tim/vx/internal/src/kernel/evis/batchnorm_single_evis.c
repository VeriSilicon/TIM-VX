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
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

typedef enum _internal_img_dim_e
{
    IMAGE = 0,
    IMAGE_2D,
} internal_img_dim_e;

#define _BATCH_NORM_KERNEL_SOURCE      "batchnorm_single"

#define STR(a) #a

// Add kernel hashtable here
#define BATCH_NORM_HASH_KEY(IN_DTYPE, OUT_DTYPE, BRDCST, _image_2d) \
        ( ( IN_DTYPE << 16 ) | ( OUT_DTYPE << 3) | ( BRDCST << 1) | (_image_2d) )

#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, BRDCST) \
        { BATCH_NORM_HASH_KEY( IN_DTYPE, OUT_DTYPE, BRDCST, IMAGE), \
        CVIVANTE_NAMESPACE("evis.batch_norm_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_brdcst"#BRDCST), \
        _BATCH_NORM_KERNEL_SOURCE}

#define PACK_KERNEL_MAP_2D( IN_DTYPE, OUT_DTYPE, BRDCST) \
        { BATCH_NORM_HASH_KEY( IN_DTYPE, OUT_DTYPE, BRDCST, IMAGE_2D), \
        CVIVANTE_NAMESPACE("evis.batch_norm_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_brdcst"#BRDCST"_2D"), \
        _BATCH_NORM_KERNEL_SOURCE}

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _batch_norm_kernel_map[] =
{
    PACK_KERNEL_MAP(F16, F16, 0),
    PACK_KERNEL_MAP(F16, I16, 0),
    PACK_KERNEL_MAP(F16, U8,  0),
    PACK_KERNEL_MAP(F16, I8,  0),
    PACK_KERNEL_MAP(U8,  U8,  0),
    PACK_KERNEL_MAP(U8,  F16, 0),
    PACK_KERNEL_MAP(I8,  I8,  0),
    PACK_KERNEL_MAP(I8,  F16, 0),
    PACK_KERNEL_MAP(I16, I16, 0),
    PACK_KERNEL_MAP(I16, F16, 0),
    PACK_KERNEL_MAP(F16, F16, 1),
    PACK_KERNEL_MAP(F16, I16, 1),
    PACK_KERNEL_MAP(F16, U8,  1),
    PACK_KERNEL_MAP(F16, I8,  1),
    PACK_KERNEL_MAP(U8,  U8,  1),
    PACK_KERNEL_MAP(U8,  F16, 1),
    PACK_KERNEL_MAP(I8,  I8,  1),
    PACK_KERNEL_MAP(I8,  F16, 1),
    PACK_KERNEL_MAP(I16, I16, 1),
    PACK_KERNEL_MAP(I16, F16, 1),

    PACK_KERNEL_MAP_2D(F16, F16, 0),
    PACK_KERNEL_MAP_2D(F16, I16, 0),
    PACK_KERNEL_MAP_2D(F16, U8 , 0),
    PACK_KERNEL_MAP_2D(F16, I8 , 0),
    PACK_KERNEL_MAP_2D(U8,  U8 , 0),
    PACK_KERNEL_MAP_2D(U8,  F16, 0),
    PACK_KERNEL_MAP_2D(I8,  I8,  0),
    PACK_KERNEL_MAP_2D(I8,  F16, 0),
    PACK_KERNEL_MAP_2D(I16, I16, 0),
    PACK_KERNEL_MAP_2D(I16, F16, 0),
    PACK_KERNEL_MAP_2D(F16, F16, 1),
    PACK_KERNEL_MAP_2D(F16, I16, 1),
    PACK_KERNEL_MAP_2D(F16, U8 , 1),
    PACK_KERNEL_MAP_2D(F16, I8 , 1),
    PACK_KERNEL_MAP_2D(U8,  U8 , 1),
    PACK_KERNEL_MAP_2D(U8,  F16, 1),
    PACK_KERNEL_MAP_2D(I8,  I8,  1),
    PACK_KERNEL_MAP_2D(I8,  F16, 1),
    PACK_KERNEL_MAP_2D(I16, I16, 1),
    PACK_KERNEL_MAP_2D(I16, F16, 1),
};

/*
 * Kernel params
 */
static vx_param_description_t _batch_norm_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _BATCH_NORM_PARAM_NUM  _cnt_of_array( _batch_norm_kernel_param_def )
#define SCALAR_INPUT_EPS          (6)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_batch_norm_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
#define _PACK_BATCH_NORM_KEY( IN_TYPE, OUT_TYPE )    \
        ( ( IN_TYPE << 16) | ( OUT_TYPE ) )

    vsi_status status = VX_SUCCESS;
    // Alignment with a power of two value.
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vx_tensor     input                         = (vx_tensor)param[BATCHNORM_INPUT];
    vx_tensor     output                        = (vx_tensor)param[BATCHNORM_INPUT_CNT];
    vsi_nn_kernel_tensor_attr_t *input_attr     = NULL;
    vsi_nn_kernel_tensor_attr_t *output_attr    = NULL;
    vsi_int_array_t             *output_shape   = NULL;
    float    input_scale                        = 1.0f;
    float    input_tail                         = 0;
    float    output_scale                       = 1.0f;
    float    output_zp                          = 0;
    uint32_t pack_key                           = 0;

    input_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input);
    CHECK_PTR_FAIL_GOTO( input_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );
    output_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output);
    CHECK_PTR_FAIL_GOTO( output_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    if( input_attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = input_attr->dfp.fl;
        if (fl > 0)
        {
            input_scale = 1.0f / (float) ((int64_t)1 << fl);
        }
        else
        {
            input_scale = (float)((int64_t)1 << -fl);
        }
    }
    else if( input_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        input_scale = input_attr->asymm.scale;
        input_tail  = 0 - input_scale * (float)input_attr->asymm.zero_point;
    }

    if( output_attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = output_attr->dfp.fl;
        if (fl > 0)
        {
            output_scale = (float) ((int64_t)1 << fl);
        }
        else
        {
            output_scale = 1.0f / (float)((int64_t)1 << -fl);
        }
    }
    else if( output_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        output_scale = 1.0f / output_attr->asymm.scale;
        output_zp    = (float)output_attr->asymm.zero_point;
    }

    pack_key = _PACK_BATCH_NORM_KEY( input_attr->dtype, output_attr->dtype );

    output_shape  = output_attr->shape;

    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.global_size[0]   = gpu_align_p2((output_shape->data[0] + gpu_param.global_scale[0] - 1)
                                             / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = output_shape->data[1];
    gpu_param.global_size[2]   = output_shape->size > 2 ? output_shape->data[2] : 1;

    switch( pack_key )
    {
        case _PACK_BATCH_NORM_KEY( F16, F16 ):
        case _PACK_BATCH_NORM_KEY( F16, I16 ):
        case _PACK_BATCH_NORM_KEY( F16, U8 ):
        case _PACK_BATCH_NORM_KEY( F16, I8 ):
        case _PACK_BATCH_NORM_KEY( I16, I16 ):
        case _PACK_BATCH_NORM_KEY( I16, F16 ):
        case _PACK_BATCH_NORM_KEY( U8,  U8 ):
        case _PACK_BATCH_NORM_KEY( U8,  F16 ):
        case _PACK_BATCH_NORM_KEY( I8,  I8 ):
        case _PACK_BATCH_NORM_KEY( I8,  F16 ):
        {
            gpu_dp_inst_t uniDatatoF32_0_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniDatatoF32_1_4x4 = {{
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
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

            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniDatatoF32_0_4x4", &uniDatatoF32_0_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniDatatoF32_1_4x4", &uniDatatoF32_1_4x4 );
            if (output_attr->dtype == F16)
            {
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniExtract8Data_2x8", &uniExtractHalf8_2x8 );
            }
            else
            {
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniExtract8Data_2x8", &uniExtractInteger_2x8 );
            }
            status |= vsi_nn_kernel_gpu_add_param( node,
                "input_scale", &input_scale );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "input_tail", &input_tail );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "output_scale", &output_scale );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "output_zp", &output_zp );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        break;
    default:
        break;
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (input_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&input_attr);
    }

    if (output_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&output_attr);
    }

#undef  _PACK_BATCH_NORM_KEY
    return status;
} /* _batch_norm_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool image_2d
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _batch_norm_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _batch_norm_kernel_map );
    vx_param_description_t * param_def  = _batch_norm_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _batch_norm_kernel_param_def );
    vx_kernel_initialize_f  initializer = _batch_norm_initializer;
    uint32_t key = 0;
    uint32_t i = 0;
    uint32_t brdcst = 0;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype   = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (inputs[BATCHNORM_INPUT]->attr.size[0] != 1 && inputs[BATCHNORM_INPUT_BETA]->attr.size[0] == 1)
    {
        brdcst = 1;
    }

    key = BATCH_NORM_HASH_KEY(in_dtype, out_dtype, brdcst, image_2d);

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }

    if( i < kernel_map_size )
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
    vsi_nn_kernel_node_param_t node_params[_BATCH_NORM_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    float eps = 0;

    eps = vsi_nn_kernel_param_get_float32(params, "eps");

    if ( (inputs[1]->attr.is_const && inputs[2]->attr.is_const)
        || (inputs[1]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT16)
        || (inputs[2]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT16)
        || (inputs[3]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT16)
        || (inputs[4]->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT32) )
    {
        return NULL;
    }

    image_2d = (inputs[0]->attr.dim_num < 3 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, inputs, outputs, image_2d);

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _BATCH_NORM_PARAM_NUM,
                    inputs, input_num, outputs, output_num );

            node_params[SCALAR_INPUT_EPS] = vsi_nn_kernel_scalar_create(
                    graph, F32, &eps );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _BATCH_NORM_PARAM_NUM );

            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_EPS] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( batchnorm_single, _setup )

