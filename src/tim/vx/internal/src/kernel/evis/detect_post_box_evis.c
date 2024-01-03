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
    INTERNAL_KERNEL_DETECT_POST_BOX,
} _internal_kernel_e;

#define _DETECT_POST_BOX_KERNEL_SOURCE      "detect_post_box"

#define STR(a) #a
// Add kernel hashtable here
#define DETECT_POST_BOX_HASH_KEY(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
        ((IN0_DTYPE << 18) | ( IN1_DTYPE << 11 ) | ( OUT_DTYPE << 4))

#define PACK_KERNEL_MAP(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
        { DETECT_POST_BOX_HASH_KEY(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE), \
        CVIVANTE_NAMESPACE("evis.detect_post_box_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)), \
        _DETECT_POST_BOX_KERNEL_SOURCE}

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _detect_post_box_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32, F32, F32 ),
    PACK_KERNEL_MAP( U8,  U8,  F32 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _detect_post_box_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _DETECT_POST_BOX_PARAM_NUM  _cnt_of_array( _detect_post_box_kernel_param_def )

#define SCALAR_SCALE_Y   (3)
#define SCALAR_SCALE_X   (4)
#define SCALAR_SCALE_H   (5)
#define SCALAR_SCALE_W   (6)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_detect_post_box_initializer)
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
    vsi_nn_kernel_tensor_attr_t * input_attr   = NULL;
    vsi_nn_kernel_tensor_attr_t * input1_attr  = NULL;
    vsi_size_array_t * in_shape                 = NULL;
    float             logE                     = (float)(log10(exp(1.0f)) / log10(2.0f));
    float     scaleIn0        = 1.0f;
    float     scaleIn1        = 1.0f;
    int32_t   input1_ZP       = 0;
    int32_t   input0_ZP       = 0;

    VSI_UNREFERENCED(param_size);

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );

    input1_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( input1_attr, "Create tensor attr buffer fail.", final );

    in_shape  = input_attr->shape;

    status  = vsi_nn_kernel_gpu_add_param( node, "logE", &logE);
    CHECK_STATUS_FAIL_GOTO(status, final );

    input0_ZP = input_attr->zero_point;
    scaleIn0  = input_attr->scale;
    input1_ZP = input1_attr->zero_point;
    scaleIn1  = input1_attr->scale;

    if ((F32 == input_attr->dtype) || (F32 == input1_attr->dtype))
    {
        gpu_dp_inst_t uniDataMerge_4x4 = {{
            0x03030303, // TCfg
            0x01010000, // ASelt
            0x00010000, 0x00010000, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00005400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status = vsi_nn_kernel_gpu_add_param( node,
                "uniDataMerge_4x4", &uniDataMerge_4x4 );
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if ((U8 == input_attr->dtype) || (U8 == input1_attr->dtype))
    {
        uint16_t  M0                 = 0;
        int32_t   postShift0         = 0;
        uint16_t  M1                 = 0;
        int32_t   postShift1         = 0;
        uint32_t  i                  = 0;
        gpu_dp_inst_t uniU8SubZptoF32Conv0_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniU8SubZptoF32Conv1_4x4 = {{
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_quantize_multiplier_16bit(scaleIn0, &M0, &postShift0);
        gpu_quantize_multiplier_16bit(scaleIn1, &M1, &postShift1);
        uniU8SubZptoF32Conv0_4x4.data[7] |= (postShift0 & 0x1F);
        uniU8SubZptoF32Conv1_4x4.data[7] |= (postShift1 & 0x1F);
        for ( i = 0; i < 8; i++ )
        {
            uniU8SubZptoF32Conv0_4x4.data[8 + i] = (((uint32_t)M0 << 16) | M0);
            uniU8SubZptoF32Conv1_4x4.data[8 + i] = (((uint32_t)M1 << 16) | M1);
        }

        status  = vsi_nn_kernel_gpu_add_param( node,
                "uniU8SubZptoF32Conv0_4x4", &uniU8SubZptoF32Conv0_4x4 );
        status |= vsi_nn_kernel_gpu_add_param( node,
                "uniU8SubZptoF32Conv1_4x4", &uniU8SubZptoF32Conv1_4x4 );
        status |= vsi_nn_kernel_gpu_add_param( node, "input0_ZP", &input0_ZP);
        status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &input1_ZP);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.dim = 2;
    gpu_param.global_size[0] = (
            (in_shape->data[1] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0]);
    gpu_param.global_size[1] = (
            (in_shape->data[2] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = 1;
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(input_attr);
    SAFE_FREE_TENSOR_ATTR(input1_attr);

    return status;
} /* _detect_post_box_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _detect_post_box_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _detect_post_box_kernel_map );
    vx_param_description_t * param_def  = _detect_post_box_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _detect_post_box_kernel_param_def );
    vx_kernel_initialize_f  initializer = _detect_post_box_initializer;
    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype  = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = DETECT_POST_BOX_HASH_KEY( in0_dtype, in1_dtype, out_dtype );

    for ( i = 0; i < (uint32_t)kernel_map_size; i++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }

    if ( i < (uint32_t)kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (vx_uint32)param_def_size;
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
    vsi_nn_kernel_node_param_t node_params[_DETECT_POST_BOX_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    float   inv_scale_y  = vsi_nn_kernel_param_get_float32( params, "inv_scale_y" );
    float   inv_scale_x  = vsi_nn_kernel_param_get_float32( params, "inv_scale_x" );
    float   inv_scale_h  = vsi_nn_kernel_param_get_float32( params, "inv_scale_h" );
    float   inv_scale_w  = vsi_nn_kernel_param_get_float32( params, "inv_scale_w" );

    status = _query_kernel( kernel, inputs, outputs);

    if ( VSI_SUCCESS == status )
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _DETECT_POST_BOX_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_SCALE_Y] = vsi_nn_kernel_scalar_create( graph, F32, &inv_scale_y );
            node_params[SCALAR_SCALE_X] = vsi_nn_kernel_scalar_create( graph, F32, &inv_scale_x );
            node_params[SCALAR_SCALE_H] = vsi_nn_kernel_scalar_create( graph, F32, &inv_scale_h );
            node_params[SCALAR_SCALE_W] = vsi_nn_kernel_scalar_create( graph, F32, &inv_scale_w );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _DETECT_POST_BOX_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCALE_Y] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCALE_X] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCALE_H] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCALE_W] );
        }
    }

    return node;

} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( detect_post_box, _setup )

