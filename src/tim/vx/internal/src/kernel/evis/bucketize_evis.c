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

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    INTERNAL_KERNEL_BUCKETIZE,
} _internal_kernel_e;

#define STR(a) #a

// Add kernel hashtable here
#define BUCKETIZE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ) \
        (( IN0_DTYPE ) | ( IN1_DTYPE << 8 ) | ( OUT_DTYPE << 16 ) | (RIGHT << 24) | (IMG_2D << 25))

#define PACK_KERNEL_2D_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ) \
        { BUCKETIZE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, RIGHT, IMG_2D ), \
        CVIVANTE_NAMESPACE("evis.bucketize_right_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
        "bucketize" }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _bucketize_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_2D_MAP( F16, F16, I32, 1, 1 ),
    PACK_KERNEL_2D_MAP( I16, I16, I32, 1, 1 ),
    PACK_KERNEL_2D_MAP( U8,  U8,  I32, 1, 1 ),
    PACK_KERNEL_2D_MAP( I8,  I8,  I32, 1, 1 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _bucketize_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _BUCKETIZE_PARAM_NUM  _cnt_of_array( _bucketize_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_bucketize_initializer)
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
    vsi_nn_kernel_tensor_attr_t * input0_attr   = NULL;
    vsi_nn_kernel_tensor_attr_t * input1_attr   = NULL;
    vsi_size_array_t * input0_shape             = NULL;
    vsi_size_array_t * input1_shape             = NULL;

    VSI_UNREFERENCED(param_size);

    input0_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input0_attr, "Create tensor attr buffer fail.", final );
    input1_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( input1_attr, "Create tensor attr buffer fail.", final );

    input0_shape  = input0_attr->shape;
    input1_shape  = input1_attr->shape;

    gpu_param.dim = 2;
    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_size[0] = gpu_align_p2(
            (input0_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (input0_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);

    {
        gpu_dp_inst_t uniDataConvert_0_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniDataConvert_1_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        int32_t boundaries_size = (int32_t)input1_shape->data[0];
        int32_t boundaries_size_x8 = (boundaries_size / 8) * 8;

        status  = vsi_nn_kernel_gpu_add_param( node, "uniDataConvert_0_4x4", &uniDataConvert_0_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniDataConvert_1_4x4", &uniDataConvert_1_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "boundaries_size_x8", &boundaries_size_x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "boundaries_size", &boundaries_size);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(input0_attr);
    SAFE_FREE_TENSOR_ATTR(input1_attr);
#undef SAFE_FREE_TENSOR_ATTR

    return status;
} /* _bucketize_initializer() */

static vsi_bool _bucketize_support_types
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_t  * input,
    vsi_nn_tensor_t  * boundaries,
    int32_t            right
    )
{
    vsi_size_t width = input->attr.size[0];
    vsi_size_t height = input->attr.size[1];
    vsi_size_t boundaries_size = boundaries->attr.size[0];
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_dtype_e in_dtype  = vsi_nn_kernel_map_dtype( input->attr.dtype.vx_type );

    image_2d = (input->attr.dim_num == 2 || input->attr.size[2] == 1);

    if ( vsi_nn_is_same_type(input, boundaries) == FALSE || right == 0 || image_2d == FALSE )
    {
        return FALSE;
    }

    if (in_dtype == F16 && graph->ctx->config.evis.ver != VSI_NN_HW_EVIS_2)
    {
        return FALSE;
    }

#define MAX_16BITS_BOUNDARIES_SIZE  (0xFFFF)
    if ( (in_dtype == F16 || in_dtype == I16) && boundaries_size > MAX_16BITS_BOUNDARIES_SIZE )
    {
        return FALSE;
    }
#undef MAX_16BITS_BOUNDARIES_SIZE

#define MAX_8BITS_BOUNDARIES_SIZE  (0xFF)
    if ( (in_dtype == I8 || in_dtype == U8) && boundaries_size > MAX_8BITS_BOUNDARIES_SIZE )
    {
        return FALSE;
    }
#undef MAX_8BITS_BOUNDARIES_SIZE

#define INPUT_SIZE_ALIGN8   (8)
    if ( width % INPUT_SIZE_ALIGN8 != 0 && height != 1 )
    {
        return FALSE;
    }
#undef INPUT_SIZE_ALIGN8

    return TRUE;
}

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t right
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _bucketize_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _bucketize_kernel_map );
    vx_param_description_t * param_def  = _bucketize_kernel_param_def;
    vx_kernel_initialize_f  initializer = _bucketize_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = BUCKETIZE_HASH_KEY( in0_dtype, in1_dtype, out_dtype, right, 1 );

    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
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
        kernel->info.numParams   = _cnt_of_array( _bucketize_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_BUCKETIZE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t right = vsi_nn_kernel_param_get_int32( params, "right" );

    if( !vsi_nn_kernel_gpu_check_shape(
        inputs[0]->attr.size, inputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    if ( _bucketize_support_types(graph, inputs[0], inputs[1], right) == FALSE )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, right );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _BUCKETIZE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _BUCKETIZE_PARAM_NUM );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( bucketize, _setup )

