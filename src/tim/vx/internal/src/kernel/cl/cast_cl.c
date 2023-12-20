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

typedef enum _internal_img_dim_e
{
    IMAGE = 0,
    IMAGE_2D,
} internal_img_dim_e;

#define _CAST_KERNEL_SOURCE      "cast"

#define STR(a) #a
// Add kernel hashtable here
#define CAST_HASH_KEY( IN_DTYPE, OUT_DTYPE, _image_2d ) \
        (( IN_DTYPE << 20 ) | ( OUT_DTYPE << 8) | (_image_2d))

#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, SOURCE ) \
        { CAST_HASH_KEY( IN_DTYPE, OUT_DTYPE, IMAGE ), \
          CVIVANTE_NAMESPACE("cl.cast_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)), \
          SOURCE }

#define PACK_KERNEL_MAP_2D( IN_DTYPE, OUT_DTYPE, SOURCE ) \
        { CAST_HASH_KEY( IN_DTYPE, OUT_DTYPE, IMAGE_2D ), \
          CVIVANTE_NAMESPACE("cl.cast_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
          SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _cast_kernel_map[] =
{
    PACK_KERNEL_MAP( F32, F32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( F32, I32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( F32, U32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( I32, F32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( I32, I32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( I32, U32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( U32, F32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( U32, I32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( U32, U32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( F32, BOOL8, _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( I32, BOOL8, _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP( U32, BOOL8, _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( F32, F32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( F32, I32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( F32, U32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( I32, F32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( I32, I32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( I32, U32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( U32, F32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( U32, I32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( U32, U32,   _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( F32, BOOL8, _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( I32, BOOL8, _CAST_KERNEL_SOURCE ),
    PACK_KERNEL_MAP_2D( U32, BOOL8, _CAST_KERNEL_SOURCE ),
};


/*
 * Kernel params
 */
static vx_param_description_t _cast_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _CAST_PARAM_NUM  _cnt_of_array( _cast_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_cast_initializer)
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
    vsi_nn_kernel_tensor_attr_t * output_attr   = NULL;
    vsi_size_array_t * out_shape                 = NULL;

    VSI_UNREFERENCED(param_size);

    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    out_shape  = output_attr->shape;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.dim = out_shape->size < 3 ? 2 : 3;
    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(output_attr);
    return status;
} /* _cast_initializer() */



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
    const _kernel_map_type * kernel_map = _cast_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _cast_kernel_map );
    vx_param_description_t * param_def  = _cast_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _cast_kernel_param_def );
    vx_kernel_initialize_f  initializer = _cast_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == in_dtype)
    {
        in_dtype = F32;
    }
    else if ((I8 == in_dtype) || (BOOL8 == in_dtype) || (I16 == in_dtype))
    {
        in_dtype = I32;
    }
    else if ((U8 == in_dtype) || (U16 == in_dtype))
    {
        in_dtype = U32;
    }

    if (F16 == out_dtype)
    {
        out_dtype = F32;
    }
    else if ((I8 == out_dtype) || (I16 == out_dtype))
    {
        out_dtype = I32;
    }
    else if ((U8 == out_dtype) || (U16 == out_dtype))
    {
        out_dtype = U32;
    }

    key = CAST_HASH_KEY( in_dtype, out_dtype, image_2d );

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
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 1,
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
    vsi_nn_kernel_node_param_t node_params[_CAST_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;

    VSI_UNREFERENCED(params);

    if( !vsi_nn_kernel_gpu_check_shape( inputs[0]->attr.size,
                inputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, inputs, outputs, image_2d );

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _CAST_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CAST_PARAM_NUM );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( cast, _setup )
