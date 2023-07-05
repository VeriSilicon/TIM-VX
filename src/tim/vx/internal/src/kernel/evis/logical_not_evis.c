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

typedef enum _internal_img_dim_e
{
    IMAGE = 0,
    IMAGE_2D,
} internal_img_dim_e;

#define _LOGICAL_OPS_KERNEL_SOURCE      "logical_not"

#define STR(a) #a

// Add kernel hashtable here
#define LOGICAL_NOT_HASH_KEY(IN_DTYPE, OUT_DTYPE, _image_2d) \
        (( IN_DTYPE << 12 ) | ( OUT_DTYPE << 4) | (_image_2d))

#define PACK_KERNEL_MAP(IN_DTYPE, OUT_DTYPE) \
        { LOGICAL_NOT_HASH_KEY(IN_DTYPE, OUT_DTYPE, IMAGE), \
        CVIVANTE_NAMESPACE("evis.logical_not_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)), \
        _LOGICAL_OPS_KERNEL_SOURCE}

#define PACK_KERNEL_MAP_2D(IN_DTYPE, OUT_DTYPE) \
        { LOGICAL_NOT_HASH_KEY(IN_DTYPE, OUT_DTYPE, IMAGE_2D), \
        CVIVANTE_NAMESPACE("evis.logical_not_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
        _LOGICAL_OPS_KERNEL_SOURCE}

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _logical_not_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( I8, I8),
    PACK_KERNEL_MAP_2D(I8, I8),
};


/*
 * Kernel params
 */
static vx_param_description_t _logical_not_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _LOGICAL_NOT_PARAM_NUM  _cnt_of_array( _logical_not_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_logical_not_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    // Alignment with a power of two value.
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vx_tensor     output            = (vx_tensor)param[1];
    vsi_nn_kernel_tensor_attr_t *output_attr   = NULL;
    vsi_size_array_t             *output_shape  = NULL;

    VSI_UNREFERENCED(param_size);

    output_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output);
    CHECK_PTR_FAIL_GOTO( output_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    output_shape  = output_attr->shape;

    gpu_param.dim = output_shape->size < 3 ? 2 : 3;
    gpu_param.global_offset[0] = 0;
    gpu_param.global_offset[1] = 0;
    gpu_param.global_offset[2] = 0;
    gpu_param.global_scale[0]  = 8;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((output_shape->data[0] + gpu_param.global_scale[0] - 1)
                                             / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (output_shape->data[1] + gpu_param.global_scale[1] - 1)
                                             / gpu_param.global_scale[1];
    gpu_param.global_size[2]   = output_shape->size > 2 ?
                                 (output_shape->data[2] + gpu_param.global_scale[2] - 1)
                                             / gpu_param.global_scale[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (output_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&output_attr);
    }
    return status;
} /* _logical_not_initializer() */



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
    const _kernel_map_type * kernel_map = _logical_not_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _logical_not_kernel_map );
    vx_param_description_t * param_def  = _logical_not_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _logical_not_kernel_param_def );
    vx_kernel_initialize_f  initializer = _logical_not_initializer;
    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (BOOL8 == in_dtype && BOOL8 == out_dtype)
    {
        in_dtype  = I8;
        out_dtype = I8;
    }

    key = LOGICAL_NOT_HASH_KEY( in_dtype, out_dtype, image_2d);

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
    vsi_nn_kernel_node_param_t node_params[_LOGICAL_NOT_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t new_rank = 0;
    vsi_bool ret = FALSE;

    VSI_UNREFERENCED(params);

    ret = vsi_nn_kernel_optimize_element_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            shape, &new_rank );

    if ( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
                inputs[0], shape, new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
                outputs[0], shape, new_rank );
    }
    else
    {
        goto final;
    }

    if ( !vsi_nn_kernel_gpu_check_shape( reshape_tensors[1]->attr.size,
                reshape_tensors[1]->attr.dim_num ) )
    {
        goto final;
    }

    image_2d = (outputs[0]->attr.dim_num == 2 || outputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, &reshape_tensors[0], &reshape_tensors[1], image_2d);
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _LOGICAL_NOT_PARAM_NUM,
                    &reshape_tensors[0], input_num, &reshape_tensors[1], output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _LOGICAL_NOT_PARAM_NUM );
        }
    }

final:
    vsi_safe_release_tensor( reshape_tensors[0] );
    vsi_safe_release_tensor( reshape_tensors[1] );

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( logical_not, _setup )

