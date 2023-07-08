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
#include "utils/vsi_nn_dtype_util.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE    "tensorstackconcat",

#define HASH_SH_KEY( IN_DTYPE, OUT_DTYPE, _image_2d ) \
    (( IN_DTYPE << 20 ) | ( OUT_DTYPE << 8) | (_image_2d))

#define PACK_KERNEL_8BITS_MAP(SRC_TYPE, OUT_TYPE) \
    {   HASH_SH_KEY(SRC_TYPE, OUT_TYPE, 0), \
        CVIVANTE_NAMESPACE("evis.tensorstackconcat_8bits"), \
        KERNEL_SOURCE },

#define PACK_KERNEL_8BITS_MAP_2D(SRC_TYPE, OUT_TYPE) \
    {   HASH_SH_KEY(SRC_TYPE, OUT_TYPE, 1), \
        CVIVANTE_NAMESPACE("evis.tensorstackconcat_8bits_2D"), \
        KERNEL_SOURCE },

#define PACK_KERNEL_16BITS_MAP(SRC_TYPE, OUT_TYPE) \
    {   HASH_SH_KEY(SRC_TYPE, OUT_TYPE, 0), \
        CVIVANTE_NAMESPACE("evis.tensorstackconcat_16bits"), \
        KERNEL_SOURCE },

#define PACK_KERNEL_16BITS_MAP_2D(SRC_TYPE, OUT_TYPE) \
    {   HASH_SH_KEY(SRC_TYPE, OUT_TYPE, 1), \
        CVIVANTE_NAMESPACE("evis.tensorstackconcat_16bits_2D"), \
        KERNEL_SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _tensorstackconcat_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_8BITS_MAP( I8, I8 )
    PACK_KERNEL_8BITS_MAP( U8, U8 )
    PACK_KERNEL_8BITS_MAP_2D( I8, I8 )
    PACK_KERNEL_8BITS_MAP_2D( U8, U8 )

    PACK_KERNEL_16BITS_MAP( F16,  F16 )
    PACK_KERNEL_16BITS_MAP( BF16, BF16 )
    PACK_KERNEL_16BITS_MAP( I16,  I16 )
    PACK_KERNEL_16BITS_MAP_2D( F16,  F16 )
    PACK_KERNEL_16BITS_MAP_2D( BF16, BF16 )
    PACK_KERNEL_16BITS_MAP_2D( I16,  I16 )
};

/*
 * Kernel params
 */
static vx_param_description_t _tensorstackconcat_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};
#define _TENSORSTACKCONCAT_PARAM_NUM  _cnt_of_array( _tensorstackconcat_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_tensorstackconcat_initializer)
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
    vsi_nn_kernel_tensor_attr_t * input_attr    = NULL;
    vsi_size_array_t * in_shape             = NULL;
    // Add initializer

    VSI_UNREFERENCED(param_size);

    input_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );

    in_shape  = input_attr->shape;

    if (input_attr->dtype == I16 || input_attr->dtype == F16)
    {
        gpu_param.global_scale[0]  = 8;
    }
    else
    {
        gpu_param.global_scale[0]  = 16;
    }
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0] = gpu_align_p2(
            (in_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = 1;
    gpu_param.global_size[2] = in_shape->size > 2 ? in_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(input_attr);
#undef SAFE_FREE_TENSOR_ATTR

    return status;
} /* _tensorstackconcat_initializer() */


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
    const _kernel_map_type * kernel_map = _tensorstackconcat_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _tensorstackconcat_kernel_map );
    vx_param_description_t * param_def  = _tensorstackconcat_kernel_param_def;
    vx_kernel_initialize_f  initializer = _tensorstackconcat_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_SH_KEY( in_dtype, out_dtype, image_2d );

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
        kernel->info.numParams   = _cnt_of_array( _tensorstackconcat_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_TENSORSTACKCONCAT_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;

    VSI_UNREFERENCED(params);

    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, inputs, outputs, image_2d );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _TENSORSTACKCONCAT_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _TENSORSTACKCONCAT_PARAM_NUM );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( tensorstackconcat, _setup )
