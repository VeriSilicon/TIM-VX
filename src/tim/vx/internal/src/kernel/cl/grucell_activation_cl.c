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
    INTERNAL_KERNEL_GRUCELL_ACTIVATION,
} _internal_kernel_e;

#define _GRUCELL_ACTIVATION_KERNEL_SOURCE      "grucell_activation"
#define _GRUCELL_ACTIVATION_KERNEL_NAME        CVIVANTE_NAMESPACE("cl.grucell_activation")

// Add kernel hashtable here
#define GRUCELL_ACTIVATION_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        (( IN_DTYPE << 8 ) | ( OUT_DTYPE ))
#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, SOURCE ) \
        { GRUCELL_ACTIVATION_HASH_KEY( IN_DTYPE, OUT_DTYPE ), _GRUCELL_ACTIVATION_KERNEL_NAME, SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _grucell_activation_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32, F32, _GRUCELL_ACTIVATION_KERNEL_SOURCE ),
};


/*
 * Kernel params
 */
static vx_param_description_t _grucell_activation_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _GRUCELL_ACTIVATION_PARAM_NUM  _cnt_of_array( _grucell_activation_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_grucell_activation_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    // vsi_nn_kernel_tensor_attr * attr[2] = { NULL };
    // attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    // attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );

    // Add initializer

    // vsi_nn_kernel_tensor_attr_release( &attr[0] );
    // vsi_nn_kernel_tensor_attr_release( &attr[1] );
    return status;
} /* _grucell_activation_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _grucell_activation_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _grucell_activation_kernel_map );
    vx_param_description_t * param_def  = _grucell_activation_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _grucell_activation_kernel_param_def );
    vx_kernel_initialize_f  initializer = _grucell_activation_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = GRUCELL_ACTIVATION_HASH_KEY( in_dtype, out_dtype );

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
    vsi_nn_kernel_node_param_t node_params[_GRUCELL_ACTIVATION_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;

    /*
    // Check if gpu can support the size
    if( !vsi_nn_kernel_gpu_check_shape(
        (int32_t*)inputs[0]->attr.size, inputs[0]->attr.dim_num ) )
    {
        return NULL;
    }
    */

    /*
     * Use kernel param
     * int32_t integer  = vsi_nn_kernel_param_get_int32( param, "data_key_i32" );
     * int64_t integer  = vsi_nn_kernel_param_get_int64( param, "hashkey" );
     * float fp         = vsi_nn_kernel_param_get_float32( param, "data_key_f32" );
     * const char * str = vsi_nn_kernel_param_get_char( param, "padding" );
     * size_t buffer_size;
     * int * buffer = (int*)vsi_nn_kernel_param_get_buffer( param, "padding", &buffer_size );
     */
    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _GRUCELL_ACTIVATION_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GRUCELL_ACTIVATION_PARAM_NUM );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( grucell_activation, _setup )

