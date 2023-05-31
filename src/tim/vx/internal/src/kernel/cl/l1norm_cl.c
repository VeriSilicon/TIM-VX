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

#define _L1NORM_KERNEL_SOURCE_NAME      "l1norm"

// Add kernel hashtable here
#define L1NORM_HASH_KEY( IN_DTYPE, OUT_DTYPE, _image_2d, AXIS) \
        (( IN_DTYPE << 24 ) | ( OUT_DTYPE << 16) | (_image_2d << 8) | (AXIS))
#define L1NORM_KERNELS( IN_DTYPE, OUT_DTYPE, AXIS ) \
        { L1NORM_HASH_KEY( IN_DTYPE, OUT_DTYPE, 0 , AXIS), \
        CVIVANTE_NAMESPACE("cl.l1norm_"#IN_DTYPE"to"#OUT_DTYPE"_axis"#AXIS), \
        _L1NORM_KERNEL_SOURCE_NAME }

#define L1NORM_KERNELS_2D( IN_DTYPE, OUT_DTYPE, AXIS ) \
        { L1NORM_HASH_KEY( IN_DTYPE, OUT_DTYPE, 1, AXIS), \
        CVIVANTE_NAMESPACE("cl.l1norm_"#IN_DTYPE"to"#OUT_DTYPE"_2D_axis"#AXIS), \
        _L1NORM_KERNEL_SOURCE_NAME }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _l1norm_kernel_map[] =
{
    // Register kernel here
    L1NORM_KERNELS( U32, U32, 0 ),
    L1NORM_KERNELS( U32, I32, 0 ),
    L1NORM_KERNELS( U32, F32, 0 ),
    L1NORM_KERNELS( I32, I32, 0 ),
    L1NORM_KERNELS( I32, U32, 0 ),
    L1NORM_KERNELS( I32, F32, 0 ),
    L1NORM_KERNELS( F32, F32, 0 ),
    L1NORM_KERNELS( F32, U32, 0 ),
    L1NORM_KERNELS( F32, I32, 0 ),

    L1NORM_KERNELS( U32, U32, 1 ),
    L1NORM_KERNELS( U32, I32, 1 ),
    L1NORM_KERNELS( U32, F32, 1 ),
    L1NORM_KERNELS( I32, I32, 1 ),
    L1NORM_KERNELS( I32, U32, 1 ),
    L1NORM_KERNELS( I32, F32, 1 ),
    L1NORM_KERNELS( F32, F32, 1 ),
    L1NORM_KERNELS( F32, U32, 1 ),
    L1NORM_KERNELS( F32, I32, 1 ),

    L1NORM_KERNELS( U32, U32, 2 ),
    L1NORM_KERNELS( U32, I32, 2 ),
    L1NORM_KERNELS( U32, F32, 2 ),
    L1NORM_KERNELS( I32, I32, 2 ),
    L1NORM_KERNELS( I32, U32, 2 ),
    L1NORM_KERNELS( I32, F32, 2 ),
    L1NORM_KERNELS( F32, F32, 2 ),
    L1NORM_KERNELS( F32, U32, 2 ),
    L1NORM_KERNELS( F32, I32, 2 ),

    L1NORM_KERNELS_2D( U32, U32, 0 ),
    L1NORM_KERNELS_2D( U32, I32, 0 ),
    L1NORM_KERNELS_2D( U32, F32, 0 ),
    L1NORM_KERNELS_2D( I32, I32, 0 ),
    L1NORM_KERNELS_2D( I32, U32, 0 ),
    L1NORM_KERNELS_2D( I32, F32, 0 ),
    L1NORM_KERNELS_2D( F32, F32, 0 ),
    L1NORM_KERNELS_2D( F32, U32, 0 ),
    L1NORM_KERNELS_2D( F32, I32, 0 ),

    L1NORM_KERNELS_2D( U32, U32, 1 ),
    L1NORM_KERNELS_2D( U32, I32, 1 ),
    L1NORM_KERNELS_2D( U32, F32, 1 ),
    L1NORM_KERNELS_2D( I32, I32, 1 ),
    L1NORM_KERNELS_2D( I32, U32, 1 ),
    L1NORM_KERNELS_2D( I32, F32, 1 ),
    L1NORM_KERNELS_2D( F32, F32, 1 ),
    L1NORM_KERNELS_2D( F32, U32, 1 ),
    L1NORM_KERNELS_2D( F32, I32, 1 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _l1norm_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}

    // Add kererl parameters here
};
#define _L1NORM_PARAM_NUM  _cnt_of_array( _l1norm_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_l1norm_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };
    vsi_status status = VSI_FAILURE;
    vx_tensor  output = (vx_tensor)param[1];
    vx_int32   axis   = 0;
    vx_int32   dim    = 0;
    vx_int32   width  = 0;
    vx_int32   height = 0;
    vx_int32   depth  = 0;

    vsi_nn_kernel_tensor_attr_t *output_attr  = NULL;
    vsi_size_array_t            *output_shape = NULL;

    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output );
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &axis);
    CHECK_PTR_FAIL_GOTO( output_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    output_shape = output_attr->shape;

    dim    = output_shape->size < 3 ? 2 : 3;
    width  = (vx_int32)output_shape->data[0];
    height = (vx_int32)output_shape->data[1];
    depth  = dim < 3 ? 1 : (vx_int32)output_shape->data[2];

    gpu_param.dim = dim;
    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    if (axis == 0)
    {
        gpu_param.local_size[0]  = 16;
        gpu_param.local_size[1]  = 1;
        gpu_param.local_size[2]  = 1;
        gpu_param.global_size[0] = 16;
        gpu_param.global_size[1] = height;
        gpu_param.global_size[2] = depth;
    }
    else if (axis == 1)
    {
        gpu_param.local_size[0]  = 1;
        gpu_param.local_size[1]  = 16;
        gpu_param.local_size[2]  = 1;
        gpu_param.global_size[0] = width;
        gpu_param.global_size[1] = 16;
        gpu_param.global_size[2] = depth;
    }
    else
    {
        gpu_param.local_size[0]  = 1;
        gpu_param.local_size[1]  = 1;
        gpu_param.local_size[2]  = 16;

        gpu_param.global_size[0]   = width;
        gpu_param.global_size[1]   = height;
        gpu_param.global_size[2]   = 16;
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (output_attr)
    {
        vsi_nn_kernel_tensor_attr_release(&output_attr);
    }

    return status;
} /* _l1norm_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool image_2d,
    int32_t axis
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _l1norm_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _l1norm_kernel_map );
    vx_param_description_t * param_def  = _l1norm_kernel_param_def;
    vx_kernel_initialize_f  initializer = _l1norm_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == in_dtype)
    {
        in_dtype = F32;
    }
    else if (U8 == in_dtype)
    {
        in_dtype = U32;
    }
    else if (I16 == in_dtype || I8 == in_dtype)
    {
        in_dtype = I32;
    }

    if (F16 == out_dtype)
    {
        out_dtype = F32;
    }
    else if (U8 == out_dtype)
    {
        out_dtype = U32;
    }
    else if (I16 == out_dtype || I8 == out_dtype)
    {
        out_dtype = I32;
    }

    key = L1NORM_HASH_KEY( in_dtype, out_dtype, image_2d, axis);

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
        kernel->info.numParams   = _cnt_of_array( _l1norm_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_L1NORM_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    int32_t axis      = vsi_nn_kernel_param_get_int32(params, "axis");
    float outputScale = vsi_nn_get_tensor_scale(outputs[0]);
    float outputTail  = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float inputZp     = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    int32_t axis_size   = (int32_t)outputs[0]->attr.size[axis];
    outputScale = 1.0f / outputScale;

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (outputs[0]->attr.dim_num == 2);

    status = _query_kernel( kernel, inputs, outputs, image_2d, axis );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            uint32_t index = 2;
            vsi_nn_kernel_node_pack_io( node_params, _L1NORM_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &inputZp );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &outputScale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &outputTail );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis_size );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _L1NORM_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( l1norm, _setup )

