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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

typedef enum _crop_and_resize_type_e
{
    nearest_neighbor = VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR,
    bilinear = VSI_NN_INTERPOLATION_BILINEAR,
}crop_and_resize_type_e;

#define _CROP_AND_RESIZE_KERNEL_SOURCE_NAME      "crop_and_resize_"

// Add kernel hashtable here
#define CROP_AND_RESIZE_HASH_KEY( IN_DTYPE, OUT_DTYPE, RESIZE_METHOD ) \
        (( IN_DTYPE << 16 ) | ( OUT_DTYPE << 8) | (RESIZE_METHOD))
#define CROP_AND_RESIZE_KERNEL( IN_DTYPE, OUT_DTYPE, RESIZE_METHOD ) \
        { CROP_AND_RESIZE_HASH_KEY( IN_DTYPE, OUT_DTYPE, RESIZE_METHOD ), \
          CVIVANTE_NAMESPACE("cl.crop_and_resize_"#RESIZE_METHOD"_"#IN_DTYPE"to"#OUT_DTYPE), \
          _CROP_AND_RESIZE_KERNEL_SOURCE_NAME#RESIZE_METHOD }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _crop_and_resize_kernel_map[] =
{
    // Register kernel here
    CROP_AND_RESIZE_KERNEL( U32, U32, nearest_neighbor ),
    CROP_AND_RESIZE_KERNEL( U32, F32, nearest_neighbor ),
    CROP_AND_RESIZE_KERNEL( F32, F32, nearest_neighbor),
    CROP_AND_RESIZE_KERNEL( F32, U32, nearest_neighbor ),
    CROP_AND_RESIZE_KERNEL( F32, I32, nearest_neighbor),
    CROP_AND_RESIZE_KERNEL( I32, I32, nearest_neighbor ),
    CROP_AND_RESIZE_KERNEL( I32, F32, nearest_neighbor),

    CROP_AND_RESIZE_KERNEL( U32, U32, bilinear),
    CROP_AND_RESIZE_KERNEL( U32, F32, bilinear),
    CROP_AND_RESIZE_KERNEL( F32, F32, bilinear),
    CROP_AND_RESIZE_KERNEL( F32, U32, bilinear),
    CROP_AND_RESIZE_KERNEL( F32, I32, bilinear),
    CROP_AND_RESIZE_KERNEL( I32, I32, bilinear),
    CROP_AND_RESIZE_KERNEL( I32, F32, bilinear),
};


/*
 * Kernel params
 */
static vx_param_description_t _crop_and_resize_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _CROP_AND_RESIZE_PARAM_NUM  _cnt_of_array( _crop_and_resize_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_crop_and_resize_initializer)
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
        {0, 0, 0},
    };

    vsi_nn_kernel_tensor_attr_t* attr[2] = {NULL, NULL};
    int32_t       crop_width  = 0;
    int32_t       crop_height = 0;
    int32_t       image_width  = 0;
    int32_t       image_height = 0;
    int32_t       batch_out = 0;
    float         width_scale = 0;
    float         height_scale = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &batch_out);
    CHECK_STATUS_FAIL_GOTO(status, final );

    image_width = (int32_t)(attr[0]->shape->data[0]);
    image_height = (int32_t)(attr[0]->shape->data[1]);
    crop_width = (int32_t)(attr[1]->shape->data[0]);
    crop_height = (int32_t)(attr[1]->shape->data[1]);

    width_scale = (crop_width > 1) ? (float)(image_width - 1) / (crop_width -1) : 0;
    height_scale = (crop_height > 1) ? (float)(image_height - 1) / (crop_height -1) : 0;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;

    gpu_param.global_size[0]   = (crop_width + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0];
    gpu_param.global_size[1]   = (crop_height + gpu_param.global_scale[1] - 1)
                                        / gpu_param.global_scale[1];
    gpu_param.global_size[2]   = (batch_out + gpu_param.global_scale[2] - 1)
                                        / gpu_param.global_scale[2];

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final);

    status = vsi_nn_kernel_gpu_add_param( node, "width_scale", &width_scale );
    status |= vsi_nn_kernel_gpu_add_param( node, "height_scale", &height_scale );
    status |= vsi_nn_kernel_gpu_add_param( node, "image_width", &image_width );
    status |= vsi_nn_kernel_gpu_add_param( node, "image_height", &image_height );
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }

    return status;
} /* _crop_and_resize_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t resize_method
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _crop_and_resize_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _crop_and_resize_kernel_map );
    vx_param_description_t * param_def  = _crop_and_resize_kernel_param_def;
    vx_kernel_initialize_f  initializer = _crop_and_resize_initializer;

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
    else if (I8 == in_dtype || I16 == in_dtype)
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
    else if (I8 == out_dtype || I16 == out_dtype)
    {
        out_dtype = I32;
    }

    key = CROP_AND_RESIZE_HASH_KEY( in_dtype, out_dtype, resize_method );

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
        kernel->info.numParams   = _cnt_of_array( _crop_and_resize_kernel_param_def );
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
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
    vsi_nn_kernel_node_param_t node_params[_CROP_AND_RESIZE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL;
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = {{0}};
    uint32_t ori_depth = (uint32_t)inputs[0]->attr.size[2];
    uint32_t ori_batchout = (uint32_t)outputs[0]->attr.size[3];
    float input_zp     = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float input_scale  = vsi_nn_get_tensor_scale(inputs[0]);
    float output_zp    = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);
    float inOutScale   = input_scale / output_scale;
    float inOutTile    = output_zp - inOutScale * input_zp;

    float extrapolation_value = vsi_nn_kernel_param_get_float32( params, "extrapolation_value" );
    int32_t resize_method = vsi_nn_kernel_param_get_int32( params, "resize_method" );

    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    shapes[0][0] = inputs[0]->attr.size[0];
    shapes[0][1] = inputs[0]->attr.size[1];
    shapes[0][2] = inputs[0]->attr.size[2] * inputs[0]->attr.size[3];

    shapes[1][0] = outputs[0]->attr.size[0];
    shapes[1][1] = outputs[0]->attr.size[1];
    shapes[1][2] = outputs[0]->attr.size[2] * outputs[0]->attr.size[3];

    rs_input = vsi_nn_kernel_tensor_reshape( inputs[0]->t, shapes[0], 3 );
    rs_output = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[1], 3 );

    if (rs_input == NULL || rs_output == NULL)
    {
        goto final;
    }

    status = _query_kernel( kernel, inputs, outputs, resize_method );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            node_params[0] = rs_input;
            node_params[1] = (vsi_nn_kernel_node_param_t)(inputs[1]->t);
            node_params[2] = (vsi_nn_kernel_node_param_t)(inputs[2]->t);
            node_params[3] = rs_output;
            node_params[4] = vsi_nn_kernel_scalar_create( graph, I32, &ori_depth );
            node_params[5] = vsi_nn_kernel_scalar_create( graph, I32, &ori_batchout );
            node_params[6] = vsi_nn_kernel_scalar_create( graph, F32, &inOutScale );
            node_params[7] = vsi_nn_kernel_scalar_create( graph, F32, &inOutTile );
            node_params[8] = vsi_nn_kernel_scalar_create( graph, F32, &extrapolation_value );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CROP_AND_RESIZE_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
        }
    }
final:
    if (rs_input)
    {
        vsi_nn_kernel_tensor_release( &rs_input );
    }
    if (rs_output)
    {
        vsi_nn_kernel_tensor_release( &rs_output );
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( crop_and_resize, _setup )

