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
#include "utils/vsi_nn_dtype_util_prv.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum _custom_warp_perspective_type_e
{
    nearest_neighbor = VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR,
    bilinear = VSI_NN_INTERPOLATION_BILINEAR,
}custom_warp_perspective_type_e;
#define _CUSTOM_WARP_PERSPECTIVE_KERNEL_SOURCE      "custom_warp_perspective"

// Add kernel hashtable here
#define CUSTOM_WARP_PERSPECTIVE_HASH_KEY( IN_DTYPE, OUT_DTYPE, TYPE, IMG_2D ) \
        (( IN_DTYPE ) | ( OUT_DTYPE << 8 ) | (TYPE << 16) | (IMG_2D << 20))
#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, TYPE ) \
        { CUSTOM_WARP_PERSPECTIVE_HASH_KEY( IN_DTYPE, OUT_DTYPE, TYPE, 0 ), \
          CVIVANTE_NAMESPACE("evis.custom_warp_perspective_"#TYPE"_"#IN_DTYPE"to"#IN_DTYPE), \
          _CUSTOM_WARP_PERSPECTIVE_KERNEL_SOURCE }
#define PACK_2D_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, TYPE ) \
        { CUSTOM_WARP_PERSPECTIVE_HASH_KEY( IN_DTYPE, OUT_DTYPE, TYPE, 1 ), \
          CVIVANTE_NAMESPACE("evis.custom_warp_perspective_"#TYPE"_"#IN_DTYPE"to"#IN_DTYPE"_2D"), \
          _CUSTOM_WARP_PERSPECTIVE_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _custom_warp_perspective_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( U8, U8, nearest_neighbor ),
    PACK_KERNEL_MAP( U8, U8, bilinear ),

    PACK_2D_KERNEL_MAP( U8, U8, nearest_neighbor ),
    PACK_2D_KERNEL_MAP( U8, U8, bilinear ),
};


/*
 * Kernel params
 */
static vx_param_description_t _custom_warp_perspective_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _CUSTOM_WARP_PERSPECTIVE_PARAM_NUM  _cnt_of_array( _custom_warp_perspective_kernel_param_def )
#define SCALAR_MATRIX_OFFSET    (2)
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_custom_warp_perspective_initializer)
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

    vsi_nn_kernel_tensor_attr_t* attr[2] = {NULL, NULL};
    vsi_size_array_t * out_shape = NULL;
    float m[9] = {0};
    float matrix0[4] = {0};
    float matrix1[4] = {0};
    float matrix2[4] = {0};
    float matrix4[4] = {0};
    int32_t i = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    for (i = 0; i < 9; i++)
    {
        status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[SCALAR_MATRIX_OFFSET + i],
            &m[i]);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    matrix0[0] = m[0]; matrix0[1] = m[1]; matrix0[2] = m[3]; matrix0[3] = m[4];
    matrix1[0] = m[6]; matrix1[1] = m[7]; matrix1[2] = m[2]; matrix1[3] = m[5];
    matrix2[0] = m[8];
    matrix4[0] = m[0]; matrix4[1] = m[1]; matrix4[2] = m[0] * 2; matrix4[3] = m[1] * 2;

    out_shape  = attr[1]->shape;

    gpu_param.global_scale[0] = 8;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_add_param( node,
        "matrix0", &matrix0 );
    status |= vsi_nn_kernel_gpu_add_param( node,
        "matrix1", &matrix1 );
    status |= vsi_nn_kernel_gpu_add_param( node,
        "matrix2", &matrix2 );
    status |= vsi_nn_kernel_gpu_add_param( node,
        "matrix4", &matrix4 );
    CHECK_STATUS_FAIL_GOTO(status, final );

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

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
} /* _custom_warp_perspective_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t type
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _custom_warp_perspective_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _custom_warp_perspective_kernel_map );
    vx_param_description_t * param_def  = _custom_warp_perspective_kernel_param_def;
    vx_kernel_initialize_f  initializer = _custom_warp_perspective_initializer;
    int32_t is_2d_img = inputs[0]->attr.dim_num < 3 || inputs[0]->attr.size[2] == 1;
    uint32_t key = 0;
    uint32_t i = 0;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = CUSTOM_WARP_PERSPECTIVE_HASH_KEY( in_dtype, out_dtype, type, is_2d_img );

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
        kernel->info.numParams   = _cnt_of_array( _custom_warp_perspective_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_CUSTOM_WARP_PERSPECTIVE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    size_t i = 0;
    size_t buffer_size = 0;
    int32_t type = vsi_nn_kernel_param_get_int32( params, "type");
    float * buffer = (float*)vsi_nn_kernel_param_get_const_buffer( params, "matrix", &buffer_size );

    if (vsi_nn_DtypeCompare(&inputs[0]->attr.dtype, &outputs[0]->attr.dtype) == FALSE)
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, type );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            vx_border_t border;
            border.mode = VX_BORDER_CONSTANT;

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _CUSTOM_WARP_PERSPECTIVE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            for (i = 0; i < buffer_size; i++)
            {
                node_params[SCALAR_MATRIX_OFFSET + i] = vsi_nn_kernel_scalar_create(
                        graph, F32, &buffer[i] );
            }
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CUSTOM_WARP_PERSPECTIVE_PARAM_NUM );
            for (i = 0; i < buffer_size; i++)
            {
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_MATRIX_OFFSET + i] );
            }
            // Set default border mode.
            border.constant_value.U32 = 0xcdcdcdcd;
            status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
            CHECK_STATUS(status);
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( custom_warp_perspective, _setup )
