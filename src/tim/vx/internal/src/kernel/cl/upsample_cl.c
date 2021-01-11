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
    INTERNAL_KERNEL_UPSAMPLE,
} _internal_kernel_e;

#define _UPSAMPLE_KERNEL_SOURCE      "upsample"

#define UPSAMPLE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, _image_2d ) \
        ((IN0_DTYPE << 20) | (IN1_DTYPE << 12) | (OUT_DTYPE << 4) | (_image_2d))


#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { UPSAMPLE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 0 ), \
          CVIVANTE_NAMESPACE("cl.upsample_"#IN0_DTYPE"_"#IN1_DTYPE"to_"#OUT_DTYPE), \
          _UPSAMPLE_KERNEL_SOURCE }


#define PACK_KERNEL_MAP_2D( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { UPSAMPLE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, 1 ), \
          CVIVANTE_NAMESPACE("cl.upsample_"#IN0_DTYPE"_"#IN1_DTYPE"to_"#OUT_DTYPE"_2D"), \
          _UPSAMPLE_KERNEL_SOURCE }


typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _upsample_kernel_map[] =
{
    PACK_KERNEL_MAP( F32, U8, F32 ),
    PACK_KERNEL_MAP( F32, U8, U8 ),
    PACK_KERNEL_MAP( U8,  U8, F32 ),
    PACK_KERNEL_MAP( U8,  U8,  U8 ),
    PACK_KERNEL_MAP( I32, U8, I32 ),
    PACK_KERNEL_MAP_2D( F32, U8, F32 ),
    PACK_KERNEL_MAP_2D( F32, U8, U8 ),
    PACK_KERNEL_MAP_2D( U8,  U8, F32 ),
    PACK_KERNEL_MAP_2D( U8,  U8,  U8 ),
    PACK_KERNEL_MAP_2D( I32, U8, I32 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _upsample_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _UPSAMPLE_PARAM_NUM  _cnt_of_array( _upsample_kernel_param_def )

#define SCALAR_SCALE          (3)
#define SCALAR_TAIL           (4)
#define SCALAR_IN_ZP          (5)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_upsample_initializer)
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

    vx_status    status             = VX_FAILURE;
    vx_tensor    input              = (vx_tensor)param[0];
    vsi_nn_kernel_tensor_attr_t * attr_in = NULL;
    vsi_int_array_t * in_shape   = NULL;
    vsi_bool          image_2d    = FALSE;

    attr_in = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)input );
    CHECK_PTR_FAIL_GOTO( attr_in, "vsi_nn_kernel_tensor_attr_create fail.", final );

    in_shape = attr_in->shape;
    image_2d = (vsi_bool)(in_shape->size < 3 || 1 == in_shape->data[2]);

    gpu_param.dim = image_2d ? 2 : 3;
    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.global_size[0]   = gpu_align_p2((in_shape->data[0] +  gpu_param.global_scale[0] - 1)
                                        /  gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (in_shape->data[1] +  gpu_param.global_scale[1] - 1)
                                        /  gpu_param.global_scale[1];
    gpu_param.global_size[2]   = image_2d ? 1 : in_shape->data[2];

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (attr_in)
    {
        vsi_nn_kernel_tensor_attr_release(&attr_in);
    }

    return status;
} /* _upsample_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool image_2d,
    vsi_bool *is_use_u8_kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _upsample_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _upsample_kernel_map );
    vx_param_description_t * param_def  = _upsample_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _upsample_kernel_param_def );
    vx_kernel_initialize_f  initializer = _upsample_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype  = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == in0_dtype)
    {
        in0_dtype   = F32;
    }

    if (F16 == out_dtype)
    {
        out_dtype   = F32;
    }

    if ((U8 != in0_dtype) && (U8 != out_dtype))
    {
        *is_use_u8_kernel = FALSE;
        param_def_size    = param_def_size - 3;
    }
    else
    {
        *is_use_u8_kernel = TRUE;
    }

    key = UPSAMPLE_HASH_KEY( in0_dtype, in1_dtype, out_dtype, image_2d );

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
    vsi_nn_kernel_node_param_t node_params[_UPSAMPLE_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t  scale_x  = 0;
    int32_t  scale_y  = 0;
    vsi_bool image_2d = FALSE;
    vsi_bool is_use_u8_kernel = FALSE;
    float    outputScale  = outputs[0]->attr.dtype.scale == 0.0f ? 1.0f : outputs[0]->attr.dtype.scale;
    float    outputTail   = (float)outputs[0]->attr.dtype.zero_point;
    float    inputScale   = inputs[0]->attr.dtype.scale == 0.0f ? 1.0f : inputs[0]->attr.dtype.scale;
    float    inputTail    = (float)inputs[0]->attr.dtype.zero_point;
    int32_t  outputZp      = outputs[0]->attr.dtype.zero_point;
    float    scale_value  = 1.0f;
    float    tail_value   = 0.0f;

    scale_x  = vsi_nn_kernel_param_get_int32(params, "scale_x");
    scale_y  = vsi_nn_kernel_param_get_int32(params, "scale_y");

    if (2 != scale_x || 2 != scale_y)
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( (int32_t*)inputs[1]->attr.size,
                inputs[1]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ))
    {
        return NULL;
    }

    scale_value = inputScale / outputScale;
    tail_value  = outputTail - inputTail * inputScale / outputScale;
    image_2d    = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status      = _query_kernel( kernel, inputs, outputs, image_2d, &is_use_u8_kernel);

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            size_t node_params_num = _UPSAMPLE_PARAM_NUM - 3;
            if (is_use_u8_kernel)
            {
                node_params[SCALAR_SCALE]  = vsi_nn_kernel_scalar_create( graph, F32, &scale_value );
                node_params[SCALAR_TAIL]   = vsi_nn_kernel_scalar_create(graph, F32, &tail_value );
                node_params[SCALAR_IN_ZP]  = vsi_nn_kernel_scalar_create(graph, I32, &outputZp );
                node_params_num = _UPSAMPLE_PARAM_NUM;
            }
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, node_params_num,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            VSI_ASSERT( status == VSI_SUCCESS );
            if (is_use_u8_kernel)
            {
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_SCALE] );
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_TAIL] );
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_IN_ZP] );
            }
        }
    }

    return node;

} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( upsample, _setup )

