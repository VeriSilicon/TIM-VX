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
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

typedef enum _internal_img_dim_e
{
    IMAGE = 0,
    IMAGE_2D,
} internal_img_dim_e;


#define _SWISH_KERNEL_SOURCE          "swish",
#define _HSWISH_KERNEL_SOURCE         "hswish",

#define STR(a) #a
// Add kernel hashtable here
#define SWISH_HASH_KEY(SWISH_TYPE, IN_DTYPE, OUT_DTYPE, _image_2d) \
        ((SWISH_TYPE << 20) | ( IN_DTYPE << 12 ) | ( OUT_DTYPE << 4) | (_image_2d))

#define SWISH_PACK_KERNEL_FLOAT_MAP( IN_DTYPE, OUT_DTYPE) \
        { SWISH_HASH_KEY(VSI_NN_SWISH, IN_DTYPE, OUT_DTYPE, IMAGE), \
        CVIVANTE_NAMESPACE("cl.swish_F32toF32"), \
        _SWISH_KERNEL_SOURCE }

#define HSWISH_PACK_KERNEL_FLOAT_MAP( IN_DTYPE, OUT_DTYPE) \
        { SWISH_HASH_KEY(VSI_NN_HSWISH, IN_DTYPE, OUT_DTYPE, IMAGE), \
        CVIVANTE_NAMESPACE("cl.hswish_F32toF32"), \
        _HSWISH_KERNEL_SOURCE }

#define SWISH_PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE) \
        { SWISH_HASH_KEY(VSI_NN_SWISH, IN_DTYPE, OUT_DTYPE, IMAGE), \
        CVIVANTE_NAMESPACE("cl.swish_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)), \
        _SWISH_KERNEL_SOURCE }

#define HSWISH_PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE) \
        { SWISH_HASH_KEY(VSI_NN_HSWISH, IN_DTYPE, OUT_DTYPE, IMAGE), \
        CVIVANTE_NAMESPACE("cl.hswish_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)), \
        _HSWISH_KERNEL_SOURCE }


#define SWISH_PACK_KERNEL_FLOAT_MAP_2D( IN_DTYPE, OUT_DTYPE) \
        { SWISH_HASH_KEY(VSI_NN_SWISH, IN_DTYPE, OUT_DTYPE, IMAGE_2D), \
        CVIVANTE_NAMESPACE("cl.swish_F32toF32_2D"), \
        _SWISH_KERNEL_SOURCE }

#define HSWISH_PACK_KERNEL_FLOAT_MAP_2D( IN_DTYPE, OUT_DTYPE) \
        { SWISH_HASH_KEY(VSI_NN_HSWISH, IN_DTYPE, OUT_DTYPE, IMAGE_2D), \
        CVIVANTE_NAMESPACE("cl.hswish_F32toF32_2D"), \
        _HSWISH_KERNEL_SOURCE }

#define SWISH_PACK_KERNEL_MAP_2D( IN_DTYPE, OUT_DTYPE) \
        { SWISH_HASH_KEY(VSI_NN_SWISH, IN_DTYPE, OUT_DTYPE, IMAGE_2D), \
        CVIVANTE_NAMESPACE("cl.swish_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
        _SWISH_KERNEL_SOURCE }

#define HSWISH_PACK_KERNEL_MAP_2D( IN_DTYPE, OUT_DTYPE) \
        { SWISH_HASH_KEY(VSI_NN_HSWISH, IN_DTYPE, OUT_DTYPE, IMAGE_2D), \
        CVIVANTE_NAMESPACE("cl.hswish_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
        _HSWISH_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _swish_kernel_map[] =
{
    SWISH_PACK_KERNEL_FLOAT_MAP(F32,  F32),
    SWISH_PACK_KERNEL_FLOAT_MAP_2D(F32,  F32),
    SWISH_PACK_KERNEL_FLOAT_MAP(F16,  F16),
    SWISH_PACK_KERNEL_FLOAT_MAP_2D(F16,  F16),
    SWISH_PACK_KERNEL_MAP(U8,  U8),
    SWISH_PACK_KERNEL_MAP_2D(U8,  U8),
    SWISH_PACK_KERNEL_MAP(I32,  I32),
    SWISH_PACK_KERNEL_MAP_2D(I32,  I32),
    SWISH_PACK_KERNEL_MAP(F32,  U8),
    SWISH_PACK_KERNEL_MAP_2D(F32,  U8),
    HSWISH_PACK_KERNEL_FLOAT_MAP(F32,  F32),
    HSWISH_PACK_KERNEL_FLOAT_MAP_2D(F32,  F32),
    HSWISH_PACK_KERNEL_FLOAT_MAP(F16,  F16),
    HSWISH_PACK_KERNEL_FLOAT_MAP_2D(F16,  F16),
    HSWISH_PACK_KERNEL_MAP(U8,  U8),
    HSWISH_PACK_KERNEL_MAP_2D(U8,  U8),
    HSWISH_PACK_KERNEL_MAP(I32,  I32),
    HSWISH_PACK_KERNEL_MAP_2D(I32,  I32),
};


/*
 * Kernel params
 */
static vx_param_description_t _swish_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define SCALAR_INPUT_SCALE           (2)
#define SCALAR_INPUT_TAIL            (3)
#define SCALAR_OUTPUT_SCALE          (4)
#define SCALAR_OUTPUT_ZP             (5)
#define SCALAR_BETA                  (6)
#define SCALAR_LOGE                  (7)
#define _SWISH_PARAM_NUM   _cnt_of_array( _swish_kernel_param_def )
#define _HSWISH_PARAM_NUM  _SWISH_PARAM_NUM - 2
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_swish_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
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
    vx_tensor    output             = (vx_tensor)param[1];
    vsi_nn_kernel_tensor_attr_t * attr_out = NULL;
    vsi_size_array_t * out_shape = NULL;

    attr_out = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)output );
    CHECK_PTR_FAIL_GOTO( attr_out, "vsi_nn_kernel_tensor_attr_create fail.", final );

    out_shape = attr_out->shape;

    gpu_param.dim = out_shape->size < 3 ? 2 : 3;
    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    gpu_param.global_size[0]   = gpu_align_p2((out_shape->data[0] +  gpu_param.global_scale[0] - 1)
                                        /  gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (out_shape->data[1] +  gpu_param.global_scale[1] - 1)
                                        /  gpu_param.global_scale[1];
    gpu_param.global_size[2]   = out_shape->size > 2 ? out_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (attr_out)
    {
        vsi_nn_kernel_tensor_attr_release(&attr_out);
    }

    return status;
} /* _swish_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool image_2d,
    vsi_nn_swish_type swish_type
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _swish_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _swish_kernel_map );
    vx_param_description_t * param_def  = _swish_kernel_param_def;
    size_t param_def_size               = _SWISH_PARAM_NUM;
    vx_kernel_initialize_f  initializer = _swish_initializer;
    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (in_dtype == F16)
        in_dtype = F32;
    if (out_dtype == F16)
        out_dtype = F32;

    key = SWISH_HASH_KEY(swish_type, in_dtype, out_dtype, image_2d);

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }

    if (VSI_NN_SWISH == (vsi_nn_swish_type)swish_type)
    {
       param_def_size   = _SWISH_PARAM_NUM;
    }
    else
    {
       param_def_size   = _HSWISH_PARAM_NUM;
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
    vsi_nn_kernel_node_param_t node_params[_SWISH_PARAM_NUM] = {NULL};
    vsi_size_t  shape[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t new_rank = 0;
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    int32_t swish_type  = vsi_nn_kernel_param_get_int32( params, "type" );
    float   beta        = 1.0f;
    float   inputScale  = vsi_nn_get_tensor_scale(inputs[0]);
    float   inputTail   = (float)vsi_nn_get_tensor_zero_point(inputs[0]) * inputScale;
    float   outputScale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);
    float   outputZP    = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    vx_float32  logE    = (vx_float32)(log10(exp(1.0f)) / log10(2.0f));
    vsi_bool ret = FALSE;

#if (VX_ACTIVATION_EXT_SUPPORT)
    if (VSI_NN_HW_EVIS_2 == graph->ctx->config.evis.ver)
    {
        return NULL;
    }
#endif

    ret = vsi_nn_kernel_optimize_element_shape(
        inputs[0]->attr.size, inputs[0]->attr.dim_num,
        shape, &new_rank );

    if( ret )
    {
        node_params[0] = vsi_nn_kernel_tensor_reshape( inputs[0]->t, shape, new_rank );
        node_params[1] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shape, new_rank );
    }

    if( !vsi_nn_kernel_gpu_check_shape( shape, new_rank ) )
    {
        return NULL;
    }

    image_2d = (new_rank == 2);

     if (VSI_NN_HSWISH == (vsi_nn_swish_type)swish_type)
     {
        beta = 1.0f / 6.0f;
     }
     else
     {
        beta = vsi_nn_kernel_param_get_float32( params, "beta" );
     }

    status = _query_kernel( kernel, inputs, outputs, image_2d, (vsi_nn_swish_type)swish_type);
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            size_t node_params_num = _SWISH_PARAM_NUM;
            node_params[SCALAR_INPUT_SCALE] = vsi_nn_kernel_scalar_create( graph, F32, &inputScale );
            node_params[SCALAR_INPUT_TAIL] = vsi_nn_kernel_scalar_create(graph, F32, &inputTail );
            node_params[SCALAR_OUTPUT_SCALE] = vsi_nn_kernel_scalar_create( graph, F32, &outputScale );
            node_params[SCALAR_OUTPUT_ZP] = vsi_nn_kernel_scalar_create(graph, F32, &outputZP );
            if (VSI_NN_SWISH == (vsi_nn_swish_type)swish_type)
            {
                node_params[SCALAR_BETA] = vsi_nn_kernel_scalar_create( graph, F32, &beta );
                node_params[SCALAR_LOGE] = vsi_nn_kernel_scalar_create( graph, F32, &logE );
            }
            else
            {
                node_params_num = _HSWISH_PARAM_NUM;
            }

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            VSI_ASSERT( status == VSI_SUCCESS );

            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_TAIL] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_ZP] );
            if (VSI_NN_SWISH == (vsi_nn_swish_type)swish_type)
            {
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_BETA] );
                vsi_nn_kernel_scalar_release( &node_params[SCALAR_LOGE] );
            }
        }
    }

    if(node_params[0])
    {
        vsi_nn_kernel_tensor_release( &node_params[0] );
    }
    if(node_params[1])
    {
        vsi_nn_kernel_tensor_release( &node_params[1] );
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( swish, _setup )
