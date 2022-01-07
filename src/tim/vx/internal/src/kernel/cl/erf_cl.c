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

/*
 * Define kernel meta.
 */
#define HASH_UNARY_KEY(_input_type, _output_type, _image_2d) \
    ( (_input_type << 12) | (_output_type << 4) | (_image_2d))

 #define VSI_NN_GEN_UNARY_KERNEL_SOURCE_NAME() \
    "erf"

#define HASH_UNARY_SH_KERNEL_NAME( SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.erf_"#SRC_TYPE"to"#DST_TYPE)

#define TENSOR_UNARY_KERNELS(SRC_TYPE, OUT_TYPE) \
    {   HASH_UNARY_KEY(SRC_TYPE, OUT_TYPE, 0), \
        HASH_UNARY_SH_KERNEL_NAME(SRC_TYPE, OUT_TYPE), \
        VSI_NN_GEN_UNARY_KERNEL_SOURCE_NAME() },

#define HASH_UNARY_SH_KERNEL_2D_NAME(SRC_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.erf_"#SRC_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_UNARY_KERNELS_2D(SRC_TYPE, OUT_TYPE) \
    {   HASH_UNARY_KEY(SRC_TYPE, OUT_TYPE, 1), \
        HASH_UNARY_SH_KERNEL_2D_NAME(SRC_TYPE, OUT_TYPE), \
        VSI_NN_GEN_UNARY_KERNEL_SOURCE_NAME() },

#define TENSOR_UNARY_KERNELS_FLOAT(SRC_TYPE, OUT_TYPE) \
    {   HASH_UNARY_KEY(SRC_TYPE, OUT_TYPE, 0), \
        HASH_UNARY_SH_KERNEL_NAME(F32, F32), \
        VSI_NN_GEN_UNARY_KERNEL_SOURCE_NAME() },

#define TENSOR_UNARY_KERNELS_FLOAT_2D(SRC_TYPE, OUT_TYPE) \
    {   HASH_UNARY_KEY(SRC_TYPE, OUT_TYPE, 1), \
        HASH_UNARY_SH_KERNEL_2D_NAME(F32, F32), \
        VSI_NN_GEN_UNARY_KERNEL_SOURCE_NAME() },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _erf_kernel_map[] =
{
    // Register kernel here
    TENSOR_UNARY_KERNELS_FLOAT(F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT(F16, F16)

    TENSOR_UNARY_KERNELS_FLOAT_2D(F32, F32)
    TENSOR_UNARY_KERNELS_FLOAT_2D(F16, F16)

    TENSOR_UNARY_KERNELS(U8,  U8)

    TENSOR_UNARY_KERNELS_2D(U8,  U8)
};

/*
 * Kernel params
 */
static vx_param_description_t _erf_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define SCALAR_INPUT_SCALE           (2)
#define SCALAR_INPUT_TAIL            (3)
#define SCALAR_OUTPUT_SCALE          (4)
#define SCALAR_OUTPUT_ZP             (5)
#define _ERF_PARAM_NUM  _cnt_of_array( _erf_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_erf_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    gpu_param_t gpu_param = {
        3,         // workdim
        {0, 0, 0}, // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0}, // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0}, // localWorkSize: local group size in thread
        {0, 0, 0}  // globalWorkSize: image size in thread
        };

    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    vsi_size_array_t * out_shape = NULL;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    out_shape  = attr[1]->shape;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if ( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(attr[0]);
    SAFE_FREE_TENSOR_ATTR(attr[1]);
#undef SAFE_FREE_TENSOR_ATTR

    return status;
} /* _erf_initializer() */


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
    const _kernel_map_type * kernel_map = _erf_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _erf_kernel_map );
    vx_param_description_t * param_def  = _erf_kernel_param_def;
    vx_kernel_initialize_f  initializer = _erf_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_UNARY_KEY( in_dtype, out_dtype, image_2d );

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
        kernel->info.numParams   = _cnt_of_array( _erf_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_ERF_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* rs_tensors[2] = { NULL };
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_size_t new_rank = 0;
    vsi_bool ret = FALSE;
    vsi_bool image_2d = FALSE;

    float inputScale = vsi_nn_get_tensor_scale(inputs[0]);
    float inputTail = (float)vsi_nn_get_tensor_zero_point(inputs[0]) * inputScale;
    float outputScale = vsi_nn_get_tensor_scale(outputs[0]);
    float outputZP = (float)vsi_nn_get_tensor_zero_point(outputs[0]) + 0.5f;

    ret = vsi_nn_kernel_optimize_element_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            shape, &new_rank );
    if ( ret )
    {
        rs_tensors[0] = vsi_nn_reshape_tensor( graph,
                inputs[0], shape, new_rank );
        rs_tensors[1] = vsi_nn_reshape_tensor( graph,
                outputs[0], shape, new_rank );
    }

    if ( !vsi_nn_kernel_gpu_check_shape( rs_tensors[0]->attr.size,
                rs_tensors[0]->attr.dim_num ) )
    {
        return NULL;
    }

    outputScale = vsi_abs(outputScale) < 1e-5 ? 0.0f : 1.0f / outputScale;

    image_2d = (rs_tensors[0]->attr.dim_num == 2 || rs_tensors[0]->attr.size[2] == 1);

    status = _query_kernel( kernel, inputs, outputs, image_2d );
    if ( VSI_SUCCESS == status )
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _ERF_PARAM_NUM,
                    rs_tensors, 1, &rs_tensors[1], 1 );

            node_params[SCALAR_INPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &inputScale );
            node_params[SCALAR_INPUT_TAIL] = vsi_nn_kernel_scalar_create(
                    graph, F32, &inputTail );
            node_params[SCALAR_OUTPUT_SCALE] = vsi_nn_kernel_scalar_create(
                    graph, F32, &outputScale );
            node_params[SCALAR_OUTPUT_ZP] = vsi_nn_kernel_scalar_create(
                    graph, F32, &outputZP );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _ERF_PARAM_NUM );
            CHECK_STATUS_FAIL_GOTO( status, OnError );
        }
    }

OnError:
    if (rs_tensors[0])
    {
        vsi_nn_ReleaseTensor( &rs_tensors[0] );
    }

    if (rs_tensors[1])
    {
        vsi_nn_ReleaseTensor( &rs_tensors[1] );
    }

    if (node_params[SCALAR_INPUT_SCALE])
    {
        vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_SCALE] );
    }

    if (node_params[SCALAR_INPUT_TAIL])
    {
        vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_TAIL] );
    }

    if (node_params[SCALAR_OUTPUT_SCALE])
    {
        vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_SCALE] );
    }

    if (node_params[SCALAR_OUTPUT_ZP])
    {
        vsi_nn_kernel_scalar_release( &node_params[SCALAR_OUTPUT_ZP] );
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( erf, _setup )
