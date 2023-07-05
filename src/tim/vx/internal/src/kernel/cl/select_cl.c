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
#include "libnnext/vx_lib_nnext.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

typedef enum _internal_img_dim_e
{
    IMAGE = 0,
    IMAGE_2D,
} internal_img_dim_e;

#define _SELECT_KERNEL_SOURCE      "select"

#define STR(a) #a

// Add kernel hashtable here
#define SELECT_HASH_KEY(COND_DTYPE, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, _image_2d) \
        ((COND_DTYPE << 25) | (IN0_DTYPE << 18) | ( IN1_DTYPE << 11 ) | ( OUT_DTYPE << 4) | (_image_2d))

#define PACK_KERNEL_MAP(COND_DTYPE, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
        { SELECT_HASH_KEY(COND_DTYPE, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, IMAGE), \
        CVIVANTE_NAMESPACE("cl.select_"STR(COND_DTYPE)"_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)), \
        _SELECT_KERNEL_SOURCE}

#define PACK_KERNEL_MAP_2D(COND_DTYPE, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
        { SELECT_HASH_KEY(COND_DTYPE, IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, IMAGE_2D), \
        CVIVANTE_NAMESPACE("cl.select_"STR(COND_DTYPE)"_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_2D"), \
        _SELECT_KERNEL_SOURCE}

#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _select_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP(I8, U8,  U8,  U8),
    PACK_KERNEL_MAP(I8, I32, I32, I32),
    PACK_KERNEL_MAP(I8, F32, F32, F32),
    PACK_KERNEL_MAP_2D(I8, U8,  U8,  U8),
    PACK_KERNEL_MAP_2D(I8, I32, I32, I32),
    PACK_KERNEL_MAP_2D(I8, F32, F32, F32),
};

/*
 * Kernel params
 */
static vx_param_description_t _select_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define SCALAR_INPUT0_SCALE           (4)
#define SCALAR_INPUT0_TAIL            (5)
#define SCALAR_INPUT1_SCALE           (6)
#define SCALAR_INPUT1_TAIL            (7)
#define _SELECT_PARAM_NUM  _cnt_of_array( _select_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_select_initializer)
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

    vx_tensor     output            = (vx_tensor)param[3];
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
    gpu_param.global_scale[0]  = 1;
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
} /* _select_initializer() */

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
    vsi_nn_kernel_dtype_e cond_dtype;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _select_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _select_kernel_map );
    vx_param_description_t * param_def  = _select_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _select_kernel_param_def );
    vx_kernel_initialize_f  initializer = _select_initializer;

    uint32_t key;
    uint32_t i;

    cond_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in0_dtype   = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    in1_dtype   = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    out_dtype   = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    cond_dtype  = (BOOL8 == cond_dtype) ? I8 : cond_dtype;
    in0_dtype   = (BOOL8 == in0_dtype)  ? I32 : in0_dtype;
    in1_dtype   = (BOOL8 == in1_dtype)  ? I32 : in1_dtype;
    out_dtype   = (BOOL8 == out_dtype)  ? I32 : out_dtype;

    in0_dtype   = (F16 == in0_dtype)  ? F32 : in0_dtype;
    in1_dtype   = (F16 == in1_dtype)  ? F32 : in1_dtype;
    out_dtype   = (F16 == out_dtype)  ? F32 : out_dtype;

    key = SELECT_HASH_KEY(cond_dtype, in0_dtype, in1_dtype, out_dtype, image_2d);

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
                "eltwise_ops_helper",
                kernel_map[i].source_name);
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
    vsi_nn_kernel_node_param_t node_params[_SELECT_PARAM_NUM] = {NULL};
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_node_t node = NULL;
    float   outputScale  = vsi_nn_get_tensor_scale(outputs[0]);
    float   outputZP     = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float   input0Scale  = vsi_nn_get_tensor_scale(inputs[1]);
    float   input0Tail   = (float)vsi_nn_get_tensor_zero_point(inputs[1]);
    float   input1Scale  = vsi_nn_get_tensor_scale(inputs[2]);
    float   input1Tail   = (float)vsi_nn_get_tensor_zero_point(inputs[2]);

    vsi_nn_tensor_t* reshape_tensors[_IO_NUM] = { NULL };
    vsi_size_t shapes[_IO_NUM][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    vsi_size_t* shapes_ptr[_IO_NUM];
    vsi_size_t* shapes_in[_INPUT_NUM];
    vsi_size_t rank_in[_INPUT_NUM];
    uint32_t new_rank = 0;
    uint32_t i = 0;
    vsi_bool ret = FALSE;

    VSI_UNREFERENCED(params);

    input0Scale = input0Scale / outputScale;
    input1Scale = input1Scale / outputScale;
    input0Tail  = outputZP - input0Tail * input0Scale;
    input1Tail  = outputZP - input1Tail * input1Scale;


    for (i = 0; i < _IO_NUM; i++)
    {
        shapes_ptr[i] = shapes[i];
    }

    for (i = 0; i < _INPUT_NUM; i++)
    {
        shapes_in[i] = inputs[i]->attr.size;
        rank_in[i]   = (vsi_size_t)inputs[i]->attr.dim_num;
    }

    ret = vsi_nn_kernel_optimize_broadcast_shape(
            (const vsi_size_t**)shapes_in, rank_in, _INPUT_NUM,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes_ptr, shapes[_INPUT_NUM], &new_rank);

    if ( ret )
    {
        for (i = 0; i < _INPUT_NUM; i++)
        {
            reshape_tensors[i] = vsi_nn_reshape_tensor( graph,
                    inputs[i], shapes[i], new_rank );
        }

        for (i = 0; i < _OUTPUT_NUM; i++)
        {
            reshape_tensors[i + _INPUT_NUM] = vsi_nn_reshape_tensor( graph,
                    outputs[i], shapes[i + _INPUT_NUM], new_rank );
        }
    }
    else
    {
        for (i = 0; i < _INPUT_NUM; i++)
        {
            reshape_tensors[i] = inputs[i];
        }
        for (i = 0; i < _OUTPUT_NUM; i++)
        {
            reshape_tensors[i + _INPUT_NUM] = outputs[i];
        }
    }

    if ( !vsi_nn_kernel_gpu_check_shape( reshape_tensors[3]->attr.size,
                reshape_tensors[3]->attr.dim_num ) )
    {
        return NULL;
    }

    image_2d = (reshape_tensors[3]->attr.dim_num == 2);
    status = _query_kernel( kernel, inputs, &reshape_tensors[3], image_2d);

    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _SELECT_PARAM_NUM,
                    &reshape_tensors[0], input_num, &reshape_tensors[3], output_num );
            node_params[SCALAR_INPUT0_SCALE] = vsi_nn_kernel_scalar_create( graph, F32, &input0Scale );
            node_params[SCALAR_INPUT0_TAIL] = vsi_nn_kernel_scalar_create(graph, F32, &input0Tail );
            node_params[SCALAR_INPUT1_SCALE] = vsi_nn_kernel_scalar_create( graph, F32, &input1Scale );
            node_params[SCALAR_INPUT1_TAIL] = vsi_nn_kernel_scalar_create(graph, F32, &input1Tail );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _SELECT_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );

            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT0_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT0_TAIL] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT1_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT1_TAIL] );
        }
    }

    if (ret)
    {
        for (i = 0; i < _IO_NUM; i++)
        {
            vsi_safe_release_tensor( reshape_tensors[i] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( select, _setup )
