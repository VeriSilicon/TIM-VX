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
typedef enum
{
    INTERNAL_KERNEL_ROPE,
} _internal_kernel_e;

#define _ROPE_KERNEL_SOURCE      "rope"
#define _ROPE_KERNEL_NAME        CVIVANTE_NAMESPACE("cl.rope")

// Add kernel hashtable here
#define STR(a) #a
#define ROPE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, AXIS ) \
        ((IN0_DTYPE) | (IN0_DTYPE << 8) | (OUT_DTYPE << 16) | (AXIS << 25))
#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, AXIS ) \
        { ROPE_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE, AXIS ), \
          CVIVANTE_NAMESPACE("cl.rope_"STR(IN0_DTYPE)"_"STR(IN1_DTYPE)"to"STR(OUT_DTYPE)"_axis"STR(AXIS)), \
         "rope_0" }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _rope_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32, F32, F32, 0 ),
    PACK_KERNEL_MAP( F32, F32, F32, 1 ),
    PACK_KERNEL_MAP( F32, F32, F32, 2 ),
    PACK_KERNEL_MAP( I32, I32, I32, 0 ),
    PACK_KERNEL_MAP( I32, I32, I32, 1 ),
    PACK_KERNEL_MAP( I32, I32, I32, 2 ),
    PACK_KERNEL_MAP( U32, U32, U32, 0 ),
    PACK_KERNEL_MAP( U32, U32, U32, 1 ),
    PACK_KERNEL_MAP( U32, U32, U32, 2 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _rope_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
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
#define _ROPE_PARAM_NUM  _cnt_of_array( _rope_kernel_param_def )
#define SCALAR_AXIS                 (4)
#define SCALAR_IN_ZP                (5)
#define SCALAR_COS_ZP               (6)
#define SCALAR_SIN_ZP               (7)
#define SCALAR_SCALE0               (8)
#define SCALAR_SCALE1               (9)
#define SCALAR_OUT_ZP               (10)
#define SCALAR_HALF_HEAD_SIZE       (11)
#define SCALAR_STEP                 (12)
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_rope_initializer)
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
    vsi_nn_kernel_tensor_attr_t* attr[2] = { NULL };
    int32_t axis = 0;
    vsi_size_array_t* out_shape = NULL;
    vsi_size_t shape[3] = { 1 };

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final);

    out_shape = attr[1]->shape;
    shape[0] = out_shape->data[0];
    shape[1] = out_shape->data[1];
    shape[2] = out_shape->data[2];
    shape[axis] = shape[axis] / 2;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = shape[0];
    gpu_param.global_size[1] = shape[1];
    gpu_param.global_size[2] = out_shape->size > 2 ? shape[2] : 1;

    status = vsi_nn_kernel_gpu_config(node, &gpu_param);

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if ( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(attr[0]);
    SAFE_FREE_TENSOR_ATTR(attr[1]);
#undef SAFE_FREE_TENSOR_ATTR

    return status;
} /* _rope_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t axis
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e in2_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _rope_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _rope_kernel_map );
    vx_param_description_t * param_def  = _rope_kernel_param_def;
    vx_kernel_initialize_f  initializer = _rope_initializer;

    uint32_t key = 0;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype(inputs[1]->attr.dtype.vx_type);
    in2_dtype = vsi_nn_kernel_map_dtype(inputs[2]->attr.dtype.vx_type);
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

#define _PACK_SELECT_KEY( in0_type, in1_type, in2_type, out_type ) \
    ((in0_type) | (in1_type << 8) | (in2_type << 16) | (out_type << 24))
    switch (_PACK_SELECT_KEY(in0_dtype, in1_dtype, in2_dtype, out_dtype))
    {
    case _PACK_SELECT_KEY(F32, F32, F32, F32):
    case _PACK_SELECT_KEY(F16, F16, F16, F16):
        key = ROPE_HASH_KEY(F32, F32, F32, axis);
        break;
    case _PACK_SELECT_KEY(U8,  U8,  U8,  U8):
    case _PACK_SELECT_KEY(U16, U16, U16, U16):
        key = ROPE_HASH_KEY(U32, U32, U32, axis);
        break;
    case _PACK_SELECT_KEY(I8,  I8,  I8,  I8):
    case _PACK_SELECT_KEY(I16, I16, I16, I16):
    case _PACK_SELECT_KEY(I32, I32, I32, I32):
        key = ROPE_HASH_KEY(I32, I32, I32, axis);
        break;
    default:
        break;
    }
#undef _PACK_SELECT_KEY

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
        kernel->info.numParams   = _cnt_of_array( _rope_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_ROPE_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t axis = vsi_nn_kernel_param_get_int32(params, "axis");
    int32_t interleaved = vsi_nn_kernel_param_get_int32(params, "interleaved");
    float in_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float cos_scale = vsi_nn_get_tensor_scale(inputs[1]);
    float sin_scale = vsi_nn_get_tensor_scale(inputs[2]);
    float out_scale = vsi_nn_get_tensor_scale(outputs[0]);
    float in_zp = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float cos_zp = (float)vsi_nn_get_tensor_zero_point(inputs[1]);
    float sin_zp = (float)vsi_nn_get_tensor_zero_point(inputs[2]);
    float output_zp = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    int32_t half_head_size = interleaved ? 1 : (int32_t)(inputs[0]->attr.size[axis] / 2);
    float scale0 = in_scale * cos_scale / out_scale;
    float scale1 = in_scale * sin_scale / out_scale;
    int32_t step = interleaved ? 2 : 1;
    int32_t i = 0;

    // Check if gpu can support the size
    if ( !vsi_nn_kernel_gpu_check_shape(
        inputs[0]->attr.size, inputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs, axis );
    if (VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _ROPE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            /* Pass parameters to node. */
            node_params[SCALAR_AXIS] = vsi_nn_kernel_scalar_create(
                graph, I32, &axis);
            node_params[SCALAR_IN_ZP] = vsi_nn_kernel_scalar_create(
                graph, F32, &in_zp);
            node_params[SCALAR_COS_ZP] = vsi_nn_kernel_scalar_create(
                graph, F32, &cos_zp);
            node_params[SCALAR_SIN_ZP] = vsi_nn_kernel_scalar_create(
                graph, F32, &sin_zp);
            node_params[SCALAR_SCALE0] = vsi_nn_kernel_scalar_create(
                graph, F32, &scale0);
            node_params[SCALAR_SCALE1] = vsi_nn_kernel_scalar_create(
                graph, F32, &scale1);
            node_params[SCALAR_OUT_ZP] = vsi_nn_kernel_scalar_create(
                graph, F32, &output_zp);
            node_params[SCALAR_HALF_HEAD_SIZE] = vsi_nn_kernel_scalar_create(
                graph, I32, &half_head_size);
            node_params[SCALAR_STEP] = vsi_nn_kernel_scalar_create(
                graph, I32, &step);
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _ROPE_PARAM_NUM );
        }
    }

    for (i = SCALAR_AXIS; i < (int32_t)_ROPE_PARAM_NUM; i++)
    {
        if (node_params[i])
        {
            vsi_nn_kernel_scalar_release(&node_params[i]);
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( rope, _setup )

