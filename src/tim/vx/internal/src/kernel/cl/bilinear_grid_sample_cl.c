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
    INTERNAL_KERNEL_BILINEAR_GRID_SAMPLE,
} _internal_kernel_e;

#define _BILINEAR_GRID_SAMPLE_KERNEL_SOURCE()      "bilinear_grid_sample"

#define STR(a) #a

// Add kernel hashtable here
#define BILINEAR_GRID_SAMPLE_HASH_KEY(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE) \
    ((IN1_DTYPE << 20) | (IN0_DTYPE << 8) | (OUT_DTYPE))

#define PACK_KERNEL_MAP(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE)                \
    {                                                                   \
        BILINEAR_GRID_SAMPLE_HASH_KEY(IN0_DTYPE, IN1_DTYPE, OUT_DTYPE), \
            CVIVANTE_NAMESPACE("cl.bilinear_grid_sample_" STR(IN0_DTYPE) "_" STR(IN1_DTYPE) "to" STR(OUT_DTYPE)), \
            _BILINEAR_GRID_SAMPLE_KERNEL_SOURCE()   \
    }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _bilinear_grid_sample_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP(F32, F32, F32 ),
    PACK_KERNEL_MAP(U8,  U8,  U8),
};


/*
 * Kernel params
 */
static vx_param_description_t _bilinear_grid_sample_kernel_param_def[] =
{
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
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _BILINEAR_GRID_SAMPLE_PARAM_NUM 8
#define _BILINEAR_GRID_SAMPLE_PARAM_QUANT_NUM \
    _cnt_of_array(_bilinear_grid_sample_kernel_param_def)

#define SCALAR_HALF_INPUT0_W (3)
#define SCALAR_HALF_INPUT0_H (4)
#define SCALAR_ADD_VALUE_W   (5)
#define SCALAR_ADD_VALUE_H   (6)
#define SCALAR_DEPTH         (7)
#define SCALAR_INPUT0_SCALE  (8)
#define SCALAR_INPUT0_TAIL   (9)
#define SCALAR_INPUT1_SCALE  (10)
#define SCALAR_INPUT1_TAIL   (11)
#define SCALAR_OUTPUT_SCALE  (12)
#define SCALAR_OUTPUT_TAIL   (13)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_bilinear_grid_sample_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {3, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    vsi_nn_kernel_tensor_attr_t* output_attr = NULL;
    vsi_size_array_t* out_shape = NULL;

    output_attr =
        vsi_nn_kernel_tensor_attr_create((vsi_nn_kernel_tensor_t)param[2]);
    CHECK_PTR_FAIL_GOTO(output_attr, "Create tensor attr buffer fail.", final);

    out_shape = output_attr->shape;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;

    gpu_param.dim = 2;
    gpu_param.global_size[0] =
        gpu_align_p2((out_shape->data[0] + gpu_param.global_scale[0] - 1) /
                         gpu_param.global_scale[0],
                     4);
    gpu_param.global_size[1] =
        ((out_shape->data[1] + gpu_param.global_scale[1] - 1) /
         gpu_param.global_scale[1]);
    gpu_param.global_size[2] = 1;
    status = vsi_nn_kernel_gpu_config(node, &gpu_param);

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR)               \
    if (_PTR) {                                   \
        vsi_nn_kernel_tensor_attr_release(&_PTR); \
        _PTR = NULL;                              \
    }
    SAFE_FREE_TENSOR_ATTR(output_attr);
    return status;
} /* _bilinear_grid_sample_initializer() */



/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    vsi_bool* is_use_u8_kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype, in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _bilinear_grid_sample_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _bilinear_grid_sample_kernel_map );
    vx_param_description_t * param_def  = _bilinear_grid_sample_kernel_param_def;
    size_t param_def_size               = _cnt_of_array(_bilinear_grid_sample_kernel_param_def);
    vx_kernel_initialize_f  initializer = _bilinear_grid_sample_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == in0_dtype) {
        in0_dtype = F32;
    }
    if (F16 == in1_dtype) {
        in1_dtype = F32;
    }
    if (F16 == out_dtype) {
        out_dtype = F32;
    }
    if ((U8 == in0_dtype) || (U8 == out_dtype)) {
        param_def_size = _BILINEAR_GRID_SAMPLE_PARAM_QUANT_NUM;
        *is_use_u8_kernel = TRUE;
    } else {
        param_def_size = _BILINEAR_GRID_SAMPLE_PARAM_NUM;
        *is_use_u8_kernel = FALSE;
    }

    key = BILINEAR_GRID_SAMPLE_HASH_KEY(in0_dtype, in1_dtype, out_dtype);

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
    vsi_nn_kernel_node_t node = NULL;
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_BILINEAR_GRID_SAMPLE_PARAM_QUANT_NUM];
    vsi_size_t final_shape[VSI_NN_MAX_DIM_NUM] = {1, 1, 1, 1};
    uint32_t final_in1_rank = 0;
    vsi_nn_tensor_t* rs_tensors = NULL;
    vsi_nn_tensor_t* final_tensors[3] = {NULL};
    vsi_size_t in0_width  = inputs[0]->attr.size[0];
    vsi_size_t in0_height = inputs[0]->attr.size[1];
    float input0_zp    = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float input0_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float input0_tail  = -(input0_zp * input0_scale);
    float input1_zp    = (float)vsi_nn_get_tensor_zero_point(inputs[1]);
    float input1_scale = vsi_nn_get_tensor_scale(inputs[1]);
    float input1_tail  = -(input1_zp * input1_scale);
    float output_zp    = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float output_scale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);
    vsi_bool is_use_u8_kernel = FALSE;
    int32_t align_corners =
        vsi_nn_kernel_param_get_int32(params, "align_corners");
    uint32_t pad_val = 0;
    int32_t  depth = 0;
    vsi_nn_kernel_dtype_e in0_dtype;

    float half_input0_w, half_input0_h, add_float_value_w, add_float_value_h;

    // Check if gpu can support the size
    if (!vsi_nn_kernel_gpu_check_shape(inputs[0]->attr.size,
                                       inputs[0]->attr.dim_num)) {
        return NULL;
    }

    if (!vsi_nn_kernel_gpu_check_shape(inputs[1]->attr.size,
                                       inputs[1]->attr.dim_num)) {
        return NULL;
    }

    final_tensors[0] = inputs[0];

    if (inputs[1]->attr.dim_num >= 3) {

        final_shape[0] = inputs[1]->attr.size[1] * inputs[1]->attr.size[0];
        final_shape[1] = inputs[1]->attr.size[2];
        final_shape[2] = 1;
        final_shape[3] = inputs[1]->attr.dim_num > 3 ? inputs[1]->attr.size[3] : 1;
        final_in1_rank =
            inputs[1]->attr.dim_num == 3 ? 2 : inputs[1]->attr.dim_num;
        if (!vsi_nn_kernel_gpu_check_shape(final_shape, final_in1_rank)) {
            return NULL;
        }

        rs_tensors = vsi_nn_reshape_tensor(graph, inputs[1], final_shape, final_in1_rank);
        final_tensors[1] = rs_tensors;
    } else {
        final_tensors[1] = inputs[1];
    }
    final_tensors[2] = outputs[0];

    if (align_corners) {
        half_input0_w     = ((float)in0_width - 1.0f) * 0.5f;
        half_input0_h     = ((float)in0_height - 1.0f) * 0.5f;
        add_float_value_w = half_input0_w;
        add_float_value_h = half_input0_h;
    } else {
        half_input0_w     = (float)in0_width * 0.5f;
        half_input0_h     = (float)in0_height * 0.5f;
        add_float_value_w = half_input0_w - 0.5f;
        add_float_value_h = half_input0_h - 0.5f;
    }

    depth = (int32_t)inputs[0]->attr.size[2];
    in0_dtype = vsi_nn_kernel_map_dtype(inputs[0]->attr.dtype.vx_type);
    if (U8 == in0_dtype) {
        pad_val = inputs[0]->attr.dtype.zero_point;
    }
    status = _query_kernel(kernel, inputs, outputs, &is_use_u8_kernel);
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            size_t node_params_num = _BILINEAR_GRID_SAMPLE_PARAM_NUM;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _BILINEAR_GRID_SAMPLE_PARAM_QUANT_NUM,
                    final_tensors, input_num, &final_tensors[2], output_num );
            node_params[SCALAR_HALF_INPUT0_W] = vsi_nn_kernel_scalar_create( graph, F32, &half_input0_w );
            node_params[SCALAR_HALF_INPUT0_H] = vsi_nn_kernel_scalar_create( graph, F32, &half_input0_h );
            node_params[SCALAR_ADD_VALUE_W]   = vsi_nn_kernel_scalar_create( graph, F32, &add_float_value_w );
            node_params[SCALAR_ADD_VALUE_H]   = vsi_nn_kernel_scalar_create( graph, F32, &add_float_value_h );
            node_params[SCALAR_DEPTH]         = vsi_nn_kernel_scalar_create( graph, I32, &depth );
            if (is_use_u8_kernel)
            {
                node_params[SCALAR_INPUT0_SCALE] = vsi_nn_kernel_scalar_create( graph, F32, &input0_scale );
                node_params[SCALAR_INPUT0_TAIL]  = vsi_nn_kernel_scalar_create( graph, F32, &input0_tail );
                node_params[SCALAR_INPUT1_SCALE] = vsi_nn_kernel_scalar_create( graph, F32, &input1_scale );
                node_params[SCALAR_INPUT1_TAIL]  = vsi_nn_kernel_scalar_create( graph, F32, &input1_tail );
                node_params[SCALAR_OUTPUT_SCALE] = vsi_nn_kernel_scalar_create( graph, F32, &output_scale );
                node_params[SCALAR_OUTPUT_TAIL]  = vsi_nn_kernel_scalar_create( graph, F32, &output_zp );
                node_params_num = _BILINEAR_GRID_SAMPLE_PARAM_QUANT_NUM;
            }
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            VSI_ASSERT(status == VSI_SUCCESS);
            vsi_nn_kernel_scalar_release(&node_params[SCALAR_HALF_INPUT0_W]);
            vsi_nn_kernel_scalar_release(&node_params[SCALAR_HALF_INPUT0_H]);
            vsi_nn_kernel_scalar_release(&node_params[SCALAR_ADD_VALUE_W]);
            vsi_nn_kernel_scalar_release(&node_params[SCALAR_ADD_VALUE_H]);
            vsi_nn_kernel_scalar_release(&node_params[SCALAR_DEPTH]);
            if (is_use_u8_kernel) {
                vsi_nn_kernel_scalar_release(&node_params[SCALAR_INPUT0_SCALE]);
                vsi_nn_kernel_scalar_release(&node_params[SCALAR_INPUT0_TAIL]);
                vsi_nn_kernel_scalar_release(&node_params[SCALAR_INPUT1_SCALE]);
                vsi_nn_kernel_scalar_release(&node_params[SCALAR_INPUT1_TAIL]);
                vsi_nn_kernel_scalar_release(&node_params[SCALAR_OUTPUT_SCALE]);
                vsi_nn_kernel_scalar_release(&node_params[SCALAR_OUTPUT_TAIL]);
            }
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                border.constant_value.U32 = pad_val;
                status = vxSetNodeAttribute(
                    (vx_node)node, VX_NODE_BORDER, &border, sizeof(border));
                CHECK_STATUS(status);
            }
        }
    }

    vsi_safe_release_tensor(rs_tensors);

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( bilinear_grid_sample, _setup )

