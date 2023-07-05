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

__BEGIN_DECLS

#define SIGNAL_FRAME_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        ( ( IN_DTYPE << 8 ) | ( OUT_DTYPE ) )
#define SIGNAL_FRAME_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
        { SIGNAL_FRAME_HASH_KEY( IN_DTYPE, OUT_DTYPE ), \
        CVIVANTE_NAMESPACE("evis.signal_frame_"#IN_DTYPE"to"#OUT_DTYPE), \
        "signal_frame" }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _signal_frame_kernel_map[] =
{
    // Register kernel here
    SIGNAL_FRAME_KERNEL_MAP( I16,  I16 ),
    SIGNAL_FRAME_KERNEL_MAP( F16,  F16 ),
    SIGNAL_FRAME_KERNEL_MAP( BF16, BF16 ),
    SIGNAL_FRAME_KERNEL_MAP( U8,   U8 ),
    SIGNAL_FRAME_KERNEL_MAP( I8,   I8 ),
};

/*
 * Kernel params
 */
static vx_param_description_t _signal_frame_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _SIGNAL_FRAME_PARAM_NUM  _cnt_of_array( _signal_frame_kernel_param_def )
#define FRAME_STEP      (2)
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_signal_frame_initializer)
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
    vsi_nn_kernel_tensor_attr_t * attr  = NULL;
    vsi_size_array_t * out_shape          = NULL;

    VSI_UNREFERENCED(param_size);

    attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr, "Create tensor attr buffer fail.", final );
    out_shape = attr->shape;

    gpu_param.global_scale[0] = 16;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    if ( attr->dtype == F16 || attr->dtype == I16 || attr->dtype == U16 || attr->dtype == BF16)
    {
        gpu_param.global_scale[0] = 8;
    }
    gpu_param.global_size[0] = (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0];
    gpu_param.global_size[1] = out_shape->data[1];
    gpu_param.global_size[2] = out_shape->data[2];

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (attr)
    {
        vsi_nn_kernel_tensor_attr_release( &attr );
        attr = NULL;
    }

    return status;
} /* _signal_frame_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    uint32_t key = 0;
    uint32_t i = 0;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = SIGNAL_FRAME_HASH_KEY( in_dtype, out_dtype );

    for ( i = 0; i < _cnt_of_array(_signal_frame_kernel_map); i ++ )
    {
        if ( _signal_frame_kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(_signal_frame_kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _signal_frame_kernel_map[i].function_name );
        kernel->info.parameters  = _signal_frame_kernel_param_def;
        kernel->info.numParams   = _cnt_of_array( _signal_frame_kernel_param_def );
        kernel->info.initialize  = _signal_frame_initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                _signal_frame_kernel_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                _signal_frame_kernel_map[i].source_name );
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
    vsi_nn_kernel_node_param_t node_params[_SIGNAL_FRAME_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t frame_length  = vsi_nn_kernel_param_get_int32( params, "frame_length" );
    int32_t frame_step  = vsi_nn_kernel_param_get_int32( params, "frame_step" );
    int32_t axis = vsi_nn_kernel_param_get_int32( params, "axis" );
    int32_t pad_end  = vsi_nn_kernel_param_get_int32( params, "pad_end" );
    float pad_value  = vsi_nn_kernel_param_get_float32( params, "pad_val" );
    vsi_size_t num_frames = outputs[0]->attr.size[axis + 1];
    int32_t rank = inputs[0]->attr.dim_num;
    vsi_size_t inner = 1;
    vsi_size_t outer = 1;
    vsi_size_t length_samples = inputs[0]->attr.size[axis];
    int32_t i = 0;
    vsi_nn_tensor_t* rs_tensors[2] = { NULL };
    vsi_size_t shape[2][VSI_NN_MAX_DIM_NUM] = {{ 0 }};

    for (i = 0; i < axis; i++)
    {
        inner *= inputs[0]->attr.size[i];
    }

    for (i = axis + 1; i < rank; i++)
    {
        outer *= inputs[0]->attr.size[i];
    }

    shape[0][0] = inner;
    shape[0][1] = length_samples;
    shape[0][2] = 1;
    shape[0][3] = outer;

    shape[1][0] = inner;
    shape[1][1] = frame_length;
    shape[1][2] = num_frames;
    shape[1][3] = outer;

    rs_tensors[0] = vsi_nn_reshape_tensor( graph,
        inputs[0], shape[0], 4 );
    rs_tensors[1] = vsi_nn_reshape_tensor( graph,
        outputs[0], shape[1], 4 );

    if ( !vsi_nn_kernel_gpu_check_shape( rs_tensors[1]->attr.size,
                rs_tensors[1]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            if ( pad_end )
            {
                // Set default border mode.
                vx_border_t border;
                uint32_t data = 0;
                uint32_t dsize = 1;

                vsi_nn_Float32ToDtype(pad_value, (uint8_t*)&data, &outputs[0]->attr.dtype);
                border.mode = VX_BORDER_CONSTANT;
                dsize = vsi_nn_GetTypeBytes( inputs[0]->attr.dtype.vx_type );
                if ( dsize == 1 )
                {
                    border.constant_value.U8 = (uint8_t)data;
                }
                else if ( dsize == 4 )
                {
                    border.constant_value.U32 = data;
                }
                else
                {
                    border.constant_value.U16 = (uint16_t)data;
                }

                status |= vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
            }
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _SIGNAL_FRAME_PARAM_NUM,
                    &rs_tensors[0], input_num, &rs_tensors[1], output_num );
            node_params[FRAME_STEP] = vsi_nn_kernel_scalar_create(
                    graph, I32, &frame_step );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _SIGNAL_FRAME_PARAM_NUM );
            CHECK_STATUS_FAIL_GOTO( status, final );
        }
    }
final:
    if (rs_tensors[0])
    {
        vsi_nn_ReleaseTensor( &rs_tensors[0] );
    }

    if (rs_tensors[1])
    {
        vsi_nn_ReleaseTensor( &rs_tensors[1] );
    }

    if (node_params[FRAME_STEP])
    {
        vsi_nn_kernel_scalar_release( &node_params[FRAME_STEP] );
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( signal_frame, _setup )
