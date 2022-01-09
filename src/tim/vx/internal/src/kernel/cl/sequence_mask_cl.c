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
#include "math.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "sequence_mask"

#define HASH_SEQUENCE_MASK_KEY(_input0_type, _output_type, _image_2d) \
    ((_input0_type << 24) | (_output_type << 8) | (_image_2d))

#define HASH_SEQUENCE_MASK_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.sequence_mask_"#SRC0_TYPE"to"#DST_TYPE)

 #define HASH_SEQUENCE_MASK_SH_2DKERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.sequence_mask_"#SRC0_TYPE"to"#DST_TYPE"_2D")

#define TENSOR_SEQUENCE_MASK_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SEQUENCE_MASK_KEY(IN0_TYPE, OUT_TYPE, 0), \
        HASH_SEQUENCE_MASK_SH_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

 #define TENSOR_SEQUENCE_MASK_2D_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SEQUENCE_MASK_KEY(IN0_TYPE, OUT_TYPE, 1), \
        HASH_SEQUENCE_MASK_SH_2DKERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } kernel_map[] =
{
    TENSOR_SEQUENCE_MASK_KERNELS(I32, U8,     KERNEL_SOURCE_1)
    TENSOR_SEQUENCE_MASK_KERNELS(I32, I32,    KERNEL_SOURCE_1)
    TENSOR_SEQUENCE_MASK_KERNELS(I32, F32,    KERNEL_SOURCE_1)
    TENSOR_SEQUENCE_MASK_2D_KERNELS(I32, U8,  KERNEL_SOURCE_1)
    TENSOR_SEQUENCE_MASK_2D_KERNELS(I32, I32, KERNEL_SOURCE_1)
    TENSOR_SEQUENCE_MASK_2D_KERNELS(I32, F32, KERNEL_SOURCE_1)
};

/*
 * Kernel params
 */
static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _CL_PARAM_NUM          _cnt_of_array(kernel_param_def)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_sequence_mask_initializer)
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

    vsi_status    status             = VSI_FAILURE;
    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_size_array_t * out_shape = NULL;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );

    out_shape  = attr[0]->shape;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((out_shape->data[0] + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = (out_shape->data[1] + gpu_param.global_scale[1] - 1)
                                        / gpu_param.global_scale[1];
    gpu_param.global_size[2]   = out_shape->size > 2 ? out_shape->data[2] : 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }

    return status;
} /* _sequence_mask_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    int32_t is2Dflg
    )
{
    vsi_nn_kernel_dtype_e input0_dtype = I32;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    vsi_status status = VSI_FAILURE;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );
    if (output_dtype == BOOL8)
    {
        output_dtype= U8;
    }

    key = HASH_SEQUENCE_MASK_KEY( input0_dtype, output_dtype, is2Dflg );

    for( i = 0; i < _cnt_of_array(kernel_map); i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(kernel_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters = kernel_param_def;
        kernel->info.numParams = _cnt_of_array( kernel_param_def );
        kernel->info.initialize = _sequence_mask_initializer;
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                kernel_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */

static int32_t _optimize_mask_shape
    (
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    int32_t max_len,
    vsi_size_t* opt_shape_in,
    vsi_size_t* opt_shape_out,
    int32_t* is2Dflg
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_size_t in_shape[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t new_rank = 0;
    uint32_t i = 0;

    for(i = 0; i < inputs[0]->attr.dim_num; i++)
    {
        in_shape[i] = inputs[0]->attr.size[i];
    }

    vsi_nn_kernel_optimize_element_shape( in_shape, inputs[0]->attr.dim_num, opt_shape_in, &new_rank );
    if (new_rank > 2)
    {
        return VSI_FAILURE;
    }

    opt_shape_out[0] = max_len;
    for(i = 0; i < (uint32_t)new_rank; i++)
    {
        opt_shape_out[i + 1] = opt_shape_in[i];
    }
    if (opt_shape_out[2] == 1)
    {
        is2Dflg[0] = 1;
    }

    return status;
}

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
    vsi_nn_kernel_node_param_t node_params[_CL_PARAM_NUM] = {NULL};
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL;
    vsi_size_t new_shape[2][VSI_NN_MAX_DIM_NUM] = {{ 1, 1, 1, 1 }, { 1, 1, 1, 1 }};
    int32_t max_len  = vsi_nn_kernel_param_get_int32( params, "max_len" );
    vsi_nn_kernel_node_t node = NULL;
    int32_t is2Dflg = 0;
    float input_zp = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float input_scale = vsi_nn_get_tensor_scale(inputs[0]);
    int32_t output_zp = vsi_nn_get_tensor_zero_point(outputs[0]);
    float output_scale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);
    float input_zpScale = 0;
    float outputVal1 = 1.0f;

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _optimize_mask_shape(inputs, outputs, max_len, new_shape[0], new_shape[1], &is2Dflg);
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }
    rs_input = vsi_nn_kernel_tensor_reshape(inputs[0]->t, new_shape[0], 2);
    rs_output = vsi_nn_kernel_tensor_reshape(outputs[0]->t, new_shape[1], 4);

    input_zpScale = input_scale * input_zp;
    outputVal1 = output_scale + (float)output_zp;

    status = _query_kernel( inputs, outputs, kernel, is2Dflg );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );

        if ( node )
        {
            uint32_t index = 0;
            node_params[index++] = rs_input;
            node_params[index++] = rs_output;
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &max_len );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &input_scale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &input_zpScale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &outputVal1 );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &output_zp );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CL_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
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

REGISTER_BACKEND_CL( sequence_mask, _setup )
