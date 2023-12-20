/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_dtype_util.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
typedef enum
{
    AXIS = 0,
    AXIS_NUM,
    ZP,
    SCALE,
    WIDTH,
    HEIGHT,
    CHN,
    DIMRATIO
} vsi_nn_kernel_param_id_e;

#define KERNEL_SOURCE_1    "moments_axis0"
#define KERNEL_SOURCE_2    "moments_axis1"
#define KERNEL_SOURCE_3    "moments_axis2"
#define KERNEL_SOURCE_4    "moments_axis01"
#define KERNEL_SOURCE_5    "moments_axis012"

// Add kernel hashtable here
#define HASH_MOMENTS_KEY(_input0_type, _output_type, _axis_num, _axis0, _axis1, _axis2, _image_2d) \
    ((_input0_type<<24) | (_output_type<<20) | (_axis_num<<16) | (_axis0<<12) | (_axis1<<8) | (_axis2<<4)|(_image_2d))

#define HASH_MOMENTS_SH_KERNEL_NAME(AXIS0, SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.moments_axis"#AXIS0"_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_MOMENTS_TWO_AXIS_SH_KERNEL_NAME(AXIS0, AXIS1, SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.moments_axis"#AXIS0#AXIS1"_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_MOMENTS_THREE_AXIS_SH_KERNEL_NAME(AXIS0, AXIS1, AXIS2, SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("cl.moments_axis"#AXIS0#AXIS1#AXIS2"_"#SRC0_TYPE"to"#DST_TYPE)

#define TENSOR_MOMENTS_KERNELS(IN0_TYPE, OUT_TYPE, AXIS0, SOURCE) \
    { HASH_MOMENTS_KEY(IN0_TYPE, OUT_TYPE, 1, AXIS0, 0, 0, 0), \
        HASH_MOMENTS_SH_KERNEL_NAME(AXIS0, IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MOMENTS_TWO_AXIS_KERNELS(IN0_TYPE, OUT_TYPE, AXIS0, AXIS1, SOURCE) \
    { HASH_MOMENTS_KEY(IN0_TYPE, OUT_TYPE, 2, AXIS0, AXIS1, 0, 0), \
        HASH_MOMENTS_TWO_AXIS_SH_KERNEL_NAME(AXIS0, AXIS1, IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MOMENTS_THREE_AXIS_KERNELS(IN0_TYPE, OUT_TYPE, AXIS0, AXIS1, AXIS2, SOURCE) \
    { HASH_MOMENTS_KEY(IN0_TYPE, OUT_TYPE, 3, AXIS0, AXIS1, AXIS2, 0), \
        HASH_MOMENTS_THREE_AXIS_SH_KERNEL_NAME(AXIS0, AXIS1, AXIS2, IN0_TYPE, OUT_TYPE), \
        SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type moments_map[] =
{
    // Register kernel here
    TENSOR_MOMENTS_KERNELS(U8,  F32, 0,        KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(F32, F32, 0,        KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(I32, F32, 0,        KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(BF16,F32, 0,        KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(U8,  F32, 1,        KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(F32, F32, 1,        KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(I32, F32, 1,        KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(BF16,F32, 1,        KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(U8,  F32, 2,        KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(F32, F32, 2,        KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(I32, F32, 2,        KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(BF16,F32, 2,        KERNEL_SOURCE_3)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(U8,  F32, 0, 1,         KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(F32, F32, 0, 1,         KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(I32, F32, 0, 1,         KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(BF16,F32, 0, 1,         KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(U8,  F32, 1, 2,         KERNEL_SOURCE_4)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(U8,  F32, 0, 1, 2,    KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(F32, F32, 0, 1, 2,    KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(I32, F32, 0, 1, 2,    KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(BF16,F32, 0, 1, 2,    KERNEL_SOURCE_5)
};

/*
 * Kernel params
 */
static vx_param_description_t _moments_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _MOMENTS_PARAM_NUM  _cnt_of_array( _moments_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_moments_initializer)
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

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_size_array_t * input_shape = NULL;
    vsi_ssize_t width = 0;
    vsi_ssize_t height = 0;
    vsi_ssize_t chn = 0;
    int32_t axis = 0;
    int32_t axis_num = 1;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &axis_num);
    CHECK_STATUS_FAIL_GOTO(status, final );

    input_shape  = attr[0]->shape;
    width = input_shape->data[0];
    height = input_shape->data[1];
    chn = input_shape->size > 2 ? input_shape->data[2] : 1;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    if (axis_num == 1 && axis == 0)
    {
        gpu_param.global_size[0]   = gpu_align_p2((height + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1]   = chn;
        gpu_param.global_size[2]   = 1;
    }
    else if (axis_num == 1 && axis == 1)
    {
        gpu_param.global_size[0]   = gpu_align_p2((width + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1]   = chn;
        gpu_param.global_size[2]   = 1;
    }
    else if (axis_num == 1 && axis == 2)
    {
        gpu_param.global_size[0]   = gpu_align_p2((width + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1]   = height;
        gpu_param.global_size[2]   = 1;
    }
    else if (axis_num == 2 && axis == 0)
    {
        gpu_param.local_size[0]  = 16;
        gpu_param.local_size[1]  = 1;
        gpu_param.local_size[2]  = 1;
        gpu_param.global_size[0]   = 16;
        gpu_param.global_size[1]   = chn;
        gpu_param.global_size[2]   = 1;
    }
    else if (axis_num == 2 && axis == 1)
    {
        gpu_param.local_size[0]  = 8;
        gpu_param.local_size[1]  = 8;
        gpu_param.local_size[2]  = 1;
        gpu_param.global_size[0] = 8;
        gpu_param.global_size[1] = 8;
        gpu_param.global_size[2] = width;
    }
    else if (axis_num == 3)
    {
        gpu_param.local_size[0]  = 16;
        gpu_param.local_size[1]  = 1;
        gpu_param.local_size[2]  = 1;
        gpu_param.global_size[0]   = 16;
        gpu_param.global_size[1]   = 1;
        gpu_param.global_size[2]   = 1;
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _instancenorm_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    const vsi_nn_kernel_param_t * params,
    int32_t* axis,
    int32_t axis_num,
    int32_t rs_flg
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    size_t i = 0;

    VSI_UNREFERENCED(params);

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (input0_dtype == I8 || input0_dtype == I16)
    {
        input0_dtype = I32;
    }
    else if (input0_dtype == F16)
    {
        input0_dtype = F32;
    }
    output_dtype = output_dtype == F16 ? F32 : output_dtype;
    key = HASH_MOMENTS_KEY( input0_dtype, output_dtype, axis_num, axis[0], axis[1], axis[2], rs_flg );

    for( i = 0; i < _cnt_of_array(moments_map); i ++ )
    {
        if ( moments_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(moments_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  moments_map[i].function_name );
        kernel->info.parameters = _moments_kernel_param_def;
        kernel->info.numParams = _MOMENTS_PARAM_NUM;
        kernel->info.initialize = _moments_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
                moments_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                moments_map[i].source_name );
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
#define INTERNAL_MOMENTS_SCALAR_NUM     (8)
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_MOMENTS_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    size_t axis_num  = 0;
    int32_t* axis = (int32_t *) vsi_nn_kernel_param_get_buffer( params, "axis", &axis_num);
    int32_t first_axis = axis[0];
    uint32_t i = 0;
    vsi_nn_kernel_scalar_t scalar_list[INTERNAL_MOMENTS_SCALAR_NUM] = {NULL};
    uint32_t axis_size = 0;
    uint32_t rank_in = 0;
    uint32_t rank_out = 0;
    vsi_bool ret = FALSE;
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 1, 1, 1, 1 } };
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    int32_t new_axis[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t input_zp = vsi_nn_get_tensor_zero_point(inputs[0]);
    float input_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float dim_ratio = 1;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    ret = vsi_nn_kernel_optimize_reduce_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            axis, (vsi_size_t)axis_num,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], &rank_in, shapes[1], &rank_out,
            new_axis, &axis_size);

    if ( ret == FALSE || axis_size > 3 || (axis_size == 3 && new_axis[0] != 0))
    {
        return NULL;
    }

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
        inputs[0], shapes[0], rank_in );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
        outputs[0], shapes[1], rank_out );
    reshape_tensors[2] = vsi_nn_reshape_tensor( graph,
        outputs[1], shapes[1], rank_out );

    first_axis = new_axis[0];

    for ( i = 0; i < axis_size; i++ )
    {
        dim_ratio = dim_ratio / (float)(shapes[0][new_axis[i]]);
    }

    if ( !vsi_nn_kernel_gpu_check_shape( shapes[0], rank_in) )
    {
        return NULL;
    }

    scalar_list[AXIS]       = vsi_nn_kernel_scalar_create( graph, I32, &first_axis );
    scalar_list[AXIS_NUM]   = vsi_nn_kernel_scalar_create( graph, I32, &axis_size );
    scalar_list[ZP]         = vsi_nn_kernel_scalar_create( graph, I32, &input_zp );
    scalar_list[SCALE]      = vsi_nn_kernel_scalar_create( graph, F32, &input_scale );
    scalar_list[WIDTH]      = vsi_nn_kernel_scalar_create( graph, I32, &shapes[0][0] );
    scalar_list[HEIGHT]     = vsi_nn_kernel_scalar_create( graph, I32, &shapes[0][1] );
    scalar_list[CHN]        = vsi_nn_kernel_scalar_create( graph, I32, &shapes[0][2] );
    scalar_list[DIMRATIO]   = vsi_nn_kernel_scalar_create( graph, F32, &dim_ratio );

    status = _query_kernel( inputs, outputs, kernel, params, new_axis, axis_size, 0 );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 0;
            vx_border_t border;
            /* Pass parameters to node. */
            node_params[index++] = reshape_tensors[0]->t;
            node_params[index++] = reshape_tensors[1]->t;
            node_params[index++] = reshape_tensors[2]->t;

            node_params[index++] = scalar_list[AXIS];
            node_params[index++] = scalar_list[AXIS_NUM];
            node_params[index++] = scalar_list[ZP];
            node_params[index++] = scalar_list[SCALE];
            node_params[index++] = scalar_list[WIDTH];
            node_params[index++] = scalar_list[HEIGHT];
            node_params[index++] = scalar_list[CHN];
            node_params[index++] = scalar_list[DIMRATIO];
            status = vsi_nn_kernel_node_pass_param( node, node_params, _MOMENTS_PARAM_NUM );
            CHECK_STATUS(status);

            // Set default border mode.
            border.mode = VX_BORDER_CONSTANT;
            vsi_nn_Float32ToDtype(0, (uint8_t*)&border.constant_value.U32, &inputs[0]->attr.dtype);
            status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
            CHECK_STATUS(status);
        }
    }

    vsi_safe_release_tensor(reshape_tensors[0]);
    vsi_safe_release_tensor(reshape_tensors[1]);
    vsi_safe_release_tensor(reshape_tensors[2]);

    /* Pass parameters to node. */
    for( i = 0; i < INTERNAL_MOMENTS_SCALAR_NUM; i ++ )
    {
        if (scalar_list[i])
        {
            vsi_nn_kernel_scalar_release( &scalar_list[i] );
        }
    }
#undef INTERNAL_MOMENTS_SCALAR_NUM

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( moments, _setup )
