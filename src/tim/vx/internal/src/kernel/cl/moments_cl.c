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
#include "kernel/vsi_nn_kernel_eltwise.h"

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
    TENSOR_MOMENTS_KERNELS(U8,  F16, 0,        KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(F16, F16, 0,        KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(F32, F32, 0,        KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(I32, F32, 0,        KERNEL_SOURCE_1)
    TENSOR_MOMENTS_KERNELS(U8,  F16, 1,        KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(F16, F16, 1,        KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(F32, F32, 1,        KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(I32, F32, 1,        KERNEL_SOURCE_2)
    TENSOR_MOMENTS_KERNELS(U8,  F16, 2,        KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(F16, F16, 2,        KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(F32, F32, 2,        KERNEL_SOURCE_3)
    TENSOR_MOMENTS_KERNELS(I32, F32, 2,        KERNEL_SOURCE_3)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(U8,  F16, 0, 1,         KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(F16, F16, 0, 1,         KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(F32, F32, 0, 1,         KERNEL_SOURCE_4)
    TENSOR_MOMENTS_TWO_AXIS_KERNELS(I32, F32, 0, 1,         KERNEL_SOURCE_4)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(U8,  F16, 0, 1, 2,    KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(F16, F16, 0, 1, 2,    KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(F32, F32, 0, 1, 2,    KERNEL_SOURCE_5)
    TENSOR_MOMENTS_THREE_AXIS_KERNELS(I32, F32, 0, 1, 2,    KERNEL_SOURCE_5)
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

static int32_t set_constant_border
    (
    vsi_nn_kernel_node_t node,
    int32_t value
    )
{
    vsi_status status = VSI_FAILURE;
    vx_border_t border;
    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.S32 = value;
    border.constant_value.U32 = (vx_uint32)value;
    border.constant_value.S16 = (vx_int16)value;
    border.constant_value.U8 = (vx_uint8)value;
    status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
    return status;
}

static int32_t get_moments_output_reshape_size
    (
    vsi_nn_tensor_t ** outputs,
    int32_t sizes[VSI_NN_MAX_DIM_NUM],
    int32_t* axis,
    int32_t axis_num
    )
{
    uint32_t out_dims_num = outputs[0]->attr.dim_num;
    uint32_t *output_size = outputs[0]->attr.size;
    uint32_t i = 0;
    int32_t out_rs_flg = 0;

    for(i = 0; i < VSI_NN_MAX_DIM_NUM; ++i)
    {
        sizes[i] = 1;
    }
    sizes[3] = out_dims_num > 3 ? output_size[3] : 1;

    if(axis_num == 1 && axis[0] == 0)
    {
        sizes[0] = output_size[1];
        sizes[1] = out_dims_num > 2 ? output_size[2] : 1;
        out_rs_flg = 1;
    }
    else if(axis_num == 1 && axis[0] == 1)
    {
        sizes[0] = output_size[0];
        sizes[1] = out_dims_num > 2 ? output_size[2] : 1;
        out_rs_flg = 1;
    }
    else if(axis_num == 2 && axis[0] == 0 && axis[1] == 1)
    {
        sizes[0] = out_dims_num > 2 ? output_size[2] : 1;
        out_rs_flg = 1;
    }

    return out_rs_flg;
} /* _get_moments_tensor_reshape_size */

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
    vsi_int_array_t * input_shape = NULL;
    int32_t width = 0;
    int32_t height = 0;
    int32_t chn = 0;
    int32_t axis = 0;
    int32_t axis_num = 1;

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
    if(axis_num == 1 && axis == 0)
    {
        gpu_param.global_size[0]   = gpu_align_p2((height + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1]   = chn;
    }
    else if(axis_num == 1 && axis == 1)
    {
        gpu_param.global_size[0]   = gpu_align_p2((width + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1]   = chn;
    }
    else if(axis_num == 1 && axis == 2)
    {
        gpu_param.global_size[0]   = gpu_align_p2((width + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
        gpu_param.global_size[1]   = height;
    }
    else if(axis_num == 2)
    {
        gpu_param.local_size[0]  = 16;
        gpu_param.local_size[1]  = 1;
        gpu_param.local_size[2]  = 1;
        gpu_param.global_size[0]   = 16;
        gpu_param.global_size[1]   = chn;
    }
    else if(axis_num == 3)
    {
        gpu_param.local_size[0]  = 16;
        gpu_param.local_size[1]  = 1;
        gpu_param.local_size[2]  = 1;
        gpu_param.global_size[0]   = 16;
        gpu_param.global_size[1]   = 1;
    }
    gpu_param.global_size[2]   = 1;

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
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_MOMENTS_KEY( input0_dtype, output_dtype, axis_num, axis[0], axis[1], axis[2], rs_flg );

    for( i = 0; i < _cnt_of_array(moments_map); i ++ )
    {
        if( moments_map[i].key == key )
        {
            break;
        }
    }

    if( i < _cnt_of_array(moments_map) )
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
    int32_t  out_shape[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t  out_rs_flg = 0;
    int32_t axis_num  = 0;
    size_t axis_num_temp = 0;
    int32_t* axis = (int32_t *) vsi_nn_kernel_param_get_buffer( params, "axis", &axis_num_temp);
    int32_t keep_dim  = vsi_nn_kernel_param_get_int32( params, "keep_dim" );
    int32_t first_axis = axis[0];
    int32_t i = 0;
    vsi_nn_kernel_scalar_t scalar_list[INTERNAL_MOMENTS_SCALAR_NUM] = {NULL};

    int32_t width = inputs[0]->attr.size[0];
    int32_t height = inputs[0]->attr.size[1];
    int32_t chn = inputs[0]->attr.size[2];
    int32_t input_zp = inputs[0]->attr.dtype.zero_point;
    float input_scale = inputs[0]->attr.dtype.scale;
    float dim_ratio = (float)1.0 / (float)(width * height);

    axis_num = (int32_t)axis_num_temp;

    if(inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        if (inputs[0]->attr.dtype.fl > 0)
        {
            input_scale = (1.0f / ((float) ((int64_t)1 << inputs[0]->attr.dtype.fl)));
        }
        else
        {
            input_scale = ((float) ((int64_t)1 << -inputs[0]->attr.dtype.fl));
        }
        input_zp = 0;
    }

    if(axis_num == 1 && axis[0] == 0)
    {
        dim_ratio = (float)1.0 / (float)(width);
    }
    else if(axis_num == 1 && axis[0] == 1)
    {
        dim_ratio = (float)1.0 / (float)(height);
    }
    else if(axis_num == 1 && axis[0] == 2)
    {
        dim_ratio = (float)1.0 / (float)(chn);
    }
    else if(axis_num == 2 && axis[0] == 0 && axis[1] == 1)
    {
        dim_ratio = (float)1.0 / (float)(width * height);
    }
    else if(axis_num == 3)
    {
        dim_ratio = (float)1.0 / (float)(width * height * chn);
    }

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    if(keep_dim)
    {
        out_rs_flg = get_moments_output_reshape_size(&outputs[0], out_shape, axis, axis_num);
    }

    scalar_list[AXIS]       = vsi_nn_kernel_scalar_create( graph, I32, &first_axis );
    scalar_list[AXIS_NUM]   = vsi_nn_kernel_scalar_create( graph, I32, &axis_num );
    scalar_list[ZP]         = vsi_nn_kernel_scalar_create( graph, I32, &input_zp );
    scalar_list[SCALE]      = vsi_nn_kernel_scalar_create( graph, F32, &input_scale );
    scalar_list[WIDTH]      = vsi_nn_kernel_scalar_create( graph, I32, &width );
    scalar_list[HEIGHT]     = vsi_nn_kernel_scalar_create( graph, I32, &height );
    scalar_list[CHN]        = vsi_nn_kernel_scalar_create( graph, I32, &chn );
    scalar_list[DIMRATIO]   = vsi_nn_kernel_scalar_create( graph, F32, &dim_ratio );

    status = _query_kernel( inputs, outputs, kernel, params, axis, axis_num, 0 );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            uint32_t index = 0;
            /* Pass parameters to node. */
            node_params[index++] = (vsi_nn_kernel_node_param_t)(inputs[0]->t);
            if(out_rs_flg)
            {
                node_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, out_shape, 4 );
                node_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[1]->t, out_shape, 4 );
            }
            else
            {
                node_params[index++] = (vsi_nn_kernel_node_param_t)(outputs[0]->t);
                node_params[index++] = (vsi_nn_kernel_node_param_t)(outputs[1]->t);
            }
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
            if(out_rs_flg)
            {
                vsi_nn_kernel_tensor_release( &node_params[1] );
                vsi_nn_kernel_tensor_release( &node_params[2] );
            }
            status = set_constant_border(node, inputs[0]->attr.dtype.zero_point);
            CHECK_STATUS(status);
        }
    }

    /* Pass parameters to node. */
    for( i = 0; i < INTERNAL_MOMENTS_SCALAR_NUM; i ++ )
    {
        if(scalar_list[i])
        {
            vsi_nn_kernel_scalar_release( &scalar_list[i] );
        }
    }
#undef INTERNAL_MOMENTS_SCALAR_NUM

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( moments, _setup )

