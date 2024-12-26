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
#include "utils/vsi_nn_dtype_util_prv.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */

#define _CUSTOM_LETTERBOX_KERNEL_SOURCE      "custom_letterbox"

// Add kernel hashtable here
#define CUSTOM_LETTERBOX_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        (( IN_DTYPE ) | ( OUT_DTYPE << 8 ))
#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
        { CUSTOM_LETTERBOX_HASH_KEY( IN_DTYPE, OUT_DTYPE ), \
          CVIVANTE_NAMESPACE("evis.custom_letterbox_"#IN_DTYPE"to"#OUT_DTYPE), \
          _CUSTOM_LETTERBOX_KERNEL_SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _custom_letterbox_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( U8, U8 ),
    PACK_KERNEL_MAP( U8, I8 ),
    PACK_KERNEL_MAP( U8, F16 ),
};

/*
 * Kernel params
 */
static vx_param_description_t _custom_letterbox_kernel_param_def[] =
{
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
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define _CUSTOM_LETTERBOX_PARAM_NUM  _cnt_of_array( _custom_letterbox_kernel_param_def )
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_custom_letterbox_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_nn_kernel_tensor_attr_t* attr[2] = {NULL, NULL};
    VSI_UNREFERENCED(param_size);
    int32_t      top = 0;
    int32_t      bottom = 0;
    int32_t      left = 0;
    int32_t      right = 0;
    float        scale_w = 0;
    float        scale_h = 0;
    int32_t      resize_w = 0;
    int32_t      resize_h = 0;
    int32_t      resize_max_w = 0;
    int32_t      resize_max_h = 0;
    float        output_scale = 1.0f;
    float        output_zp = 0;
    float        out_scale_r = 0;
    float        out_zp_r = 0;
    float        out_scale_g = 0;
    float        out_zp_g = 0;
    float        out_scale_b = 0;
    float        out_zp_b = 0;
    float        pad_v_r = 0;
    float        pad_v_g = 0;
    float        pad_v_b = 0;
    int32_t      in_width  = 0;
    int32_t      in_height = 0;
    int32_t      out_width  = 0;
    int32_t      out_height = 0;
    float        mean_r = 0;
    float        mean_g = 0;
    float        mean_b = 0;
    float        scale_r = 0;
    float        scale_g = 0;
    float        scale_b = 0;
    vx_int32     pad_value_r = 0;
    vx_int32     pad_value_g = 0;
    vx_int32     pad_value_b = 0;
    vx_int32     r_order = 0;
    vx_int32     b_order = 0;
    vx_int32     reverse_channel = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &top);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &bottom);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &left);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &right);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[6], &mean_r);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[7], &mean_g);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[8], &mean_b);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[9], &scale_r);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[10], &scale_g);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[11], &scale_b);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[12], &pad_value_r);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[13], &pad_value_g);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[14], &pad_value_b);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[15], &reverse_channel);
    CHECK_STATUS_FAIL_GOTO(status, final );

    in_width = (int32_t)attr[0]->shape->data[0] / 3;
    in_height = (int32_t)attr[0]->shape->data[1];
    out_width = (int32_t)attr[1]->shape->data[0];
    out_height = (int32_t)attr[1]->shape->data[1] / 3;

    output_scale = 1.0f / attr[1]->scale;
    output_zp = (float)(attr[1]->zero_point);

    resize_w = out_width - left - right;
    resize_h = out_height - top - bottom;
    resize_max_w = out_width - right;
    resize_max_h = out_height - bottom;
    scale_w = (float)in_width / resize_w;
    scale_h = (float)in_height / resize_h;
    out_scale_r = scale_r / output_scale;
    out_zp_r = output_zp - out_scale_r * mean_r;
    out_scale_g = scale_g / output_scale;
    out_zp_g = output_zp - out_scale_g * mean_g;
    out_scale_b = scale_b / output_scale;
    out_zp_b = output_zp - out_scale_b * mean_b;
    pad_v_r = pad_value_r * out_scale_r + out_zp_r;
    pad_v_g = pad_value_g * out_scale_g + out_zp_g;
    pad_v_b = pad_value_b * out_scale_b + out_zp_b;

    if (reverse_channel)
    {
        r_order = out_height * 2;
        b_order = 0;
    }
    else
    {
        r_order = 0;
        b_order = out_height * 2;
    }

    {
        gpu_dp_inst_t uniU8RightSubLeft_4x4 = {{
            0x00090909, // TCfg
            0x00000000, // ASelt
            0x00140003, 0x00000025, // ABin
            0x000a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000,
            0x00010001, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniLeftToFloat32_4x4 = {{
            0x00010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00000002, // ABin
            0x00020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtactHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtract8Data_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status |= vsi_nn_kernel_gpu_add_param( node, "uniU8RightSubLeft_4x4", &uniU8RightSubLeft_4x4 );
        status |= vsi_nn_kernel_gpu_add_param( node, "uniLeftToFloat32_4x4", &uniLeftToFloat32_4x4 );
        status |= vsi_nn_kernel_gpu_add_param( node, "uniExtactHalf8_2x8", &uniExtactHalf8_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node, "uniExtract8Data_2x8", &uniExtract8Data_2x8 );
    }
    status |= vsi_nn_kernel_gpu_add_param( node, "top", &top );
    status |= vsi_nn_kernel_gpu_add_param( node, "left", &left );
    status |= vsi_nn_kernel_gpu_add_param( node, "out_scale_r", &out_scale_r );
    status |= vsi_nn_kernel_gpu_add_param( node, "out_scale_g", &out_scale_g );
    status |= vsi_nn_kernel_gpu_add_param( node, "out_scale_b", &out_scale_b );
    status |= vsi_nn_kernel_gpu_add_param( node, "out_zp_r", &out_zp_r );
    status |= vsi_nn_kernel_gpu_add_param( node, "out_zp_g", &out_zp_g );
    status |= vsi_nn_kernel_gpu_add_param( node, "out_zp_b", &out_zp_b );
    status |= vsi_nn_kernel_gpu_add_param( node, "pad_v_r", &pad_v_r );
    status |= vsi_nn_kernel_gpu_add_param( node, "pad_v_g", &pad_v_g );
    status |= vsi_nn_kernel_gpu_add_param( node, "pad_v_b", &pad_v_b );
    status |= vsi_nn_kernel_gpu_add_param( node, "scale_w", &scale_w );
    status |= vsi_nn_kernel_gpu_add_param( node, "scale_h", &scale_h );
    status |= vsi_nn_kernel_gpu_add_param( node, "resize_max_w", &resize_max_w );
    status |= vsi_nn_kernel_gpu_add_param( node, "resize_max_h", &resize_max_h );
    status |= vsi_nn_kernel_gpu_add_param( node, "out_height", &out_height );
    status |= vsi_nn_kernel_gpu_add_param( node, "r_order", &r_order );
    status |= vsi_nn_kernel_gpu_add_param( node, "b_order", &b_order );

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_size[0] = out_width;
    gpu_param.global_size[1] = out_height;

    status |= vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }

    return status;
} /* _custom_warp_affine_initializer() */

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
    const _kernel_map_type * kernel_map = _custom_letterbox_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _custom_letterbox_kernel_map );
    vx_param_description_t * param_def  = _custom_letterbox_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _custom_letterbox_kernel_param_def );
    vx_kernel_initialize_f  initializer = _custom_letterbox_initializer;
    uint32_t key = 0;
    uint32_t i = 0;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = CUSTOM_LETTERBOX_HASH_KEY( in_dtype, out_dtype );

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
        kernel->info.numParams   = (vx_uint32)param_def_size;
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
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
    vsi_nn_kernel_node_param_t node_params[_CUSTOM_LETTERBOX_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    size_t i = 0;

    int32_t top = vsi_nn_kernel_param_get_int32( params, "top");
    int32_t bottom = vsi_nn_kernel_param_get_int32( params, "bottom");
    int32_t left = vsi_nn_kernel_param_get_int32( params, "left");
    int32_t right = vsi_nn_kernel_param_get_int32( params, "right");
    float   mean_r = vsi_nn_kernel_param_get_float32( params, "mean_r");
    float   mean_g = vsi_nn_kernel_param_get_float32( params, "mean_g");
    float   mean_b = vsi_nn_kernel_param_get_float32( params, "mean_b");
    float   scale_r = vsi_nn_kernel_param_get_float32( params, "scale_r");
    float   scale_g = vsi_nn_kernel_param_get_float32( params, "scale_g");
    float   scale_b = vsi_nn_kernel_param_get_float32( params, "scale_b");
    int32_t pad_value_r = vsi_nn_kernel_param_get_int32( params, "pad_value_r");
    int32_t pad_value_g = vsi_nn_kernel_param_get_int32( params, "pad_value_g");
    int32_t pad_value_b = vsi_nn_kernel_param_get_int32( params, "pad_value_b");
    int32_t reverse_channel = vsi_nn_kernel_param_get_int32( params, "reverse_channel");
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 0 } };

    uint32_t param_num = _CUSTOM_LETTERBOX_PARAM_NUM;
    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);
    shapes[0][0] = inputs[0]->attr.size[1] * 3;
    shapes[0][1] = inputs[0]->attr.size[2];
    shapes[1][0] = outputs[0]->attr.size[0];
    shapes[1][1] = outputs[0]->attr.size[1] * 3;

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
            inputs[0], shapes[0], 2 );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
            outputs[0], shapes[1], 2 );

    if (reshape_tensors[0] == NULL ||
        reshape_tensors[1] == NULL)
    {
        goto final;
    }

    if (reverse_channel)
    {
        float mean_temp = mean_r;
        float scale_temp = scale_r;
        int32_t pad_value_temp = pad_value_r;
        mean_r = mean_b;
        mean_b = mean_temp;
        scale_r = scale_b;
        scale_b = scale_temp;
        pad_value_r = pad_value_b;
        pad_value_b = pad_value_temp;
    }

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 2;

            vsi_nn_kernel_node_pack_io( node_params, param_num,
                    reshape_tensors, 1, &reshape_tensors[1], 1 );

            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &top );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &bottom );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &left );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &right );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &mean_r );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &mean_g );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &mean_b );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &scale_r );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &scale_g );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &scale_b );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_value_r );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_value_g );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_value_b );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse_channel );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, param_num );
            vsi_nn_kernel_scalar_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
            vsi_nn_kernel_scalar_release( &node_params[10] );
            vsi_nn_kernel_scalar_release( &node_params[11] );
            vsi_nn_kernel_scalar_release( &node_params[12] );
            vsi_nn_kernel_scalar_release( &node_params[13] );
            vsi_nn_kernel_scalar_release( &node_params[14] );
            vsi_nn_kernel_scalar_release( &node_params[15] );

            CHECK_STATUS(status);
        }
    }

final:
    for (i = 0; i < 2; i++)
    {
        vsi_safe_release_tensor(reshape_tensors[i]);
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( custom_letterbox, _setup )
