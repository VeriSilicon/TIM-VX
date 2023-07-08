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
    INTERNAL_KERNEL_TINY_YOLOV4_POSTPROCESS_BOX,
} _internal_kernel_e;

#define _SOURCE         "tiny_yolov4_postprocess_box"
#define _KERNEL_NAME    CVIVANTE_NAMESPACE("evis.tiny_yolov4_postprocess_box_U8_U8toU8")

// Add kernel hashtable here
#define TINY_YOLOV4_POSTPROCESS_BOX_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        (( IN0_DTYPE ) | ( IN1_DTYPE << 8 ) | ( OUT_DTYPE << 16 ))
#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ) \
        { TINY_YOLOV4_POSTPROCESS_BOX_HASH_KEY( IN0_DTYPE, IN1_DTYPE, OUT_DTYPE ), \
        _KERNEL_NAME, _SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _tiny_yolov4_postprocess_box_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( U8, U8, U8 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _tiny_yolov4_postprocess_box_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _TINY_YOLOV4_POSTPROCESS_BOX_PARAM_NUM  _cnt_of_array( _tiny_yolov4_postprocess_box_kernel_param_def )
#define SCALAR_BIAS_0_VALUE          (3)
#define SCALAR_BIAS_1_VALUE          (4)
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_tiny_yolov4_postprocess_box_initializer)
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
    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    float CONST2 = 16.0f;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    // Add initializer
    gpu_param.dim = 2;
    gpu_param.global_scale[0] = 4;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_size[0] = gpu_align_p2(
            (attr[0]->shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 8);
    gpu_param.global_size[1] = 1;

    if (attr[0]->shape->data[0] == 13 * 13)
    {
        CONST2 = 32.0f;
    }

    if (attr[0]->dtype == U8 && attr[1]->dtype == U8 && attr[2]->dtype == U8)
    {
        float input0_scale = attr[0]->scale;
        float input0_tail = 0 - (float)attr[0]->zero_point * input0_scale;
        float input1_scale = attr[1]->scale;
        float input1_tail = 0 - (float)attr[1]->zero_point * input1_scale;
        float output_scale = 1.0f / attr[2]->scale;
        float output_zp = (float)attr[2]->zero_point;
        gpu_dp_inst_t uniExtract8Data_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniDatatoFloat32_0_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniDatatoFloat32_1_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniDataTranspose_0_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0c080400, 0x0d090501, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniDataTranspose_1_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0e0a0602, 0x0f0b0703, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        status  = vsi_nn_kernel_gpu_add_param( node, "uniDatatoFloat32_0_4x4", &uniDatatoFloat32_0_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniDatatoFloat32_1_4x4", &uniDatatoFloat32_1_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniExtract8Data_2x8", &uniExtract8Data_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniDataTranspose_0_2x8", &uniDataTranspose_0_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniDataTranspose_1_2x8", &uniDataTranspose_1_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "input0_scale", &input0_scale);
        status |= vsi_nn_kernel_gpu_add_param( node, "input0_tail", &input0_tail);
        status |= vsi_nn_kernel_gpu_add_param( node, "input1_scale", &input1_scale);
        status |= vsi_nn_kernel_gpu_add_param( node, "input1_tail", &input1_tail);
        status |= vsi_nn_kernel_gpu_add_param( node, "output_scale", &output_scale);
        status |= vsi_nn_kernel_gpu_add_param( node, "output_zp", &output_zp);
        status |= vsi_nn_kernel_gpu_add_param( node, "CONST2", &CONST2);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
    }

    return status;
} /* _tiny_yolov4_postprocess_box_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _tiny_yolov4_postprocess_box_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _tiny_yolov4_postprocess_box_kernel_map );
    vx_param_description_t * param_def  = _tiny_yolov4_postprocess_box_kernel_param_def;
    vx_kernel_initialize_f  initializer = _tiny_yolov4_postprocess_box_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = TINY_YOLOV4_POSTPROCESS_BOX_HASH_KEY( in0_dtype, in1_dtype, out_dtype );

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
        kernel->info.numParams   = _cnt_of_array( _tiny_yolov4_postprocess_box_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_TINY_YOLOV4_POSTPROCESS_BOX_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t shape[3][VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    float bias_0 = vsi_nn_kernel_param_get_float32( params, "bias_0" );
    float bias_1 = vsi_nn_kernel_param_get_float32( params, "bias_1" );

    VSI_UNREFERENCED(params);

    memcpy(shape[0], inputs[0]->attr.size, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    shape[0][0] = shape[0][0] * shape[0][1];
    shape[0][1] = shape[0][2];
    shape[0][2] = 1;

    memcpy(shape[1], inputs[1]->attr.size, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    shape[1][0] = shape[1][0] * shape[1][1];
    shape[1][1] = shape[1][2];
    shape[1][2] = 1;

    memcpy(shape[2], outputs[0]->attr.size, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    shape[2][0] = shape[2][0];
    shape[2][1] = shape[2][2] * shape[2][1];
    shape[2][2] = 1;

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
            inputs[0], shape[0], inputs[0]->attr.dim_num );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
            inputs[1], shape[1], inputs[1]->attr.dim_num );
    reshape_tensors[2] = vsi_nn_reshape_tensor( graph,
            outputs[0], shape[2], outputs[0]->attr.dim_num );

    if ( !vsi_nn_kernel_gpu_check_shape(
        reshape_tensors[0]->attr.size, reshape_tensors[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _TINY_YOLOV4_POSTPROCESS_BOX_PARAM_NUM,
                    reshape_tensors, input_num, &reshape_tensors[2], output_num );
            /* Pass parameters to node. */
            node_params[SCALAR_BIAS_0_VALUE] = vsi_nn_kernel_scalar_create( graph, F32, &bias_0 );
            node_params[SCALAR_BIAS_1_VALUE] = vsi_nn_kernel_scalar_create( graph, F32, &bias_1 );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _TINY_YOLOV4_POSTPROCESS_BOX_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_BIAS_0_VALUE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_BIAS_1_VALUE] );
        }
    }

    vsi_safe_release_tensor( reshape_tensors[0] );
    vsi_safe_release_tensor( reshape_tensors[1] );
    vsi_safe_release_tensor( reshape_tensors[2] );

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( tiny_yolov4_postprocess_box, _setup )

