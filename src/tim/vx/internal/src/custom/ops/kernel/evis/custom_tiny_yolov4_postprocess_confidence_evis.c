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
    INTERNAL_KERNEL_TINY_YOLOV4_POSTPROCESS_CONFIDENCE,
} _internal_kernel_e;

#define _SOURCE         "tiny_yolov4_postprocess_confidence"
#define _KERNEL_NAME    CVIVANTE_NAMESPACE("evis.tiny_yolov4_postprocess_conf_U8toU8")

// Add kernel hashtable here
#define _CONFIDENCE_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        (( IN_DTYPE << 8 ) | ( OUT_DTYPE ))
#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
        { _CONFIDENCE_HASH_KEY( IN_DTYPE, OUT_DTYPE ), \
         _KERNEL_NAME, _SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _tiny_yolov4_postprocess_confidence_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( U8, U8 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _tiny_yolov4_postprocess_confidence_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _TINY_YOLOV4_POSTPROCESS_CONFIDENCE_PARAM_NUM  \
    _cnt_of_array( _tiny_yolov4_postprocess_confidence_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_tiny_yolov4_postprocess_confidence_initializer)
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
    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    gpu_param.dim = 2;
    gpu_param.global_scale[0] = 4;
    gpu_param.global_scale[1] = 4;
    gpu_param.global_size[0] = gpu_align_p2(
            (attr[0]->shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (attr[1]->shape->data[0] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);

    if (attr[0]->dtype == U8 && attr[1]->dtype == U8)
    {
        float output_scale = attr[0]->scale * attr[0]->scale / attr[1]->scale;
        int output_zp = attr[1]->zero_point;
        uint16_t   M0                 = 0;
        int32_t    postShift          = 0;
        int32_t i = 0;

        gpu_dp_inst_t uniU8TimesU8_0_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x01010101, // BSelt
            0x00010000, 0x00030002, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniU16TimesMultiplier_PostShift_2x8 = {{
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniU8PlusU8_trans_0_2x8 = {{
            0xffffffff, // TCfg
            0x44444444, // ASelt
            0x0c080400, 0x0d090501, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniU8PlusU8_trans_1_2x8 = {{
            0xffffffff, // TCfg
            0x44444444, // ASelt
            0x0e0a0602, 0x0f0b0703, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_quantize_multiplier_16bit((double)output_scale, &M0, &postShift);

        uniU16TimesMultiplier_PostShift_2x8.data[7] |= (postShift & 0x1F);
        for ( i = 8; i < 16; i++ )
        {
            uniU16TimesMultiplier_PostShift_2x8.data[i] = M0;
        }

        status  = vsi_nn_kernel_gpu_add_param( node, "uniU8TimesU8_0_4x4", &uniU8TimesU8_0_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniU16TimesMultiplier_PostShift_2x8",
            &uniU16TimesMultiplier_PostShift_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniU8PlusU8_trans_0_2x8", &uniU8PlusU8_trans_0_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniU8PlusU8_trans_1_2x8", &uniU8PlusU8_trans_1_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "output_zp", &output_zp);
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

    return status;
} /* _tiny_yolov4_postprocess_confidence_initializer() */


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
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _tiny_yolov4_postprocess_confidence_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _tiny_yolov4_postprocess_confidence_kernel_map );
    vx_param_description_t * param_def  = _tiny_yolov4_postprocess_confidence_kernel_param_def;
    vx_kernel_initialize_f  initializer = _tiny_yolov4_postprocess_confidence_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = _CONFIDENCE_HASH_KEY( in_dtype, out_dtype );

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
        kernel->info.numParams   = _cnt_of_array( _tiny_yolov4_postprocess_confidence_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_TINY_YOLOV4_POSTPROCESS_CONFIDENCE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t shape[2][VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };

    VSI_UNREFERENCED(params);

    memcpy(shape[0], inputs[0]->attr.size, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    shape[0][0] = shape[0][0] * shape[0][1];
    shape[0][1] = shape[0][2];
    shape[0][2] = 1;

    memcpy(shape[1], outputs[0]->attr.size, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    shape[1][0] = shape[1][0];
    shape[1][1] = shape[1][2] * shape[1][1];
    shape[1][2] = 1;

    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
            inputs[0], shape[0], inputs[0]->attr.dim_num );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
            outputs[0], shape[1], outputs[0]->attr.dim_num );

    if ( !vsi_nn_kernel_gpu_check_shape(
        reshape_tensors[0]->attr.size, reshape_tensors[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _TINY_YOLOV4_POSTPROCESS_CONFIDENCE_PARAM_NUM,
                    reshape_tensors, input_num, &reshape_tensors[1], output_num );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params,
                _TINY_YOLOV4_POSTPROCESS_CONFIDENCE_PARAM_NUM );
        }
    }

    vsi_safe_release_tensor(reshape_tensors[0]);
    vsi_safe_release_tensor(reshape_tensors[1]);

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( tiny_yolov4_postprocess_confidence, _setup )

