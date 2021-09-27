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
#include "libnnext/vx_lib_nnext.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

#define _ROI_ALIGN_KERNEL_SOURCE(_input_type)      "roi_align"

#define STR(a) #a
// Add kernel hashtable here
#define ROI_ALIGN_HASH_KEY( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, _image_2d ) \
        (( IN0_DTYPE ) | ( IN1_DTYPE << 7) | (IN2_DTYPE << 14) | (OUT_DTYPE << 21) | (_image_2d << 28))

#define PACK_KERNEL_MAP( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE ) \
        { ROI_ALIGN_HASH_KEY( IN0_DTYPE, IN1_DTYPE, IN2_DTYPE, OUT_DTYPE, 0 ), \
          CVIVANTE_NAMESPACE("cl.roi_align_"STR(IN0_DTYPE)"to"STR(OUT_DTYPE)), \
          _ROI_ALIGN_KERNEL_SOURCE(IN0_DTYPE) }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _roi_align_kernel_map[] =
{
    PACK_KERNEL_MAP(F32, F32, I32, F32),
};


/*
 * Kernel params
 */
static vx_param_description_t _roi_align_kernel_param_def[] =
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
#define _ROI_ALIGN_PARAM_NUM  _cnt_of_array( _roi_align_kernel_param_def )

#define SCALAR_SPATIAL_X_SCALE          (4)
#define SCALAR_SPATIAL_Y_SCALE          (5)
#define SCALAR_INPUT_WIDTH              (6)
#define SCALAR_INPUT_HEIGHT             (7)
#define SCALAR_RCP_OF_OUTPUT_WIDTH      (8)
#define SCALAR_RCP_OF_OUTPUT_HEIGHT     (9)
#define SCALAR_SAMPLING_X_RATIO         (10)
#define SCALAR_SAMPLING_Y_RATIO         (11)
#define SCALAR_DEPTH                    (12)

#define ROI_ALIGN_PARAM_NUM         13
#define ROI_ALIGN_QUANT_PARAM_NUM   _cnt_of_array( _roi_align_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_roi_align_initializer)
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
    vsi_nn_kernel_tensor_attr_t * rois_attr     = NULL;
    vsi_nn_kernel_tensor_attr_t * output_attr   = NULL;
    vsi_size_array_t * rois_shape                = NULL;
    vsi_size_array_t * out_shape                 = NULL;

    rois_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( rois_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    rois_shape = rois_attr->shape;
    out_shape  = output_attr->shape;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.dim = 3;
    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = rois_shape->data[1];
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(output_attr);

    return status;
} /* _roi_align_initializer() */



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
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_nn_kernel_dtype_e in2_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _roi_align_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _roi_align_kernel_map );
    vx_param_description_t * param_def  = _roi_align_kernel_param_def;
    size_t param_def_size               = ROI_ALIGN_QUANT_PARAM_NUM;
    vx_kernel_initialize_f  initializer = _roi_align_initializer;

    uint32_t key;
    uint32_t i;

    in0_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    in1_dtype  = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    in2_dtype  = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    out_dtype  = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    in0_dtype = in0_dtype == F16 ? F32 : in0_dtype;
    in1_dtype = in1_dtype == F16 ? F32 : in1_dtype;

    key = ROI_ALIGN_HASH_KEY( in0_dtype, in1_dtype, in2_dtype, out_dtype, image_2d );

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

#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)

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
    vsi_nn_kernel_node_param_t node_params[_ROI_ALIGN_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    uint32_t rank[_IO_NUM] = {0};
    vsi_size_t  shapes[_IO_NUM][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    vsi_nn_tensor_t* reshape_tensors[_IO_NUM] = { NULL };
    int32_t i = 0;
    float   width_ratio         = vsi_nn_kernel_param_get_float32( params, "width_ratio" );
    float   height_ratio        = vsi_nn_kernel_param_get_float32( params, "height_ratio" );
    int32_t width_sample_num    = vsi_nn_kernel_param_get_int32( params, "width_sample_num" );
    int32_t height_sample_num   = vsi_nn_kernel_param_get_int32( params, "height_sample_num" );
    float   width_scale         = 1.0f / width_ratio;
    float   height_scale        = 1.0f / height_ratio;
    float   in_width            = (float)(inputs[0]->attr.size[0]);
    float   in_height           = (float)(inputs[0]->attr.size[1]);
    float   rcp_of_out_width    = 1.0f / (float)(outputs[0]->attr.size[0]);
    float   rcp_of_out_height   = 1.0f / (float)(outputs[0]->attr.size[1]);
    float   sampling_x_ratio    = width_sample_num > 0 ? (float)width_sample_num : 0;
    float   sampling_y_ratio    = height_sample_num > 0 ? (float)height_sample_num : 0;
    vsi_size_t     depth               = inputs[0]->attr.size[2];

    vsi_nn_kernel_optimize_nchw2xhw_shape( (const vsi_size_t*)inputs[0]->attr.size, inputs[0]->attr.dim_num,
                                              shapes[0], &rank[0]);
    vsi_nn_kernel_optimize_1d_tensor_shape( (const vsi_size_t*)inputs[1]->attr.size, inputs[1]->attr.dim_num,
                                              shapes[1], &rank[1]);
    vsi_nn_kernel_optimize_1d_tensor_shape( (const vsi_size_t*)inputs[2]->attr.size, inputs[2]->attr.dim_num,
                                              shapes[2], &rank[2]);
    vsi_nn_kernel_optimize_nchw2xhw_shape( (const vsi_size_t*)outputs[0]->attr.size, outputs[0]->attr.dim_num,
                                              shapes[3], &rank[3]);

    for (i = 0; i < _INPUT_NUM; i++)
    {
        reshape_tensors[i] = vsi_nn_reshape_tensor( graph,
            inputs[i], shapes[i], rank[i] );
    }
    reshape_tensors[_INPUT_NUM] = vsi_nn_reshape_tensor( graph,
        outputs[0], shapes[_INPUT_NUM], rank[_INPUT_NUM] );

    if ( !vsi_nn_kernel_gpu_check_shape( reshape_tensors[0]->attr.size,
                inputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( kernel, reshape_tensors, &reshape_tensors[_INPUT_NUM], image_2d);

    if ( VSI_SUCCESS == status )
    {
        size_t node_params_num = ROI_ALIGN_PARAM_NUM;

        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _ROI_ALIGN_PARAM_NUM,
                reshape_tensors, input_num, &reshape_tensors[_INPUT_NUM], output_num );

            node_params[SCALAR_SPATIAL_X_SCALE]      = vsi_nn_kernel_scalar_create( graph, F32, &width_scale );
            node_params[SCALAR_SPATIAL_Y_SCALE]      = vsi_nn_kernel_scalar_create( graph, F32, &height_scale );
            node_params[SCALAR_INPUT_WIDTH]          = vsi_nn_kernel_scalar_create( graph, F32, &in_width );
            node_params[SCALAR_INPUT_HEIGHT]         = vsi_nn_kernel_scalar_create( graph, F32, &in_height );
            node_params[SCALAR_RCP_OF_OUTPUT_WIDTH]  = vsi_nn_kernel_scalar_create( graph, F32, &rcp_of_out_width );
            node_params[SCALAR_RCP_OF_OUTPUT_HEIGHT] = vsi_nn_kernel_scalar_create( graph, F32, &rcp_of_out_height );
            node_params[SCALAR_SAMPLING_X_RATIO]     = vsi_nn_kernel_scalar_create( graph, F32, &sampling_x_ratio );
            node_params[SCALAR_SAMPLING_Y_RATIO]     = vsi_nn_kernel_scalar_create( graph, F32, &sampling_y_ratio );
            node_params[SCALAR_DEPTH]                = vsi_nn_kernel_scalar_create( graph, I32, &depth );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SPATIAL_X_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SPATIAL_Y_SCALE] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_WIDTH] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_HEIGHT] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_RCP_OF_OUTPUT_WIDTH] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_RCP_OF_OUTPUT_HEIGHT] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SAMPLING_X_RATIO] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_SAMPLING_Y_RATIO] );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_DEPTH] );
        }
    }

    for (i = 0; i < _IO_NUM; i++)
    {
        if (reshape_tensors[i])
        {
            vsi_nn_ReleaseTensor( &reshape_tensors[i] );
        }
    }

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CL( roi_align, _setup )

