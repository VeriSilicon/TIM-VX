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

#define _RESIZE_CUBIC_KERNEL_SOURCE()      "resize_cubic"

#define STR(a) #a
// Add kernel hashtable here
#define RESIZE_CUBIC_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \
        (( IN_DTYPE << 20 ) | ( OUT_DTYPE << 8) )

#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE ) \
        { RESIZE_CUBIC_HASH_KEY( IN_DTYPE, OUT_DTYPE ), \
          CVIVANTE_NAMESPACE("evis.resize_cubic_"STR(IN_DTYPE)"to"STR(OUT_DTYPE)), \
          _RESIZE_CUBIC_KERNEL_SOURCE() }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _resize_cubic_kernel_map[] =
{
    PACK_KERNEL_MAP(F16, F16),
    PACK_KERNEL_MAP(I16, I16),
    PACK_KERNEL_MAP(F16, I16),
    PACK_KERNEL_MAP(I16, F16),
    PACK_KERNEL_MAP(I8,  I8),
    PACK_KERNEL_MAP(F16, I8),
    PACK_KERNEL_MAP(I8,  F16),
    PACK_KERNEL_MAP(U8,  U8),
    PACK_KERNEL_MAP(F16, U8),
    PACK_KERNEL_MAP(U8,  F16),
};


/*
 * Kernel params
 */
static vx_param_description_t _resize_cubic_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

#define RESIZE_CUBIC_NUM   _cnt_of_array( _resize_cubic_kernel_param_def )


/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_resize_cubic_initializer)
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
    vsi_nn_kernel_tensor_attr_t *input_attr     = NULL;
    vsi_nn_kernel_tensor_attr_t *output_attr    = NULL;
    vsi_size_array_t * out_shape                = NULL;

    float       input_scale        = 1.0;
    float       input_tail         = 0;
    float       output_scale       = 1.0;
    float       output_tail        = 0;

    VSI_UNREFERENCED(param_size);

    input_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0]);
    CHECK_PTR_FAIL_GOTO( input_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );
    output_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1]);
    CHECK_PTR_FAIL_GOTO( output_attr, "vsi_nn_kernel_tensor_attr_create fail.", final );

    out_shape  = output_attr->shape;

    if ( input_attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = input_attr->dfp.fl;
        if (fl > 0)
        {
            input_scale = 1.0f / (float) ((int64_t)1 << fl);
        }
        else
        {
            input_scale = (float)((int64_t)1 << -fl);
        }
    }
    else if ( input_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        input_scale = input_attr->asymm.scale;
        input_tail  = 0 - input_scale * (float)input_attr->asymm.zero_point;
    }

    if ( output_attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        int32_t fl = output_attr->dfp.fl;
        if (fl > 0)
        {
            output_scale = (float) ((int64_t)1 << fl);
        }
        else
        {
            output_scale = 1.0f / (float)((int64_t)1 << -fl);
        }
    }
    else if ( output_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        output_scale = 1.0f / output_attr->asymm.scale;
        output_tail    = (float)output_attr->asymm.zero_point;
    }

    gpu_param.global_scale[0]  = 4;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    {
        gpu_dp_inst_t uniFp16ToFp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_dp_inst_t uniExtract8Bit_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        gpu_dp_inst_t uniExtractHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16};
        status  = vsi_nn_kernel_gpu_add_param( node, "uniFp16ToFp32_4x4", &uniFp16ToFp32_4x4);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniExtractHalf8_2x8", &uniExtractHalf8_2x8);
        status |= vsi_nn_kernel_gpu_add_param( node, "uniExtract8Bit_2x8", &uniExtract8Bit_2x8);
    }
    status |= vsi_nn_kernel_gpu_add_param( node, "input_scale", &input_scale);
    status |= vsi_nn_kernel_gpu_add_param( node, "input_tail", &input_tail);
    status |= vsi_nn_kernel_gpu_add_param( node, "output_scale", &output_scale);
    status |= vsi_nn_kernel_gpu_add_param( node, "output_tail", &output_tail);


    gpu_param.global_size[0] = gpu_align_p2(
            (out_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (out_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = out_shape->size > 2 ? out_shape->data[2] : 1;
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );

final:
    if (input_attr) vsi_nn_kernel_tensor_attr_release( &input_attr );
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );
    return status;
} /* _resize_cubic_initializer() */



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
    const _kernel_map_type * kernel_map = _resize_cubic_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _resize_cubic_kernel_map );
    vx_param_description_t * param_def  = _resize_cubic_kernel_param_def;
    size_t param_def_size               = RESIZE_CUBIC_NUM;
    vx_kernel_initialize_f  initializer = _resize_cubic_initializer;

    uint32_t key = 0;
    uint32_t i = 0;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = RESIZE_CUBIC_HASH_KEY( in_dtype, out_dtype );

    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_def_size;
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

static vsi_nn_tensor_t* _create_scale_tensor
    (
    vsi_nn_graph_t  *graph,
    vsi_size_t       output_size,
    float            scale_factor,
    float            half_pixel_value,
    vsi_nn_tensor_t** index
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t*  scale           = NULL;
    vsi_size_t   i                    = 0;
    float       *scale_data_ptr       = NULL;
    int         *index_data_ptr       = NULL;
    float        scale_value          = 0;
    vsi_ssize_t  data                 = 0;
    int          idx                  = 0;
    float        delta_v              = 0;
    float        cubic_coeff_a        = -0.5f;
    vsi_size_t   item_count           = 4 * output_size;
    scale_data_ptr = (float *)malloc(item_count * sizeof(float));
    if (scale_data_ptr == NULL)
    {
        VSILOGE("allocate memory fail at function %s line %d", __FUNCTION__, __LINE__);
        goto OnError;
    }

    index_data_ptr = (int *)malloc(output_size * sizeof(int));
    if (index_data_ptr == NULL)
    {
        VSILOGE("allocate memory fail at function %s line %d", __FUNCTION__, __LINE__);
        goto OnError;
    }

    for (i = 0; i < output_size; i ++)
    {
        scale_value = ((float)i + half_pixel_value) * scale_factor - half_pixel_value;
        data = (vsi_ssize_t)scale_value;
        delta_v = scale_value - (float)data;
        idx   = (int)data - 1;

        index_data_ptr[i] = idx;
        scale_data_ptr[i * 4 + 0] = cubic_coeff_a * (((delta_v - 4) * (delta_v + 1) + 8) * (delta_v + 1) - 4);
        scale_data_ptr[i * 4 + 1] = ((cubic_coeff_a + 2) * delta_v - (cubic_coeff_a + 3)) * delta_v *delta_v + 1;
        scale_data_ptr[i * 4 + 2] = ((cubic_coeff_a + 2) * (1 - delta_v) - (cubic_coeff_a + 3))
                                  * (1 - delta_v) * (1 - delta_v) + 1;
        scale_data_ptr[i * 4 + 3] = cubic_coeff_a * ((( 2 - delta_v - 5) * (2 - delta_v) + 8) * (2 - delta_v) - 4);
    }
    attr.size[0] = item_count;
    attr.dim_num = 1;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.vtl = FALSE;

    scale = vsi_nn_CreateTensorFromData(graph, (uint8_t *)scale_data_ptr, &attr);
    if (scale_data_ptr)
    {
        free (scale_data_ptr);
        scale_data_ptr = NULL;
    }

    attr.size[0] = output_size;
    attr.dim_num = 1;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.vtl = FALSE;

    *index = vsi_nn_CreateTensorFromData(graph, (uint8_t *)index_data_ptr, &attr);
    if (index_data_ptr)
    {
        free (index_data_ptr);
        index_data_ptr = NULL;
    }

OnError:
    return scale;
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
    vsi_nn_kernel_node_param_t node_params[RESIZE_CUBIC_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t align_corners       = vsi_nn_kernel_param_get_int32( params, "align_corners" );
    int32_t half_pixel_centers  = vsi_nn_kernel_param_get_int32( params, "half_pixel_centers" );
    vsi_size_t in_width     = inputs[0]->attr.size[0];
    vsi_size_t in_height    = inputs[0]->attr.size[1];
    vsi_size_t out_width    = outputs[0]->attr.size[0];
    vsi_size_t out_height   = outputs[0]->attr.size[1];
    float   half_pixel_value = 0.0f;
    float   width_scale = 0.0f;
    float   height_scale = 0.0f;
    vsi_nn_tensor_t* scale_w = NULL;
    vsi_nn_tensor_t* scale_h = NULL;
    vsi_nn_tensor_t* index_w = NULL;
    vsi_nn_tensor_t* index_h = NULL;

    if (align_corners && out_width > 1)
    {
        width_scale = ((vx_float32)(in_width - 1) * 1.0f) / (vx_float32)(out_width - 1);
    }
    else
    {
        width_scale = ((vx_float32)in_width * 1.0f) / (vx_float32)out_width;
    }

    if (align_corners && out_height > 1)
    {
        height_scale = ((vx_float32)(in_height - 1) * 1.0f) / (vx_float32)(out_height - 1);
    }
    else
    {
        height_scale = ((vx_float32)in_height * 1.0f) / (vx_float32)out_height;
    }

    if (half_pixel_centers)
    {
        half_pixel_value = 0.5f;
    }
    else
    {
        half_pixel_value = 0.0f;
    }

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            size_t node_params_num = RESIZE_CUBIC_NUM;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, RESIZE_CUBIC_NUM,
                    inputs, input_num, outputs, output_num );
            scale_w = _create_scale_tensor(graph, out_width,\
                              width_scale, half_pixel_value, &index_w);
            CHECK_PTR_FAIL_GOTO( scale_w, "Create buffer fail.", final );
            CHECK_PTR_FAIL_GOTO( index_w, "Create buffer fail.", final );
            scale_h = _create_scale_tensor(graph, out_height,\
                              height_scale, half_pixel_value, &index_h);
            CHECK_PTR_FAIL_GOTO( scale_h, "Create buffer fail.", final );
            CHECK_PTR_FAIL_GOTO( index_h, "Create buffer fail.", final );
            node_params[2] = (vsi_nn_kernel_node_param_t)(scale_w->t);
            node_params[3] = (vsi_nn_kernel_node_param_t)(scale_h->t);
            node_params[4] = (vsi_nn_kernel_node_param_t)(index_w->t);
            node_params[5] = (vsi_nn_kernel_node_param_t)(index_h->t);
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, node_params_num );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
        }
    }

final:
    vsi_safe_release_tensor(scale_w);
    vsi_safe_release_tensor(scale_h);
    vsi_safe_release_tensor(index_w);
    vsi_safe_release_tensor(index_h);
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( resize_cubic, _setup )
