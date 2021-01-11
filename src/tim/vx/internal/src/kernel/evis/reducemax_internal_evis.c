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
#include "vsi_nn_error.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

#define HASH_REDUCEMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, _image_2d) \
    ((AXIS << 20) | (IN_DTYPE << 12) | (OUT_DTYPE << 4) | (_image_2d))

 #define HASH_REDUCEMAX_KERNEL_SOURCE_NAME(AXIS) \
    "reducemax_internal_axis"#AXIS

#define HASH_REDUCEMAX_KERNELS( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_REDUCEMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 0), \
        CVIVANTE_NAMESPACE("evis.reducemax_axis"#AXIS"_"#IN_DTYPE"to"#OUT_DTYPE), \
        HASH_REDUCEMAX_KERNEL_SOURCE_NAME(AXIS) },

#define HASH_REDUCEMAX_KERNELS_2D( AXIS, IN_DTYPE, OUT_DTYPE) \
        { HASH_REDUCEMAX_HASH_KEY(AXIS, IN_DTYPE, OUT_DTYPE, 1), \
        CVIVANTE_NAMESPACE("evis.reducemax_axis"#AXIS"_"#IN_DTYPE"to"#OUT_DTYPE"_2D"), \
        HASH_REDUCEMAX_KERNEL_SOURCE_NAME(AXIS) },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _reducemax_internal_kernel_map[] =
{
    HASH_REDUCEMAX_KERNELS( 0, F16, F16 )
    HASH_REDUCEMAX_KERNELS( 0, F16, I16 )
    HASH_REDUCEMAX_KERNELS( 0, F16, I8  )
    HASH_REDUCEMAX_KERNELS( 0, F16, U8  )
    HASH_REDUCEMAX_KERNELS( 0, I16, I16 )
    HASH_REDUCEMAX_KERNELS( 0, I8,  I8  )
    HASH_REDUCEMAX_KERNELS( 0, U8,  U8  )
    HASH_REDUCEMAX_KERNELS( 0, I16, F16 )
    HASH_REDUCEMAX_KERNELS( 0, I8,  F16 )
    HASH_REDUCEMAX_KERNELS( 0, U8,  F16 )
    HASH_REDUCEMAX_KERNELS( 1, F16, F16 )
    HASH_REDUCEMAX_KERNELS( 1, F16, I16 )
    HASH_REDUCEMAX_KERNELS( 1, F16, I8  )
    HASH_REDUCEMAX_KERNELS( 1, F16, U8  )
    HASH_REDUCEMAX_KERNELS( 1, I16, I16 )
    HASH_REDUCEMAX_KERNELS( 1, I8,  I8  )
    HASH_REDUCEMAX_KERNELS( 1, U8,  U8  )
    HASH_REDUCEMAX_KERNELS( 1, I16, F16 )
    HASH_REDUCEMAX_KERNELS( 1, I8,  F16 )
    HASH_REDUCEMAX_KERNELS( 1, U8,  F16 )
    HASH_REDUCEMAX_KERNELS( 2, F16, F16 )
    HASH_REDUCEMAX_KERNELS( 2, F16, I16 )
    HASH_REDUCEMAX_KERNELS( 2, F16, I8  )
    HASH_REDUCEMAX_KERNELS( 2, F16, U8  )
    HASH_REDUCEMAX_KERNELS( 2, I16, I16 )
    HASH_REDUCEMAX_KERNELS( 2, I8,  I8  )
    HASH_REDUCEMAX_KERNELS( 2, U8,  U8  )
    HASH_REDUCEMAX_KERNELS( 2, I16, F16 )
    HASH_REDUCEMAX_KERNELS( 2, I8,  F16 )
    HASH_REDUCEMAX_KERNELS( 2, U8,  F16 )

    HASH_REDUCEMAX_KERNELS_2D( 0, F16, F16 )
    HASH_REDUCEMAX_KERNELS_2D( 0, F16, I16 )
    HASH_REDUCEMAX_KERNELS_2D( 0, F16, I8  )
    HASH_REDUCEMAX_KERNELS_2D( 0, F16, U8  )
    HASH_REDUCEMAX_KERNELS_2D( 0, I16, I16 )
    HASH_REDUCEMAX_KERNELS_2D( 0, I8,  I8  )
    HASH_REDUCEMAX_KERNELS_2D( 0, U8,  U8  )
    HASH_REDUCEMAX_KERNELS_2D( 0, I16, F16 )
    HASH_REDUCEMAX_KERNELS_2D( 0, I8,  F16 )
    HASH_REDUCEMAX_KERNELS_2D( 0, U8,  F16 )
    HASH_REDUCEMAX_KERNELS_2D( 1, F16, F16 )
    HASH_REDUCEMAX_KERNELS_2D( 1, F16, I16 )
    HASH_REDUCEMAX_KERNELS_2D( 1, F16, I8  )
    HASH_REDUCEMAX_KERNELS_2D( 1, F16, U8  )
    HASH_REDUCEMAX_KERNELS_2D( 1, I16, I16 )
    HASH_REDUCEMAX_KERNELS_2D( 1, I8,  I8  )
    HASH_REDUCEMAX_KERNELS_2D( 1, U8,  U8  )
    HASH_REDUCEMAX_KERNELS_2D( 1, I16, F16 )
    HASH_REDUCEMAX_KERNELS_2D( 1, I8,  F16 )
    HASH_REDUCEMAX_KERNELS_2D( 1, U8,  F16 )
};

/*
 * Kernel params
 */
static vx_param_description_t _reducemax_internal_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _REDUCEMAX_INTERNAL_PARAM_NUM  _cnt_of_array( _reducemax_internal_kernel_param_def )

#define SCALAR_INPUT_AXIS          (2)
/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_reducemax_internal_initializer)
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
    int32_t     axis                           = 0;
    vsi_nn_kernel_tensor_attr_t *input_attr    = NULL;
    vsi_nn_kernel_tensor_attr_t *output_attr   = NULL;
    vsi_int_array_t * input_shape              = NULL;
    vsi_int_array_t * output_shape             = NULL;
    int32_t  input_fl = 0, output_fl = 0;
    int32_t  axisSize = 0;
    float    inputScale                        = 1.0f;
    float    input_offset_asymmetric           = 0.0f;
    float    outputScale                       = 1.0f;
    float    output_offset_asymmetric          = 0.0f;

    input_attr  = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( input_attr, "Create tensor attr buffer fail.", final );
    output_attr = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( output_attr, "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );

    output_shape  = output_attr->shape;
    input_shape   = input_attr->shape;

    if (axis == 0)
    {
        gpu_param.global_scale[0]  = 1;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_scale[2]  = 1;
    }
    else
    {
        gpu_param.global_scale[0]  = 8;
        gpu_param.global_scale[1]  = 1;
        gpu_param.global_scale[2]  = 1;
    }

    gpu_param.dim = 2;
    gpu_param.global_size[0] = gpu_align_p2(
            (output_shape->data[0] + gpu_param.global_scale[0] - 1)
            / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1] = (
            (output_shape->data[1] + gpu_param.global_scale[1] - 1)
            / gpu_param.global_scale[1]);
    gpu_param.global_size[2] = 1;
    axisSize = input_shape->data[axis];

    {
        gpu_dp_inst_t uniPackMaxData_2x8 = {{
            0x00000111, // TCfg
            0x00000000, // ASelt
            0x00050300, 0x00000000, // ABin
            0x00000222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00004400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniGetLoData_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniGetHiData_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16};

        if (0 == axis)
        {
            status  = vsi_nn_kernel_gpu_add_param( node,
                    "uniPackMaxData_2x8", &uniPackMaxData_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniGetLoData_4x4", &uniGetLoData_4x4 );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
        else if (1 == axis || 2 == axis)
        {
            status  = vsi_nn_kernel_gpu_add_param( node,
                    "uniGetLoData_4x4", &uniGetLoData_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniGetHiData_4x4", &uniGetHiData_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, final );
        }
    }

    if( input_attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        input_fl = input_attr->dfp.fl;
        if (input_fl > 0)
        {
            inputScale = 1.0f / (float) ((int64_t)1 << input_fl);
        }
        else
        {
            inputScale = (float)((int64_t)1 << -input_fl);
        }
        status  = vsi_nn_kernel_gpu_add_param( node, "inputScale", &inputScale );
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if( input_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        inputScale              = input_attr->asymm.scale;
        input_offset_asymmetric = (float)(input_attr->asymm.zero_point);
        status  = vsi_nn_kernel_gpu_add_param( node, "inputScale", &inputScale );
        status |= vsi_nn_kernel_gpu_add_param( node, "input_offset_asymmetric", &input_offset_asymmetric );
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    if( output_attr->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        output_fl = output_attr->dfp.fl;
        if (output_fl > 0)
        {
            outputScale = (float) ((int64_t)1 << output_fl);
        }
        else
        {
            outputScale = 1.0f / (float)((int64_t)1 << -output_fl);
        }
        status  = vsi_nn_kernel_gpu_add_param( node, "outputScale", &outputScale );
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if( output_attr->quant == VSI_NN_KERNEL_QUANT_ASYMM )
    {
        outputScale              = 1.0f / output_attr->asymm.scale;
        output_offset_asymmetric = (float)(output_attr->asymm.zero_point);
        status  = vsi_nn_kernel_gpu_add_param( node, "outputScale", &outputScale );
        status |= vsi_nn_kernel_gpu_add_param( node, "output_offset_asymmetric", &output_offset_asymmetric );
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    status  = vsi_nn_kernel_gpu_add_param( node, "axisSize", &axisSize );
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
final:
    if (input_attr) vsi_nn_kernel_tensor_attr_release( &input_attr );
    if (output_attr) vsi_nn_kernel_tensor_attr_release( &output_attr );

    return status;
} /* _reducemax_internal_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs,
    int32_t axis,
    vsi_bool image_2d
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _reducemax_internal_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _reducemax_internal_kernel_map );
    vx_param_description_t * param_def  = _reducemax_internal_kernel_param_def;
    size_t param_def_size               = _cnt_of_array( _reducemax_internal_kernel_param_def );
    vx_kernel_initialize_f  initializer = _reducemax_internal_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if ((BF16 == in_dtype) && (BF16 == out_dtype))
    {
        in_dtype  = F16;
        out_dtype = F16;
    }

    key = HASH_REDUCEMAX_HASH_KEY( axis, in_dtype, out_dtype, image_2d );

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }

    if( i < kernel_map_size )
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
    vsi_nn_kernel_node_param_t node_params[_REDUCEMAX_INTERNAL_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    int32_t  axis = 0;

    axis = vsi_nn_kernel_param_get_int32(params, "axis");

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)inputs[0]->attr.size,
                inputs[0]->attr.dim_num )
     || !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num )
     || axis > 2)
    {
        return NULL;
    }
    image_2d = (inputs[0]->attr.dim_num == 2 || inputs[0]->attr.size[2] == 1);
    status = _query_kernel( kernel, inputs, outputs, axis, image_2d );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _REDUCEMAX_INTERNAL_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_INPUT_AXIS] = vsi_nn_kernel_scalar_create(
                    graph, I32, &axis );
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _REDUCEMAX_INTERNAL_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[SCALAR_INPUT_AXIS] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( reducemax_internal, _setup )

