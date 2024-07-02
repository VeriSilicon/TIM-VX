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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

#define _COL2IM_KERNEL_SOURCE_NAME      "col2im"

// Add kernel hashtable here
#define COL2IM_HASH_KEY( IN_DTYPE, OUT_DTYPE, _image_2d) \
        (( IN_DTYPE << 16 ) | ( OUT_DTYPE << 8 | (_image_2d)))
#define COL2IM_KERNELS( IN_DTYPE, OUT_DTYPE ) \
        { COL2IM_HASH_KEY( IN_DTYPE, OUT_DTYPE , 0), \
         CVIVANTE_NAMESPACE("cl.col2im_"#IN_DTYPE"to"#OUT_DTYPE), \
         _COL2IM_KERNEL_SOURCE_NAME }

#define COL2IM_KERNELS_2D( IN_DTYPE, OUT_DTYPE ) \
        { COL2IM_HASH_KEY( IN_DTYPE, OUT_DTYPE , 1), \
         CVIVANTE_NAMESPACE("cl.col2im_"#IN_DTYPE"to"#OUT_DTYPE"_2D"), \
         _COL2IM_KERNEL_SOURCE_NAME }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _col2im_kernel_map[] =
{
    // Register kernel here
    COL2IM_KERNELS( F32, F32 ),
    COL2IM_KERNELS( F32, U32 ),
    COL2IM_KERNELS( F32, I32 ),
    COL2IM_KERNELS( U32, U32 ),
    COL2IM_KERNELS( U32, F32 ),
    COL2IM_KERNELS( U32, I32 ),
    COL2IM_KERNELS( I32, I32 ),
    COL2IM_KERNELS( I32, U32 ),
    COL2IM_KERNELS( I32, F32 ),

    COL2IM_KERNELS_2D( F32, F32 ),
    COL2IM_KERNELS_2D( F32, U32 ),
    COL2IM_KERNELS_2D( F32, I32 ),
    COL2IM_KERNELS_2D( U32, U32 ),
    COL2IM_KERNELS_2D( U32, F32 ),
    COL2IM_KERNELS_2D( U32, I32 ),
    COL2IM_KERNELS_2D( I32, I32 ),
    COL2IM_KERNELS_2D( I32, U32 ),
    COL2IM_KERNELS_2D( I32, F32 ),
};


/*
 * Kernel params
 */
static vx_param_description_t _col2im_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _COL2IM_PARAM_NUM  _cnt_of_array( _col2im_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_col2im_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,         // workdim
        {0, 0, 0}, // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0}, // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0}, // localWorkSize: local group size in thread
        {0, 0, 0}  // globalWorkSize: image size in thread
        };
    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    vsi_size_array_t * in_shape = NULL;
    int32_t stride_w = 1, stride_h = 1;
    int32_t dilation_w = 1, dilation_h = 1, dilation_d = 1;
    int32_t pad_w_front = 0, pad_w_end = 0, pad_h_front = 0, pad_h_end = 0, pad_d_front = 0, pad_d_end = 0;
    int32_t kernel_w = 1, kernel_h = 1, kernel_d = 1;
    int32_t move_time_x = 0;
    int32_t move_time_y = 0;
    int32_t width_pad = 0;
    int32_t height_pad = 0;
    int32_t depth_pad = 0;
    int32_t kernel_x_new = 1;
    int32_t kernel_y_new = 1;
    int32_t kernel_z_new = 1;
    int32_t batch = 1;
    int32_t width = 1;
    int32_t height = 1;
    int32_t depth = 1;

    VSI_UNREFERENCED(param_size);
    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &stride_w);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &stride_h);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &dilation_w);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &dilation_h);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &dilation_d);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &pad_w_front);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[9], &pad_w_end);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[10], &pad_h_front);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[11], &pad_h_end);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[12], &pad_d_front);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[13], &pad_d_end);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[14], &kernel_w);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[15], &kernel_h);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[16], &kernel_d);
    CHECK_STATUS_FAIL_GOTO(status, final );

    batch = (int32_t)(attr[0]->shape->data[2]);
    width = (int32_t)(attr[1]->shape->data[0]);
    height = (int32_t)(attr[1]->shape->data[1]);
    depth  = (int32_t)(attr[1]->shape->data[2]) / batch;
    width_pad = width + pad_w_front + pad_w_end;
    height_pad = height + pad_h_front + pad_h_end;
    depth_pad = depth + pad_d_front + pad_d_end;
    move_time_x = (width_pad - ((kernel_w - 1) * dilation_w + 1) + stride_w) / stride_w;
    move_time_y = (height_pad - ((kernel_h - 1) * dilation_h + 1) + stride_h) / stride_h;
    kernel_x_new = (kernel_w - 1) * dilation_w + 1;
    kernel_y_new = (kernel_h - 1) * dilation_h + 1;
    kernel_z_new = (kernel_d - 1) * dilation_d + 1;

    status = vsi_nn_kernel_gpu_add_param( node, "width_pad", &width_pad );
    status |= vsi_nn_kernel_gpu_add_param( node, "height_pad", &height_pad );
    status |= vsi_nn_kernel_gpu_add_param( node, "depth_pad", &depth_pad );
    status |= vsi_nn_kernel_gpu_add_param( node, "move_time_x", &move_time_x );
    status |= vsi_nn_kernel_gpu_add_param( node, "move_time_y", &move_time_y );
    status |= vsi_nn_kernel_gpu_add_param( node, "kernel_x_new", &kernel_x_new );
    status |= vsi_nn_kernel_gpu_add_param( node, "kernel_y_new", &kernel_y_new );
    status |= vsi_nn_kernel_gpu_add_param( node, "kernel_z_new", &kernel_z_new );
    status |= vsi_nn_kernel_gpu_add_param( node, "depth", &depth );
    CHECK_STATUS_FAIL_GOTO(status, final );

    in_shape  = attr[1]->shape;

    gpu_param.global_scale[0] = 1;
    gpu_param.global_scale[1] = 1;
    gpu_param.global_scale[2] = 1;
    gpu_param.global_size[0] = in_shape->data[0];
    gpu_param.global_size[1] = in_shape->data[1];
    gpu_param.global_size[2] = in_shape->data[2];

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
} /* _col2im_initializer() */

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
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _col2im_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _col2im_kernel_map );
    vx_param_description_t * param_def  = _col2im_kernel_param_def;
    vx_kernel_initialize_f  initializer = _col2im_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (F16 == in_dtype)
    {
        in_dtype = F32;
    }
    else if (U8 == in_dtype)
    {
        in_dtype = U32;
    }
    else if (I8 == in_dtype || I16 == in_dtype)
    {
        in_dtype = I32;
    }

    if (F16 == out_dtype)
    {
        out_dtype = F32;
    }
    else if (U8 == out_dtype)
    {
        out_dtype = U32;
    }
    else if (I8 == out_dtype || I16 == out_dtype)
    {
        out_dtype = I32;
    }

    key = COL2IM_HASH_KEY( in_dtype, out_dtype ,image_2d);

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
        kernel->info.numParams   = _cnt_of_array( _col2im_kernel_param_def );
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "eltwise_ops_helper",
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
    vsi_nn_kernel_node_param_t node_params[_COL2IM_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    vsi_bool image_2d = FALSE;
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL;
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = {{0}};
    float inputScale = vsi_nn_get_tensor_scale(inputs[0]);
    float outputScale = vsi_nn_get_tensor_scale(outputs[0]);
    float inputZp  = (float)vsi_nn_get_tensor_zero_point(inputs[0]);
    float outputZp  = (float)vsi_nn_get_tensor_zero_point(outputs[0]);
    float inOutScale = inputScale / outputScale;
    float inOutTile = outputZp - inOutScale * inputZp;
    int32_t stride_w = vsi_nn_kernel_param_get_int32( params, "stride_w" );
    int32_t stride_h = vsi_nn_kernel_param_get_int32( params, "stride_h" );
    int32_t stride_d = vsi_nn_kernel_param_get_int32( params, "stride_d" );
    int32_t dilation_w = vsi_nn_kernel_param_get_int32( params, "dilation_w" );
    int32_t dilation_h = vsi_nn_kernel_param_get_int32( params, "dilation_h" );
    int32_t dilation_d = vsi_nn_kernel_param_get_int32( params, "dilation_d" );
    int32_t pad_w_front = vsi_nn_kernel_param_get_int32( params, "pad_w_front" );
    int32_t pad_w_end = vsi_nn_kernel_param_get_int32( params, "pad_w_end" );
    int32_t pad_h_front = vsi_nn_kernel_param_get_int32( params, "pad_h_front" );
    int32_t pad_h_end = vsi_nn_kernel_param_get_int32( params, "pad_h_end" );
    int32_t pad_d_front = vsi_nn_kernel_param_get_int32( params, "pad_d_front" );
    int32_t pad_d_end = vsi_nn_kernel_param_get_int32( params, "pad_d_end" );
    size_t dim_num = 0;
    int32_t* block_shape = (int32_t *) vsi_nn_kernel_param_get_buffer( params, "block_shape", &dim_num);
    int32_t kernel_w = block_shape[0];
    int32_t kernel_h = dim_num > 1 ? block_shape[1] : 1;
    int32_t kernel_d = dim_num > 2 ? block_shape[2] : 1;

    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    image_2d = dim_num > 2 ? FALSE : TRUE;

    shapes[0][0] = inputs[0]->attr.size[0];
    shapes[0][1] = inputs[0]->attr.size[1] / outputs[0]->attr.size[dim_num];
    shapes[0][2] = inputs[0]->attr.size[2] * outputs[0]->attr.size[dim_num];

    shapes[1][0] = outputs[0]->attr.size[0];
    shapes[1][1] = outputs[0]->attr.size[1];
    if (image_2d)
    {
        shapes[1][2] = outputs[0]->attr.size[2] * outputs[0]->attr.size[3];
    }
    else
    {
        shapes[1][2] = outputs[0]->attr.size[2] * outputs[0]->attr.size[3] * outputs[0]->attr.size[4];
    }

    rs_input = vsi_nn_kernel_tensor_reshape( inputs[0]->t, shapes[0], 3 );
    rs_output = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[1], 3 );

    if (rs_input == NULL || rs_output == NULL)
    {
        goto final;
    }

    status = _query_kernel( kernel, inputs, outputs, image_2d );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            node_params[0] = rs_input;
            node_params[1] = rs_output;
            node_params[2] = vsi_nn_kernel_scalar_create( graph, I32, &stride_w );
            node_params[3] = vsi_nn_kernel_scalar_create( graph, I32, &stride_h );
            node_params[4] = vsi_nn_kernel_scalar_create( graph, I32, &stride_d );
            node_params[5] = vsi_nn_kernel_scalar_create( graph, I32, &dilation_w );
            node_params[6] = vsi_nn_kernel_scalar_create( graph, I32, &dilation_h );
            node_params[7] = vsi_nn_kernel_scalar_create( graph, I32, &dilation_d );
            node_params[8] = vsi_nn_kernel_scalar_create( graph, I32, &pad_w_front );
            node_params[9] = vsi_nn_kernel_scalar_create( graph, I32, &pad_w_end );
            node_params[10] = vsi_nn_kernel_scalar_create( graph, I32, &pad_h_front );
            node_params[11] = vsi_nn_kernel_scalar_create( graph, I32, &pad_h_end );
            node_params[12] = vsi_nn_kernel_scalar_create( graph, I32, &pad_d_front );
            node_params[13] = vsi_nn_kernel_scalar_create( graph, I32, &pad_d_end );
            node_params[14] = vsi_nn_kernel_scalar_create( graph, I32, &kernel_w );
            node_params[15] = vsi_nn_kernel_scalar_create( graph, I32, &kernel_h );
            node_params[16] = vsi_nn_kernel_scalar_create( graph, I32, &kernel_d );
            node_params[17] = vsi_nn_kernel_scalar_create( graph, F32, &inOutScale );
            node_params[18] = vsi_nn_kernel_scalar_create( graph, F32, &inOutTile );

            status  = vsi_nn_kernel_node_pass_param( node, node_params, _COL2IM_PARAM_NUM );
            CHECK_STATUS(status);
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
            vsi_nn_kernel_scalar_release( &node_params[16] );
            vsi_nn_kernel_scalar_release( &node_params[17] );
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

REGISTER_BACKEND_CL( col2im, _setup )

