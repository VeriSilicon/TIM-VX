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
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vsi_nn_vxkernel.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _CPU_ARG_NUM            (3)
#define _CPU_INPUT_NUM          (1)
#define _CPU_OUTPUT_NUM         (2)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("cpu.moments")

DEF_KERNEL_EXECUTOR(_moments_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t i = 0;
    int32_t axis_first = 0;
    int32_t axis_num = 0;
    uint32_t mask = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2]  = (vsi_nn_kernel_tensor_t)param[2];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &axis_first);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &axis_num);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_uint32((vsi_nn_kernel_scalar_t)param[5], &mask);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    buffer[2] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create output buffer fail.", final );
    memset( buffer[2], 0, out_elements * sizeof(float) );

    if(mask == 0)
    {
        int32_t  outerSize = 1;
        int32_t  axisSize  = 1;
        int32_t  innerSize = 1;
        int32_t  inner     = 0;
        int32_t  outer     = 0;

        for (i = 0; i < (uint32_t)axis_first; i++)
        {
            innerSize *= attr[0]->shape->data[i];
        }

        for(i = 0; i < (uint32_t)axis_num; i++)
        {
            axisSize *= attr[0]->shape->data[axis_first + i];
        }

        for (i = (uint32_t)axis_first + axis_num; i < attr[0]->shape->size; i++)
        {
            outerSize *= attr[0]->shape->data[i];
        }

        for ( outer = 0; outer < outerSize; ++outer)
        {
            for ( inner = 0; inner < innerSize; ++inner)
            {
                float sum = .0f;
                float sumsq = .0f;
                float mean = .0f;
                float vari = .0f;

                for (i = 0; i < (uint32_t)axisSize; ++i)
                {
                    float value = buffer[0][(outer * axisSize + i) * innerSize + inner];
                    sum += value;
                    sumsq += (value * value);
                }
                mean = sum / (axisSize);
                vari = sumsq / (axisSize) - mean * mean;
                buffer[1][outer * innerSize + inner] = (float)mean;
                buffer[2][outer * innerSize + inner] = (float)vari;
            }
        }
    }
    else
    {
        int32_t  width   = attr[0]->shape->data[0];
        int32_t  height  = attr[0]->shape->size > 1 ? attr[0]->shape->data[1] : 1;
        int32_t  channel = attr[0]->shape->size > 2 ? attr[0]->shape->data[2] : 1;
        int32_t  batch   = attr[0]->shape->size > 3 ? attr[0]->shape->data[3] : 1;
        int32_t  width_o = attr[1]->shape->data[0];
        int32_t  height_o  = attr[1]->shape->size > 1 ? attr[1]->shape->data[1] : 1;
        int32_t  channel_o = attr[1]->shape->size > 2 ? attr[1]->shape->data[2] : 1;
        int32_t b = 0, c = 0, h = 0;
        int32_t  wh_offset = width * height;
        int32_t  axisSize  = width * channel;
        int32_t  vol = width_o * height_o * channel_o;

        for(b = 0; b < batch; b++)
        {
            for(h = 0; h < height; h++)
            {
                float sum = .0f;
                float sumsq = .0f;
                float mean = .0f;
                float vari = .0f;
                int h_offset = h * width;
                for(c = 0; c < channel; c++)
                {
                    int offset = h_offset + c * wh_offset;
                    for(i = 0; i < (uint32_t)width; i++)
                    {
                        float value = buffer[0][i + offset];
                        sum += value;
                        sumsq += (value * value);
                    }
                }
                mean = sum / (axisSize);
                vari = sumsq / (axisSize) - mean * mean;
                buffer[1][b * vol + h] = (float)mean;
                buffer[2][b * vol + h] = (float)vari;
            }
        }
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            buffer[1], out_elements );
    status |= vsi_nn_kernel_tensor_write_from_float( tensors[2], attr[2],
            buffer[2], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
        }
    }
    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if(attr[i]) { vsi_nn_kernel_tensor_attr_release( &attr[i] ); }
    }
    return status;
} /* _pre_process_yuv420_exec() */
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
    // Add kererl parameters here
};
#define _MOMENTS_PARAM_NUM  _cnt_of_array( _moments_kernel_param_def )

static const vx_kernel_description_t _kernel_info =
{
    KERNEL_ID_PLACEHOLDER,
    _KERNEL_NAME,
    _moments_exec,
    _moments_kernel_param_def,
    _cnt_of_array( _moments_kernel_param_def ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    memmove( &kernel->info, &_kernel_info, sizeof(vx_kernel_description_t) );
    return VSI_SUCCESS;
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
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t axis_num  = 0;
    size_t axis_num_temp = 0;
    int32_t* axis = (int32_t *) vsi_nn_kernel_param_get_buffer( params, "axis", &axis_num_temp);
    vsi_bool is_continue_axis = TRUE;
    uint32_t mask = 0;
    int32_t i = 0;

    axis_num = (int32_t)axis_num_temp;

    for ( i = 1; i < axis_num; i++)
    {
        if ( axis[i] != (axis[i - 1] + 1) && axis[0] == 0)
        {
            is_continue_axis = FALSE;
            break;
        }
    }

    if (is_continue_axis == FALSE)
    {
        for(i = 0; i < axis_num; i++)
        {
            mask |= (1 << axis[i]);
        }
    }

    status = _query_kernel( inputs, outputs, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            int32_t axis_first  = axis[0];
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );
            backend_params[3] = vsi_nn_kernel_scalar_create( graph, I32, &axis_first );
            backend_params[4] = vsi_nn_kernel_scalar_create( graph, I32, &axis_num );
            backend_params[5] = vsi_nn_kernel_scalar_create( graph, U32, &mask );

            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
            CHECK_STATUS( status );
            vsi_nn_kernel_scalar_release( &backend_params[3] );
            vsi_nn_kernel_scalar_release( &backend_params[4] );
            vsi_nn_kernel_scalar_release( &backend_params[5] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( moments, _setup )

