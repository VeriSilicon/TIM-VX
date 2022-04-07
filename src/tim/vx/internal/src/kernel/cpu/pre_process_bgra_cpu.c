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
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

#define _CPU_ARG_NUM            (10)
#define _CPU_INPUT_NUM          (1)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("cpu.pre_process_bgra_sw")

#define DESCALE(x) (((x) + (1<<19)) >> 20)

DEF_KERNEL_EXECUTOR(_pre_process_bgra_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VX_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    float * outBuffer = NULL;
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t i = 0;
    int32_t xRatio = 0, yRatio = 0, xOffset = 0, yOffset = 0;
    float rMean = 0, gMean = 0, bMean = 0, var = 0;
    int32_t order = 0, trans = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );

    i = 2;
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &xRatio);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &yRatio);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &xOffset);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &yOffset);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &rMean);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &gMean);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &bMean);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &var);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &order);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &trans);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    if(trans)
    {
        outBuffer = (float *)malloc( out_elements * sizeof(float) );
        CHECK_PTR_FAIL_GOTO( outBuffer, "Create output buffer fail.", final );
        memset( outBuffer, 0, out_elements * sizeof(float) );
    }

    {
        int32_t elementSize = 4;
        int32_t rline1[2], rline2[2];
        int32_t gline1[2], gline2[2];
        int32_t bline1[2], bline2[2];
        int32_t dx = 0, dy = 0, dz = 0;
        int32_t src_stride = (int32_t)attr[0]->shape->data[0];
        int32_t src_width = (int32_t)(src_stride / elementSize);
        int32_t src_height = (int32_t)attr[0]->shape->data[1];
        int32_t dst_width = (int32_t)(trans ? attr[1]->shape->data[1] : attr[1]->shape->data[0]);
        int32_t dst_height = (int32_t)(trans ? attr[1]->shape->data[2] : attr[1]->shape->data[1]);
        int32_t stride = (int32_t)(dst_width * dst_height);
        int32_t bOffset = 0;
        int32_t gOffset = 1 * stride;
        int32_t rOffset = 2 * stride;
        uint8_t R = 0, G = 0, B = 0;

        if(order)
        {
            bOffset = 2 * stride;
            rOffset = 0;
        }

        for ( dz = 0; dz < 1; dz ++)
        {
            for ( dy = 0; dy < (int32_t)dst_height; dy ++)
            {
                for ( dx = 0; dx < (int32_t)dst_width; dx ++)
                {
                    int32_t source_index = 0;
                    int32_t output_index = dx + dy * dst_width;
                    int32_t dstR_idx = output_index + rOffset;
                    int32_t dstG_idx = output_index + gOffset;
                    int32_t dstB_idx = output_index + bOffset;
                    float finalVal = 0;

                    if(xRatio != (1 << 15) || yRatio != (1 << 15))
                    {
                        int32_t fx = (dx * xRatio + (xRatio >> 1)) - (1 << 14);
                        int32_t sx = fx & 0xffff8000; // Floor
                        int32_t fy = 0, sy = 0;
                        int32_t temp1 = 0, temp2 = 0;

                        fx -= sx;
                        sx = sx >> 15;

                        sx = sx < 0 ? 0 : sx;
                        sx = sx > src_width ? src_width - 1: sx;

                        fx = (fx +(1 << 4)) >> 5;

                        // for y
                        fy = (dy * yRatio + (yRatio >> 1)) - (1<< 14);
                        sy = fy & 0xffff8000; // Floor
                        fy -= sy;
                        sy = sy >> 15;

                        sy = sy < 0 ? 0 : sy;
                        fy = fy < 0 ? 0 : fy;

                        fy = (fy + (1<< 4)) >> 5;

                        sx += xOffset;
                        sy += yOffset;
                        source_index = (sx + sy * src_width + dz * src_width * src_height) * elementSize;

                        bline1[0] = (int32_t)buffer[0][source_index];
                        bline1[1] = (int32_t)buffer[0][source_index + elementSize];
                        bline2[0] = (int32_t)buffer[0][source_index + src_stride];
                        bline2[1] = (int32_t)buffer[0][source_index + src_stride + elementSize];

                        gline1[0] = (int32_t)buffer[0][source_index + 1];
                        gline1[1] = (int32_t)buffer[0][source_index + elementSize + 1];
                        gline2[0] = (int32_t)buffer[0][source_index + src_stride + 1];
                        gline2[1] = (int32_t)buffer[0][source_index + src_stride + elementSize + 1];

                        rline1[0] = (int32_t)buffer[0][source_index + 2];
                        rline1[1] = (int32_t)buffer[0][source_index + elementSize + 2];
                        rline2[0] = (int32_t)buffer[0][source_index + src_stride + 2];
                        rline2[1] = (int32_t)buffer[0][source_index + src_stride + elementSize + 2];

                        // B
                        temp1 = fx * (bline1[1] - bline1[0]) + (bline1[0] << 10);
                        temp2 = fx * (bline2[1] - bline2[0]) + (bline2[0] << 10);
                        temp1 = fy * (temp2 - temp1) + (temp1 << 10);
                        B = (uint8_t)(DESCALE(temp1));
                        finalVal = (B - bMean) * var;
                        buffer[1][dstB_idx] = finalVal;

                        // R
                        temp1 = fx * (rline1[1] - rline1[0]) + (rline1[0] << 10);
                        temp2 = fx * (rline2[1] - rline2[0]) + (rline2[0] << 10);
                        temp1 = fy * (temp2 - temp1) + (temp1 << 10);
                        R = (uint8_t)(DESCALE(temp1));
                        finalVal = (R - rMean) * var;
                        buffer[1][dstR_idx] = finalVal;

                        // G
                        temp1 = fx * (gline1[1] - gline1[0]) + (gline1[0] << 10);
                        temp2 = fx * (gline2[1] - gline2[0]) + (gline2[0] << 10);
                        temp1 = fy * (temp2 - temp1) + (temp1 << 10);
                        G = (uint8_t)(DESCALE(temp1));
                        finalVal = (G - gMean) * var;
                        buffer[1][dstG_idx] = finalVal;
                    }
                    else //copy
                    {
                        int32_t offset = xOffset + yOffset * src_width;
                        source_index = (dx + dy * src_width + offset) * elementSize;

                        finalVal = (buffer[0][source_index] - bMean) * var;
                        buffer[1][dstB_idx] = finalVal;

                        finalVal = (buffer[0][source_index + 1] - gMean) * var;
                        buffer[1][dstG_idx] = finalVal;

                        finalVal = (buffer[0][source_index + 2] - rMean) * var;
                        buffer[1][dstR_idx] = finalVal;
                    }
                }
            }
        }
    }

    if(trans)
    {
        vsi_size_t shape[] = {attr[1]->shape->data[0], attr[1]->shape->data[1], attr[1]->shape->data[2], 1};
        vsi_size_t perm[] = {1, 2, 0, 3};
        vsi_nn_Transpose((uint8_t*)outBuffer, (uint8_t*)buffer[1],
                        shape, (uint32_t)attr[1]->shape->size, perm, VSI_NN_TYPE_FLOAT32);

        status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            outBuffer, out_elements );
    }
    else
    {
        status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
                buffer[1], out_elements );
    }
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    if(outBuffer)
    {
        free(outBuffer);
    }

    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
        }
        if(attr[i]) { vsi_nn_kernel_tensor_attr_release( &attr[i] ); }
    }
    return status;
} /* _pre_process_bgra_exec() */

static vx_param_description_t kernel_param_def[] =
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
};

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _pre_process_bgra_exec;
    kernel->info.parameters  = kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( kernel_param_def );

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
    vsi_status status = VX_FAILURE;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;

    status = _query_kernel( inputs, outputs, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            uint32_t index = 2;
            int32_t scale_x  = vsi_nn_kernel_param_get_int32( params, "scale_x" );
            int32_t scale_y  = vsi_nn_kernel_param_get_int32( params, "scale_y" );
            int32_t left     = vsi_nn_kernel_param_get_int32( params, "left" );
            int32_t top      = vsi_nn_kernel_param_get_int32( params, "top" );
            float r_mean     = vsi_nn_kernel_param_get_float32( params, "r_mean" );
            float g_mean     = vsi_nn_kernel_param_get_float32( params, "g_mean" );
            float b_mean     = vsi_nn_kernel_param_get_float32( params, "b_mean" );
            float bgra_scale  = vsi_nn_kernel_param_get_float32( params, "rgb_scale" );
            int32_t reverse  = vsi_nn_kernel_param_get_int32( params, "reverse" );
            int32_t trans    = vsi_nn_kernel_param_get_int32( params, "enable_perm" );

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );

            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_x );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_y );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &left );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &top );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &r_mean );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &g_mean );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &b_mean );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &bgra_scale );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &trans );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
            CHECK_STATUS( status );
            vsi_nn_kernel_scalar_release( &backend_params[2] );
            vsi_nn_kernel_scalar_release( &backend_params[3] );
            vsi_nn_kernel_scalar_release( &backend_params[4] );
            vsi_nn_kernel_scalar_release( &backend_params[5] );
            vsi_nn_kernel_scalar_release( &backend_params[6] );
            vsi_nn_kernel_scalar_release( &backend_params[7] );
            vsi_nn_kernel_scalar_release( &backend_params[8] );
            vsi_nn_kernel_scalar_release( &backend_params[9] );
            vsi_nn_kernel_scalar_release( &backend_params[10] );
            vsi_nn_kernel_scalar_release( &backend_params[11] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( pre_process_bgra, _setup )
