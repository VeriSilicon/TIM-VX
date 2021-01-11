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
#include <limits.h>
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_error.h"
#include "vsi_nn_types.h"
#include "kernel/vsi_nn_kernel.h"

vsi_nn_kernel_tensor_t kernel_pad_node
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_tensor_t tensor,
    int32_t * pad_front,
    int32_t * pad_end,
    size_t pad_size,
    vsi_nn_pad_mode_e mode,
    int32_t pad_value,
    vsi_nn_kernel_node_t * out_node
    )
{
    vsi_nn_kernel_tensor_attr_t * attr = NULL;
    vsi_nn_kernel_tensor_t out_tensor = NULL;
    vsi_nn_kernel_node_t node = NULL;
    vx_nn_pad_params_t pad_param;
    int32_t i;

    // Compute pad size
    for( i = (int32_t)pad_size - 1; i >= 0; i -- )
    {
        if( pad_front[i] > 0 || pad_end[i] > 0 )
        {
            break;
        }
    }
    pad_size = (size_t)i + 1;
    if( pad_size > 2 )
    {
        VSILOGE("Not support pad size > 2.");
        return NULL;
    }
    else if( pad_size == 0 )
    {
        VSILOGE("No need to pad.");
        return NULL;
    }
    memset( &pad_param, 0, sizeof( pad_param ) );

    switch( mode )
    {
        case VSI_NN_PAD_MODE_CONSTANT:
            pad_param.pad_mode = VX_PAD_CONSTANT;
            break;
        case VSI_NN_PAD_MODE_REPLICATE:
            pad_param.pad_mode = VX_PAD_REPLICATE;
            break;
        case VSI_NN_PAD_MODE_SYMMETRIC:
            pad_param.pad_mode = VX_PAD_MIRROR_SYMMETRIC;
            break;
        case VSI_NN_PAD_MODE_REFLECT:
            pad_param.pad_mode = VX_PAD_MIRROR_REFLECT;
            break;
        default:
            VSILOGE("Wrong pad_mode %d", mode);
            break;
    }
    pad_param.pad_const = (vx_scalar)vsi_nn_kernel_scalar_create( graph, I32, &pad_value );
    pad_param.numViewDimensions = (vx_uint8)pad_size;
    pad_param.pad_front_array = pad_front;
    pad_param.pad_back_array = pad_end;

    attr = vsi_nn_kernel_tensor_attr_create( tensor );
    CHECK_PTR_FAIL_GOTO( attr, "Create tensor attr buffer fail.", final );
    // Compute new size
    if( pad_size > attr->shape->size )
    {
        VSILOGE("Pad size %lu is greater than tensor's rank %lu",
                pad_size, attr->shape->size );
        goto final;
    }
    for( i = 0; i < (int32_t)pad_size; i ++ )
    {
        attr->shape->data[i] += pad_front[i] + pad_end[i];
    }
    out_tensor = vsi_nn_kernel_tensor_create( graph->g, attr, TRUE );
    CHECK_PTR_FAIL_GOTO( out_tensor, "Create pad tensor fail.", final );
    node = (vsi_nn_kernel_node_t)vxTensorPadNode( graph->g,
            (vx_tensor)tensor, (vx_tensor)out_tensor,
            &pad_param, sizeof( pad_param ) );
final:
    if( NULL == node ) {
        VSILOGW("Create pad node fail.");
        if( out_tensor )
        {
            vsi_nn_kernel_tensor_release( &out_tensor );
        }
    }
    else
    {
        if( out_node )
        {
            *out_node = node;
        }
        else
        {
            vxReleaseNode( (vx_node*)&node );
        }
    }
    if( pad_param.pad_const )
    {
        vsi_nn_kernel_scalar_release( (vsi_nn_kernel_scalar_t*)&pad_param.pad_const );
    }
    if( attr )
    {
        vsi_nn_kernel_tensor_attr_release( &attr );
    }
    return out_tensor;
} /* kernel_pad_node() */

