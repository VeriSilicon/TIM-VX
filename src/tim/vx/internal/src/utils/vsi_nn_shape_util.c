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
#include "vsi_nn_log.h"
#include "utils/vsi_nn_shape_util.h"

void vsi_nn_shape_get_stride
    (
    const int32_t * shape,
    size_t rank,
    size_t * out_stride
    )
{
    uint32_t i;
    if( !shape || !out_stride )
    {
        return;
    }

    out_stride[0] = 1;
    for( i = 1; i < rank; i ++ )
    {
        out_stride[i] = shape[i - 1] * out_stride[i - 1];
    }
} /* vsi_nn_shape_get_stride() */

size_t vsi_nn_shape_get_size
    (
    const int32_t * shape,
    size_t rank
    )
{
    size_t size = 0;
    uint32_t i;
    if( !shape )
    {
        return size;
    }
    size = 1;
    for( i = 0; i < rank; i ++ )
    {
        if( shape[i] > 0 )
        {
            size *= shape[i];
        }
        else
        {
            VSILOGE("Got invalid dim: %d at %d.", shape[i], i);
            size = 0;
            break;
        }
    }
    return size;
} /* vsi_nn_shape_get_size() */

