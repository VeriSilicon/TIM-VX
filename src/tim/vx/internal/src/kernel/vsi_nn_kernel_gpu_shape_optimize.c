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
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

static vsi_bool compute_gpu_divisor
    (
    const vsi_size_t input_value,
    const vsi_size_t limit,
    const int32_t gcd,
    vsi_size_t* divisor
    );

static vsi_size_t element_fill_dim
    (
    vsi_size_t* shape_x, vsi_size_t rank_x,
    vsi_size_t max_rank, vsi_size_t size_x
    );

static vsi_bool compute_gpu_divisor
    (
    const vsi_size_t input_value,
    const vsi_size_t limit,
    const int32_t gcd,
    vsi_size_t* divisor
    )
{
    vsi_size_t i = 0;
    for( i = vsi_nn_min( input_value, limit - 1 ); i > 0; i -- )
    {
        if ( ( i % gcd == 0 ) && ( input_value % i == 0 ) )
        {
            *divisor = i;
            return TRUE;
        }
    }
    return FALSE;
} /* compute_gpu_divisor */

static vsi_size_t element_fill_dim
    (
    vsi_size_t* shape_x, vsi_size_t rank_x,
    vsi_size_t max_rank, vsi_size_t size_x
    )
{
    vsi_size_t cost_size = 1;
    VSI_ASSERT( rank_x <= max_rank );

    if (size_x == 1)
        return 0;

    if ( size_x < GPU_TENSOR_MAX_WIDTH)
    {
        shape_x[rank_x] = size_x;
    }
    else
    {
        vsi_size_t divisor = 0;
        vsi_size_t remainder = 0;
        compute_gpu_divisor( size_x, GPU_TENSOR_MAX_WIDTH, 1, &divisor );
        remainder = size_x / divisor;
        if ( remainder > GPU_TENSOR_MAX_WIDTH || rank_x >= max_rank)
        {
            // Cannot optimize.
            shape_x[rank_x] = size_x;
        }
        else
        {
            /*
             * We've limit the max size to 2^32 -1(Almost 4G * sizeof(data type)),
             * so it should be always 2.
             */
            cost_size = 2;
            if ( size_x > 1 )
            {
                shape_x[rank_x]  = divisor;
                shape_x[rank_x + 1] = remainder;
            }
            else
            {
                shape_x[rank_x] = 1;
                shape_x[rank_x + 1] = 1;
            }
        }
    }
    return cost_size;
} /* element_fill_dim() */

/*only for continuous axises or one axis*/
vsi_bool vsi_nn_kernel_optimize_reduce_shape
    (
    const vsi_size_t* shape_x, const vsi_size_t rank_x,
    const int32_t *axis, const vsi_size_t axis_size,
    const vsi_size_t* shape_output, const vsi_size_t rank_output,
    vsi_size_t* out_shape_x, uint32_t* out_rank_x,
    vsi_size_t* out_shape_output, uint32_t* out_rank_output,
    int32_t* out_axis, uint32_t* out_axis_size
    )
{
    vsi_bool ret                        = TRUE;
    vsi_size_t   i                          = 0;
    vsi_size_t   rank_in                    = 0;
    vsi_size_t   rank_out                   = 0;
    vsi_size_t   dims                       = 0;
    vsi_size_t  innerSize                  = 1;
    vsi_size_t  outerSize                  = 1;
    vsi_size_t  axisSize                   = 1;

    for (i = 0; i < axis_size; i++)
    {
        axisSize *= shape_x[axis[i]];
    }

    for (i = 0; i < (size_t)axis[0]; i++)
    {
        innerSize *= shape_x[i];
    }

    for (i = axis[axis_size - 1] + 1; i < rank_x; i++)
    {
        outerSize *= shape_x[i];
    }

    rank_in += element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, innerSize);
    rank_out += element_fill_dim(out_shape_output, rank_out, GPU_TENSOR_MAX_WIDTH, innerSize);
    dims = element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, axisSize);
    if (dims == 0)
    {
        out_axis[0] = (int32_t)rank_in;
        *out_axis_size = 1;
        out_shape_x[rank_in ++] = 1;
    }
    else
    {
        *out_axis_size = (uint32_t)dims;
        for (i = 0; i < dims; i++)
        {
            out_axis[i] = (int32_t)rank_in + (int32_t)i;
        }
    }

    rank_in += dims;

    rank_in += element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, outerSize);
    rank_out += element_fill_dim(out_shape_output, rank_out, GPU_TENSOR_MAX_WIDTH, outerSize);

    if ( 0 == rank_in )
    {
        out_shape_x[0] = 1;
        out_shape_x[1] = 1;
        rank_in = 2;
    }
    else if ( 1 == rank_in )
    {
        out_shape_x[1] = 1;
        rank_in = 2;
    }

    if ( 0 == rank_out )
    {
        out_shape_output[0] = 1;
        out_shape_output[1] = 1;
        rank_out = 2;
    }
    else if ( 1 == rank_out )
    {
        out_shape_output[1] = 1;
        rank_out = 2;
    }

    *out_rank_x = (uint32_t)rank_in;
    *out_rank_output = (uint32_t)rank_out;

    return ret;
} /* vsi_nn_kernel_optimize_reduce_shape() */

vsi_bool vsi_nn_kernel_optimize_tensor_shape
    (
    const vsi_size_t* shape_x, const vsi_size_t rank_x,
    const int32_t *axis, const vsi_size_t axis_size,
    vsi_size_t* out_shape_x, uint32_t* out_rank_x,
    int32_t* out_axis, uint32_t* out_axis_size
    )
{
    vsi_bool ret                        = TRUE;
    vsi_size_t   i                          = 0;
    vsi_size_t   rank_in                    = 0;
    vsi_size_t   dims                       = 0;
    vsi_size_t  innerSize                  = 1;
    vsi_size_t  outerSize                  = 1;
    vsi_size_t  axisSize                   = 1;

    for (i = 0; i < axis_size; i++)
    {
        axisSize *= shape_x[axis[i]];
    }

    for (i = 0; i < (size_t)axis[0]; i++)
    {
        innerSize *= shape_x[i];
    }

    for (i = axis[axis_size - 1] + 1; i < rank_x; i++)
    {
        outerSize *= shape_x[i];
    }

    rank_in += element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, innerSize);
    dims = element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, axisSize);
    if (dims == 0)
    {
        out_axis[0] = (int32_t)rank_in;
        *out_axis_size = 1;
        out_shape_x[rank_in ++] = 1;
    }
    else
    {
        *out_axis_size = (uint32_t)dims;
        for (i = 0; i < dims; i++)
        {
            out_axis[i] = (int32_t)rank_in + (int32_t)i;
        }
    }

    rank_in += dims;

    rank_in += element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, outerSize);

    if ( 0 == rank_in )
    {
        out_shape_x[0] = 1;
        out_shape_x[1] = 1;
        rank_in = 2;
    }
    else if ( 1 == rank_in )
    {
        out_shape_x[1] = 1;
        rank_in = 2;
    }

    *out_rank_x = (uint32_t)rank_in;

    return ret;
} /* vsi_nn_kernel_optimize_reduce_shape() */

vsi_bool vsi_nn_kernel_optimize_element_shape
    (
    const vsi_size_t* shape_x, const vsi_size_t rank_x,
    vsi_size_t* out_shape_x, vsi_size_t* out_rank_x
    )
{
    vsi_bool ret                        = TRUE;
    uint32_t  i                         = 0;
    vsi_size_t   rank_in                    = 0;
    vsi_size_t  element_num                = 1;

    for (i = 0; i < rank_x; i++)
    {
        element_num *= shape_x[i];
    }

    rank_in += element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, element_num);

    if ( 0 == rank_in )
    {
        out_shape_x[0] = 1;
        out_shape_x[1] = 1;
        rank_in = 2;
    }
    else if ( 1 == rank_in )
    {
        out_shape_x[1] = 1;
        rank_in = 2;
    }

    *out_rank_x = (size_t)rank_in;

    return ret;
} /* vsi_nn_kernel_optimize_element_shape() */

vsi_bool vsi_nn_kernel_optimize_softmax_shape
    (
    const vsi_size_t* shape_x, const vsi_size_t rank_x, const int32_t axis,
    vsi_size_t* out_shape_x, uint32_t* out_rank_x,int32_t* out_axis
    )
{
    vsi_bool ret                        = TRUE;
    vsi_size_t   i                          = 0;
    vsi_size_t   rank_in                    = 0;
    vsi_size_t   dims                       = 0;
    vsi_size_t  innerSize                  = 1;
    vsi_size_t  outerSize                  = 1;
    vsi_size_t  axisSize                   = shape_x[axis];

    for (i = 0; i < (size_t)axis; i++)
    {
        innerSize *= shape_x[i];
    }

    for (i = axis + 1; i < rank_x; i++)
    {
        outerSize *= shape_x[i];
    }

    rank_in += element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, innerSize);
    dims = element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, axisSize);
    if (dims == 0)
    {
        *out_axis = (int32_t)rank_in;
        out_shape_x[rank_in ++] = 1;
    }
    else
    {
        *out_axis = (int32_t)rank_in;
    }

    rank_in += dims;

    rank_in += element_fill_dim(out_shape_x, rank_in, GPU_TENSOR_MAX_WIDTH, outerSize);

    if ( 0 == rank_in )
    {
        out_shape_x[0] = 1;
        out_shape_x[1] = 1;
        rank_in = 2;
    }
    else if ( 1 == rank_in )
    {
        out_shape_x[1] = 1;
        rank_in = 2;
    }

    *out_rank_x = (uint32_t)rank_in;

    return ret;
} /* vsi_nn_kernel_optimize_softmax_shape() */


typedef enum
{
    TILE_STATE_AXIS_X  = 0,
    TILE_STATE_AXIS_Y  = 1,
    TILE_STATE_AXIS_XY = 2,
    TILE_STATE_NO_AXIS = 4,
    TILE_STATE_EMPTY   = 8,
} tile_axis_state_e;

static vsi_size_t tile_fill_dim
    (
    vsi_size_t* shape_x, vsi_size_t* shape_y,
    vsi_size_t* shape_output, vsi_size_t rank,
    vsi_size_t max_rank, vsi_size_t size_x, vsi_size_t size_y,
    vsi_size_t size_output
    )
{
    vsi_size_t cost_size = 1;
    VSI_ASSERT( rank <= max_rank );
    if ( size_output < GPU_TENSOR_MAX_WIDTH )
    {
        shape_x[rank] = size_x;
        shape_y[rank] = size_y;
        shape_output[rank] = size_output;
    }
    else
    {
        vsi_size_t divisor = 0;
        vsi_size_t remainder = 0;
        compute_gpu_divisor( size_output, GPU_TENSOR_MAX_WIDTH, 1, &divisor );
        remainder = size_output / divisor;
        if ( remainder > GPU_TENSOR_MAX_WIDTH || rank >= max_rank )
        {
            // Cannot optimize.
            shape_x[rank] = size_x;
            shape_y[rank] = size_y;
            shape_output[rank] = size_output;
        }
        else
        {
            /*
             * We've limit the max size to 2^32 -1(Almost 4G * sizeof(data type)),
             * so it should be always 2.
             */
            cost_size = 2;
            if ( size_x > 1 )
            {
                shape_x[rank]  = divisor;
                shape_x[rank + 1] = remainder;
            }
            else
            {
                shape_x[rank] = 1;
                shape_x[rank + 1] = 1;
            }
            if ( size_y > 1 )
            {
                shape_y[rank]  = divisor;
                shape_y[rank + 1] = remainder;
            }
            else
            {
                shape_y[rank] = 1;
                shape_y[rank + 1] = 1;
            }
            shape_output[rank] = divisor;
            shape_output[rank + 1] = remainder;
        }
    }
    return cost_size;
} /* eltwise_fill_dim() */

vsi_bool vsi_nn_kernel_optimize_tile_shape
    (
    const vsi_size_t* shape_x,   const vsi_size_t rank_x,
    const vsi_size_t* multiples, const vsi_size_t rank,
    const vsi_size_t* shape_output, const vsi_size_t rank_output,
    vsi_size_t* out_shape_x, vsi_size_t* out_shape_y,
    vsi_size_t* out_shape_output, vsi_size_t* out_rank_output
    )
{
    vsi_bool ret                        = TRUE;
    vsi_bool append_dim                 = FALSE;
    vsi_size_t   i                          = 0;
    vsi_size_t   dims                       = 0;
    vsi_size_t  effective_size_x           = 1;
    vsi_size_t  effective_size_y           = 1;
    vsi_size_t  effective_size_z           = 1;
    vsi_size_t  sx                         = 0;
    vsi_size_t  sy                         = 0;
    vsi_size_t  sz                         = 0;
    tile_axis_state_e state             = TILE_STATE_EMPTY;
    tile_axis_state_e next_state        = TILE_STATE_EMPTY;

#define _swap_size(a, b, tmp)  \
    do { \
        tmp = a; \
        a = b; \
        b = tmp; \
    } while(0)
    for( i = 0; i < rank_output; i++ )
    {
        sx = shape_x[i];
        sy = multiples[i];
        sz = shape_output[i];
        /*
         * Skip dim if the size is equal to 1
         * Also skip if ( sx == 1 && sy == 1 )
         */
        if ( shape_output[i] == 1 )
        {
            continue;
        }

        // Update state
        state = TILE_STATE_EMPTY;
        if ( sx == sz )
        {
            state = TILE_STATE_NO_AXIS;
        }
        else if ( sx != sz )
        {
            state = TILE_STATE_AXIS_X;
        }
        else
        {
            VSI_ASSERT( FALSE );
        }

        next_state = (i + 1) < rank_output ?
            (multiples[i + 1] == 1 ? TILE_STATE_NO_AXIS : TILE_STATE_AXIS_X) : TILE_STATE_EMPTY;

        append_dim = FALSE;
#define _pack_state( cur_state, next_state )    (next_state << 16 | cur_state)
        switch( _pack_state( state, next_state ) )
        {
            case _pack_state( TILE_STATE_NO_AXIS, TILE_STATE_NO_AXIS ):
            case _pack_state( TILE_STATE_NO_AXIS, TILE_STATE_EMPTY ):
                effective_size_x *= sx;
                effective_size_y *= sy;
                effective_size_z *= sz;
                break;
            /*
             * ...,x1,x2,...
             * ...,y1,y2,...
             */
            case _pack_state( TILE_STATE_AXIS_X, TILE_STATE_AXIS_X ):
            case _pack_state( TILE_STATE_AXIS_X, TILE_STATE_NO_AXIS ):
            case _pack_state( TILE_STATE_AXIS_X, TILE_STATE_EMPTY ):
                append_dim = TRUE;
                break;
            /*
             * ...,x1, 1,...
             * ...,y1,y2,...
             *
             * ..., 1,x2,...
             * ...,y1,y2,...
             *
             */
            case _pack_state( TILE_STATE_NO_AXIS, TILE_STATE_AXIS_X ):
                effective_size_x *= sx;
                effective_size_y *= sy;
                effective_size_z *= sz;
                sx = effective_size_x;
                sy = effective_size_y;
                sz = effective_size_z;
                effective_size_x = 1;
                effective_size_y = 1;
                effective_size_z = 1;
                append_dim = TRUE;
                break;
            default:
                VSILOGE("Get error state (%d -> %d) while computing broadcast shape.",
                        state, next_state);
                VSI_ASSERT( FALSE );
                break;
        }
#undef _pack_state
        if ( append_dim )
        {
            dims += tile_fill_dim( out_shape_x, out_shape_y, out_shape_output,
                    dims, VSI_NN_MAX_DIM_NUM, sx, sy, sz );
        }
    }
    if ( ret )
    {
        /* Append the last dim */
        if ( i == rank_output )
        {
            sx = effective_size_x;
            sy = effective_size_y;
            sz = effective_size_z;
            dims += tile_fill_dim( out_shape_x, out_shape_y, out_shape_output,
                    dims, VSI_NN_MAX_DIM_NUM, sx, sy, sz );
        }
        /* Avoid 1D shape*/
        if ( 1 == dims )
        {
            out_shape_x[1] = 1;
            out_shape_y[1] = 1;
            out_shape_output[1] = 1;
            dims = 2;
        }
        /* For debug */
#if DEBUG
        vsi_nn_print_size_array( out_shape_x, dims );
        vsi_nn_print_size_array( out_shape_y, dims );
        vsi_nn_print_size_array( out_shape_output, dims );
#endif
        *out_rank_output = (uint32_t)dims;
    }
#undef _swap_size
    return ret;
} /* vsi_nn_kernel_optimize_eltwise_shape() */

vsi_bool vsi_nn_kernel_optimize_1d_tensor_shape
    (
    const vsi_size_t* shape, const uint32_t rank,
    vsi_size_t* out_shape, uint32_t* out_rank
    )
{
    memcpy(out_shape, shape, sizeof(vsi_size_t) * rank);
    *out_rank = vsi_nn_max(rank, 2);

    out_shape[1] = rank == 1 ? 1 : out_shape[1];

    return TRUE;
}

vsi_bool vsi_nn_kernel_optimize_nchw2xhw_shape
    (
    const vsi_size_t* shape, const uint32_t rank,
    vsi_size_t* out_shape, uint32_t* out_rank
    )
{
    uint32_t dim_num = 0;
    uint32_t i = 0;

    vsi_nn_kernel_optimize_1d_tensor_shape( shape,
        rank, out_shape, &dim_num);

    for (i = 3; i < dim_num; i++)
    {
        out_shape[2] *= out_shape[i];
    }

    *out_rank = vsi_nn_min(dim_num, 3);

    return TRUE;
}