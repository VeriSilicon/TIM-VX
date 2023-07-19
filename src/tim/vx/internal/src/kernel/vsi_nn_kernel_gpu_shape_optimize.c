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
#include "kernel/vsi_nn_kernel_eltwise.h"

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

    if ( size_x < max_rank)
    {
        shape_x[rank_x] = size_x;
    }
    else
    {
        vsi_size_t divisor = 0;
        vsi_size_t remainder = 0;
        compute_gpu_divisor( size_x, max_rank, 1, &divisor );
        if (divisor == 0)
        {
            VSILOGE( "divisor might be used in a division by zero." );
            cost_size =  (vsi_size_t)-1;
            goto final;
        }
        remainder = size_x / divisor;
        if ( remainder > max_rank || rank_x >= max_rank)
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
final:
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

    VSI_UNREFERENCED(shape_output);
    VSI_UNREFERENCED(rank_output);

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
        if (divisor == 0)
        {
            VSILOGE( "divisor might be used in a division by zero." );
            cost_size =  (vsi_size_t)-1;
            goto final;
        }
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
final:
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
    vsi_bool    ret                        = TRUE;
    vsi_bool    append_dim                 = FALSE;
    vsi_size_t  i                          = 0;
    vsi_size_t  j                          = 0;
    vsi_size_t  dims                       = 0;
    vsi_size_t  effective_size_x           = 1;
    vsi_size_t  effective_size_y           = 1;
    vsi_size_t  effective_size_z           = 1;
    vsi_size_t  sx                         = 0;
    vsi_size_t  sy                         = 0;
    vsi_size_t  sz                         = 0;
    int32_t     idx_start                  = -1;
    int32_t     idx_end                    = 0;
    tile_axis_state_e state             = TILE_STATE_EMPTY;
    tile_axis_state_e next_state        = TILE_STATE_EMPTY;
    vsi_size_t* temp_shape_x            = NULL;
    vsi_size_t* temp_shape_y            = NULL;
    vsi_size_t* temp_shape_output       = NULL;
    vsi_size_t  temp_rank               = 0;

#define _swap_size(a, b, tmp)  \
    { \
        tmp = a; \
        a = b; \
        b = tmp; \
    }

    VSI_UNREFERENCED(rank_x);
    VSI_UNREFERENCED(rank);

    temp_shape_x = (vsi_size_t*)malloc(rank * sizeof(vsi_size_t));
    if (temp_shape_x == NULL)
    {
        VSILOGE( "malloc temp_shape_x error." );
        ret = FALSE;
        goto final;
    }

    temp_shape_y = (vsi_size_t*)malloc(rank * sizeof(vsi_size_t));
    if (temp_shape_y == NULL)
    {
        VSILOGE( "malloc temp_shape_y error." );
        ret = FALSE;
        goto final;
    }

    temp_shape_output = (vsi_size_t*)malloc(rank * sizeof(vsi_size_t));
    if (temp_shape_output == NULL)
    {
        VSILOGE( "malloc temp_shape_output error." );
        ret = FALSE;
        goto final;
    }
    memcpy(temp_shape_x, shape_x, rank * sizeof(vsi_size_t));
    memcpy(temp_shape_y, multiples, rank * sizeof(vsi_size_t));
    memcpy(temp_shape_output, shape_output, rank * sizeof(vsi_size_t));

    for (i = 0, temp_rank = 0; i < rank_output; i++)
    {
        if (i == rank_output - 1 && temp_shape_x[i] == 1)
        {
            if (idx_start >= 0)
            {
               sx = 1;
               sy = temp_shape_y[idx_start];
               sz = temp_shape_output[idx_start];
               idx_end = (int32_t)i ;
               for (j = (vsi_size_t)idx_start + 1; j <= (vsi_size_t)idx_end; j++)
               {
                   sy *= temp_shape_y[j];
                   sz *= temp_shape_output[j];
               }
               temp_rank += tile_fill_dim( temp_shape_x, temp_shape_y, temp_shape_output,
                       temp_rank, VSI_NN_MAX_DIM_NUM, sx, sy, sz );
               idx_start = -1;
            }
            else
            {
                temp_shape_x[temp_rank] = temp_shape_x[i];
                temp_shape_y[temp_rank] = temp_shape_y[i];
                temp_shape_output[temp_rank++] = temp_shape_output[i];
            }
        }
        else if (temp_shape_x[i] != 1)
        {
            idx_end = (int32_t)i - 1;
            if (idx_start >= 0)
            {
               sx = 1;
               sy = temp_shape_y[idx_start];
               sz = temp_shape_output[idx_start];
               for (j = (vsi_size_t)idx_start + 1; j <= (vsi_size_t)idx_end; j++)
               {
                   sy *= temp_shape_y[j];
                   sz *= temp_shape_output[j];
               }
               temp_rank += tile_fill_dim( temp_shape_x, temp_shape_y, temp_shape_output,
                       temp_rank, VSI_NN_MAX_DIM_NUM, sx, sy, sz );
               idx_start = -1;
            }
            temp_shape_x[temp_rank] = temp_shape_x[i];
            temp_shape_y[temp_rank] = temp_shape_y[i];
            temp_shape_output[temp_rank++] = temp_shape_output[i];
        }
        else if (idx_start == -1)
        {
            idx_start = (int32_t)i;
        }
    }

    for( i = 0; i < temp_rank; i++ )
    {
        sx = temp_shape_x[i];
        sy = temp_shape_y[i];
        sz = temp_shape_output[i];
        /*
         * Skip dim if the size is equal to 1
         * Also skip if ( sx == 1 && sy == 1 )
         */
        if ( temp_shape_output[i] == 1 )
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

        next_state = (i + 1) < temp_rank ?
            (temp_shape_y[i + 1] == 1 ? TILE_STATE_NO_AXIS : TILE_STATE_AXIS_X) : TILE_STATE_EMPTY;

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
            case _pack_state( TILE_STATE_AXIS_X, TILE_STATE_EMPTY ):
                effective_size_x = sx;
                effective_size_y = sy;
                effective_size_z = sz;
                break;
            case _pack_state( TILE_STATE_AXIS_X, TILE_STATE_AXIS_X ):
            case _pack_state( TILE_STATE_AXIS_X, TILE_STATE_NO_AXIS ):
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
        if ( i == temp_rank )
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
final:
    if (temp_shape_x)
    {
        free( temp_shape_x);
        temp_shape_x = NULL;
    }
    if (temp_shape_y)
    {
        free( temp_shape_y);
        temp_shape_y = NULL;
    }
    if (temp_shape_output)
    {
        free( temp_shape_output);
        temp_shape_output = NULL;
    }

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

vsi_bool vsi_nn_kernel_optimize_element_shape_with_max_rank
    (
    const vsi_size_t* shape_x, const vsi_size_t rank_x,
    vsi_size_t* out_shape_x, vsi_size_t* out_rank_x, vsi_size_t max_rank
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

    rank_in += element_fill_dim(out_shape_x, rank_in, max_rank, element_num);

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

vsi_bool vsi_nn_kernel_optimize_group_norm_shape
    (
    const vsi_size_t* shape, const uint32_t rank, int32_t groups,
    int32_t is_sp_kernel, vsi_size_t* out_shape
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t i = 0;
    vsi_size_t out_rank = 0;
    vsi_size_t group_shape[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t max_rank = GPU_TENSOR_MAX_WIDTH;
    group_shape[0] = shape[0];
    group_shape[1] = shape[1];
    group_shape[2] = shape[2] / groups;

#define NN_INPUT_SIZE_MAX      ((1 << 13) - 1)
    if (is_sp_kernel)
    {
        max_rank = NN_INPUT_SIZE_MAX;
    }
#undef NN_INPUT_SIZE_MAX

    vsi_nn_kernel_optimize_element_shape_with_max_rank( group_shape, 3,
        out_shape, &out_rank, max_rank);

    if (!is_sp_kernel && out_shape[1] == 1 && out_rank < 3)
    {
        out_shape[1] = groups;
        out_shape[2] = 1;
        out_shape[3] = 1;
        for (i = 3; i < rank; i++)
        {
            out_shape[3] = out_shape[3] * shape[i];
        }
    }
    else if (out_rank == 2)
    {
        out_shape[2] = groups;
        out_shape[3] = 1;
        for (i = 3; i < rank; i++)
        {
            out_shape[3] = out_shape[3] * shape[i];
        }
    }
    else
    {
        status = VSI_FAILURE;
    }

    return status;
}

vsi_bool vsi_nn_kernel_optimize_scatter_elements_shape
    (
    const vsi_size_t* shape_x, const vsi_size_t rank_x, const int32_t axis,
    vsi_size_t* out_shape_x, uint32_t* out_rank_x, int32_t* out_axis, vsi_size_t max_size
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

    rank_in += element_fill_dim(out_shape_x, rank_in, max_size, innerSize);
    dims = element_fill_dim(out_shape_x, rank_in, max_size, axisSize);
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

    rank_in += element_fill_dim(out_shape_x, rank_in, max_size, outerSize);

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
} /* vsi_nn_kernel_optimize_scatter_elements_shape() */


vsi_bool vsi_nn_kernel_optimize_matrixmul_broadcast_shape
    (
    const vsi_size_t * shape_x,
    const vsi_size_t * shape_y,
    const vsi_size_t * shape_output,
    vsi_size_t dim_x,
    vsi_size_t dim_y,
    vsi_size_t dim_out,
    vsi_size_t* out_shape_x,
    vsi_size_t* out_shape_y,
    vsi_size_t* out_shape_output,
    uint32_t* new_rank_out,
    uint32_t* cross_flg,
    uint32_t* size_axis_inner_outer,
    uint32_t* strides_axis_inner_outer
    )
{
    vsi_bool     ret = FALSE;
    vsi_size_t   rank_in[2] = {0, 0};
    vsi_size_t   rank_out = 0;
    vsi_size_t   shapes_in_broadcast_part[2][VSI_NN_MAX_DIM_NUM] = {{1}};
    vsi_size_t*  shapes_in_broadcast_part_ptr[2]                 = {NULL, NULL};
    vsi_size_t   shapes_out_broadcast_part[VSI_NN_MAX_DIM_NUM]   = {1};
    vsi_size_t   out_shape_in[2][VSI_NN_MAX_DIM_NUM]             = {{1}};
    vsi_size_t*  out_shape_in_ptr[2]                             = {NULL, NULL};
    vsi_size_t   out_shape_boradcast_output[VSI_NN_MAX_DIM_NUM]  = {1};
    uint32_t     new_rank   = 0;
    uint32_t     i          = 0;
    vsi_size_t   outer0 = 1;
    vsi_size_t   outer1 = 1;
    vsi_size_t   outer2 = 1;
    vsi_size_t   axis_size = 0;
    vsi_size_t   inner_size = 1;
    vsi_size_t   outer_size = 1;
    vsi_size_t   axis_size0 = 1;
    vsi_size_t   axis_size1 = 1;
    vsi_size_t   axis_size2 = 1;
    vsi_size_t   inner_size0 = 0;
    vsi_size_t   inner_size1 = 0;
    vsi_size_t   inner_size2 = 0;
    vsi_size_t   outer_size0 = 0;
    vsi_size_t   outer_size1 = 0;
    vsi_size_t   outer_size2 = 0;
    uint32_t ne_flg = 0;
    uint32_t axis = 0;
    uint32_t outer_flg = 0;
    uint32_t outer_axis = 0;
    uint32_t first_flg = 0;
    cross_flg[0] = 0;

    if (dim_x > 2 && dim_y > 2)
    {
        for (i = 2; i < dim_x; i++)
        {
            outer0 *= shape_x[i];
        }
        for (i = 2; i < dim_y; i++)
        {
            outer1 *= shape_y[i];
        }
        for (i = 2; i < dim_out; i++)
        {
            outer2 *= shape_output[i];
        }

        for (i = 2; i < vsi_nn_min(dim_x, dim_y); i++)
        {
            if (shape_x[i] != shape_y[i] && first_flg == 0)
            {
                if (shape_x[i] == 1)
                {
                    ne_flg = 1;
                    inner_size = shape_y[i];
                }
                else
                {
                    ne_flg = 2;
                    inner_size = shape_x[i];
                }
                first_flg = 1;
                continue;
            }
            else if (ne_flg == 1 && shape_x[i] != shape_y[i] && shape_x[i] == 1 && first_flg == 1)
            {
                inner_size *= shape_y[i];
            }
            else if (ne_flg == 2 && shape_x[i] != shape_y[i] && shape_y[i] == 1 && first_flg == 1)
            {
                inner_size *= shape_x[i];
            }
            else if (ne_flg == 1 && shape_x[i] != shape_y[i] && shape_x[i] != 1 && first_flg == 1)
            {
                outer_flg = 1;
                outer_axis = i;
                break;
            }
            else if (ne_flg == 2 && shape_x[i] != shape_y[i] && shape_y[i] != 1 && first_flg == 1)
            {
                outer_flg = 2;
                outer_axis = i;
                break;
            }
            else if (i > 2 && shape_x[i] == shape_y[i] && shape_y[i] != 1 && first_flg == 1)
            {
                first_flg = 2;
            }
            else if (shape_x[i] != shape_y[i] && shape_x[i] != 1 && first_flg == 2)
            {
                outer_flg = 1;
                outer_axis = i;
                break;
            }
            else if (shape_x[i] != shape_y[i] && shape_y[i] != 1 && first_flg == 2)
            {
                outer_flg = 2;
                outer_axis = i;
                break;
            }
            else if (i == 2 && shape_x[i] == shape_y[i] && shape_y[i] != 1)
            {
                /*axis = 2;
                axis_size = shape_x[i];*/
            }
        }

        if (ne_flg > 0 && outer0 > 1 && outer1 > 1)
        {
            for (i = 2; i < vsi_nn_min(dim_x, dim_y); i++)
            {
                if (shape_x[i] == shape_y[i] && shape_x[i] != 1)
                {
                    cross_flg[0] = 1;
                    axis = i;
                    axis_size = shape_x[i];
                    break;
                }
            }
        }

        if (cross_flg[0] == 1) // cross
        {
            if (outer_flg == 1)
            {
                for (i = outer_axis; i < dim_x; i++)
                {
                    outer_size *= shape_x[i];
                }
            }
            else if (outer_flg == 2)
            {
                for (i = outer_axis; i < dim_y; i++)
                {
                    outer_size *= shape_y[i];
                }
            }
            else
            {
                outer_size = 1;
            }

            axis_size0 = 1;
            axis_size1 = 1;
            axis_size2 = 1;
            if (axis > 2 && ne_flg == 1)
            {
                axis_size1 = inner_size;
                axis_size2 = inner_size;
            }
            else if (axis > 2 && ne_flg == 2)
            {
                axis_size0 = inner_size;
                axis_size2 = inner_size;
            }

            inner_size0 = 0;
            inner_size1 = 0;
            inner_size2 = 1;
            if (axis == 2 && ne_flg == 1)
            {
                inner_size1 = axis_size;
                inner_size2 = axis_size;
            }
            else if (axis > 2 && ne_flg == 1)
            {
                inner_size1 = 1;
            }
            else if (axis == 2 && ne_flg == 2)
            {
                inner_size0 = axis_size;
                inner_size2 = axis_size;
            }
            else if (axis > 2 && ne_flg == 2)
            {
                inner_size0 = 1;
            }

            outer_size0 = 0;
            outer_size1 = 0;
            outer_size2 = axis_size * inner_size;
            if (outer_flg == 1)
            {
                outer_size0 = axis_size0 * axis_size;
            }
            else if (outer_flg == 2)
            {
                outer_size1 = axis_size1 * axis_size;
            }

            for (i = 0; i < 2; i++)
            {
                out_shape_x[i] = shape_x[i];
                out_shape_y[i] = shape_y[i];
                out_shape_output[i] = shape_output[i];
            }
            out_shape_x[2] = outer0;
            out_shape_x[3] = 1;
            out_shape_y[2] = outer1;
            out_shape_output[2] = outer2;
            new_rank_out[0] = 4;
            new_rank_out[1] = 3;
            new_rank_out[2] = 3;

            size_axis_inner_outer[0] = (uint32_t)axis_size;
            size_axis_inner_outer[1] = (uint32_t)inner_size;
            size_axis_inner_outer[2] = (uint32_t)outer_size;

            strides_axis_inner_outer[0] = (uint32_t)axis_size0;
            strides_axis_inner_outer[1] = (uint32_t)inner_size0;
            strides_axis_inner_outer[2] = (uint32_t)outer_size0;

            strides_axis_inner_outer[3] = (uint32_t)axis_size1;
            strides_axis_inner_outer[4] = (uint32_t)inner_size1;
            strides_axis_inner_outer[5] = (uint32_t)outer_size1;

            strides_axis_inner_outer[6] = (uint32_t)axis_size2;
            strides_axis_inner_outer[7] = (uint32_t)inner_size2;
            strides_axis_inner_outer[8] = (uint32_t)outer_size2;

            return TRUE;
        }
        else if (outer0 > 1 && outer1 > 1 && ne_flg > 0 && cross_flg[0] == 0)
        {
            cross_flg[0] = 2;
        }
    }

    if (cross_flg[0] == 2) // merge
    {
        for (i = 0; i < 2; i++)
        {
            out_shape_x[i] = shape_x[i];
            out_shape_y[i] = shape_y[i];
            out_shape_output[i] = shape_output[i];
        }
        out_shape_output[2] = outer2;
        new_rank_out[2] = 3;
        if (ne_flg == 1)
        {
            out_shape_x[2] = outer0;
            out_shape_x[3] = 1;
            out_shape_y[2] = outer1;

            new_rank_out[0] = 4;
            new_rank_out[1] = 3;
        }
        else if (ne_flg == 2)
        {
            out_shape_x[2] = outer0;
            out_shape_y[2] = outer1;
            out_shape_y[3] = 1;

            new_rank_out[0] = 3;
            new_rank_out[1] = 4;
        }

        return TRUE;
    }
    else if (dim_x == 1 && dim_y > 1)
    {
        out_shape_x[0]    = shape_x[0];
        out_shape_x[1]    = 1;

        out_shape_y[0]    = shape_y[0];
        out_shape_y[1]    = shape_y[1];

        out_shape_output[0]   = shape_output[0];
        out_shape_output[1]   = 1;

        if (dim_y > 2)
        {
            shapes_in_broadcast_part[0][0] = 1;
            rank_in[0] = 1;

            for (i = 2; i <= dim_y; i++)
            {
                shapes_in_broadcast_part[1][i - 2] = shape_y[i];
            }
            rank_in[1] = dim_y - 2;

            for(i = 1; i <= dim_out; i++)
            {
                shapes_out_broadcast_part[i - 1] = shape_output[i];
            }
            rank_out = dim_out - 1;
        }
    }
    else if (dim_y == 1 && dim_x > 1)
    {
        out_shape_y[0]    = 1;
        out_shape_y[1]    = shape_y[0];

        out_shape_x[0]    = shape_x[0];
        out_shape_x[1]    = shape_x[1];

        out_shape_output[0]   = 1;
        out_shape_output[1]   = shape_output[0];

        if (dim_x > 2)
        {
            shapes_in_broadcast_part[1][0] = 1;
            rank_in[1] = 1;

            for (i = 2; i <= dim_x; i++)
            {
                shapes_in_broadcast_part[0][i - 2] = shape_x[i];
            }
            rank_in[0] = dim_x - 2;

            for(i = 1; i <= dim_out; i++)
            {
                shapes_out_broadcast_part[i - 1] = shape_output[i];
            }
            rank_out = dim_out - 1;
        }
    }
    else
    {
        out_shape_x[0]    = shape_x[0];
        out_shape_x[1]    = shape_x[1];

        out_shape_y[0]    = shape_y[0];
        out_shape_y[1]    = shape_y[1];

        out_shape_output[0]    = shape_output[0];
        out_shape_output[1]    = shape_output[1];

        for (i = 2; i < dim_x; i++)
        {
            shapes_in_broadcast_part[0][i - 2] = shape_x[i];
        }
        for (i = 2; i < dim_y; i++)
        {
            shapes_in_broadcast_part[1][i - 2] = shape_y[i];
        }
        for (i = 2; i < dim_out; i++)
        {
            shapes_out_broadcast_part[i - 2] = shape_output[i];
        }
        rank_in[0] = dim_x - 2;
        rank_in[1] = dim_y - 2;
        rank_out = dim_out - 2;

    }

    shapes_in_broadcast_part_ptr[0] = shapes_in_broadcast_part[0];
    shapes_in_broadcast_part_ptr[1] = shapes_in_broadcast_part[1];
    out_shape_in_ptr[0] = out_shape_in[0];
    out_shape_in_ptr[1] = out_shape_in[1];

    ret = vsi_nn_kernel_optimize_broadcast_shape(
            (const vsi_size_t **)shapes_in_broadcast_part_ptr, rank_in, 2,
            shapes_out_broadcast_part, rank_out,
            (vsi_size_t **)out_shape_in_ptr, out_shape_boradcast_output, &new_rank);

    if (ret)
    {
        int32_t j = 0;

        new_rank_out[0] = new_rank + 2;
        new_rank_out[1] = new_rank + 2;
        new_rank_out[2] = new_rank + 2;

        j = new_rank - 1;
        while (out_shape_in[0][j] == 1 && j >= 0) {
            new_rank_out[0]--;
            j--;
        }

        j = new_rank - 1;
        while (out_shape_in[1][j] == 1 && j >= 0) {
            new_rank_out[1]--;
            j--;
        }

        j = new_rank - 1;
        while (out_shape_boradcast_output[j] == 1 && j >= 0) {
            new_rank_out[2]--;
            j--;
        }

        for (i = 0; i < new_rank; i++)
        {
            out_shape_x[i + 2] = out_shape_in[0][i];
            out_shape_y[i + 2] = out_shape_in[1][i];
            out_shape_output[i + 2] = out_shape_boradcast_output[i];
        }
    }

    return ret;
}
