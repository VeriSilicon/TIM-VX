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
#include "vsi_nn_error.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

typedef enum
{
    ELTWISE_BROADCAST_STATE_BROADCAST_X  = 0,
    ELTWISE_BROADCAST_STATE_BROADCAST_Y  = 1,
    ELTWISE_BROADCAST_STATE_BROADCAST_XY = 2,
    ELTWISE_BROADCAST_STATE_NO_BROADCAST = 4,
    ELTWISE_BROADCAST_STATE_EMPTY        = 8,
} eltwise_broadcast_state_e;

#if 0
static size_t vsi_nn_compute_element_num
    ( const int32_t* shape, const size_t rank);
#endif

static vsi_size_t eltwise_fill_dim
    (
    vsi_size_t* shape_x, vsi_size_t* shape_y,
    vsi_size_t* shape_output, vsi_size_t rank,
    vsi_size_t max_rank, vsi_size_t size_x, vsi_size_t size_y,
    vsi_size_t size_output
    );

static vsi_bool compute_gpu_divisor
    (
    const vsi_size_t input_value,
    const vsi_size_t limit,
    const int32_t gcd,
    vsi_size_t* divisor
    );

#if 0
static size_t vsi_nn_compute_element_num
    ( const int32_t* shape, const size_t rank)
{
    size_t i;
    size_t element = 1;
    for( i = 0; i < rank; i ++ )
    {
        element *= shape[i];
    }
    return element;
}
#endif

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
        if( ( i % gcd == 0 ) && ( input_value % i == 0 ) )
        {
            *divisor = i;
            return TRUE;
        }
    }
    return FALSE;
} /* compute_gpu_divisor */

static vsi_size_t eltwise_fill_dim
    (
    vsi_size_t* shape_x, vsi_size_t* shape_y,
    vsi_size_t* shape_output, vsi_size_t rank,
    vsi_size_t max_rank, vsi_size_t size_x, vsi_size_t size_y,
    vsi_size_t size_output
    )
{
    vsi_size_t cost_size = 1;
    VSI_ASSERT( rank <= max_rank );
    if( size_output < GPU_TENSOR_MAX_WIDTH )
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
        if( remainder > GPU_TENSOR_MAX_WIDTH || rank >= max_rank )
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
            if( size_x > 1 )
            {
                shape_x[rank]  = divisor;
                shape_x[rank + 1] = remainder;
            }
            else
            {
                shape_x[rank] = 1;
                shape_x[rank + 1] = 1;
            }
            if( size_y > 1 )
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

vsi_bool vsi_nn_kernel_optimize_eltwise_shape
    (
    const vsi_size_t* shape_x, const vsi_size_t rank_x,
    const vsi_size_t* shape_y, const vsi_size_t rank_y,
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
    vsi_size_t  tmp_sz                     = 0;
    vsi_size_t  sx                         = 0;
    vsi_size_t  sy                         = 0;
    eltwise_broadcast_state_e state     = ELTWISE_BROADCAST_STATE_EMPTY;
    eltwise_broadcast_state_e prv_state = ELTWISE_BROADCAST_STATE_EMPTY;

#define _swap_size(a, b, tmp)  \
    do { \
        tmp = a; \
        a = b; \
        b = tmp; \
    } while(0)
    for( i = 0; i < rank_output; i++ )
    {
        sx = i < rank_x ? shape_x[i] : 1;
        sy = i < rank_y ? shape_y[i] : 1;

        /*
         * Skip dim if the size is equal to 1
         * Also skip if( sx == 1 && sy == 1 )
         */
        if( shape_output[i] == 1 )
        {
            continue;
        }
        // Invalid shape for broadcasting
        if( sx != sy && sx > 1 && sy > 1 )
        {
            ret = FALSE;
            break;
        }
        // Update state
        state = ELTWISE_BROADCAST_STATE_EMPTY;
        if( sx == sy )
        {
            state = ELTWISE_BROADCAST_STATE_NO_BROADCAST;
        }
        else if( sx == 1 )
        {
            state = ELTWISE_BROADCAST_STATE_BROADCAST_X;
        }
        else if( sy == 1 )
        {
            state = ELTWISE_BROADCAST_STATE_BROADCAST_Y;
        }
        else
        {
            VSI_ASSERT( FALSE );
        }
        if( prv_state == ELTWISE_BROADCAST_STATE_EMPTY )
        {
            effective_size_x *= sx;
            effective_size_y *= sy;
            prv_state = state;
            continue;
        }
        append_dim = FALSE;
#define _pack_state( prev_state, cur_state )    (prev_state << 16 | cur_state)
        switch( _pack_state( prv_state, state ) )
        {
            /*
             * ...,x1,x2,...
             * ...,y1,y2,...
             */
            case _pack_state( ELTWISE_BROADCAST_STATE_NO_BROADCAST, ELTWISE_BROADCAST_STATE_NO_BROADCAST ):
                effective_size_x *= sx;
                effective_size_y *= sy;
                break;
            /*
             * ..., 1, 1,...
             * ...,y1,y2,...
             */
            case _pack_state( ELTWISE_BROADCAST_STATE_BROADCAST_X, ELTWISE_BROADCAST_STATE_BROADCAST_X ):
                effective_size_y *= sy;
                break;
            /*
             * ...,x1,x2,...
             * ..., 1, 1,...
             */
            case _pack_state( ELTWISE_BROADCAST_STATE_BROADCAST_Y, ELTWISE_BROADCAST_STATE_BROADCAST_Y ):
                effective_size_x *= sx;
                break;

            /*
             * ...,x1, 1,...
             * ...,y1,y2,...
             *
             * ...,x1,x2,...
             * ...,y1, 1,...
             *
             * ..., 1,x2,...
             * ...,y1, 1,...
             *
             * ..., 1,x2,...
             * ...,y1,y2,...
             *
             * ...,x1, 1,...
             * ..., 1,y2,...
             *
             * ...,x1,x2,...
             * ..., 1,y2,...
             */
            case _pack_state( ELTWISE_BROADCAST_STATE_NO_BROADCAST, ELTWISE_BROADCAST_STATE_BROADCAST_X ):
            case _pack_state( ELTWISE_BROADCAST_STATE_NO_BROADCAST, ELTWISE_BROADCAST_STATE_BROADCAST_Y ):
            case _pack_state( ELTWISE_BROADCAST_STATE_BROADCAST_X, ELTWISE_BROADCAST_STATE_BROADCAST_Y ):
            case _pack_state( ELTWISE_BROADCAST_STATE_BROADCAST_X, ELTWISE_BROADCAST_STATE_NO_BROADCAST ):
            case _pack_state( ELTWISE_BROADCAST_STATE_BROADCAST_Y, ELTWISE_BROADCAST_STATE_BROADCAST_X ):
            case _pack_state( ELTWISE_BROADCAST_STATE_BROADCAST_Y, ELTWISE_BROADCAST_STATE_NO_BROADCAST ):
                _swap_size(sx, effective_size_x, tmp_sz);
                _swap_size(sy, effective_size_y, tmp_sz);
                append_dim = TRUE;
                break;
            default:
                VSILOGE("Get error state (%d -> %d) while computing broadcast shape.",
                        prv_state, state);
                VSI_ASSERT( FALSE );
                break;
        }
#undef _pack_state
        prv_state = state;
        if( append_dim )
        {
            dims += eltwise_fill_dim( out_shape_x, out_shape_y, out_shape_output,
                    dims, VSI_NN_MAX_DIM_NUM, sx, sy, vsi_nn_max( sx, sy ) );
        }
    }
    if( ret )
    {
        /* Append the last dim */
        if( i == rank_output )
        {
            sx = effective_size_x;
            sy = effective_size_y;
            dims += eltwise_fill_dim( out_shape_x, out_shape_y, out_shape_output,
                    dims, VSI_NN_MAX_DIM_NUM, sx, sy, vsi_nn_max( sx, sy ) );
        }
        /* Avoid 1D shape*/
        if( 1 == dims )
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
        *out_rank_output = (size_t)dims;
    }
#undef _swap_size
    return ret;
} /* vsi_nn_kernel_optimize_eltwise_shape() */



static vsi_size_t broadcast_fill_dim
    (
    vsi_size_t** shape_in, int32_t input_num,
    vsi_size_t* shape_output, vsi_size_t rank,
    vsi_size_t max_rank, vsi_size_t* size_in,
    vsi_size_t size_output
    )
{
    int32_t i         = 0;
    vsi_size_t cost_size = 1;
    VSI_ASSERT( rank <= max_rank );
    if( size_output < GPU_TENSOR_MAX_WIDTH )
    {
        for (i = 0; i < input_num; i++)
        {
            shape_in[i][rank] = size_in[i];
        }
        shape_output[rank] = size_output;
    }
    else
    {
        vsi_size_t divisor = 0;
        vsi_size_t remainder = 0;
        compute_gpu_divisor( size_output, GPU_TENSOR_MAX_WIDTH, 1, &divisor );
        remainder = size_output / divisor;
        if( remainder > GPU_TENSOR_MAX_WIDTH || rank >= max_rank )
        {
            // Cannot optimize.
            for (i = 0; i < input_num; i++)
            {
                shape_in[i][rank] = size_in[i];
            }
            shape_output[rank] = size_output;
        }
        else
        {
            /*
             * We've limit the max size to 2^32 -1(Almost 4G * sizeof(data type)),
             * so it should be always 2.
             */
            cost_size = 2;
            for (i = 0; i < input_num; i++)
            {
                if (size_in[i] > 1)
                {
                    shape_in[i][rank]     = divisor;
                    shape_in[i][rank + 1] = remainder;
                }
                else
                {
                    shape_in[i][rank]     = 1;
                    shape_in[i][rank + 1] = 1;
                }
            }
            shape_output[rank] = divisor;
            shape_output[rank + 1] = remainder;
        }
    }
    return cost_size;
} /* broadcast_fill_dim() */

vsi_bool vsi_nn_kernel_optimize_broadcast_shape
    (
    const vsi_size_t** shape_in, const vsi_size_t* rank_in,
    const int32_t   input_num,
    const vsi_size_t*  shape_output, const vsi_size_t rank_output,
    vsi_size_t** out_shape_in,
    vsi_size_t* out_shape_output, uint32_t* out_rank_output
    )
{
#define MAX_INPUT_NUM    30
    vsi_bool ret                              = TRUE;
    vsi_bool append_dim                       = FALSE;
    vsi_size_t   i                                = 0;
    vsi_size_t   j                                = 0;
    vsi_size_t   k                                = 0;
    vsi_size_t   dims                             = 0;
    vsi_size_t  effective_size[MAX_INPUT_NUM]    = {1};
    vsi_size_t  tmp_sz                           = 0;
    vsi_size_t  size_in[MAX_INPUT_NUM]           = {0};
    int32_t  state_mask                       = 0;
    int32_t  prv_state_mask                   = -1;

#define _swap_size(a, b, tmp)  \
    do { \
        tmp = a; \
        a = b; \
        b = tmp; \
    } while(0)

    if (input_num > MAX_INPUT_NUM)
    {
        VSILOGE("Max support input num is %d, while input num is %d.",
                MAX_INPUT_NUM, input_num);
        ret = FALSE;
        goto final;
    }

    for (i = 0; i < (vsi_size_t)input_num; i++)
    {
        effective_size[i] = 1;
    }

    for( i = 0; i < rank_output; i++ )
    {
        for (j = 0; j < (vsi_size_t)input_num; j++)
        {
            size_in[j] = i < rank_in[j] ? shape_in[j][i] : 1;
        }
        /*
         * Skip dim if the size is equal to 1
         */
        if( shape_output[i] == 1 )
        {
            continue;
        }

        // Invalid shape for broadcasting
        k = 0;
        for (j = 0; j < (vsi_size_t)input_num; j++)
        {
            if (size_in[j] > 1)
            {
                k = j;
                break;
            }
        }

        for (j = 0; j < (vsi_size_t)input_num; j++)
        {
            if ((size_in[k] != size_in[j])
             && (size_in[j] > 1))
            {
                ret = FALSE;
                goto final;
            }
        }

        state_mask = 0;
        for (j = 0; j < (vsi_size_t)input_num; j++)
        {
            if (1 == size_in[j])
            {
                state_mask |= (1 << j);
            }
        }

        append_dim = FALSE;

        if ((-1 == prv_state_mask) || (state_mask == prv_state_mask))
        {
            for (j = 0; j < (vsi_size_t)input_num; j++)
            {
                effective_size[j] *= size_in[j];
            }
        }
        else
        {
            for (j = 0; j < (vsi_size_t)input_num; j++)
            {
                _swap_size(size_in[j], effective_size[j], tmp_sz);
            }
            append_dim = TRUE;
        }

        prv_state_mask = state_mask;

        if( append_dim )
        {
            vsi_size_t size_output;
            size_output = size_in[0];
            for (j = 1; j < (vsi_size_t)input_num; j++)
            {
                size_output = vsi_nn_max(size_output, size_in[j]);
            }
            dims += broadcast_fill_dim(out_shape_in, input_num, out_shape_output,
                    dims, VSI_NN_MAX_DIM_NUM, size_in, size_output);
        }
    }

    if( ret )
    {
        /* Append the last dim */
        if( i == rank_output )
        {
            vsi_size_t size_output;
            size_output = effective_size[0];
            for (j = 1; j < (size_t)input_num; j++)
            {
                size_output = vsi_nn_max(size_output, effective_size[j]);
            }
            dims += broadcast_fill_dim(out_shape_in, input_num, out_shape_output,
                    dims, VSI_NN_MAX_DIM_NUM, effective_size, size_output);
        }
        /* Avoid 1D shape*/
        if( 1 == dims )
        {
            for (j = 0; j < (vsi_size_t)input_num; j++)
            {
                out_shape_in[j][1] = 1;
            }
            out_shape_output[1] = 1;
            dims = 2;
        }
        else
        {
            for (j = 0; j < (vsi_size_t)input_num; j++)
            {
                for ( i = 0; i < dims; i++)
                {
                    if ( out_shape_in[j][i] == 0 )
                        out_shape_in[j][i] = 1;
                }
            }
        }

        *out_rank_output = (uint32_t)dims;
    }

#undef _swap_size
#undef MAX_INPUT_NUM
final:
    return ret;
} /* vsi_nn_kernel_optimize_broadcast_shape() */