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
#include <string.h>
#include <math.h>
#include "vsi_nn_tensor.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_map.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

static void _compute_stride
    (
    uint32_t * shape,
    uint32_t   dim_num,
    uint32_t * stride
    );

static void _compute_stride
    (
    uint32_t * shape,
    uint32_t   dim_num,
    uint32_t * stride
    )
{
    int i;
    uint32_t s;
    s = 1;
    for( i = dim_num - 1; i >= 0; i -- )
    {
        stride[i] = s;
        s *= shape[i];
    }
} /* _compute_stride() */

void vsi_nn_Transpose
    (
    uint8_t  * dst,
    uint8_t  * data,
    uint32_t * shape,
    uint32_t   dim_num,
    uint32_t * perm,
    vsi_nn_type_e type
    )
{
    uint32_t i;
    uint32_t i_dst;
    uint32_t i_org;
    uint32_t i_t;
    uint32_t size;
    uint32_t unit_bytes;
    uint32_t org_stride[VSI_NN_MAX_DIM_NUM];
    uint32_t dst_stride[VSI_NN_MAX_DIM_NUM];
    uint32_t dst_shape[VSI_NN_MAX_DIM_NUM];

    if( NULL == data || NULL == dst || NULL == shape || NULL == perm
        || 0 == dim_num || dim_num > VSI_NN_MAX_DIM_NUM )
    {
        return;
    }
    if( 1 == dim_num )
    {
        VSILOGW( "Transpose error, incorrect dim %d", dim_num );
        return;
    }
    for( i = 0; i < dim_num; i ++ )
    {
        if( perm[i] >= dim_num )
        {
            VSILOGW( "Incorrect perm %d", perm[i] );
            return;
        }
        dst_shape[i] = shape[perm[i]];
    }
    unit_bytes = vsi_nn_GetTypeBytes( type );
    _compute_stride( shape, dim_num, org_stride );
    _compute_stride( dst_shape, dim_num, dst_stride );
    size = vsi_nn_ShapeProduct( shape, dim_num );
    for( i_dst = 0; i_dst < size; i_dst ++ )
    {
        i_org = 0;
        i_t = i_dst;
        for( i = 0; i < dim_num; i ++ )
        {
            i_org += ( i_t / dst_stride[i] ) * org_stride[perm[i]];
            i_t %= dst_stride[i];
        }
        memcpy( &dst[i_dst * unit_bytes], &data[i_org * unit_bytes], unit_bytes );
        //dst[i_dst] = data[i_org];
    }
} /* vsi_nn_Transpose() */

void vsi_nn_Permute
    (
    uint8_t  * dst,
    uint8_t  * data,
    uint32_t * shape,
    uint32_t   dim_num,
    uint32_t * perm,
    vsi_nn_type_e type
    )
{
    uint32_t unit_bytes, i;
    uint32_t org_stride[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t dst_stride[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t dst_shape[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t dim_stack[VSI_NN_MAX_DIM_NUM] = {0};
    uint8_t * in_addr_stack[VSI_NN_MAX_DIM_NUM] = {NULL};
    uint8_t * out_addr_stack[VSI_NN_MAX_DIM_NUM] = {NULL};
    uint8_t * in_addr_tmp = NULL;
    uint8_t * out_addr_tmp = NULL;
    uint32_t current = 0;
    vsi_bool back = FALSE;
    uint32_t layer = dim_num - 1;

    if( NULL == data || NULL == dst || NULL == shape || NULL == perm
        || 0 == dim_num || dim_num > VSI_NN_MAX_DIM_NUM )
    {
        return;
    }
    if( 1 == dim_num )
    {
        VSILOGW( "Permute error, incorrect dim %d", dim_num );
        return;
    }

    for( i = 0; i < dim_num; i ++ )
    {
        if( perm[i] >= dim_num )
        {
            VSILOGW( "Incorrect perm %d", perm[i] );
            return;
        }
        dst_shape[i] = shape[perm[i]];
    }
    unit_bytes = vsi_nn_GetTypeBytes( type );
    vsi_nn_GetStrideSizeBySize( shape, dim_num, type, org_stride );
    vsi_nn_GetStrideSizeBySize( dst_shape, dim_num, type, dst_stride );

    in_addr_tmp = data;
    out_addr_tmp = dst;

    for (;;)
    {
        in_addr_stack[current] = in_addr_tmp;
        out_addr_stack[current] = out_addr_tmp;

        if (layer == 1)
        {
            uint32_t x, y;
            uint8_t* new_out_addr = out_addr_tmp;
            for (y = 0; y < shape[perm[1]]; y++)
            {
                for (x = 0; x < shape[perm[0]]; x++)
                {
                    uint8_t* new_in_addr = in_addr_tmp + (y * org_stride[perm[1]] + x * org_stride[perm[0]]);
                    memcpy(new_out_addr, new_in_addr, unit_bytes);
                    new_out_addr += unit_bytes;
                }
            }

            if (!current) break;
            current--;
            layer++;
            back = TRUE;
        }
        else if (!back)
        {
            current++;
            layer--;
        }
        else
        {
            dim_stack[current]++;
            if (dim_stack[current] < shape[perm[layer]])
            {
                in_addr_tmp += org_stride[perm[layer]];
                out_addr_tmp += dst_stride[layer];
                back = FALSE;
            }
            else
            {
                dim_stack[current] = 0;
                if (!current) break;
                current--;
                layer++;
                in_addr_tmp = in_addr_stack[current];
                out_addr_tmp = out_addr_stack[current];
            }
        }
    }
} /* vsi_nn_Permute() */

void vsi_nn_SqueezeShape
    (
    uint32_t * shape,
    uint32_t * dim_num
    )
{
    int i;
    int origin_count;
    int count;
    int start;
    count = *dim_num;
    origin_count = count;
    if( 1 == count )
    {
        return;
    }
    start = 0;
    for( i = 0; i < count; i ++ )
    {
        if( 1 == shape[i] )
        {
            continue;
        }
        else if( i > start )
        {
            memmove( &shape[start], &shape[i], (count - i) * sizeof( uint32_t ) );
            count -= i - start;
            start += i - start;
        }
        else
        {
            start = i + 1;
        }
    }
    *dim_num = count;
    memset( &shape[count], 0, sizeof( uint32_t ) * ( origin_count - count ) );
} /* vsi_nn_SqueezeShape() */

uint32_t vsi_nn_ShapeProduct
    (
    uint32_t * shape,
    uint32_t   dim_num
    )
{
    uint32_t i;
    uint32_t res;
    res = 1;
    for ( i = 0; i < dim_num; i++ )
    {
        res *= shape[i];
    }
    return res;
} /* vsi_nn_ShapeProduct() */

void vsi_nn_InvertShape
    (
    uint32_t * in,
    uint32_t   dim_num,
    uint32_t * out
    )
{
    uint32_t i;
    for ( i = 0; i < dim_num; i++ )
    {
        out[i] = in[dim_num - 1 - i];
    }
} /* vsi_nn_InvertShape() */

void vsi_nn_InvertPermuteShape
    (
    uint32_t * in,
    uint32_t   dim_num,
    uint32_t * out
    )
{
    uint32_t i;
    for ( i = 0; i < dim_num; i++ )
    {
        out[in[i]] = i;
    }
} /* vsi_nn_InvertPermuteShape() */

double vsi_nn_Rint
    (
    double x
    )
{
    return vsi_rint(x);
} /* vsi_nn_Rint() */

// Implement the Philox algorithm to generate random numbers in parallel.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
//   http://www.thesalmons.org/john/random123/papers/random123sc11.pdf

// This source code only implement philox_4x32_10 algorithm.
// ---------------------philox_4x32_10 algorithm beginning-------------
#ifndef PHILOX_W32_0
#define PHILOX_W32_0 ((uint32_t)0x9E3779B9)
#endif
#ifndef PHILOX_W32_1
#define PHILOX_W32_1 ((uint32_t)0xBB67AE85)
#endif

#ifndef PHILOX_M4x32_0
#define PHILOX_M4x32_0 ((uint32_t)0xD2511F53)
#endif
#ifndef PHILOX_M4x32_1
#define PHILOX_M4x32_1 ((uint32_t)0xCD9E8D57)
#endif

struct r123array4x32{
    uint32_t v[4];
};

struct r123array2x32 {
    uint32_t v[2];
};

typedef struct r123array4x32 philox4x32_ctr_t;
typedef struct r123array2x32 philox4x32_key_t;
typedef struct r123array2x32 philox4x32_ukey_t;

uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* hip)
{
    uint64_t product = ((uint64_t)a)*((uint64_t)b);
    *hip = product>>32;
    return (uint32_t)product;
}

philox4x32_key_t philox4x32keyinit(philox4x32_ukey_t uk)
{
    return uk;
}

struct r123array2x32 _philox4x32bumpkey(struct r123array2x32 key)
{
    key.v[0] += PHILOX_W32_0;
    key.v[1] += PHILOX_W32_1;
    return key;
}

struct r123array4x32 _philox4x32round(struct r123array4x32 ctr, struct r123array2x32 key)
{
    uint32_t hi0;
    uint32_t hi1;
    uint32_t lo0 = mulhilo32(PHILOX_M4x32_0, ctr.v[0], &hi0);
    uint32_t lo1 = mulhilo32(PHILOX_M4x32_1, ctr.v[2], &hi1);
    struct r123array4x32 out = {{hi1^ctr.v[1]^key.v[0], lo1,
                              hi0^ctr.v[3]^key.v[1], lo0}};
    return out;
}

philox4x32_ctr_t philox4x32_R(uint32_t R, philox4x32_ctr_t ctr, philox4x32_key_t key)
{
    uint32_t i;
    for (i = 0; i < R; i++)
    {
        if (i != 0)
        {
            key = _philox4x32bumpkey(key);
        }
        ctr = _philox4x32round(ctr, key);
    }
    return ctr;
}

philox4x32_ctr_t g_ctr;
philox4x32_key_t g_key;

void vsi_nn_random_init_for_philox_4x32_10
    (
    uint32_t low,
    uint32_t high
    )
{
    philox4x32_ukey_t uk;
    uk.v[0] = low;
    uk.v[1] = high;
    g_key = philox4x32keyinit(uk);
}

void vsi_nn_random_generate_by_philox_4x32_10
    (
    uint32_t *random_buf,
    uint32_t len
    )
{
    uint32_t i;
    for (i = 0; i < len / 4; i++)
    {
        g_ctr = philox4x32_R(10, g_ctr, g_key);
        memcpy(&(random_buf[i * 4]), &g_ctr, 4 * sizeof(uint32_t));
    }
    i = len % 4;
    if (i)
    {
        g_ctr = philox4x32_R(10, g_ctr, g_key);
        memcpy(&(random_buf[(len / 4) * 4]), &g_ctr, i * sizeof(uint32_t));
    }
}
// ---------------------philox_4x32_10 algorithm end-------------------

void vsi_nn_random_uniform_transform
    (
    uint32_t *random_buf,
    float *uniform_buf,
    uint32_t len
    )
{
    float rand_max = (float)(pow(2.0,32));
    uint32_t i;
    for (i = 0; i < len; i++)
    {
        uniform_buf[i] = random_buf[i] / rand_max;
    }
}
