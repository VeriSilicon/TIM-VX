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
#ifndef _VSI_NN_MATH_H
#define _VSI_NN_MATH_H

#include <math.h>
#include <stdlib.h>
#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define vsi_nn_abs(x)               (((x) < 0)    ? -(x) :  (x))
#define vsi_nn_max(a,b)             (((a) > (b)) ? (a) : (b))
#define vsi_nn_min(a,b)             (((a) < (b)) ? (a) : (b))
#define vsi_nn_clamp(x, min, max)   (((x) < (min)) ? (min) : \
                                 ((x) > (max)) ? (max) : (x))
#define vsi_nn_float_compare(a,b,diff) (vsi_nn_abs((a) - (b)) < (diff) ? TRUE : FALSE)
#define vsi_abs(x)                  vsi_nn_abs(x)
#define vsi_clamp(x, min, max)      vsi_nn_clamp(x, min, max)
#define vsi_rtne(x)                 vsi_rint(x)

#define VSI_NN_FLOAT32_INF          (0x7F800000)
#define VSI_NN_FLOAT32_NAN          (0x7FC00000)
#define VSI_NN_FLOAT64_INF          (0x7FF0000000000000)
#define VSI_NN_FLOAT64_NAN          (0x7FF8000000000000)


#define DEFINE_ARRAY_TYPE( NAME, TYPE ) \
    typedef struct { \
        size_t size; \
        TYPE data[0]; \
    } vsi_##NAME##_array_t; \
    static inline vsi_##NAME##_array_t * vsi_##NAME##_array_create( size_t size ) { \
        vsi_##NAME##_array_t * array = (vsi_##NAME##_array_t *)malloc( \
                sizeof(vsi_##NAME##_array_t) + sizeof(TYPE) * size ); \
        array->size = size; \
        return array; \
    } \
    static inline void vsi_##NAME##_array_release( vsi_##NAME##_array_t ** array ) \
        { \
            if( array && *array ) { \
                free( *array ); \
                *array = NULL; \
            } \
        }
DEFINE_ARRAY_TYPE( int, int32_t )
DEFINE_ARRAY_TYPE( float, float )

#undef DEFINE_ARRAY_TYPE

OVXLIB_API void vsi_nn_Transpose
    (
    uint8_t  * dst,
    uint8_t  * data,
    uint32_t * shape,
    uint32_t   dim_num,
    uint32_t * perm,
    vsi_nn_type_e type
    );

OVXLIB_API void vsi_nn_Permute
    (
    uint8_t  * dst,
    uint8_t  * data,
    uint32_t * shape,
    uint32_t   dim_num,
    uint32_t * perm,
    vsi_nn_type_e type
    );

OVXLIB_API void vsi_nn_SqueezeShape
    (
    uint32_t * shape,
    uint32_t * dim_num
    );

OVXLIB_API uint32_t vsi_nn_ShapeProduct
    (
    uint32_t * shape,
    uint32_t   dim_num
    );

//shape: row first <--> column first
OVXLIB_API void vsi_nn_InvertShape
    (
    uint32_t * in,
    uint32_t   dim_num,
    uint32_t * out
    );

//Permute shape: row first <--> column first
OVXLIB_API void vsi_nn_InvertPermuteShape
    (
    uint32_t * in,
    uint32_t   dim_num,
    uint32_t * out
    );

OVXLIB_API double vsi_nn_Rint
    (
    double x
    );

/**
* Set Seeds for philox_4x32_10 algorithm
* philox_4x32_10 algorithm need 2 uint32_t as seeds.
*
* @param[in] the low uint32_t of the seed.
* @param[in] the high uint32_t of the seed.
*/
OVXLIB_API void vsi_nn_random_init_for_philox_4x32_10
    (
    uint32_t low,
    uint32_t high
    );

/**
* Random Number Generator By philox_4x32_10 algorithm
* Random Number(uint32_t) Generator By philox_4x32_10 algorithm
*
* @param[out] the buffer for RNG output.
* @param[in] the number of generated random numbers.
*/
OVXLIB_API void vsi_nn_random_generate_by_philox_4x32_10
    (
    uint32_t *random_buf,
    uint32_t len
    );

/**
* Uniform Transform
* Transform the random uint32_t to Uniform float in [0, 1).
*
* @param[in] the buffer for random uint32_t.
* @param[out] the buffer for uniform float in [0, 1).
* @param[in] the number of random numbers.
*/
OVXLIB_API void vsi_nn_random_uniform_transform
    (
    uint32_t *random_buf,
    float *uniform_buf,
    uint32_t len
    );

static inline double copy_sign
    (
    double number,
    double sign
    )
{
    double value = vsi_nn_abs(number);
    return (sign > 0) ? value : (-value);
} /* copy_sign() */

static inline float simple_round
    (
    float x
    )
{
    return (float) copy_sign(floorf(fabsf(x) + 0.5f), x);
} /* simple_round() */

static inline double vsi_rint
    (
    double x
    )
{
#define _EPSILON 1e-8
    double decimal;
    double inter;

    decimal = modf((double)x, &inter);
    if( vsi_nn_abs((vsi_nn_abs(decimal) - 0.5f)) < _EPSILON )
    {
        inter += (int32_t)(inter) % 2;
    }
    else
    {
        return simple_round( (float)x );
    }
    return inter;
} /* vsi_rint() */

#ifdef __cplusplus
}
#endif

#endif
