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

#include <stdio.h>
#include <stdint.h>
#include "kernel/vsi_nn_gpu.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_error.h"

void gpu_dp_inst_update_postshfit
    (
    gpu_dp_inst_t * dp_inst,
    int32_t shift
    )
{
    if( !dp_inst )
    {
        return;
    }
    VSI_ASSERT( dp_inst->type == GPU_DP_TYPE_16 );
    if( shift < 0 )
    {
        const uint32_t multiplier = gpu_multiplier( 1 << (-shift) );
        gpu_dp_inst_update_multiplier( dp_inst, 0, 8, multiplier );
    }
    else
    {
        const int32_t index = 7;
        const uint8_t postshift = (uint8_t)gpu_postshift( shift );
        // clear postshift
        dp_inst->data[index] &= ~((uint32_t)0x1F);
        // set postshift
        dp_inst->data[index] |= (postshift & 0x1F);
    }
} /* gpu_dp_inst_update_postshfit() */

void gpu_dp_inst_update_multiplier
    (
    gpu_dp_inst_t * dp_inst,
    int32_t start,
    int32_t end,
    int32_t multiplier
    )
{
    const int32_t multiplier_pos = 8;
    const int32_t start_pos = multiplier_pos + start;
    const int32_t end_pos = multiplier_pos + end;
    int32_t i;

    for( i = start_pos; i < end_pos; i ++ )
    {
        dp_inst->data[i] = multiplier;
    }
}

void gpu_quantize_multiplier_32bit
    (
    double double_multiplier,
    uint32_t * quantize_multiplier,
    int32_t * shift
    )
{
    double q;
    int64_t q_fixed;
    const int32_t bit = 32;
    if( vsi_abs(double_multiplier - 0.0) < 1e-5 )
    {
        *quantize_multiplier = 0;
        *shift = bit - 0;
    }
    else
    {
        q = frexp( double_multiplier, shift );
        q_fixed = (int64_t)(vsi_rint(q * (1ll << 31)));
        VSI_ASSERT( q_fixed <= (1ll << 31) );
        if( q_fixed == (1ll << 31) )
        {
            q_fixed /= 2;
            *shift += 1;
        }
        if( *shift < -31 )
        {
            *shift = 0;
            q_fixed = 0;
        }
        *quantize_multiplier = (uint32_t)q_fixed;
    }
    if( 0 == *quantize_multiplier )
    {
        *shift = 0;
    }
    else
    {
        *shift = bit - *shift;
    }
} /* gpu_quantize_multiplier_32_bit() */

void gpu_quantize_multiplier_16bit
    (
    double double_multiplier,
    uint16_t * quantize_multiplier,
    int32_t * shift
    )
{
    uint32_t multiplier_32bit = 0;
    const int32_t bit = 16;
    gpu_quantize_multiplier_32bit( double_multiplier, &multiplier_32bit, shift );
    *quantize_multiplier = (uint16_t)(multiplier_32bit >> (bit - 1));
    if( *quantize_multiplier == 0 )
    {
        *shift = 0;
    }
    else
    {
        *shift -= bit;
    }
} /* gpu_quantize_multiplier_16bit() */

