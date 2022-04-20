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

#ifndef _VSI_NN_OP_PRE_PROCESS_RGB888_PLANAR_H
#define _VSI_NN_OP_PRE_PROCESS_RGB888_PLANAR_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _vsi_nn_pre_process_rgb888_planar_param
{
    struct _pre_process_rgb888_planar_local_data_t* local;
    // Add parameters here
    struct
    {
        uint32_t left;
        uint32_t top;
        uint32_t width;
        uint32_t height;
    } rect;

    struct
    {
        vsi_size_t *size;
        uint32_t   dim_num;
    } output_attr;

    float r_mean;
    float g_mean;
    float b_mean;
    float scale;
} vsi_nn_pre_process_rgb888_planar_param;
_compiler_assert(offsetof(vsi_nn_pre_process_rgb888_planar_param, local) == 0, \
    vsi_nn_pre_process_rgb888_planar_h );

#ifdef __cplusplus
}
#endif

#endif
