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
#ifndef _VSI_NN_OP_PRE_PROCESS_GRAY_H
#define _VSI_NN_OP_PRE_PROCESS_GRAY_H

#include "vsi_nn_types.h"

enum
{
    PRE_PROCESS_GRAY_INPUT = 0,

    PRE_PROCESS_GRAY_INPUT_CNT,

    PRE_PROCESS_GRAY_OUTPUT = 0,

    PRE_PROCESS_GRAY_OUTPUT_CNT
};

typedef struct _vsi_nn_pre_process_gray_lcl_data
{
    int32_t scale_x;
    int32_t scale_y;
    vsi_bool enable_copy;
} vsi_nn_pre_process_gray_lcl_data;

typedef struct _vsi_nn_pre_process_gray_param
{
    struct
    {
        uint32_t left;
        uint32_t top;
        uint32_t width;
        uint32_t height;
    } rect;

    struct
    {
        uint32_t   *size;
        uint32_t   dim_num;
    } output_attr;

    float mean;
    float scale;

    /* pre process gray layer local data structure */
    vsi_nn_pre_process_gray_lcl_data local;
} vsi_nn_pre_process_gray_param;

#endif

