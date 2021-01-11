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
#ifndef _VSI_NN_OP_PRE_PROCESS_H
#define _VSI_NN_OP_PRE_PROCESS_H

#include "vsi_nn_types.h"
#include "vsi_nn_pre_post_process.h"

typedef  vsi_nn_preprocess_source_format_e vsi_nn_pre_process_type_e;

enum
{
    PRE_PROCESS_INPUT0 = 0,

    PRE_PROCESS_INPUT1,
    PRE_PROCESS_INPUT2,

    PRE_PROCESS_INPUT_CNT,

    PRE_PROCESS_OUTPUT = 0,

    PRE_PROCESS_OUTPUT_CNT
};

#define _VSI_NN_PRE_PROCESS_LOCAL_TENSOR_NUM 10
typedef struct _vsi_nn_pre_process_lcl_data
{
    vsi_nn_tensor_t *local_tensor[_VSI_NN_PRE_PROCESS_LOCAL_TENSOR_NUM];
} vsi_nn_pre_process_lcl_data;

typedef struct _vsi_nn_pre_process_param
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

    uint32_t * perm;
    uint32_t   dim_num;

    struct
    {
        float   mean[3];
        float   scale;
    } norm;

    vsi_bool reverse_channel;

    vsi_nn_pre_process_type_e type;

    vsi_nn_pre_process_lcl_data *local;
} vsi_nn_pre_process_param;
#endif

