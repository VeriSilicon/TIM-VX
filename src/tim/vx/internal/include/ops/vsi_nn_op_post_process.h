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
#ifndef _VSI_NN_OP_POST_PROCESS_H
#define _VSI_NN_OP_POST_PROCESS_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    POST_PROCESS_INPUT = 0,

    POST_PROCESS_INPUT_CNT,

    POST_PROCESS_OUTPUT = 0,

    POST_PROCESS_OUTPUT_CNT
};

typedef struct _vsi_nn_post_process_lcl_data
{
    vsi_bool initialized;
    vsi_bool enable_data_conv;
    vsi_bool enable_perm;
} vsi_nn_post_process_lcl_data;

typedef struct _vsi_nn_post_process_param
{
    uint32_t * perm;
    uint32_t   dim_num;

    /* post process layer local data structure */
    vsi_nn_post_process_lcl_data local;
} vsi_nn_post_process_param;

#ifdef __cplusplus
}
#endif

#endif
