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
#ifndef _VSI_NN_OP_SOFTMAX_INTERNAL_H
#define _VSI_NN_OP_SOFTMAX_INTERNAL_H

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "utils/vsi_nn_link_list.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _vsi_nn_softmax_internal_lcl_data
{
    vsi_nn_link_list_t link_list;
    vx_node            node;
    vx_tensor          src_tensor;
    vx_tensor          dst_tensor;
} vsi_nn_softmax_internal_lcl_data;

typedef struct _vsi_nn_softmax_internal_param
{
    vsi_nn_softmax_internal_lcl_data *data;
    float beta;
    int32_t axis;
} vsi_nn_softmax_internal_param;

#ifdef __cplusplus
}
#endif

#endif
