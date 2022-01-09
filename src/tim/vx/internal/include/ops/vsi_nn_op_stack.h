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
#ifndef _VSI_NN_OP_STACK_H
#define _VSI_NN_OP_STACK_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VSI_NN_STACK_MAX_INPUTS (16)

typedef struct _vsi_nn_stack_lcl_data
{
    vsi_nn_link_list_t link_list;
    union
    {
        /* used for optimze concat to tensor view */
        struct
        {
            vx_node            cp_node;
            vx_tensor          src_tensor;
            vx_tensor          dst_tensor;
            vsi_nn_tensor_t *  src_in;
        };
        /* used for vxConcatIndefiniteLayer */
        struct
        {
            vx_object_array    array;
        };
    };
} vsi_nn_stack_lcl_data;

typedef struct _vsi_nn_stack_lcl
{
    uint32_t block_size;
    uint32_t block_num;
    vx_tensor local_tensor[VSI_NN_STACK_MAX_INPUTS];
} vsi_nn_stack_lcl;

typedef struct _vsi_nn_stack_param
{
    /* local data must be the first. */
    vsi_nn_stack_lcl_data * lcl_data;
    vsi_nn_stack_lcl local;
    uint32_t axis;
} vsi_nn_stack_param;

#ifdef __cplusplus
}
#endif

#endif
