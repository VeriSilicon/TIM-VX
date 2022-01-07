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
#ifndef _VSI_NN_OP_RNNCELL_OVXLIB_H
#define _VSI_NN_OP_RNNCELL_OVXLIB_H

#include "vsi_nn_types.h"
#include "vsi_nn_op_rnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _vsi_nn_rnncell_ovxlib_lcl_data_t
{
    vsi_bool multi_batch;
} vsi_nn_rnncell_ovxlib_lcl_data_t;

typedef struct _vsi_nn_rnncell_ovxlib_param
{
    vsi_nn_rnncell_ovxlib_lcl_data_t* local;

    vsi_nn_activation_e activation;
    vsi_nn_dtype_t* internal_dtype;
} vsi_nn_rnncell_ovxlib_param;

#ifdef __cplusplus
}
#endif

#endif
