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

#ifndef _VSI_NN_OP_BATCHNORM_SINGLE_H
#define _VSI_NN_OP_BATCHNORM_SINGLE_H

#include "vsi_nn_types.h"

/* enum for inputs/outputs */
enum
{
    BATCHNORM_INPUT             = 0,
    BATCHNORM_INPUT_MEAN        = 1,

    BATCHNORM_INPUT_VARIANCE    = 2,
    BATCHNORM_INPUT_GAMMA       = 3,
    BATCHNORM_INPUT_BETA        = 4,

    BATCHNORM_INPUT_CNT,

    BATCHNORM_OUTPUT            = 0,

    BATCHNORM_OUTPUT_CNT
};

typedef struct _vsi_nn_batchnorm_single_param
{
    // Add parameters here
    float eps;
} vsi_nn_batchnorm_single_param;

#endif

