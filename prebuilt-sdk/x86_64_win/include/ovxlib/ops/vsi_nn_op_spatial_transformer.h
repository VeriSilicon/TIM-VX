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
#ifndef _VSI_NN_OP_SPATIAL_TRANSFORMER_H
#define _VSI_NN_OP_SPATIAL_TRANSFORMER_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define _VSI_NN_SPATIAL_TRANSFORMER_LOCAL_TENSOR_NUM 2

typedef struct _vsi_nn_spatial_transformer_lcl_data
{
    vsi_nn_tensor_t  *local_tensor;
    vx_scalar scl;
} vsi_nn_spatial_transformer_lcl_data;


typedef struct _vsi_nn_spatial_transformer_param
{
    int32_t       output_H;
    int32_t       output_W;
    int32_t       has_theta_1_1;
    int32_t       has_theta_1_2;
    int32_t       has_theta_1_3;
    int32_t       has_theta_2_1;
    int32_t       has_theta_2_2;
    int32_t       has_theta_2_3;
    float         theta_1_1;
    float         theta_1_2;
    float         theta_1_3;
    float         theta_2_1;
    float         theta_2_2;
    float         theta_2_3;
    vsi_bool      align_corners;
} vsi_nn_spatial_transformer_param;

#ifdef __cplusplus
}
#endif

#endif
