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

#ifndef _VSI_NN_OP_RESIZE_H
#define _VSI_NN_OP_RESIZE_H

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define _VSI_NN_RESIZE_LOCAL_TENSOR_NUM 2

typedef uint32_t vsi_nn_interpolation_type_t; enum
{
    VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR = 0,
    VSI_NN_INTERPOLATION_BILINEAR,
    VSI_NN_INTERPOLATION_AREA,
    VSI_NN_INTERPOLATION_CUBIC
};

typedef uint32_t vsi_nn_resize_layout_type_t; enum
{
    VSI_NN_RESIZE_LAYOUT_NCHW = 0,
    VSI_NN_RESIZE_LAYOUT_NHWC
};

typedef struct _vsi_nn_resize_local_data {
    vsi_bool use_internal_node;
} vsi_nn_resize_local_data;

typedef struct _vsi_nn_resize_param
{
    vsi_enum     type;
    float        factor;
    int32_t      size[2];

    /* resize layer local data structure */
    union
    {
        vsi_nn_resize_local_data *lcl_data;
        struct {
            vx_tensor   local_tensor[_VSI_NN_RESIZE_LOCAL_TENSOR_NUM];
        } reserved;
    };
    vsi_bool    align_corners;
    vsi_bool    half_pixel_centers;
    vsi_enum    layout;
} vsi_nn_resize_param;

#ifdef __cplusplus
}
#endif

#endif
