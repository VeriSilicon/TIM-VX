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
#ifndef _VSI_NN_OP_IMAGEPROCESS_H
#define _VSI_NN_OP_IMAGEPROCESS_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t vsi_nn_imageprocess_resize_type_e; enum
{
    VSI_NN_IMAGEPROCESS_RESIZE_NONE = 0,
    VSI_NN_IMAGEPROCESS_RESIZE_BILINEAR,
};

typedef uint32_t vsi_nn_imageprocess_mean_type_e; enum
{
    VSI_NN_IMAGEPROCESS_MEAN_NONE = 0,
    VSI_NN_IMAGEPROCESS_MEAN_CHANNEL,
    VSI_NN_IMAGEPROCESS_MEAN_PIXEL
};

typedef struct _vsi_nn_imageprocess_param
{
    vsi_nn_platform_e platform_type;
    struct
    {
        vx_bool enable;
        int32_t dim_num;
        int32_t* start;
        int32_t* length;
    } crop;
    struct
    {
        vsi_nn_imageprocess_resize_type_e type;
        int32_t dim_num;
        int32_t* length;
    } resize;
    vx_bool reverse_channel;
    struct
    {
        vsi_nn_imageprocess_mean_type_e type;
        float scale;
        int32_t mean_value_size;
        float* mean_value;
    } mean;
} VSI_PUBLIC_TYPE vsi_nn_imageprocess_param;

/**
* Insert imageprocess op for image pre process
* @deprecated
* @see vsi_nn_InsertImageprocessSingleNode
*
* @param[in] graph.
* @param[in] the attr of the input tensor of graph.
* @param[in] the parameters of imageprocess.
* @param[in] bmp buffer of input image.
* @param[in] output tensor.
*/
OVXLIB_API vsi_status vsi_nn_op_imageprocess_single_node
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_attr_t *attr,
    vsi_nn_imageprocess_param *p,
    uint8_t *data,
    vsi_nn_tensor_t *tensor_out
    );

/**
* Insert imageprocess op for image pre process.
*
* @param[in] graph.
* @param[in] the attr of the input tensor of graph.
* @param[in] the parameters of imageprocess.
* @param[in] bmp buffer of input image.
* @param[in] output tensor.
* @param[in] id. There may be multi models in a process. Each one has a uniqe id.\n
*            But repeatedly running one model with different images should share the same id.
*/
OVXLIB_API vsi_status vsi_nn_InsertImageprocessSingleNode
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_attr_t *attr,
    vsi_nn_imageprocess_param *p,
    uint8_t *data,
    vsi_nn_tensor_t *tensor_out,
    int32_t id
    );

/**
* Release the resource of imageprocess op.
*/
OVXLIB_API vsi_status vsi_nn_ReleaseImageprocessSingleNode();

#ifdef __cplusplus
}
#endif

#endif

