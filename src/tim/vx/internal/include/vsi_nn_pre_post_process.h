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
#ifndef VSI_NN_PRE_POST_PROCESS_H
#define VSI_NN_PRE_POST_PROCESS_H

#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"

#define VSI_NN_PREPROCESS_IMMATERIAL (0)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Preprocess type
 */
typedef enum
{
    VSI_NN_PREPROCESS_SOURCE_LAYOUT = 0,
    VSI_NN_PREPROCESS_SET_SOURCE_FORMAT,
    VSI_NN_PREPROCESS_IMAGE_SIZE,
    VSI_NN_PREPROCESS_CROP,
    VSI_NN_PREPROCESS_MEAN_AND_SCALE,
    VSI_NN_PREPROCESS_PERMUTE,
    VSI_NN_PREPROCESS_REVERSE_CHANNEL,
    VSI_NN_PREPROCESS_IMAGE_RESIZE_BILINEAR,
    VSI_NN_PREPROCESS_IMAGE_RESIZE_NEAREST,
    VSI_NN_PREPROCESS_DTYPE_CONVERT,
    VSI_NN_PREPROCESS_MEANS_AND_SCALES,
} vsi_nn_preprocess_type_e;

/**
 * Postprocess type
 */
typedef enum
{
    VSI_NN_POSTPROCESS_PERMUTE = 0,
    VSI_NN_POSTPROCESS_DTYPE_CONVERT,
} vsi_nn_postprocess_type_e;

typedef enum
{
    VSI_NN_SOURCE_LAYOUT_NHWC = 0,
    VSI_NN_SOURCE_LAYOUT_NCHW,
} vsi_nn_preprocess_source_layout_e;

typedef enum
{
    VSI_NN_DEST_LAYOUT_NHWC = 0,
    VSI_NN_DEST_LAYOUT_NCHW,
} vsi_nn_preprocess_dest_layout_e;

/**
 * Input source format
 */
typedef enum
{
    VSI_NN_SOURCE_FORMAT_TENSOR = 0,
    VSI_NN_SOURCE_FORMAT_IMAGE_GRAY,
    VSI_NN_SOURCE_FORMAT_IMAGE_RGB,
    VSI_NN_SOURCE_FORMAT_IMAGE_YUV420,
    VSI_NN_SOURCE_FORMAT_IMAGE_BGRA,
    VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR,
    VSI_NN_SOURCE_FORMAT_IMAGE_YUV444,
    VSI_NN_SOURCE_FORMAT_IMAGE_NV12,
    VSI_NN_SOURCE_FORMAT_IMAGE_RGB888_PLANAR_SEP,
    VSI_NN_SOURCE_FORMAT_IMAGE_YUYV422,
    VSI_NN_SOURCE_FORMAT_IMAGE_UYVY422,
    VSI_NN_SOURCE_FORMAT_IMAGE_NV21,
} vsi_nn_preprocess_source_format_e;

/**
 * Preprocess base structure
 */
typedef struct
{
    /** Preprocess type*/
    vsi_nn_preprocess_type_e type;
    /** Preprocess paramters */
    void* param;
} VSI_PUBLIC_TYPE vsi_nn_preprocess_base_t;

/**
 * Postprocess base structure
 */
typedef struct
{
    /** Postprocess type*/
    vsi_nn_postprocess_type_e type;
    /** Postrocess paramters */
    void* param;
} VSI_PUBLIC_TYPE vsi_nn_postprocess_base_t;

/**
 * Process dtype convert parameter structure
 */
typedef struct {
     vsi_nn_dtype_t dtype;
     /** Reserve some more btyes for future features. */
     char reserved[40];
} vsi_nn_process_dtype_convert_t;

typedef vsi_nn_process_dtype_convert_t vsi_nn_preprocess_dtype_convert_t;
typedef vsi_nn_process_dtype_convert_t vsi_nn_postprocess_dtype_convert_t;

/**
 * Process crop parameter structure
 */
typedef struct
{
    /** Crop begin for each dim */
    int32_t* begin;
    /** Crop size for each dim */
    int32_t* size;
    /** Image dim */
    int32_t dim;
}vsi_nn_preprocess_crop_t;

/**
 * Process mean and scale parameter structure
 */
typedef struct
{
    /** Mean value for each channel */
    float* channel_mean;
    /*Channel length */
    int32_t channel_len;
    /** Scale value */
    float scale;
}vsi_nn_process_mean_and_scale_t;

/**
 * Process mean and scale parameter structure
 */
typedef struct
{
    /** Mean value for each channel */
    float* channel_mean;
    /*Channel length */
    int32_t channel_len;
    /** Scale value */
    float* scale;
    /** Scale length */
    int32_t scale_len;
}vsi_nn_process_means_and_scales_t;

typedef vsi_nn_process_mean_and_scale_t vsi_nn_preprocess_mean_and_scale_t;
typedef vsi_nn_process_means_and_scales_t vsi_nn_preprocess_means_and_scales_t;
typedef vsi_nn_process_mean_and_scale_t vsi_nn_postprocess_mean_and_scale_t;
typedef vsi_nn_process_means_and_scales_t vsi_nn_postprocess_means_and_scales_t;

/**
 * Process permute parameter structure
 */
typedef struct
{
    /** Permute value for each channel */
    int32_t* perm;
    /** Permute dim */
    int32_t dim;
}vsi_nn_process_permute_t;

typedef vsi_nn_process_permute_t vsi_nn_preprocess_permute_t;
typedef vsi_nn_process_permute_t vsi_nn_postprocess_permute_t;

/**
 * Preprocess image resize parameter structure
 */
typedef struct {
    /** Width */
    uint32_t w;
    /** Height */
    uint32_t h;
    /** Channel */
    uint32_t c;
} vsi_nn_preprocess_image_resize_t;

typedef vsi_nn_preprocess_image_resize_t vsi_nn_preprocess_image_size_t;

vsi_status vsi_nn_add_single_preproc_node
    (
    vsi_nn_graph_t* graph,
    uint32_t input_idx,
    vsi_nn_tensor_id_t input,
    vsi_nn_node_t** first_node,
    uint32_t nodes_count,
    vsi_nn_preprocess_base_t* preprocess,
    uint32_t proc_count
    );

vsi_status vsi_nn_add_single_postproc_node
    (
    vsi_nn_graph_t* graph,
    uint32_t output_idx,
    vsi_nn_node_t* last_node,
    vsi_nn_postprocess_base_t* postprocess,
    uint32_t proc_count
    );

/**
 * Add preprocess node in for specified input
 *
 * @param[in] graph Graph to be added node in.
 * @param[in] input_idx Input tensor port.
 * @param[in] preprocess Preprocess task handle.
 * @param[in] count Preprocess task count.
 *
 * @return VSI_SUCCESS on success, or appropriate error code otherwise.
 *
 */
OVXLIB_API vsi_status vsi_nn_AddGraphPreProcess
    (
    vsi_nn_graph_t* graph,
    uint32_t input_idx,
    vsi_nn_preprocess_base_t* preprocess,
    uint32_t count
    );

/**
 * Add postprocess node in for specified output
 *
 * @param[in] graph Graph to be added node in.
 * @param[in] input_idx Input tensor port.
 * @param[in] postprocess Postprocess task handle.
 * @param[in] count Postprocess task count.
 *
 * @return VSI_SUCCESS on success, or appropriate error code otherwise.
 *
 */
OVXLIB_API vsi_status vsi_nn_AddGraphPostProcess
    (
    vsi_nn_graph_t* graph,
    uint32_t output_idx,
    vsi_nn_postprocess_base_t* postprocess,
    uint32_t count
    );

OVXLIB_API vsi_status vsi_nn_AddBinaryGraphInputsWithCropParam
    (
        vsi_nn_graph_t* graph,
        vsi_nn_node_id_t* enable_nodes,
        uint32_t enable_nodes_count
    );

OVXLIB_API vsi_status vsi_nn_UpdateCropParamsForBinaryGraph
    (
        vsi_nn_graph_t* graph,
        uint32_t enabled_crop_input_idx,
        uint32_t start_x,
        uint32_t start_y,
        uint32_t crop_w,
        uint32_t crop_h,
        uint32_t dst_w,
        uint32_t dst_h
    );

#ifdef __cplusplus
}
#endif

#endif
