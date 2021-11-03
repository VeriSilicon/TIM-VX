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
#ifndef _VSI_NN_POST_CMUPOSE_H_
#define _VSI_NN_POST_CMUPOSE_H_

#include "utils/vsi_nn_link_list.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _vsi_nn_subset_data_t
{
    float idx[20];
}vsi_nn_subset_data_t;

typedef struct _vsi_nn_subset_t
{
    vsi_nn_link_list_t link_list;
    vsi_nn_subset_data_t data;
}vsi_nn_subset_t;

typedef struct _vsi_nn_peaks_data_t
{
    uint32_t location[2];
    float score;
    uint32_t id;
}vsi_nn_peaks_data_t;

typedef struct _vsi_nn_peaks_t
{
    vsi_nn_link_list_t link_list;
    vsi_nn_peaks_data_t peak;
}vsi_nn_peaks_t;

typedef struct _vsi_nn_conncection_data_t
{
    uint32_t x;
    uint32_t y;
    float score;
    uint32_t i;
    uint32_t j;
}vsi_nn_connection_data_t;

typedef struct _vsi_nn_connection_t
{
    vsi_nn_link_list_t link_list;
    vsi_nn_connection_data_t data;
}vsi_nn_connection_t;

typedef struct _vsi_nn_con_candidate_data_t
{
    uint32_t i;
    uint32_t j;
    float score;
    float candAB;
}vsi_nn_con_candidate_data_t;

typedef struct _vsi_nn_con_candidate_t
{
    vsi_nn_link_list_t link_list;
    vsi_nn_con_candidate_data_t data;
}vsi_nn_con_candidate_t;

typedef struct _vsi_nn_cmupose_multiplier_t
{
    float *size;
    uint32_t num;
}vsi_nn_cmupose_multiplier_t;

typedef struct _vsi_nn_cmupose_image_t
{
    uint32_t width;
    uint32_t height;
    uint32_t channel;
}vsi_nn_cmupose_image_t;

typedef struct _vsi_nn_cmupose_scale_search_t
{
    float *size;
    uint32_t num;
}vsi_nn_cmupose_scale_search_t;

typedef struct _vsi_nn_cmupose_model_t
{
    uint32_t boxsize;
    uint32_t stride;
    uint32_t padValue;
}vsi_nn_cmupose_model_t;

typedef struct _vsi_nn_cmupose_param_t
{
    float thre1;
    float thre2;
    float thre3;
    uint32_t mid_num;
    vsi_nn_cmupose_scale_search_t scale_search;
}vsi_nn_cmupose_param_t;

typedef struct _vsi_nn_cmupose_inputs_t
{
    vsi_nn_tensor_t *net_out;
}vsi_nn_cmupose_inputs_t;

typedef struct _vsi_nn_cmupose_config_t
{
    vsi_nn_cmupose_inputs_t inputs;
    vsi_nn_cmupose_param_t  param;
    vsi_nn_cmupose_model_t  model;
    vsi_nn_cmupose_image_t  image;
}vsi_nn_cmupose_config_t;

OVXLIB_API vsi_status vsi_nn_CMUPose_Post_Process
    (
    float *net_out,
    vsi_nn_cmupose_config_t *config,
    vsi_nn_peaks_t ***all_peaks_out,
    uint32_t *all_peaks_num_out,
    vsi_nn_subset_t **subset_list_out,
    vsi_nn_peaks_data_t **peak_candidate_out,
    uint32_t *peak_candidate_num_out
    );

OVXLIB_API vsi_status vsi_nn_CMUPose_PostProcess
    (
    vsi_nn_graph_t *graph,
    vsi_nn_cmupose_inputs_t *inputs,
    vsi_nn_cmupose_image_t *image,
    vsi_nn_cmupose_param_t *param,
    vsi_nn_cmupose_model_t *model,
    vsi_nn_peaks_t ***all_peaks,
    uint32_t *all_peaks_num,
    vsi_nn_peaks_data_t **candidate,
    uint32_t *candidate_num,
    vsi_nn_subset_t **subset
    );

#ifdef __cplusplus
}
#endif

#endif
