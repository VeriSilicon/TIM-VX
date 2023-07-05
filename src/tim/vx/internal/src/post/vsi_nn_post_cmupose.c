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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vsi_nn_context.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_node_attr_template.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_util.h"
#include "post/vsi_nn_post_cmupose.h"
#include "vsi_nn_error.h"

static const int32_t limbSeq[19][2] = {{2,3},   {2,6},   {3,4},  {4,5},   {6,7},
                                       {7,8},   {2,9},   {9,10}, {10,11}, {2,12},
                                       {12,13}, {13,14}, {2,1},  {1,15},  {15,17},
                                       {1,16},  {16,18}, {3,17}, {6,18}};
static const int32_t mapIdx[19][2] = {{31,32}, {39,40}, {33,34}, {35,36}, {41,42},
                                      {43,44}, {19,20}, {21,22}, {23,24}, {25,26},
                                      {27,28}, {29,30}, {47,48}, {49,50}, {53,54},
                                      {51,52}, {55,56}, {37,38}, {45,46}};
uint32_t peak_id = 0;

// this is a temp test function
#if 0
static void savetxt(char *filename, float *buffer, uint32_t sz)
{
#define _MAX_BUFFER_SZ  (512)
    FILE * fp;
    const float c_flush_th = 0.7f;
    uint8_t buf[_MAX_BUFFER_SZ];
    uint32_t count,i;

    count = 0;
    fp = vsi_nn_fopen( filename, "w" );
    for( i = 0; i < sz; i ++ )
    {
        count += snprintf( (char *)&buf[count], _MAX_BUFFER_SZ - count,
            "%f\n", buffer[i]);
        if( ((float)count / _MAX_BUFFER_SZ) > c_flush_th )
        {
            fwrite( buf, count, 1, fp );
            count = 0;
        }
    }

    fwrite( buf, count, 1, fp );
    fclose( fp );
}
#endif

static void _init_subset(vsi_nn_link_list_t *node)
{
    vsi_nn_subset_t *ptr = NULL;
    ptr = (vsi_nn_subset_t *)node;
    ptr->link_list.next = NULL;
    ptr->link_list.prev = NULL;
    memset(&ptr->data.idx, 0, sizeof(float) * 20);
}

static void _init_candidate(vsi_nn_link_list_t *node)
{
    vsi_nn_con_candidate_t *ptr = NULL;
    ptr = (vsi_nn_con_candidate_t *)node;
    ptr->link_list.next = NULL;
    ptr->link_list.prev = NULL;
    memset(&ptr->data, 0, sizeof(vsi_nn_con_candidate_data_t));
}

static void _init_connection(vsi_nn_link_list_t *node)
{
    vsi_nn_connection_t *ptr = NULL;
    ptr = (vsi_nn_connection_t *)node;
    ptr->link_list.next = NULL;
    ptr->link_list.prev = NULL;
    memset(&ptr->data, 0, sizeof(vsi_nn_connection_data_t));
}

static void _init_peak(vsi_nn_link_list_t *node)
{
    vsi_nn_peaks_t *box = NULL;
    box = (vsi_nn_peaks_t *)node;
    box->link_list.next = NULL;
    box->link_list.prev = NULL;
    memset(&box->peak, 0, sizeof(vsi_nn_peaks_data_t));
}

static vsi_status _cmupose_init_multiplier
    (
    vsi_nn_cmupose_config_t *config,
    vsi_nn_cmupose_multiplier_t *multiplier
    )
{
    uint32_t i,num;
    uint32_t boxsize,width;
    vsi_status status = VSI_FAILURE;

    if(NULL == config || NULL == multiplier)
    {
        return VSI_FAILURE;
    }

    num = config->param.scale_search.num;
    multiplier->size = (float *)malloc(num * sizeof(float));
    CHECK_PTR_FAIL_GOTO( multiplier->size, "Create buffer fail.", final );
    status = VSI_SUCCESS;
    multiplier->num = num;

    boxsize = config->model.boxsize;
    width = config->image.width;
    memset(multiplier->size, 0, sizeof(float) * num);
    for(i=0; i<num; i++)
    {
        float x = config->param.scale_search.size[i];
        multiplier->size[i] = x * boxsize / width;
    }

final:

    return status;
}

static vsi_status _cmupose_init_heatmap_avg
    (
    vsi_nn_cmupose_config_t *config,
    float **heatmap_avg
    )
{
    uint32_t w,h,channel;
    vsi_size_t sz;
    vsi_status status = VSI_FAILURE;

    if(NULL == config || NULL == heatmap_avg)
    {
        return VSI_FAILURE;
    }

#define VSI_NN_POST_CMUPOSE_DEF_HEATMAP 19
    channel = VSI_NN_POST_CMUPOSE_DEF_HEATMAP;
    w = config->image.width;
    h = config->image.height;
    sz = channel * w * h;
    *heatmap_avg = (float *)malloc(sizeof(float) * sz);
    CHECK_PTR_FAIL_GOTO( *heatmap_avg, "Create buffer fail.", final );
    status = VSI_SUCCESS;
    memset(*heatmap_avg, 0, sizeof(float) * sz);
final:

    return status;
}

static vsi_status _cmupose_init_paf_avg
    (
    vsi_nn_cmupose_config_t *config,
    float **paf_avg
    )
{
    uint32_t w,h,channel;
    vsi_size_t sz;
    vsi_status status = VSI_FAILURE;

    if(NULL == config || NULL == paf_avg)
    {
        return VSI_FAILURE;
    }

#define VSI_NN_POST_CMUPOSE_DEF_PAF 38
    channel = VSI_NN_POST_CMUPOSE_DEF_PAF;
    w = config->image.width;
    h = config->image.height;
    sz = channel * w * h;
    *paf_avg = (float *)malloc(sizeof(float) * sz);
    CHECK_PTR_FAIL_GOTO( *paf_avg, "Create buffer fail.", final );
    status = VSI_SUCCESS;
    memset(*paf_avg, 0, sizeof(float) * sz);

final:

    return status;
}

static void _cmupose_deinit
    (
    vsi_nn_cmupose_multiplier_t *multiplier,
    float *heatmap_avg,
    float *paf_avg
    )
{
    if(multiplier->size)free(multiplier->size);
    if(heatmap_avg)free(heatmap_avg);
    if(paf_avg)free(paf_avg);
}

static vsi_status _cmupose_init
    (
    vsi_nn_cmupose_config_t *config,
    vsi_nn_cmupose_multiplier_t *multiplier,
    float **heatmap_avg,
    float **paf_avg
    )
{
    vsi_status status = VSI_FAILURE;

    if(NULL == config || NULL == multiplier
       || NULL == heatmap_avg || NULL == paf_avg)
    {
        return status;
    }

    status = _cmupose_init_multiplier(config, multiplier);
    if(VSI_SUCCESS != status)
    {
        goto error;
    }
    status = _cmupose_init_heatmap_avg(config, heatmap_avg);
    if(VSI_SUCCESS != status)
    {
        goto error;
    }
    status = _cmupose_init_paf_avg(config, paf_avg);
    if(VSI_SUCCESS != status)
    {
        goto error;
    }

    return status;
error:
    _cmupose_deinit(multiplier, *heatmap_avg, *paf_avg);
    return status;
}

#if 0
static vx_status resize_binlinear
    (
    float *src,
    uint32_t *src_size,
    float *dst,
    uint32_t *dst_size
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t i,j,n,k;
    float *src_ptr = NULL, *dst_ptr = NULL;
    uint32_t src_w, src_h, dst_w, dst_h;
    float xRatio, yRatio;
    uint32_t LineSize,GrayStride;
    uint32_t index00,index01,index10,index11;
    float value00,value01,value10,value11;
    double temp;

    src_ptr = src;
    dst_ptr = dst;

    src_w = src_size[0];
    src_h = src_size[1];
    dst_w = dst_size[0];
    dst_h = dst_size[1];

    xRatio = (float)src_w / dst_w;
    yRatio = (float)src_h / dst_h;

    GrayStride = 19; /* in gray, channel = 1*/
    LineSize = src_w * GrayStride;

    n = 1;
    for(i = 0; i < dst_h; i++)
    {
        float srcY = i * yRatio;
        uint32_t IntY = (uint32_t)srcY;
        float v = srcY - IntY;
        float v1 = 1.f - v;

        for(j = 0; j < dst_w; j++)
        {
            float srcX = j * xRatio;
            uint32_t IntX = (uint32_t)srcX;
            float u = srcX - IntX;
            float u1 = 1.f - u;

            /*
                index00 -------- index01
                  |                 |
                  |    index(x,y)   |
                  |                 |
                  |                 |
                index10 -------- index11
            */
            index00 = IntY * LineSize + IntX * GrayStride;
            index10;
            if(IntY < src_h - 1)
                index10 = index00 + LineSize;
            else
                index10 = index00;

            index01,index11;
            if(IntX < src_w)
            {
                index01 = index00 + GrayStride;
                index11 = index10 + GrayStride;
            }
            else
            {
                index01 = index00;
                index11 = index10;
            }

            for(k=0; k<19; k++)
            {
                value00 = src_ptr[index00 + k];
                value01 = src_ptr[index01 + k];
                value10 = src_ptr[index10 + k];
                value11 = src_ptr[index11 + k];
                temp = v1 * (u * value01 + u1 * value00) + v * (u * value11 + u1 * value10);

                *dst_ptr = (float)temp;
                dst_ptr++;
            }

            n++;
        }
    }

    return VSI_SUCCESS;
}
#endif

static vx_status resize_nearest
    (
    float *src,
    uint32_t *src_size,
    float *dst,
    uint32_t *dst_size
    )
{
    uint32_t w = 0, h = 0;

    uint32_t output_depth = dst_size[2];
    uint32_t output_height = dst_size[1];
    uint32_t output_width = dst_size[0];

    uint32_t input_depth = src_size[2];
    uint32_t input_height = src_size[1];
    uint32_t input_width = src_size[0];

    float width_scale = (input_width * 1.0f)/output_width;
    float height_scale = (input_height * 1.0f)/output_height;

    uint32_t depthf = output_depth*sizeof(float);
    uint32_t stride_out = output_depth * output_width;
    uint32_t stride_in  = input_depth * input_width;

    for(h = 0; h < output_height; h ++)
    {
        uint32_t in_y = vsi_nn_min((uint32_t)(h * height_scale), input_height - 1);
        for (w = 0; w < output_width; w ++)
        {
            uint32_t in_x = vsi_nn_min((uint32_t)(w * width_scale), input_width - 1);

            uint32_t index_out,index_in;
            index_out = stride_out * h + output_depth * w;
            index_in = stride_in * in_y + input_depth * in_x;

            memcpy(dst + index_out, src + index_in, depthf);
        }
    }

    return VSI_SUCCESS;
}

static double *create_gaussian_kernel
    (
    float sigma,
    int32_t *size
    )
{
    double *kernel = NULL;
    int32_t ksz,i;
    double sum,scale2X;
    static double kernel_25[25] = {0.000045, 0.000160, 0.000514, 0.001477, 0.003799,
                                  0.008741, 0.017997, 0.033160, 0.054672, 0.080659,
                                  0.106486, 0.125798, 0.132985, 0.125798, 0.106486,
                                  0.080659, 0.054672, 0.033160, 0.017997, 0.008741,
                                  0.003799, 0.001477, 0.000514, 0.000160, 0.000045};

    ksz = vsi_nn_max(1, ((int32_t)(4.0 * sigma + 1.0 - 1e-8))) * 2 + 1;
    kernel = (double *)malloc(sizeof(double) * ksz);
    CHECK_PTR_FAIL_GOTO( kernel, "Create buffer fail.", final );
    memset(kernel, 0, sizeof(double) * ksz);

    if(ksz == 25)
    {
        memcpy(kernel, kernel_25, sizeof(double) * ksz);
        goto final;
    }

    sum = 0.f;
    scale2X = -0.5 / (sigma * sigma);
    for(i=0; i<ksz; i++)
    {
        double x = i - (ksz - 1) * 0.5;
        double t = exp(scale2X * x * x);
        kernel[i] = t;
        sum += kernel[i];
    }

    sum = 1./sum;
    for(i=0; i<ksz; i++)
    {
        kernel[i] = kernel[i] * sum;
    }

#if 0
    for(i=0; i<ksz; i++)
    {
        printf("kernel[%u] = %lf\n", i, kernel[i]);
    }
#endif

final:
    *size = ksz;
    return kernel;
}

// only support convolve1d = 'same'
static void _convolve_same
    (
    float *input,
    uint32_t input_size,
    double *kernel,
    int32_t kernel_size,
    float *output
    )
{
    uint32_t pad = 0, pad_input_size = 0;
    uint32_t i = 0, offset = 0;
    int32_t k = 0;
    float *pad_input = NULL;
    double sum = 0;

    uint32_t pad_input_sizef,input_sizef;
    if(NULL == input || NULL == kernel || NULL == output)
    {
        return ;
    }

    // init input pad
    pad = (kernel_size - 1) / 2;
    pad_input_size = 2 * pad + input_size;
    pad_input_sizef = sizeof(float) * pad_input_size;
    input_sizef = input_size * sizeof(float);
    pad_input = (float *)malloc(pad_input_sizef);
    CHECK_PTR_FAIL_GOTO( pad_input, "Create buffer fail.", final );
    memset(pad_input, 0, pad_input_sizef);
    memcpy(pad_input + pad, input, input_sizef);

    // init output buffer
    memset(output, 0, input_sizef);

    //compute convolve
    offset = 0;
    for(i=0; i<input_size; i++)
    {
        offset = i;
        sum = 0.f;
        for(k=0; k<kernel_size; k++)
        {
            double t_i = pad_input[offset++];
            double t_k = kernel[k];
            sum += t_i * t_k;
        }
        output[i] = (float)sum;
    }

final:
    vsi_nn_safe_free(pad_input);
}

static void get_cols
    (
    float *inputs,
    uint32_t height,
    uint32_t width,
    uint32_t cols_index,
    float *output
    )
{
    uint32_t w;

    if(NULL == inputs || NULL == output)
    {
        return ;
    }

    memset(output, 0, sizeof(sizeof(float) * height));

    for(w=0; w<width; w++)
    {
        output[w] = inputs[w * width + cols_index];
    }
}

static void set_cols
    (
    float *data,
    float *cols,
    uint32_t height,
    uint32_t width,
    uint32_t cols_index
    )
{
    uint32_t w;

    VSI_UNREFERENCED(height);

    if(NULL == data || cols == NULL)
    {
        return ;
    }

    for(w=0; w<width; w++)
    {
        data[w * width + cols_index] = cols[w];
    }
}

static vsi_status gaussian_filter
    (
    float *inputs,
    float sigma,
    vsi_nn_cmupose_config_t *config,
    float *output
    )
{
    double *kernel = NULL;
    float *temp = NULL,*rows = NULL,*cols = NULL;
    float *conv_buffer = NULL, *conv_buffer1 = NULL;
    int32_t ksz;
    uint32_t w,h,i;

    uint32_t szwf,szhf;
    vsi_size_t sz,szf;

    kernel = NULL, ksz = 0;
    kernel = create_gaussian_kernel(sigma, &ksz);
    CHECK_PTR_FAIL_GOTO( kernel, "Create buffer fail.", final );

    w = config->image.width;
    h = config->image.height;
    sz = w * h;
    szf = sizeof(float) * sz;
    szwf = sizeof(float) * w;
    szhf = sizeof(float) * h;
    temp = (float *)malloc(szf);
    CHECK_PTR_FAIL_GOTO( temp, "Create buffer fail.", final );
    memset(temp, 0, szf);

    rows = NULL;
    conv_buffer = (float *)malloc(szwf);
    CHECK_PTR_FAIL_GOTO( conv_buffer, "Create buffer fail.", final );
    for(i=0; i<h; i++)
    {
        rows = inputs + i * w;
        _convolve_same(rows, w, kernel, ksz, conv_buffer);
        memcpy(temp + i * w, conv_buffer, szwf);
    }

    conv_buffer1 = (float *)malloc(szhf);
    CHECK_PTR_FAIL_GOTO( conv_buffer1, "Create buffer fail.", final );
    cols = (float *)malloc(szhf);
    CHECK_PTR_FAIL_GOTO( cols, "Create buffer fail.", final );
    for(i=0; i<w; i++)
    {
        get_cols(temp, h, w, i, cols);
        _convolve_same(cols, h, kernel, ksz, conv_buffer1);
        set_cols(output, conv_buffer1, h, w, i);
    }

final:
    vsi_nn_safe_free(cols);
    vsi_nn_safe_free(conv_buffer);
    vsi_nn_safe_free(conv_buffer1);
    vsi_nn_safe_free(kernel);
    vsi_nn_safe_free(temp);

    return VSI_SUCCESS;
}

static vsi_nn_peaks_t *_compute_peaks
    (
    float *map_ori,
    float *map,
    float *map_left,
    float *map_right,
    float *map_up,
    float *map_down,
    vsi_nn_cmupose_config_t *config
    )
{
    uint32_t i,j,index;
    uint32_t width,height;
    float thre1;
    float score;
    vsi_nn_peaks_t *peak = NULL, *peak_list = NULL;

    thre1 = config->param.thre1;
    width = config->image.width;
    height = config->image.height;

    for(i=0; i<height; i++)
    {
        for(j=0; j<width; j++)
        {
            index = i * width + j;
            if(map[index] >= map_left[index] &&
                map[index] >= map_right[index] &&
                map[index] >= map_up[index] &&
                map[index] >= map_down[index] &&
                map[index] >= thre1)
            {
                peak = (vsi_nn_peaks_t *)
                    vsi_nn_LinkListNewNode(sizeof(vsi_nn_peaks_t), _init_peak);
                CHECK_PTR_FAIL_GOTO( peak, "get point fail.", final );
                score = map_ori[i * width + j];

                peak->peak.id = peak_id;
                peak->peak.score = score;
                peak->peak.location[0] = j;
                peak->peak.location[1] = i;

                vsi_nn_LinkListPushEnd(
                    (vsi_nn_link_list_t **)&peak_list,
                    (vsi_nn_link_list_t *)peak );

                #if 0
                printf("peak[%u %u %f %u]\n",
                    peak->peak.location[0],
                    peak->peak.location[1],
                    peak->peak.score,
                    peak->peak.id);
                #endif
                peak_id++;
            }
        }
    }

final:
    return peak_list;
}

static vsi_status _get_score_mid
    (
    float *paf_avg,
    const int32_t *mapIdx_k,
    vsi_nn_cmupose_config_t *config,
    float *score_mid
    )
{
    uint32_t w,h,width,height;
    uint32_t s1,s2,index_src,index_out;
    uint32_t c1,c2;
    uint32_t sz_c1_w,sz_c2_w;

    if(NULL == paf_avg || NULL == mapIdx_k || NULL == config || NULL == score_mid)
    {
        return VSI_FAILURE;
    }

    s1 = mapIdx_k[0] - 19;
    s2 = mapIdx_k[1] - 19;
    width = config->image.width;
    height = config->image.height;

    c1 = 2;
    c2 = 38;
    sz_c1_w = c1 * width;
    sz_c2_w = c2 * width;
    memset(score_mid, 0, sizeof(float) * width * height * c1);
    for(h=0; h<height; h++)
    {
        for(w=0; w<width; w++)
        {
            index_out = h * sz_c1_w + w * c1;
            index_src = h * sz_c2_w + w * c2;
            score_mid[index_out + 0] = paf_avg[index_src + s1];
            score_mid[index_out + 1] = paf_avg[index_src + s2];
        }
    }

    return VSI_SUCCESS;
}

static vsi_status _get_peaks
    (
    vsi_nn_peaks_t **all_peaks,
    const int32_t index,
    vsi_nn_peaks_t **candX,
    uint32_t *num
    )
{
    vsi_nn_peaks_t *iter, *peak;
    uint32_t n;

    if(NULL == all_peaks || NULL == candX)
    {
        return VSI_FAILURE;
    }

    peak = all_peaks[index - 1];
    iter = peak;
    n = 0;
    while(iter)
    {
    #if 0
        printf("peak[%u %u %f %u]\n",
            iter->peak.location[0], iter->peak.location[1], iter->peak.score, iter->peak.id);
    #endif
        iter = (vsi_nn_peaks_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter);
        n++;
    }
    *candX = peak;
    *num = n;

    return VSI_SUCCESS;
}

static vsi_status _get_peak_data
    (
    vsi_nn_peaks_t *peaks,
    uint32_t index,
    vsi_nn_peaks_data_t *data
    )
{
    vsi_nn_peaks_t *iter;
    uint32_t n;
    vsi_status status;

    status = VSI_FAILURE;
    if(NULL == peaks || NULL == data)
    {
        return status;
    }

    n = 0;
    iter = peaks;
    memset(data, 0, sizeof(vsi_nn_peaks_data_t));
    while (iter)
    {
        if(n == index)
        {
            data->id = iter->peak.id;
            data->score = iter->peak.score;
            data->location[0] = iter->peak.location[0];
            data->location[1] = iter->peak.location[1];
            status = VSI_SUCCESS;
            break;
        }
        iter = (vsi_nn_peaks_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter);
        n++;
    }

    return status;
}

static vsi_status _line_space
    (
    float start,
    float end,
    uint32_t num,
    float *outputs
    )
{
    vsi_status status;
    float step,sum;
    uint32_t i;

    status = VSI_FAILURE;
    if(NULL == outputs)
    {
        return status;
    }

    memset(outputs, 0, sizeof(float) * num);
    step = (end - start) / (num - 1);
    sum = start;
    for(i=0; i<num; i++)
    {
        outputs[i] = sum;
        sum += step;
    }

    status = VSI_SUCCESS;
    return status;
}

static vsi_bool _compute_criterion1
    (
    float *score_midpts,
    uint32_t num,
    float thre2
    )
{
    uint32_t i,nonzero;

    nonzero = 0;
    for(i=0; i<num; i++)
    {
        if(score_midpts[i] > thre2)
        {
            nonzero++;
        }
    }

    if(nonzero > (0.8 * num))
    {
        return TRUE;
    }

    return FALSE;
}

static void _sort_con_candidate
    (
    vsi_nn_con_candidate_t *node
    )
{
    vsi_nn_con_candidate_t *p = NULL, *q = NULL;
    vsi_nn_con_candidate_data_t temp;

    p = node;
    while (p)
    {
        q = (vsi_nn_con_candidate_t *)p->link_list.next;
        while (q)
        {
            if(q->data.score > p->data.score)
            {
                memmove(&temp, &q->data, sizeof(vsi_nn_con_candidate_data_t));
                memmove(&q->data, &p->data, sizeof(vsi_nn_con_candidate_data_t));
                memmove(&p->data, &temp, sizeof(vsi_nn_con_candidate_data_t));
            }
            q = (vsi_nn_con_candidate_t *)q->link_list.next;
        }
        p = (vsi_nn_con_candidate_t *)p->link_list.next;
    }
}

static vsi_nn_con_candidate_t *_get_connection_candidate
    (
    float *score_mid,
    vsi_nn_peaks_t *candA,
    vsi_nn_peaks_t *candB,
    uint32_t nA,
    uint32_t nB,
    vsi_nn_cmupose_config_t *config,
    uint32_t *candidate_sum
    )
{
    uint32_t i,j,x,sum;
    vsi_nn_peaks_data_t candA_data,candB_data;
    float norm,vec[2],vec_x[10],vec_y[10],score_midpts[10];
    float score_midpts_sum,score_with_dist_prior;
    int32_t veci[2],r0,r1;
    uint32_t mid_num;
    uint32_t height,width,score_mid_depth,stride;
    vsi_bool criterion1 = FALSE, criterion2 = FALSE;
    float linespace1[10],linespace2[10],startend[10][2];
    vsi_nn_con_candidate_t *con_candidate,*con_candidate_list;

    height = config->image.height;
    width  = config->image.width;
    mid_num = _cnt_of_array(linespace1);  //config->param.mid_num;
    score_mid_depth = 2;
    stride = width * score_mid_depth;
    vsi_nn_LinkListInitRoot(con_candidate_list);
    sum = 0;
    for(i=0; i<nA; i++)
    {
        for(j=0; j<nB; j++)
        {
            _get_peak_data(candB, j, &candB_data);
            _get_peak_data(candA, i, &candA_data);
            veci[0] = candB_data.location[0] - candA_data.location[0];
            veci[1] = candB_data.location[1] - candA_data.location[1];
            norm = sqrtf((float)(veci[0] * veci[0] + veci[1] * veci[1]));
            vec[0] = veci[0] / norm;
            vec[1] = veci[1] / norm;

            _line_space((float)candA_data.location[0], (float)candB_data.location[0],
                mid_num, linespace1);
            _line_space((float)candA_data.location[1], (float)candB_data.location[1],
                mid_num, linespace2);

            score_midpts_sum = 0;
            for(x=0; x<mid_num; x++)
            {
                startend[x][0] = linespace1[x];
                startend[x][1] = linespace2[x];
                //printf("x=%u [ %f %f ]\n", x, linespace1[x], linespace2[x]);

                r0 = (int32_t)vsi_rint(startend[x][1]);
                r1 = (int32_t)vsi_rint(startend[x][0]);
                //printf("r0=%d, r1=%d\n", r0, r1);

                vec_x[x] = score_mid[stride * r0 + score_mid_depth * r1 + 0];
                vec_y[x] = score_mid[stride * r0 + score_mid_depth * r1 + 1];
                //printf("vec_x[%u]=%f, vec_y[%u]=%f\n", x, vec_x[x], x, vec_y[x]);

                score_midpts[x] = vec_x[x] * vec[0] + vec_y[x] * vec[1];
                //printf("score_midpts[%u] = %f\n", x, score_midpts[x]);

                score_midpts_sum += score_midpts[x];
            }
            score_with_dist_prior =
                score_midpts_sum / 10 + vsi_nn_min(0.5f * height / norm - 1, 0);

            criterion1 = _compute_criterion1(score_midpts, 10, config->param.thre2);
            if(score_with_dist_prior > 0)
            {
                criterion2 = TRUE;
            }

            if(criterion1 && criterion2)
            {
                con_candidate = (vsi_nn_con_candidate_t *)
                    vsi_nn_LinkListNewNode(sizeof(vsi_nn_con_candidate_t), _init_candidate);
                CHECK_PTR_FAIL_GOTO( con_candidate, "null point.", final );

                sum++;
                con_candidate->data.i = i;
                con_candidate->data.j = j;
                con_candidate->data.score = score_with_dist_prior;
                con_candidate->data.candAB =
                    score_with_dist_prior + candA_data.score + candB_data.score;

                vsi_nn_LinkListPushEnd(
                    (vsi_nn_link_list_t **)&con_candidate_list,
                    (vsi_nn_link_list_t *)con_candidate );
            }
        }
    }

    *candidate_sum = sum;

final:
    return con_candidate_list;
}

static vsi_status _get_connection_candidate_data
    (
    vsi_nn_con_candidate_t *con_candidate,
    uint32_t index,
    vsi_nn_con_candidate_data_t *data
    )
{
    vsi_nn_con_candidate_t *iter;
    uint32_t n;

    if(NULL == con_candidate || NULL == data)
    {
        return VSI_FAILURE;
    }

    n = 0;
    iter = con_candidate;
    memset(data, 0, sizeof(vsi_nn_con_candidate_data_t));
    while(iter)
    {
        if(n == index)
        {
            data->i = iter->data.i;
            data->j = iter->data.j;
            data->score = iter->data.score;
            data->candAB = iter->data.candAB;
        }
        n++;
        iter = (vsi_nn_con_candidate_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter);
    }

    return VSI_SUCCESS;
}

static vsi_bool _check_connection_candidate_ij
    (
    vsi_nn_con_candidate_data_t *data,
    vsi_nn_connection_t *connection
    )
{
    vsi_nn_connection_t *iter;
    uint32_t i,j;
    if(NULL == data)
    {
        return FALSE;
    }

    i = data->i;
    j = data->j;
    iter = connection;
    while (iter)
    {
        if(iter->data.i == i || iter->data.j == j)
        {
            return FALSE;
        }
        iter = (vsi_nn_connection_t *)vsi_nn_LinkListNext((vsi_nn_link_list_t *)iter);
    }

    return TRUE;
}

static vsi_nn_connection_t *_get_connection
    (
    vsi_nn_con_candidate_t *con_candidate,
    uint32_t candidate_sum,
    vsi_nn_peaks_t *candA,
    vsi_nn_peaks_t *candB,
    uint32_t nA,
    uint32_t nB,
    uint32_t *connection_sum
    )
{
    uint32_t c,sum;
    vsi_nn_connection_t *connection_list,*connection;
    vsi_nn_con_candidate_data_t candidate_data;
    vsi_nn_peaks_data_t candA_data,candB_data;
    vsi_bool ret;

    if(NULL == con_candidate ||
       NULL == candA ||
       NULL == candB ||
       NULL == connection_sum)
    {
        return NULL;
    }

    sum = 0;
    vsi_nn_LinkListInitRoot(connection_list);
    for(c=0; c<candidate_sum; c++)
    {
        _get_connection_candidate_data(con_candidate, c, &candidate_data);

        ret = _check_connection_candidate_ij(&candidate_data, connection_list);
        if(ret == TRUE)
        {
            _get_peak_data(candB, candidate_data.j, &candB_data);
            _get_peak_data(candA, candidate_data.i, &candA_data);
            connection = (vsi_nn_connection_t *)
                vsi_nn_LinkListNewNode(sizeof(vsi_nn_connection_t), _init_connection);
            CHECK_PTR_FAIL_GOTO( connection, "get point fail.", final );

            connection->data.i = candidate_data.i;
            connection->data.j = candidate_data.j;
            connection->data.score = candidate_data.score;
            connection->data.x = candA_data.id;
            connection->data.y = candB_data.id;

            vsi_nn_LinkListPushEnd(
                (vsi_nn_link_list_t **)&connection_list,
                (vsi_nn_link_list_t *)connection );
            sum++;

            #if 0
            connection = connection_list;
            printf("======================= c=%u\n",c);
            while (connection)
            {
                printf("[ %u %u %f %u %u ]\n",
                    connection->data.x, connection->data.y, connection->data.score,
                    connection->data.i, connection->data.j);
                connection = (vsi_nn_connection_t *)
                    vsi_nn_LinkListNext( (vsi_nn_link_list_t *)connection);
            }
            #endif

            if(sum >= vsi_nn_min(nA, nB))
            {
                break;
            }
        }
    }

    #if 0
    connection = connection_list;
    printf("============================\n");
    while (connection)
    {
        printf("[ %u %u %f %u %u ]\n",
            connection->data.x, connection->data.y, connection->data.score,
            connection->data.i, connection->data.j);
        connection = (vsi_nn_connection_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)connection);
    }
    #endif

final:
    return connection_list;
}

static vsi_nn_peaks_data_t *_get_peak_candidate
    (
    vsi_nn_peaks_t **all_peaks,
    uint32_t all_peaks_num,
    uint32_t peak_counter
    )
{
    uint32_t i,j;
    vsi_nn_peaks_data_t *candidate,*iter;
    vsi_nn_peaks_t *peaks;

    if(NULL == all_peaks)
    {
        return NULL;
    }
    candidate = (vsi_nn_peaks_data_t *)malloc(sizeof(vsi_nn_peaks_data_t) * peak_counter);
    CHECK_PTR_FAIL_GOTO( candidate, "Create buffer fail.", final );

    iter = candidate;
    for(i=0,j=0; i< all_peaks_num; i++,j++)
    {
        peaks = all_peaks[i];
        while (peaks)
        {
            memcpy(iter, &peaks->peak, sizeof(vsi_nn_peaks_data_t));
            peaks = (vsi_nn_peaks_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)peaks);
            iter++;
        }
    }

final:
    return candidate;
}

static vsi_bool _check_special_k
    (
    int32_t *special_k,
    uint32_t special_k_num,
    uint32_t index
    )
{
    uint32_t i;

    for(i=0; i<special_k_num; i++)
    {
        if(special_k[i] == (int32_t)index)
        {
            return FALSE;
        }
    }

    return TRUE;
}

static vsi_status _get_partXs
    (
    vsi_nn_connection_t *connection_k,
    uint32_t index,
    int32_t *parts
    )
{
    vsi_nn_connection_t *iter;
    uint32_t num;

    if(NULL == connection_k)
    {
        return VSI_FAILURE;
    }

    num = 0;
    iter = connection_k;
    while (iter)
    {
        if(index == 0)
        {
            parts[num++] = iter->data.x;
        }
        else if(index == 1)
        {
            parts[num++] = iter->data.y;
        }
        else
        {
            return VSI_FAILURE;
        }
        iter = (vsi_nn_connection_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter);
    }

    return VSI_SUCCESS;
}

static vsi_bool _need_merge
    (
    vsi_nn_subset_t *j1_subset,
    vsi_nn_subset_t *j2_subset
    )
{
    uint32_t i,n;

    if(NULL == j1_subset || NULL == j2_subset)
    {
        return FALSE;
    }

    n = _cnt_of_array(j1_subset->data.idx);

    for(i=0; i<n-2; i++)
    {
        int32_t idx1,idx2;

        idx1 = 0, idx2 = 0;
        if(j1_subset->data.idx[i] >= 0)
        {
            idx1 = 1;
        }
        if(j2_subset->data.idx[i] >= 0)
        {
            idx2 = 1;
        }

        if(idx1 == 1 && idx2 == 1)
        {
            return TRUE;
        }
    }

    return FALSE;
}

static void _release_all_connection
    (
    vsi_nn_connection_t **all_connection,
    uint32_t connection_list_num
    )
{
    uint32_t i;
    vsi_nn_connection_t *connection;
    for(i=0; i<connection_list_num; i++)
    {
        connection = all_connection[i];
        vsi_nn_LinkListDeinit((vsi_nn_link_list_t *)connection, NULL);
    }
    if(all_connection)free(all_connection);
}

static vsi_nn_subset_t *_compute_subset
    (
    vsi_nn_connection_t **all_connection,
    uint32_t all_connection_num,
    vsi_nn_peaks_data_t *candidate,
    int32_t *special_k,
    uint32_t *subset_num
    )
{
    uint32_t i,j,k,n,num;
    uint32_t mapIdx_len,indexA,indexB;
    vsi_bool ret;
    vsi_nn_connection_t *connection_k = NULL;
    vsi_nn_subset_t *subset_list = NULL, *subset = NULL;
    uint32_t *deleteIdx = NULL;

    VSI_UNREFERENCED(all_connection_num);

    if(NULL == all_connection ||
       NULL == candidate ||
       NULL == special_k ||
       NULL == subset_num)
    {
        return NULL;
    }

    connection_k = NULL;
    mapIdx_len = _cnt_of_array(mapIdx);
    num = 0;
    vsi_nn_LinkListInitRoot(subset_list);

    for(k=0; k<mapIdx_len; k++)
    {
        ret = _check_special_k(special_k, mapIdx_len, k);
        if(ret == TRUE)
        {
            int32_t partAs[32];
            int32_t partBs[32];

            connection_k = all_connection[k];

            memset(partAs, -1, sizeof(int32_t) * 32);
            memset(partBs, -1, sizeof(int32_t) * 32);
            _get_partXs(connection_k, 0, partAs);
            _get_partXs(connection_k, 1, partBs);

            indexA = limbSeq[k][0] - 1;
            indexB = limbSeq[k][1] - 1;

            n = vsi_nn_LinkListGetNodeNumber((vsi_nn_link_list_t *)connection_k);
            for(i=0; i<n; i++)
            {
                int32_t found = 0;
                int32_t subset_idx[2] = {-1};
                vsi_nn_subset_t *sig_subset = NULL;
                vsi_nn_connection_t *sig_connect = NULL;

                for(j=0; j<num; j++)
                {
                    sig_subset= (vsi_nn_subset_t *)
                        vsi_nn_LinkListGetIndexNode((vsi_nn_link_list_t *)subset_list, j);
                    CHECK_PTR_FAIL_GOTO( sig_subset, "null point.", final );

                    if(sig_subset->data.idx[indexA] == partAs[i] ||
                       sig_subset->data.idx[indexB] == partBs[i])
                    {
                        subset_idx[found] = j;
                        found += 1;
                    }
                }

                if(found == 1)
                {
                    j = subset_idx[0];
                    sig_subset= (vsi_nn_subset_t *)
                        vsi_nn_LinkListGetIndexNode((vsi_nn_link_list_t *)subset_list, j);
                    CHECK_PTR_FAIL_GOTO( sig_subset, "get point fail.", final );
                    if(sig_subset->data.idx[indexB] != partBs[i])
                    {
                        int32_t ii = partBs[i];
                        sig_connect = (vsi_nn_connection_t *)
                            vsi_nn_LinkListGetIndexNode((vsi_nn_link_list_t *)connection_k, i);
                        CHECK_PTR_FAIL_GOTO( sig_connect, "get point fail.", final );

                        sig_subset->data.idx[indexB] = (float)ii;
                        sig_subset->data.idx[20 - 1] += 1;
                        sig_subset->data.idx[20 - 2] +=
                            candidate[ii].score + sig_connect->data.score;
                    }
                }
                else if(found == 2)
                {
                    int32_t j1 = subset_idx[0];
                    int32_t j2 = subset_idx[1];
                    vsi_nn_subset_t *j1_subset = (vsi_nn_subset_t *)
                        vsi_nn_LinkListGetIndexNode((vsi_nn_link_list_t *)subset_list, j1);
                    vsi_nn_subset_t *j2_subset = NULL;
                    CHECK_PTR_FAIL_GOTO( j1_subset, "get point fail.", final );
                    j2_subset = (vsi_nn_subset_t *)
                        vsi_nn_LinkListGetIndexNode((vsi_nn_link_list_t *)subset_list, j2);
                    CHECK_PTR_FAIL_GOTO( j2_subset, "get point fail.", final );
                    if(_need_merge(j1_subset, j2_subset) == FALSE)
                    {
                        uint32_t ii;
                        vsi_nn_subset_t *j1_iter = j1_subset;
                        vsi_nn_subset_t *j2_iter = j2_subset;
                        sig_connect = (vsi_nn_connection_t *)
                            vsi_nn_LinkListGetIndexNode((vsi_nn_link_list_t *)connection_k, i);
                        CHECK_PTR_FAIL_GOTO( sig_connect, "get point fail.", final );

                        for(ii=0; ii<(20-2); ii++)
                        {
                            j1_iter->data.idx[ii] += j2_iter->data.idx[ii] + 1;
                        }
                        for(ii=(20-2); ii<20; ii++)
                        {
                            j1_iter->data.idx[ii] += j2_iter->data.idx[ii];
                        }
                        j1_iter->data.idx[20 - 2] += sig_connect->data.score;
                        vsi_nn_LinkListDelIndexNode((vsi_nn_link_list_t **)&subset_list, j2);
                        num--;
                    }
                    else
                    {
                        float sum = 0.f;
                        int32_t ii = partBs[i];
                        sig_connect = (vsi_nn_connection_t *)
                            vsi_nn_LinkListGetIndexNode((vsi_nn_link_list_t *)connection_k, i);
                        CHECK_PTR_FAIL_GOTO( sig_connect, "get point fail.", final );

                        sum = candidate[ii].score + sig_connect->data.score;
                        j1_subset->data.idx[indexB] = (float)ii;
                        j1_subset->data.idx[20 - 1] += 1;
                        j1_subset->data.idx[20 - 2] += sum;
                    }
                }
                else if(found == 0 && k < 17)
                {
                    float sum = 0.f;
                    float row[20] = {-1.0};
                    uint32_t t1 = 0, t2 = 0;
                    uint32_t l = 0;
                    float s1 = 0.f, s2 = 0.f;
                    sig_connect = (vsi_nn_connection_t *)
                        vsi_nn_LinkListGetIndexNode((vsi_nn_link_list_t *)connection_k, i);
                    CHECK_PTR_FAIL_GOTO( sig_connect, "get point fail.", final );
                    t1 = sig_connect->data.x;
                    t2 = sig_connect->data.y;
                    s1 = candidate[t1].score;
                    s2 = candidate[t2].score;
                    sum = s1 + s2 + sig_connect->data.score;

                    for(l=0; l<_cnt_of_array(row); l++)
                    {
                        row[l] = -1.0;
                    }
                    row[indexA] = (float)partAs[i];
                    row[indexB] = (float)partBs[i];
                    row[20 - 1] = 2.f;
                    row[20 - 2] = sum;

                    subset = (vsi_nn_subset_t *)
                        vsi_nn_LinkListNewNode(sizeof(vsi_nn_subset_t), _init_subset);
                    CHECK_PTR_FAIL_GOTO( subset, "null point.", final );
                    memcpy(&subset->data, row, sizeof(float) * 20);

                    vsi_nn_LinkListPushEnd(
                        (vsi_nn_link_list_t **)&subset_list,
                        (vsi_nn_link_list_t *)subset );
                    num++;
                }

                connection_k = (vsi_nn_connection_t *)
                    vsi_nn_LinkListNext( (vsi_nn_link_list_t *)connection_k);
            } /* end for(i=0; i<n; i++) */
        } /* end if(ret == TRUE) */
    } /* end for(k=0; k<mapIdx_len; k++) */

    deleteIdx = (uint32_t *)malloc(sizeof(uint32_t) * num);
    CHECK_PTR_FAIL_GOTO( deleteIdx, "Create buffer fail.", final );
    memset(deleteIdx, -1, sizeof(uint32_t) * num);

    subset = subset_list;
    CHECK_PTR_FAIL_GOTO( subset, "null point.", final );
    for(i=0,j=0; i<num; i++)
    {
        float tmp1 = subset->data.idx[20 - 1];
        float tmp2 = subset->data.idx[20 - 2];
        if(tmp1 < 4 || (tmp2 / tmp1) < 0.4)
        {
            deleteIdx[j++] = i;
        }
        subset = (vsi_nn_subset_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)subset);
    }
    for(i=0; i<num; i++)
    {
        if(deleteIdx[i] != (uint32_t)-1)
        {
            vsi_nn_LinkListDelIndexNode((vsi_nn_link_list_t **)&subset_list, deleteIdx[i]);
            num--;
        }
    }

final:
    if(deleteIdx)free(deleteIdx);
    return subset_list;
}

static vsi_nn_connection_t **_compute_all_connetion
    (
    float *paf_avg,
    vsi_nn_peaks_t **all_peaks,
    vsi_nn_cmupose_config_t *config,
    uint32_t *connection_list_num,
    int32_t **special
    )
{
    uint32_t k;
    float *score_mid;
    vsi_nn_peaks_t *candA = NULL, *candB = NULL;
    uint32_t nA,nB;
    uint32_t height,width,score_mid_depth;
    vsi_nn_con_candidate_t *con_candidate_list;
    vsi_nn_connection_t **connection_all = NULL;
    int32_t *special_k = NULL;
    uint32_t mapIdx_len,candidate_sum,connection_sum;

    mapIdx_len = _cnt_of_array(mapIdx);
    height = config->image.height;
    width  = config->image.width;
    score_mid_depth = 2;

    score_mid = (float *)malloc(sizeof(float) * height * width * score_mid_depth);
    CHECK_PTR_FAIL_GOTO( score_mid, "Create buffer fail.", final );
    connection_all = (vsi_nn_connection_t **)malloc(sizeof(vsi_nn_connection_t *) * mapIdx_len);
    CHECK_PTR_FAIL_GOTO( connection_all, "Create buffer fail.", final );
    special_k = (int32_t *)malloc(sizeof(int32_t) * mapIdx_len);
    CHECK_PTR_FAIL_GOTO( special_k, "Create buffer fail.", final );

    memset(connection_all, 0, sizeof(vsi_nn_connection_t *) * mapIdx_len);
    memset(special_k, -1, sizeof(int32_t) * mapIdx_len);
    for(k=0; k<mapIdx_len; k++)
    {
        _get_score_mid(paf_avg, mapIdx[k], config, score_mid);
        //savetxt("sheldon/score_mid.txt", score_mid, 320*320*2);

        candA = NULL, candB = NULL;
        nA = 0, nB = 0;
        _get_peaks(all_peaks, limbSeq[k][0], &candA, &nA);
        _get_peaks(all_peaks, limbSeq[k][1], &candB, &nB);

        if(nA != 0 && nB != 0)
        {
            candidate_sum = 0;
            con_candidate_list = _get_connection_candidate(score_mid,
                candA, candB,
                nA, nB,
                config,
                &candidate_sum);

#if 0
            printf("=======================================\n");
            iter = con_candidate_list;
            while (iter)
            {
                printf("con_candidate[ %u %u %f %f ]\n",
                    iter->data.i,iter->data.j,iter->data.score,iter->data.candAB);
                iter = (vsi_nn_con_candidate_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter);
            }
#endif
            _sort_con_candidate(con_candidate_list);
#if 0
            printf("======================================= k=%u\n",k);
            iter = con_candidate_list;
            while (iter)
            {
                printf("con_candidate[ %u %u %f %f ]\n",
                    iter->data.i,iter->data.j,iter->data.score,iter->data.candAB);
                iter = (vsi_nn_con_candidate_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter);
            }
#endif

            connection_sum = 0;
            connection_all[k] = _get_connection(con_candidate_list,
                candidate_sum,
                candA, candB,
                nA, nB,
                &connection_sum);
#if 0
            printf("=======================================\n",k);
            iter = connection_all[k];
            while (iter)
            {
                printf("connection[%u] = [ %u %u %f %u %u ]\n",
                    iter->data.x,iter->data.y,iter->data.score,iter->data.i,iter->data.j);
                iter = (vsi_nn_connection_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter);
            }
#endif

            vsi_nn_LinkListDeinit((vsi_nn_link_list_t *)con_candidate_list, NULL);
        }
        else
        {
            special_k[k] = k;
            connection_all[k] = NULL;
        }
    }

    *connection_list_num = mapIdx_len;
    *special = special_k;

final:
    vsi_nn_safe_free(score_mid);
    return connection_all;
}

static vsi_nn_peaks_t **_compute_all_peaks
    (
    float *heatmap_avg,
    vsi_nn_cmupose_config_t *config,
    uint32_t *peak_conter,
    uint32_t *peak_list_num
    )
{
    uint32_t i, part, loop;
    uint32_t width,height;
    vsi_size_t sz, j;
    float *map_ori = NULL, *map = NULL;
    float *pheatmap_avg = NULL, *pmap_ori = NULL;
    float *map_left = NULL, *map_right = NULL, *map_up = NULL, *map_down = NULL;
    vsi_nn_peaks_t **all_peaks = NULL;

    vsi_size_t szf;

    if(NULL == heatmap_avg || NULL == peak_conter)
    {
        return NULL;
    }

    width = config->image.width;
    height = config->image.height;
    sz = width * height;
    szf = sizeof(float) * sz;
    map_ori   = (float *)malloc(szf);
    CHECK_PTR_FAIL_GOTO( map_ori, "Create buffer fail.", final );
    map       = (float *)malloc(szf);
    CHECK_PTR_FAIL_GOTO( map, "Create buffer fail.", final );
    map_left  = (float *)malloc(szf);
    CHECK_PTR_FAIL_GOTO( map_left, "Create buffer fail.", final );
    map_right = (float *)malloc(szf);
    CHECK_PTR_FAIL_GOTO( map_right, "Create buffer fail.", final );
    map_up    = (float *)malloc(szf);
    CHECK_PTR_FAIL_GOTO( map_up, "Create buffer fail.", final );
    map_down  = (float *)malloc(szf);
    CHECK_PTR_FAIL_GOTO( map_down, "Create buffer fail.", final );

    loop = 19 -1; // why?
    all_peaks = (vsi_nn_peaks_t **)malloc(sizeof(vsi_nn_peaks_t *) * loop);
    CHECK_PTR_FAIL_GOTO( all_peaks, "Create buffer fail.", final );
    memset(all_peaks, 0, sizeof(vsi_nn_peaks_t *) * loop);
    peak_id = 0;

    //heatmap_avg[320][320][19]
    for(part=0; part<loop; part++)
    {
        memset(map_ori,   0, szf);
        memset(map,       0, szf);

        pheatmap_avg = heatmap_avg + part;
        pmap_ori = map_ori;
        for(j=0; j<sz; j++)
        {
            *pmap_ori = *pheatmap_avg;
            pmap_ori += 1;
            pheatmap_avg += 19;
        }
        //savetxt("sheldon/map_ori.txt", map_ori, sz);

        gaussian_filter(map_ori, 3.f, config, map);
        //savetxt("sheldon/map.txt", map, sz);

        memset(map_left,  0, szf);
        memcpy(map_left + width, map, sizeof(float) * (sz - width));
        //savetxt("sheldon/map_left.txt", map_left, sz);

        memset(map_right, 0, szf);
        memcpy(map_right, map + width, sizeof(float) * (sz - width));
        //savetxt("sheldon/map_right.txt", map_right, sz);

        memset(map_up,    0, szf);
        memset(map_down,  0, szf);
        for(i=0; i<height; i++)
        {
            memcpy(map_up + i * width + 1, map + i * width, sizeof(float) * (width - 1));
            memcpy(map_down + i * width, map + i * width + 1, sizeof(float) * (width - 1));
        }
        //savetxt("sheldon/map_up.txt", map_up, sz);
        //savetxt("sheldon/map_down.txt", map_down, sz);

        all_peaks[part] = _compute_peaks(map_ori,
            map,
            map_left,
            map_right,
            map_up,
            map_down,
            config);
    }

    *peak_list_num = loop;
    *peak_conter = peak_id;

final:
    vsi_nn_safe_free(map);
    vsi_nn_safe_free(map_ori);
    vsi_nn_safe_free(map_left);
    vsi_nn_safe_free(map_right);
    vsi_nn_safe_free(map_up);
    vsi_nn_safe_free(map_down);
    return all_peaks;
}

static void _fill_heatmap_avg
    (
    float *net_out,
    vsi_nn_cmupose_config_t *config,
    float *heatmap_avg
    )
{
    uint32_t net_out_c,width,height,channel;
    uint32_t i,j,index_net_out,index_buffer;
    float *buffer = NULL;
    vsi_size_t sz,szf,channelf;
    uint32_t stride_h_nc,stride_h_c;

    // shape = [width][height][channel]
    uint32_t size1[3] = {0};
    uint32_t size2[3] = {0};
    size1[0] = (uint32_t)config->inputs.net_out->attr.size[1];
    size1[1] = (uint32_t)config->inputs.net_out->attr.size[0];
    size1[2] = 19;

    size2[0] = config->image.height;
    size2[1] = config->image.width;
    size2[2] = 19;

    net_out_c = 57;
    width = size1[1];
    height = size1[0];
    channel = 19;
    sz = width * height * channel;
    szf = sizeof(float) * sz;
    channelf = channel*sizeof(float);
    buffer = (float *)malloc(szf);
    CHECK_PTR_FAIL_GOTO( buffer, "Create buffer fail.", final );
    memset(buffer, 0, szf);

    stride_h_nc = height * net_out_c;
    stride_h_c  = height * channel;
    for(i=0; i<height; i++)
    {
        for(j=0; j<width; j++)
        {
            index_net_out = i * stride_h_nc + j * net_out_c;
            index_buffer = i * stride_h_c + j * channel;
            memcpy(buffer + index_buffer, net_out + index_net_out, channelf);
        }
    }
    //savetxt("sheldon/heatmap.txt", buffer, sz);
    resize_nearest(buffer, size1, heatmap_avg, size2);
    //savetxt("sheldon/heatmap_avg.txt", heatmap_avg, config->image.height*config->image.width*19);

final:
    vsi_nn_safe_free(buffer);
}

static void _fill_paf_avg
    (
    float *net_out,
    vsi_nn_cmupose_config_t *config,
    float *paf_avg
    )
{
    uint32_t net_out_c,width,height,channel;
    vsi_size_t i,j,index_net_out,index_buffer;
    float *buffer = NULL;

    vsi_size_t sz,szf,cf,stride_h_nc,stride_h_c;

    //[width, height, channel]
    uint32_t size1[3] = {0};
    uint32_t size2[3] = {0};
    size1[0] = (uint32_t)config->inputs.net_out->attr.size[1];
    size1[1] = (uint32_t)config->inputs.net_out->attr.size[0];
    size1[2] = 38;

    size2[0] = config->image.height;
    size2[1] = config->image.width;
    size2[2] = 38;

    net_out_c = 57;
    width = size1[1];
    height = size1[0];
    channel = 38;
    sz = width * height * channel;
    szf = sizeof(float) * sz;
    cf = channel*sizeof(float);
    buffer = (float *)malloc(szf);
    CHECK_PTR_FAIL_GOTO( buffer, "Create buffer fail.", final );
    memset(buffer, 0, szf);

    stride_h_nc = height * net_out_c;
    stride_h_c  = height * channel;
    for(i=0; i<height; i++)
    {
        for(j=0; j<width; j++)
        {
            index_net_out = i * stride_h_nc + j * net_out_c + 19; // 19 = 57 - 38
            index_buffer = i * stride_h_c + j * channel;
            memcpy(buffer + index_buffer, net_out + index_net_out, cf);
        }
    }
    //savetxt("sheldon/paf.txt", buffer, sz);
    resize_nearest(buffer, size1, paf_avg, size2);
    //savetxt("sheldon/paf_avg.txt", paf_avg, config->image.width*config->image.height*38);

final:
    vsi_nn_safe_free(buffer);
}

vsi_status vsi_nn_CMUPose_Post_Process
    (
    float *net_out,
    vsi_nn_cmupose_config_t *config,
    vsi_nn_peaks_t ***all_peaks_out,
    uint32_t *all_peaks_num_out,
    vsi_nn_subset_t **subset_list_out,
    vsi_nn_peaks_data_t **peak_candidate_out,
    uint32_t *peak_candidate_num_out
    )
{
    vsi_status status;
    vsi_nn_cmupose_multiplier_t multiplier;
    float *heatmap_avg = NULL, *paf_avg = NULL;
    vsi_nn_peaks_t **all_peaks = NULL;
    vsi_nn_peaks_data_t *peak_candidate = NULL;
    vsi_nn_connection_t **all_connection = NULL;
    vsi_nn_subset_t *subset_list = NULL;
    int32_t *special_k = NULL;
    uint32_t peak_counter = 0;
    uint32_t peak_list_num = 0, connection_list_num = 0, subset_num = 0;
    //uint32_t n;

    status = VSI_FAILURE;
    if(NULL == config ||
        NULL == all_peaks_out ||
        NULL == all_peaks_num_out ||
        NULL == subset_list_out ||
        NULL == peak_candidate_out ||
        NULL == peak_candidate_num_out)
    {
        return status;
    }

    status = _cmupose_init(config, &multiplier, &heatmap_avg, &paf_avg);
    if(VSI_SUCCESS != status)
    {
        return status;
    }

    _fill_heatmap_avg(net_out, config, heatmap_avg);
    _fill_paf_avg(net_out, config, paf_avg);

    all_peaks = _compute_all_peaks(heatmap_avg, config, &peak_counter, &peak_list_num);
    CHECK_PTR_FAIL_GOTO( all_peaks, "Create buffer fail.", final );
#if 0
    for(n=0; n<peak_list_num; n++)
    {
        vsi_nn_peaks_t *iter = all_peaks[n];
        printf("peak_list[%u]=================\n", n);
        while(iter)
        {
            printf("peak[%u %u %f %u]\n",
                iter->peak.location[0], iter->peak.location[1], iter->peak.score, iter->peak.id);
            iter = (vsi_nn_peaks_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter);
        }
    }
#endif

    all_connection = _compute_all_connetion(paf_avg, all_peaks, config, &connection_list_num, &special_k);
    CHECK_PTR_FAIL_GOTO( all_connection, "Create buffer fail.", final );
#if 0
    for(n=0; n<connection_list_num; n++)
    {
        vsi_nn_connection_t *iter = all_connection[n];
        printf("all_connection[%u]=================total[%u]\n", n, connection_list_num);
        while(iter)
        {
            printf("connection[%u %u %f %u %u]\n",
                iter->data.x, iter->data.y, iter->data.score, iter->data.i, iter->data.j);
            iter = (vsi_nn_connection_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter);
        }
    }
#endif

    peak_candidate = _get_peak_candidate(all_peaks, peak_list_num, peak_counter);
#if 0
    for(n=0; n<peak_counter; n++)
    {
        printf("peak_candidate[%u] = [ %u %u %f %u ]\n", n,
            peak_candidate[n].location[0],peak_candidate[n].location[1],
            peak_candidate[n].score,peak_candidate[n].id);
    }
#endif

    subset_list = _compute_subset(all_connection,
        connection_list_num,
        peak_candidate,
        special_k,
        &subset_num);

    *all_peaks_out = all_peaks;
    *all_peaks_num_out = peak_list_num;
    *subset_list_out = subset_list;
    *peak_candidate_out = peak_candidate;
    *peak_candidate_num_out = peak_counter;
    status = VSI_SUCCESS;

final:
    _cmupose_deinit(&multiplier, heatmap_avg, paf_avg);
    if (all_connection)
    {
        _release_all_connection(all_connection, connection_list_num);
    }
    vsi_nn_safe_free(special_k);

    return status;
}

static float *_get_net_out_data
    (
    vsi_nn_graph_t *graph,
    vsi_nn_cmupose_config_t *config
    )
{
    vsi_nn_tensor_t tensor,*net_out;
    vsi_size_t perm[3] = {1,2,0};
    float *buffer;
    vsi_bool ret = FALSE;

    net_out = config->inputs.net_out;
    memset(&tensor, 0, sizeof(vsi_nn_tensor_t));
    tensor.attr.dim_num = net_out->attr.dim_num;
    tensor.attr.size[0] = net_out->attr.size[2];
    tensor.attr.size[1] = net_out->attr.size[1];
    tensor.attr.size[2] = net_out->attr.size[0];
    memcpy(&tensor.attr.dtype, &net_out->attr.dtype, sizeof(vsi_nn_dtype_t));
    ret = vsi_nn_ReshapeTensor(graph,
                               net_out,
                               &tensor,
                               tensor.attr.size,
                               tensor.attr.dim_num);
    if(ret != TRUE)
    {
        return NULL;
    }

    vsi_nn_TransposeTensor(graph, &tensor, perm, tensor.attr.dim_num, NULL);
    buffer  = vsi_nn_ConvertTensorToFloat32Data(graph, &tensor);

    // for test
    //savetxt("sheldon/net_out.txt", buffer, (net_out->attr.size[0]*net_out->attr.size[1]*57));

    //vxReleaseTensor(&tensor.t);
    return buffer;
}

static vsi_status _auto_fill_cmupose
    (
    vsi_nn_graph_t *graph,
    vsi_nn_cmupose_inputs_t *inputs,
    vsi_nn_cmupose_image_t *image,
    vsi_nn_cmupose_param_t *param,
    vsi_nn_cmupose_model_t *model,
    vsi_nn_cmupose_config_t *cmupose_config
    )
{
    vsi_status status;
    vsi_nn_tensor_t *net_in = NULL;
    static float default_scale_search[1] = {1};

    VSI_UNREFERENCED(image);
    VSI_UNREFERENCED(param);
    VSI_UNREFERENCED(model);

    status = VSI_FAILURE;
    if(NULL == graph)
    {
        return status;
    }

    // fill input
    if(inputs == NULL)
    {
        cmupose_config->inputs.net_out = vsi_nn_GetTensor(graph, graph->output.tensors[0]);
    }
    else
    {
        cmupose_config->inputs.net_out = inputs->net_out;
    }

    // fill image
    net_in = vsi_nn_GetTensor(graph, graph->input.tensors[0]);
    CHECK_PTR_FAIL_GOTO( net_in, "Create tensor fail.", final );
    cmupose_config->image.width = (uint32_t)net_in->attr.size[0];
    cmupose_config->image.height = (uint32_t)net_in->attr.size[1];
    cmupose_config->image.channel = (uint32_t)net_in->attr.size[2];

    // fill param
    cmupose_config->param.scale_search.size = default_scale_search;
    cmupose_config->param.scale_search.num = _cnt_of_array(default_scale_search);
    cmupose_config->param.mid_num = 10;
    cmupose_config->param.thre1 = 0.1f;
    cmupose_config->param.thre2 = 0.05f;
    cmupose_config->param.thre3 = 0.5f;

    // fill model
    cmupose_config->model.boxsize = 368;
    cmupose_config->model.stride = 8;
    cmupose_config->model.padValue = 128;

    status = VSI_SUCCESS;

final:
    return status;
}

vsi_status vsi_nn_CMUPose_PostProcess
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
    )
{
    vsi_status status;
    float *net_out;
    vsi_nn_cmupose_config_t cmupose_config;

    status = VSI_FAILURE;
    net_out = NULL;

    memset(&cmupose_config, 0, sizeof(vsi_nn_cmupose_config_t));
    status = _auto_fill_cmupose(graph,
        inputs,
        image,
        param,
        model,
        &cmupose_config);
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    net_out = _get_net_out_data(graph, &cmupose_config);
    if(NULL == net_out)
    {
        goto final;
    }

    status = vsi_nn_CMUPose_Post_Process(net_out,
                                         &cmupose_config,
                                         all_peaks,
                                         all_peaks_num,
                                         subset,
                                         candidate,
                                         candidate_num);
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    status = VSI_SUCCESS;
final:
    if(net_out)free(net_out);
    return status;
}
