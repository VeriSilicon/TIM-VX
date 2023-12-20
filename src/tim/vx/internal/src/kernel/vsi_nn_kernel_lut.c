/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "vsi_nn_context.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include <float.h>
#include "vsi_nn_error.h"
#include "utils/vsi_nn_dtype_util_prv.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_lut.h"
#include "utils/vsi_nn_dtype_util.h"

static int32_t _comparator(const void *pa, const void *pb)
{
    vsi_nn_kernel_lut_t a = *(vsi_nn_kernel_lut_t *)pa;
    vsi_nn_kernel_lut_t b = *(vsi_nn_kernel_lut_t *)pb;
    float diff = a.index - b.index;

    if ( diff > 0)
    {
        return 1;
    }
    else if ( diff < 0)
    {
        return -1;
    }

    return 0;
}

static float exp_eval(float val)
{
    return expf(val);
}

static float log_eval(float data)
{
    return logf(data);
}

static float selu_eval(float data, vsi_nn_kernel_lut_params *lut_param)
{
    float alpha = lut_param->params[0];
    float gamma = lut_param->params[1];
    return data >=0 ? data * gamma : expf(data) * alpha * gamma - alpha * gamma;
}

static float neg_eval(float data)
{
    return data * -1.0f;
}

static float hsigmoid_eval(float data, vsi_nn_kernel_lut_params *lut_param)
{
    float alpha = lut_param->params[0];
    float beta = lut_param->params[1];

    data = (float)(alpha * data + beta);
    data = vsi_nn_clamp(data, 0, 1);

    return data;
}

static float soft_plus_eval(float data)
{
    return log_eval(exp_eval(data) + 1);
}

static float mish_eval(float data)
{
    data = (float)(data * tanh(soft_plus_eval(data)));

    return data;
}

static float erf_eval(float x)
{
    float res = 0;
    float tmp = x;
    float factorial = 1; /*n!*/
    float x_pow = x;
    int32_t one = 1;
    int32_t n = 1;

    if (x <= -3)
    {
        return -1;
    }
    else if (x >= 3)
    {
        return 1;
    }

    while (vsi_abs(tmp) > 1e-5)
    {
        res += tmp;

        factorial *= n;
        one *= -1;
        x_pow *= x * x;
        tmp = one / factorial * x_pow / ( 2 * n + 1);

        n ++;
    }
#define VSI_MUL2_RSQRTPI    (1.1283791670955126f)

    res *= VSI_MUL2_RSQRTPI;

    return res;
}

static float gelu_eval(float data)
{
    data = (float)(0.5f * data * (1 + erf_eval(data / (float)sqrt(2.0f))));

    return data;
}

#define VSI_SQRT_2_RCP_PI  0.7978845834732056f
static float hgelu_eval(float data)
{
    float cdf = (float)(0.5f * (1.0f + tanh((VSI_SQRT_2_RCP_PI *
        (data + 0.044715f * data * data * data)))));

    return data * cdf;
}

static float relu_keras_eval(float val, vsi_nn_kernel_lut_params *lut_param)
{
    float alpha = lut_param->params[0];
    float max = lut_param->params[1];
    float threshold = lut_param->params[2];

    val = vsi_nn_min(val, max);
    val = val < threshold ? alpha * (val - threshold) : val;
    return val;
}

static float clip_eval(float val, vsi_nn_kernel_lut_params *lut_param)
{
    float min = lut_param->params[0];
    float max = lut_param->params[1];

    return vsi_nn_clamp(val, min, max);
}

static float square_eval(float x)
{
    return x * x;
}

static float celu_eval(float x, vsi_nn_kernel_lut_params *lut_param)
{
    float alpha = lut_param->params[0];
    float positive = vsi_nn_max(0, x);
    float negative = vsi_nn_min(alpha * (expf(x / alpha) - 1), 0);

    return positive + negative;
}

static float rcp_eval(float x)
{
    return 1.0f / x;
}

static float softsign_eval(float x)
{
    return x / (1 + vsi_abs(x));
}

static float linear_exp_eval(float x, vsi_nn_kernel_lut_params *lut_param)
{
    float a = lut_param->params[0];
    float b = lut_param->params[1];

    return expf(x * a + b);
}

static float linear_rsqrt_eval(float x, vsi_nn_kernel_lut_params *lut_param)
{
    float a = lut_param->params[0];
    float b = lut_param->params[1];
    float scale = lut_param->params[2];

    return scale / sqrtf(a * x + b);
}

static float linear_sigmoid_eval(float x, vsi_nn_kernel_lut_params *lut_param)
{
    float a = lut_param->params[0];
    float b = lut_param->params[1];

    return 1.0f / (1 + expf(a * x + b));
}

static float atan_eval(float x)
{
    return atanf(x);
}

static float atanh_eval(float x)
{
    return (log_eval(1 + x) - log_eval(1 - x)) / 2;
}

static float acosh_eval(float x)
{
    return (log_eval(x + (float)sqrt(x * x - 1)));
}

static float inverse_sigmoid_eval(float x, vsi_nn_kernel_lut_params *lut_param)
{
    float eps = lut_param->params[0];
    float x1, x2;
    x = vsi_nn_clamp(x, 0, 1);
    x1 = vsi_nn_clamp(x, eps, 1);
    x2 = vsi_nn_clamp((1 - x), eps, 1);

    return log_eval(x1 / x2);
}

static float vsi_nn_kernel_lut_activation(float data, vsi_nn_kernel_lut_params *lut_param)
{
    float result = 0;

    switch (lut_param->act_type)
    {
    case VSI_NN_KERNEL_LUT_MISH:
        result =  mish_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_LOG:
        result =  log_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_EXP:
        result =  exp_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_SELU:
        result =  selu_eval(data, lut_param);
        break;
    case VSI_NN_KERNEL_LUT_NEG:
        result =  neg_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_HSIGMOID:
        result =  hsigmoid_eval(data, lut_param);
        break;
    case VSI_NN_KERNEL_LUT_SOFT_PLUS:
        result =  soft_plus_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_ERF:
        result =  erf_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_GELU:
        result =  gelu_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_HGELU:
        result =  hgelu_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_RELU_KERAS:
        result =  relu_keras_eval(data, lut_param);
        break;
    case VSI_NN_KERNEL_LUT_CLIP:
        result =  clip_eval(data, lut_param);
        break;
    case VSI_NN_KERNEL_LUT_SQUARE:
        result =  square_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_CELU:
        result =  celu_eval(data, lut_param);
        break;
    case VSI_NN_KERNEL_LUT_RCP:
        result =  rcp_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_SOFTSIGN:
        result = softsign_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_LINEAR_EXP:
        result = linear_exp_eval(data, lut_param);
        break;
    case VSI_NN_KERNEL_LUT_LINEAR_RSQRT:
        result = linear_rsqrt_eval(data, lut_param);
        break;
    case VSI_NN_KERNEL_LUT_LINEAR_SIGMOID:
        result = linear_sigmoid_eval(data, lut_param);
        break;
    case VSI_NN_KERNEL_LUT_ATAN:
        result = atan_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_ATANH:
        result = atanh_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_ACOSH:
        result = acosh_eval(data);
        break;
    case VSI_NN_KERNEL_LUT_INVERSE_SIGMOID:
        result = inverse_sigmoid_eval(data, lut_param);
        break;
    default:
        VSILOGE( "unsupported activation function:%d", lut_param->act_type );
        break;
    }

    return result;
}

vsi_status vsi_nn_kernel_lut_positive
    (
    vx_lut index_lut,
    vx_lut output_lut,
    vsi_nn_kernel_lut_params *param
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_lut_t *lut = NULL;
    uint32_t i = 0;
    float clamp_min = 0;
    float index[VSI_NN_KERNEL_LUT_MAX_SIZE] = {0};
    float value[VSI_NN_KERNEL_LUT_MAX_SIZE] = {0};

    if (index_lut == NULL || output_lut == NULL || param == NULL)
    {
        return VSI_FAILURE;
    }

    lut = (vsi_nn_kernel_lut_t *)calloc(VSI_NN_KERNEL_LUT_MAX_SIZE, sizeof(vsi_nn_kernel_lut_t));
    CHECK_PTR_FAIL_GOTO( lut, "Create LUT buffer fail.", final );
    memset(lut, 0, sizeof(vsi_nn_kernel_lut_t) * VSI_NN_KERNEL_LUT_MAX_SIZE);

    clamp_min = param->clamp_min;

    for ( i = 0; i < VSI_NN_KERNEL_LUT_MAX_SIZE; i++)
    {
        int16_t val = (int16_t)(i << 5);
        float fidx = fp16_to_fp32(val);

        if (val < 0)
        {
            continue;
        }

        if (param->pwl_sign_remove_support && fidx < clamp_min)
        {
            fidx = clamp_min;
        }

        lut[i].index = fidx;
        lut[i].val = vsi_nn_kernel_lut_activation(lut[i].index, param);
    }

    for (i = 992; i < VSI_NN_KERNEL_LUT_MAX_SIZE; i++)
    {
        lut[i].index = VSI_NN_KERNEL_LUT_FP16_MAX;
        lut[i].val = vsi_nn_kernel_lut_activation(lut[i].index, param);
    }

    qsort(lut, VSI_NN_KERNEL_LUT_MAX_SIZE, sizeof(vsi_nn_kernel_lut_t), _comparator);

    for ( i = 0; i < VSI_NN_KERNEL_LUT_MAX_SIZE; i++)
    {
        index[i] = lut[i].index;
        value[i] = lut[i].val;
    }

    status  = vxCopyLUT(index_lut, (void*)&index, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyLUT(output_lut, (void*)&value, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
final:
    vsi_nn_safe_free(lut);

    return status;
}

vsi_status vsi_nn_kernel_lut_all
    (
    vx_lut index_lut,
    vx_lut output_lut,
    vsi_nn_kernel_lut_params *param
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_lut_t *lut = NULL;
    uint32_t i = 0;
    float clamp_min = 0;
    float index[VSI_NN_KERNEL_LUT_MAX_SIZE] = {0};
    float value[VSI_NN_KERNEL_LUT_MAX_SIZE] = {0};

    if (index_lut == NULL || output_lut == NULL || param == NULL)
    {
        return VSI_FAILURE;
    }

    lut = (vsi_nn_kernel_lut_t *)calloc(VSI_NN_KERNEL_LUT_MAX_SIZE, sizeof(vsi_nn_kernel_lut_t));
    CHECK_PTR_FAIL_GOTO( lut, "Create LUT buffer fail.", final );
    memset(lut, 0, sizeof(vsi_nn_kernel_lut_t) * VSI_NN_KERNEL_LUT_MAX_SIZE);

    clamp_min = param->clamp_min;

    for ( i = 0; i < VSI_NN_KERNEL_LUT_MAX_SIZE; i++)
    {
        int16_t val = (int16_t)(i << 6);
        float fidx = fp16_to_fp32(val);
        if (param->pwl_sign_remove_support && fidx < clamp_min)
        {
            fidx = clamp_min;
        }

        lut[i].index = fidx;
        lut[i].val = vsi_nn_kernel_lut_activation(lut[i].index, param);
    }

    for (i = 0x0; i < 0x10; i++)
    {
        float fidx = 0;
        if (param->pwl_sign_remove_support && fidx < clamp_min)
        {
            fidx = clamp_min;
        }

        lut[i].index = fidx;
        lut[i].val = vsi_nn_kernel_lut_activation(lut[i].index, param);
    }

    for (i = 0x1F0; i < 0x200; i++)
    {
        lut[i].index = VSI_NN_KERNEL_LUT_FP16_MAX;
        lut[i].val = vsi_nn_kernel_lut_activation(lut[i].index, param);
    }

    for (i = 0x3F0; i < 0x400; i++)
    {
        if (param->pwl_sign_remove_support)
        {
            lut[i].index = clamp_min;
        }
        else
        {
            lut[i].index = VSI_NN_KERNEL_LUT_FP16_MIN;
        }

        lut[i].val = vsi_nn_kernel_lut_activation(lut[i].index, param);
    }

    qsort(lut, VSI_NN_KERNEL_LUT_MAX_SIZE, sizeof(vsi_nn_kernel_lut_t), _comparator);

    for ( i = 0; i < VSI_NN_KERNEL_LUT_MAX_SIZE; i++)
    {
        index[i] = lut[i].index;
        value[i] = lut[i].val;
    }

    status  = vxCopyLUT(index_lut, (void*)&index, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyLUT(output_lut, (void*)&value, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
final:
    vsi_nn_safe_free(lut);

    return status;
}

vsi_status vsi_nn_kernel_lut
    (
    vx_lut index_lut,
    vx_lut output_lut,
    vsi_nn_kernel_lut_params *param
    )
{
    vsi_status status = VSI_SUCCESS;
    float clamp_min = 0;

    if (param == NULL)
    {
        return VSI_FAILURE;
    }

    clamp_min = param->clamp_min;

    if (param->pwl_sign_remove_support && clamp_min >= 0)
    {
        status = vsi_nn_kernel_lut_positive(index_lut, output_lut, param);
    }
    else
    {
        status = vsi_nn_kernel_lut_all(index_lut, output_lut, param);
    }

    return status;
}
