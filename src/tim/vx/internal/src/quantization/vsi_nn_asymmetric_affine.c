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
#include "vsi_nn_log.h"
#include "quantization/vsi_nn_asymmetric_affine.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_dtype_util_prv.h"

vsi_status vsi_nn_QuantAffineCalParam
    (
    vsi_nn_type_e  type,
    float     max_data,
    float     min_data,
    float   * scale,
    int32_t    * zero_point
    )
{
    double max_range, min_range;
    int32_t tmp;
    max_range = 0.0;
    min_range = 0.0;

    switch( type )
    {
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_UINT16:
    case VSI_NN_TYPE_UINT32:
        break;
    default:
        VSILOGW("Not support type %#x", type);
        return VSI_FAILURE;
    }
    type_get_range( type, &max_range, &min_range );
    *scale = ( max_data - min_data ) / (float)( max_range - min_range );
    tmp = (int32_t)vsi_rint( (float)min_range - min_data / *scale );
    *zero_point = (int32_t)vsi_nn_min( (int32_t)max_range,
        vsi_nn_max( (int32_t)min_range, tmp ) );
    return VSI_SUCCESS;
} /* vsi_nn_QuantAffineCalParam() */

vsi_bool vsi_nn_QuantAffineCheck
    (
    vsi_nn_tensor_t *input,
    vsi_nn_tensor_t *weight,
    vsi_nn_tensor_t *bias
    )
{
    vsi_bool ret;
    vsi_nn_type_e dtype;
    const double diff_scale = (double)1e-5;

    ret = FALSE;
    dtype = input->attr.dtype.vx_type;

    switch (dtype)
    {
        case VSI_NN_TYPE_UINT4:
        case VSI_NN_TYPE_UINT8:
        case VSI_NN_TYPE_UINT16:
        case VSI_NN_TYPE_UINT32:
        case VSI_NN_TYPE_INT4:
        case VSI_NN_TYPE_INT8:
        case VSI_NN_TYPE_INT16:
        {
            double product_scale = (double)input->attr.dtype.scale * (double)weight->attr.dtype.scale;
            const double acuity_round_decimals = 1e-8;
            if(bias && bias->attr.dtype.scale)
            {
                double tmp0,tmp1;
                double bias_scale = bias->attr.dtype.scale;
                tmp0 = vsi_nn_abs(product_scale - bias_scale);
                tmp1 = vsi_nn_min(product_scale, bias_scale) * diff_scale;
                tmp1 = vsi_nn_max(tmp1, acuity_round_decimals);
                if(tmp0 <= tmp1)
                {
                    ret = TRUE;
                }
            }
            else
            {
                ret = TRUE;
            }
        }
        break;
    default:
        VSILOGW("input dtype error %#x", dtype);
        break;
    }

    return ret;
} /* vsi_nn_QuantAffineCheck() */
