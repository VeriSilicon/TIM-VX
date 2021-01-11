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
#include "quantization/vsi_nn_perchannel_symmetric_affine.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_limits.h"

vsi_status vsi_nn_QuantAffinePerchannelCalParam
    (
    vsi_nn_type_e  type,
    float     max_data,
    float     min_data,
    float   * scales
    //int32_t    * zero_point
    )
{
    double max_range, min_range;
    //int32_t tmp;
    max_range = 0.0;
    min_range = 0.0;

    switch( type )
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_UINT16:
    case VSI_NN_TYPE_UINT32:
        break;
    default:
        VSILOGW("Not support type %#x", type);
        return VSI_FAILURE;
    }
    vsi_nn_TypeGetRange( type, &max_range, &min_range );
    *scales = ( max_data - min_data ) / (float)( max_range - min_range );
    //tmp = (int32_t)vsi_nn_Rint( (float)min_range - min_data / *scales );
    //*zero_point = (int32_t)vsi_nn_min( (int32_t)max_range,
    //    vsi_nn_max( (int32_t)min_range, tmp ) );
    return VSI_SUCCESS;
} /* vsi_nn_QuantAffinePerchannelCalParam() */

vsi_bool vsi_nn_QuantAffinePerchannelCheck
    (
    vsi_nn_tensor_t *input,
    vsi_nn_tensor_t *weight,
    vsi_nn_tensor_t *bias
    )
{
#ifdef VSI_PERCHANNEL_QUANTIZATION_SUPPORT
    vsi_bool ret;
    vsi_nn_type_e dtype;
    const float diff_scale = (float)1e-5;
    ret = FALSE;
    dtype = input->attr.dtype.vx_type;

    switch (dtype)
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_UINT16:
    case VSI_NN_TYPE_UINT32:
        {
            float input_scale = input->attr.dtype.scale;
            const float *w = NULL;
            const float *b = NULL;
            int i = 0;
            w = weight->attr.dtype.scales;
            if(bias && bias->attr.dtype.scales)
            {
                b = bias->attr.dtype.scales;
                for (i=0; i < weight->attr.dtype.scale_dim; i++)
                {
                    float weight_scale = *(w+i);
                    float bias_scale = *(b+i);
                    float iw_scale = input_scale * weight_scale;
                    float diff = vsi_nn_abs(bias_scale - iw_scale);
                    if(diff <= diff_scale)
                    {
                        ret = TRUE;
                    }
                    else
                    {
                        break;
                    }
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
#else
    vsi_bool ret;
    ret = FALSE;
#endif
    return ret;
} /* vsi_nn_QuantAffinePerchannelCheck() */
