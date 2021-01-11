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
#include <math.h>
#include "vsi_nn_log.h"
#include "quantization/vsi_nn_dynamic_fixed_point.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

vsi_status vsi_nn_QuantDFPCalParam
    (
    vsi_nn_type_e dtype,
    float    max_data,
    float    min_data,
    int8_t     * fl
    )
{
    int32_t tmp;
    int32_t bits;

    switch( dtype )
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_INT32:
        break;
    default:
        VSILOGW("Not support dtype %#x", dtype);
        return VSI_FAILURE;
    }
    max_data = vsi_nn_max( vsi_nn_abs( max_data ), vsi_nn_abs( min_data ) );
    bits = vsi_nn_GetTypeBytes( dtype ) * 8;
    tmp = (int32_t)ceil( log( max_data ) / log( 2 ) );
    *fl = (int8_t)(bits - 1 - tmp);
    return VSI_SUCCESS;
} /* vsi_nn_QuantDFPCalParam() */

vsi_bool vsi_nn_QuantDFPCheck
(
    vsi_nn_tensor_t *input,
    vsi_nn_tensor_t *weight,
    vsi_nn_tensor_t *bias
)
{
    vsi_bool ret;
    vsi_nn_type_e dtype;

    ret = FALSE;
    dtype = input->attr.dtype.vx_type;

    switch (dtype)
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_INT32:
        {
            if(bias)
            {
                int8_t input_fl = input->attr.dtype.fl;
                int8_t weight_fl = weight->attr.dtype.fl;
                int8_t bias_fl = bias->attr.dtype.fl;
                if(bias_fl == (input_fl + weight_fl))
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
} /* vsi_nn_QuantDFPCheck() */
