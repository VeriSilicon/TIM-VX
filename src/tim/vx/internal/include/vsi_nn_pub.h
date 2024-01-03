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

#ifndef _VSI_NN_PUB_H
#define _VSI_NN_PUB_H

#if !defined(OVXLIB_API)
    #if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
        #define OVXLIB_API __declspec(dllimport)
    #else
        #define OVXLIB_API __attribute__((visibility("default")))
    #endif
#endif

#include "vsi_nn_log.h"
#include "vsi_nn_context.h"
#include "vsi_nn_client_op.h"
#include "vsi_nn_node.h"
#include "vsi_nn_node_attr_template.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_types.h"
#include "vsi_nn_version.h"
#include "vsi_nn_assert.h"
#include "vsi_nn_post.h"
#include "vsi_nn_rnn.h"
#include "vsi_nn_test.h"
#include "vsi_nn_pre_post_process.h"
#include "utils/vsi_nn_code_generator.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_dtype_util.h"
#include "quantization/vsi_nn_asymmetric_affine.h"
#include "quantization/vsi_nn_dynamic_fixed_point.h"

#if defined(VSI_ENABLE_LCOV_TEST) && VSI_ENABLE_LCOV_TEST
#include "lcov/vsi_nn_coverage.h"
#endif

#endif

