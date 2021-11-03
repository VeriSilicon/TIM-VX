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
#ifndef _VSI_NN_TEST_H
#define _VSI_NN_TEST_H

#include "vsi_nn_log.h"
#include "utils/vsi_nn_util.h"

#if defined(__cplusplus)
extern "C"{
#endif

#define TEST_CHECK_TENSOR_ID( id, lbl )      do {\
    if( VSI_NN_TENSOR_ID_NA == id ) {\
        VSILOGE("CHECK TENSOR ID %d", __LINE__);\
        goto lbl;\
        }\
    } while(0)

#define TEST_CHECK_PTR( ptr, lbl )      do {\
    if( NULL == ptr ) {\
        VSILOGE("CHECK PTR %d", __LINE__);\
        goto lbl;\
    }\
} while(0)

#define TEST_CHECK_STATUS( stat, lbl )  do {\
    if( VSI_SUCCESS != stat ) {\
        VSILOGE("CHECK STATUS(%d:%s)", (stat), vsi_nn_DescribeStatus(stat));\
        goto lbl;\
    }\
} while(0)

#if defined(__cplusplus)
}
#endif

#endif

