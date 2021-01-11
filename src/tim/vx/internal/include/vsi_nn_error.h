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
#ifndef _VSI_NN_ERROR_H
#define _VSI_NN_ERROR_H

#include <assert.h>
#include "vsi_nn_log.h"
#include "utils/vsi_nn_util.h"

#define VSI_ASSERT( cond )  assert(cond)

#define VSI_CHECK_PTR( pointer, msg, retval ) \
    do { \
        if( pointer == NULL ) { \
            VSILOGD("%s",msg); \
            VSI_ASSERT(FALSE); \
        } \
    } while(0)


#define CHECK_STATUS_FAIL_GOTO( stat, lbl )  do {\
    if( VSI_SUCCESS != stat ) {\
        VSILOGE("CHECK STATUS(%d:%s)", (stat), vsi_nn_DescribeStatus(stat));\
        goto lbl;\
    }\
} while(0)

#define CHECK_STATUS( stat )  do {\
    if( VSI_SUCCESS != stat ) {\
        VSILOGE("CHECK STATUS(%d:%s)", (stat), vsi_nn_DescribeStatus(stat));\
    }\
} while(0)

#define CHECK_PTR_FAIL_GOTO( pointer, msg, lbl ) \
    do { \
        if( pointer == NULL ) { \
            VSILOGD("CHECK POINTER %s", msg); \
            goto lbl; \
        } \
    } while(0)

#endif
