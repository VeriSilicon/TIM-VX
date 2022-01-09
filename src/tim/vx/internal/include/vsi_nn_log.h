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

#ifndef _VSI_NN_LOG_H
#define _VSI_NN_LOG_H
#include <stdio.h>

#if defined(__cplusplus)
extern "C"{
#endif

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

typedef enum _vsi_nn_log_level_e
{
    VSI_NN_LOG_UNINIT = -1,
    VSI_NN_LOG_CLOSE,
    VSI_NN_LOG_ERROR,
    VSI_NN_LOG_WARN,
    VSI_NN_LOG_INFO,
    VSI_NN_LOG_DEBUG
}vsi_nn_log_level_e;

#define VSI_NN_MAX_DEBUG_BUFFER_LEN 1024
#define VSILOGE( fmt, ... ) \
    vsi_nn_LogMsg(VSI_NN_LOG_ERROR, "E [%s:%s:%d]" fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define VSILOGW( fmt, ... ) \
    vsi_nn_LogMsg(VSI_NN_LOG_WARN,  "W [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define VSILOGI( fmt, ... ) \
    vsi_nn_LogMsg(VSI_NN_LOG_INFO,  "I [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define VSILOGD( fmt, ... ) \
    vsi_nn_LogMsg(VSI_NN_LOG_DEBUG, "D [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define _LOG_( fmt, ... ) \
    vsi_nn_LogMsg(VSI_NN_LOG_DEBUG, "[%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)

OVXLIB_API void vsi_nn_LogMsg
    (
    vsi_nn_log_level_e level,
    const char *fmt,
    ...
    );

#if defined(__cplusplus)
}
#endif

#endif

