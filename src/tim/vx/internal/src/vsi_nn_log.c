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
#include <stdarg.h>
#include <string.h>

#include "vsi_nn_log.h"
#include "vsi_nn_types.h"

static vsi_bool _check_log_level
    (
    vsi_nn_log_level_e level
    )
{
    char *env_level_s;
    static vsi_nn_log_level_e env_level = VSI_NN_LOG_UNINIT;

    if(env_level == VSI_NN_LOG_UNINIT)
    {
        env_level_s = getenv("VSI_NN_LOG_LEVEL");
        if(env_level_s)
        {
            env_level = (vsi_nn_log_level_e)atoi(env_level_s);
        }
        else
        {
            env_level = VSI_NN_LOG_WARN;
        }
    }

    if(env_level >= level)
    {
        return TRUE;
    }

    return FALSE;
}

void vsi_nn_LogMsg
    (
    vsi_nn_log_level_e level,
    const char *fmt,
    ...
    )
{
    char arg_buffer[VSI_NN_MAX_DEBUG_BUFFER_LEN] = {0};
    va_list arg;

    if(_check_log_level(level) == FALSE)
    {
        return ;
    }

    va_start(arg, fmt);
    vsnprintf(arg_buffer, VSI_NN_MAX_DEBUG_BUFFER_LEN, fmt, arg);
    va_end(arg);

    fprintf(stderr, "%s\n", arg_buffer);
} /* vsi_nn_LogMsg() */
