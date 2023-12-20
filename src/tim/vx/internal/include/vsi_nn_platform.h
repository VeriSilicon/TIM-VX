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
#ifndef _VSI_NN_PLATFORM_H
#define _VSI_NN_PLATFORM_H

#include "vsi_nn_feature_config.h"

#include <VX/vx_khr_cnn.h>
#include <VX/vx_helper.h>
#include <VX/vx_ext_program.h>
#include <VX/vx_api.h>
#include <VX/vx_compatibility.h>
#include <VX/vx_khr_import_kernel.h>
#if defined(VX_KHR_COMPATIBILITY) && (0x1==VX_KHR_COMPATIBILITY)
#include <VX/vx_khr_compatible.h>
#endif

/*
    This is a compatibility head file for backward compatibility OpenVX 1.1 spec
*/
#include "vsi_nn_compatibility.h"

#endif
