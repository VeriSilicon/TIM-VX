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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "vsi_nn_version.h"

#define MACRO_TO_STRING(M) #M
#define VERSION_PREFIX "OVXLIB_VERSION=="
#define DEF_VERSION(a,b,c) VERSION_PREFIX MACRO_TO_STRING(a)"." MACRO_TO_STRING(b)"." MACRO_TO_STRING(c)
#define DEF_VERSION_STR DEF_VERSION(VSI_NN_VERSION_MAJOR,VSI_NN_VERSION_MINOR,VSI_NN_VERSION_PATCH)

const char *vsi_nn_GetVersion(void)
{
    static const char *version = DEF_VERSION_STR;
    return version;
}

uint32_t vsi_nn_GetVersionMajor(void)
{
    return VSI_NN_VERSION_MAJOR;
}

uint32_t vsi_nn_GetVersionMinor(void)
{
    return VSI_NN_VERSION_MINOR;
}

uint32_t vsi_nn_GetVersionPatch(void)
{
    return VSI_NN_VERSION_PATCH;
}