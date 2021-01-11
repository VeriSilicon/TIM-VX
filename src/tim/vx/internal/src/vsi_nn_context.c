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
#include "vsi_nn_types.h"
#include "vsi_nn_test.h"
#include "vsi_nn_context.h"
#include "vsi_nn_platform.h"

static vsi_status query_hardware_caps
    (
    vsi_nn_context_t context
    )
{
    vsi_status status = VSI_FAILURE;
    vx_hardware_caps_params_t param;

#if VX_HARDWARE_CAPS_PARAMS_EXT_SUPPORT
    vx_hardware_caps_params_ext_t paramExt;

    memset(&paramExt, 0, sizeof(vx_hardware_caps_params_ext_t));
    status = vxQueryHardwareCaps(context->c, (vx_hardware_caps_params_t*)(&paramExt),
                sizeof(vx_hardware_caps_params_ext_t));
    param.evis1 = paramExt.base.evis1;
    param.evis2 = paramExt.base.evis2;
#else
    memset(&param, 0, sizeof(vx_hardware_caps_params_t));
    status = vxQueryHardwareCaps(context->c, &param, sizeof(vx_hardware_caps_params_t));
#endif
    TEST_CHECK_STATUS(status, final);

#if VX_HARDWARE_CAPS_PARAMS_EXT_SUPPORT
    context->config.subGroupSize = paramExt.subGroupSize;
#endif
    if(param.evis1 == TRUE && param.evis2 == FALSE)
    {
        context->config.evis.ver = VSI_NN_HW_EVIS_1;
    }
    else if(param.evis1 == FALSE && param.evis2 == TRUE)
    {
        context->config.evis.ver = VSI_NN_HW_EVIS_2;
    }
    else
    {
        context->config.evis.ver = VSI_NN_HW_EVIS_NONE;
        VSILOGW("Unsupported evis version");
    }

final:
    return status;
}

vsi_nn_context_t vsi_nn_CreateContext
    ( void )
{
    vsi_nn_context_t context = NULL;
    vx_context c = NULL;

    context = (vsi_nn_context_t)malloc(sizeof(struct _vsi_nn_context_t));
    if(NULL == context)
    {
        return NULL;
    }
    c = vxCreateContext();
    if(NULL == c)
    {
        free(context);
        return NULL;
    }

    memset(context, 0, sizeof(struct _vsi_nn_context_t));
    context->c = c;
    if(query_hardware_caps(context) != VSI_SUCCESS)
    {
        vsi_nn_ReleaseContext(&context);
        return NULL;
    }

    return context;
} /* vsi_nn_CreateContext() */

void vsi_nn_ReleaseContext
    ( vsi_nn_context_t * ctx )
{
    if( NULL != ctx && NULL != *ctx )
    {
        vsi_nn_context_t context = *ctx;
        if(context->c)
        {
            vxReleaseContext( &context->c);
        }
        free(context);
        *ctx = NULL;
    }
} /* vsi_nn_ReleaseContext() */

