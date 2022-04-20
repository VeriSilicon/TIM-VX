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
#if VX_STREAM_PROCESSOR_SUPPORT
    vx_hardware_caps_params_ext2_t paramExt2;
#endif
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
#ifdef VSI_40BIT_VA_SUPPORT
    context->config.use_40bits_va = paramExt.supportVA40;
#endif
#if VX_STREAM_PROCESSOR_SUPPORT
    memset(&paramExt2, 0, sizeof(vx_hardware_caps_params_ext2_t));
    status = vxQueryHardwareCaps(context->c, (vx_hardware_caps_params_t*)(&paramExt2),
                sizeof(vx_hardware_caps_params_ext2_t));
    context->config.support_stream_processor = paramExt.supportStreamProcessor;
    context->config.sp_exec_count = paramExt2.streamProcessorExecCount;
#endif

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

int32_t vsi_nn_getEnv(const char* name, char** env_s) {
    int32_t ret = 0;
    *env_s = vsi_nn_getenv(name);
    if (*env_s) {
        ret = TRUE;
    }
    return ret;
}

static vsi_status vsi_nn_initOptions
    (
    vsi_nn_runtime_option_t *options
    )
{
    char* env_s = NULL;

    env_s = NULL;
    options->enable_shader = 1;
    if (vsi_nn_getEnv("VIV_VX_ENABLE_SHADER", &env_s) && env_s)
    {
        options->enable_shader = atoi(env_s);
    }

    env_s = NULL;
    options->enable_opcheck = 1;
    if (vsi_nn_getEnv("VSI_NN_ENABLE_OPCHECK", &env_s) && env_s)
    {
        options->enable_opcheck = atoi(env_s);
    }

    env_s = NULL;
    options->enable_concat_optimize = 1;
    if (vsi_nn_getEnv("VSI_NN_ENABLE_CONCAT_OPTIMIZE", &env_s) && env_s)
    {
        options->enable_concat_optimize = atoi(env_s);
    }

    env_s = NULL;
    options->enable_asymi8_to_u8 = 1;
    if (vsi_nn_getEnv("VSI_NN_ENABLE_I8TOU8", &env_s) && env_s)
    {
        options->enable_asymi8_to_u8 = atoi(env_s);
    }

    return VSI_SUCCESS;
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

    if (vsi_nn_initOptions(&context->options) != VSI_SUCCESS)
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
