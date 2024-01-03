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
    if (context->options.enable_stream_processor)
    {
        context->config.support_stream_processor = paramExt.supportStreamProcessor;
        context->config.sp_exec_count = paramExt2.streamProcessorExecCount;
        context->config.sp_vector_depth = paramExt2.streamProcessorVectorSize;
        if (context->config.sp_exec_count > 0)
        {
            context->config.sp_per_core_vector_depth =
                context->config.sp_vector_depth / context->config.sp_exec_count;
        }
    }
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

#if (defined(__ANDROID__)) && (ANDROID_SDK_VERSION >= 30)
static const char* ENV_ENABLE_SHADER = "vendor.VIV_VX_ENABLE_SHADER";
static const char* ENV_ENABLE_OPCHECK = "vendor.VSI_NN_ENABLE_OPCHECK";
static const char* ENV_ENABLE_CONCAT_OPTIMIZE = "vendor.VSI_NN_ENABLE_CONCAT_OPTIMIZE";
static const char* ENV_ENABLE_I8TOU8 = "vendor.VSI_NN_ENABLE_I8TOU8";
static const char* ENV_ENABLE_DATACONVERT_OPTIMIZE = "vendor.VSI_NN_ENABLE_DATACONVERT_OPTIMIZE";
static const char* ENV_ENABLE_STREAM_PROCESSOR = "vendor.VSI_VX_ENABLE_STREAM_PROCESSOR";
static const char* ENV_FORCE_RGB888_OUT_NHWC = "vendor.VSI_NN_FORCE_RGB888_OUT_NHWC";
static const char* ENV_ENABLE_SLICE_OPTIMIZE = "vendor.VSI_NN_ENABLE_SLICE_OPTIMIZE";
static const char* ENV_ENABLE_BATCH_OPT = "vendor.VSI_VX_ENABLE_BATCH_OPT";
#else
static const char* ENV_ENABLE_SHADER = "VIV_VX_ENABLE_SHADER";
static const char* ENV_ENABLE_OPCHECK = "VSI_NN_ENABLE_OPCHECK";
static const char* ENV_ENABLE_CONCAT_OPTIMIZE = "VSI_NN_ENABLE_CONCAT_OPTIMIZE";
static const char* ENV_ENABLE_I8TOU8 = "VSI_NN_ENABLE_I8TOU8";
static const char* ENV_ENABLE_DATACONVERT_OPTIMIZE = "VSI_NN_ENABLE_DATACONVERT_OPTIMIZE";
static const char* ENV_ENABLE_STREAM_PROCESSOR = "VSI_VX_ENABLE_STREAM_PROCESSOR";
static const char* ENV_FORCE_RGB888_OUT_NHWC = "VSI_NN_FORCE_RGB888_OUT_NHWC";
static const char* ENV_ENABLE_SLICE_OPTIMIZE = "VSI_NN_ENABLE_SLICE_OPTIMIZE";
static const char* ENV_ENABLE_BATCH_OPT = "VSI_VX_ENABLE_BATCH_OPT";
#endif
static vsi_status vsi_nn_initOptions
    (
    vsi_nn_runtime_option_t *options
    )
{
    int32_t default_value = 1;

    options->enable_shader = vsi_nn_getenv_asint(ENV_ENABLE_SHADER, 1);
    options->enable_opcheck = vsi_nn_getenv_asint(ENV_ENABLE_OPCHECK, 1);
#if (VX_CONCAT_OPT_SUPPORT)
    default_value = 0;
#else
    default_value = 1;
#endif
    options->enable_concat_optimize = vsi_nn_getenv_asint(ENV_ENABLE_CONCAT_OPTIMIZE, default_value);
    options->enable_asymi8_to_u8 = vsi_nn_getenv_asint(ENV_ENABLE_I8TOU8, 1);
    options->enable_dataconvert_optimize = vsi_nn_getenv_asint(ENV_ENABLE_DATACONVERT_OPTIMIZE, 1);
    options->enable_stream_processor = vsi_nn_getenv_asint(ENV_ENABLE_STREAM_PROCESSOR, 1);
    options->enable_rgb88_planar_nhwc = vsi_nn_getenv_asint(ENV_FORCE_RGB888_OUT_NHWC, 0);
#if (VX_STRIDED_SLICE_OPT_SUPPORT)
    default_value = 0;
#else
    default_value = 1;
#endif
    options->enable_slice_optimize = vsi_nn_getenv_asint(ENV_ENABLE_SLICE_OPTIMIZE, default_value);
    options->enable_batch_opt = vsi_nn_getenv_asint(ENV_ENABLE_BATCH_OPT, 0);

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

    if (vsi_nn_initOptions(&context->options) != VSI_SUCCESS)
    {
        vsi_nn_ReleaseContext(&context);
        return NULL;
    }

    if (query_hardware_caps(context) != VSI_SUCCESS)
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
