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
/** @file */
#ifndef _VSI_NN_CONTEXT_H
#define _VSI_NN_CONTEXT_H

#include "vsi_nn_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Max size of HW target name */
#define VSI_NN_MAX_TARGET_NAME 32

/**
 * Hardware evis version.
 */
typedef enum
{
    VSI_NN_HW_EVIS_NONE,
    VSI_NN_HW_EVIS_1,
    VSI_NN_HW_EVIS_2
} vsi_nn_hw_evis_version_e;

/**
 * Structure to store hardware evis version.
 */
typedef struct _vsi_nn_hw_evis_t
{
    vsi_nn_hw_evis_version_e ver;
} vsi_nn_hw_evis_t;

/**
 * Hardware config.
 * It stores hardware name and evis version.
 */
typedef struct _vsi_nn_hw_config_t
{
    char target_name[VSI_NN_MAX_TARGET_NAME];
    vsi_nn_hw_evis_t evis;
#if VX_HARDWARE_CAPS_PARAMS_EXT_SUPPORT
    uint32_t subGroupSize;
#endif
    uint32_t use_40bits_va;
    uint32_t support_stream_processor;
    uint32_t sp_exec_count;
    uint32_t sp_vector_depth;
    uint32_t sp_per_core_vector_depth;
} vsi_nn_hw_config_t;

typedef struct _vsi_nn_runtime_option_t
{
    int32_t enable_shader;
    int32_t enable_opcheck;
    int32_t enable_concat_optimize;
    int32_t enable_asymi8_to_u8;
    int32_t enable_dataconvert_optimize;
    int32_t enable_stream_processor;
    int32_t enable_rgb88_planar_nhwc;
} vsi_nn_runtime_option_t;

/**
 * Ovxlib NN runtime context.
 */
typedef struct _vsi_nn_context_t
{
    vx_context c;
    vsi_nn_hw_config_t config;
    vsi_nn_runtime_option_t options;
} VSI_PUBLIC_TYPE *vsi_nn_context_t;

/**
 * Create context
 * Create ovxlib NN runtime context.
 * @return Context handle on success, or NULL otherwise.
 */
OVXLIB_API vsi_nn_context_t vsi_nn_CreateContext
    ( void );

/**
 * Release context
 * Release ovxlib NN runtime resource and reset context handle to NULL.
 *
 * @param[in] ctx Pointer to context handle.
 */
OVXLIB_API void vsi_nn_ReleaseContext
    ( vsi_nn_context_t * ctx );

#ifdef __cplusplus
}
#endif

#endif
