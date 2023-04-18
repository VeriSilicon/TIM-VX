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
#ifndef _VSI_NN_VXKERNEL_H
#define _VSI_NN_VXKERNEL_H

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _vx_kernel_type_e
{
    VX_KERNEL_TYPE_CPU,
    VX_KERNEL_TYPE_VX,
    VX_KERNEL_TYPE_BIN
} vx_kernel_type_e;

typedef struct vsi_nn_kernel_info
{
    char **resource_name;
    uint8_t resource_num;
    vx_kernel_type_e type;
    vx_kernel_description_t ** kernel;
    uint8_t kernel_index;
    uint8_t init_index;
} VSI_PUBLIC_TYPE vsi_nn_kernel_info_t;

uint8_t * vsi_nn_LoadBinarySource
    (
    uint8_t * file,
    int32_t * sz
    );

vsi_status vsi_nn_RegisterClientKernel
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_info_t * kernel_info
    );

/*
 * Deprecated(vsi_nn_RegisterClientKernelAndCreateNode): use vsi_nn_RegisterClientKernelAndNewNode() insteatd.
*/
OVXLIB_API vx_node vsi_nn_RegisterClientKernelAndCreateNode
    (
    vsi_nn_graph_t * graph,
    vx_kernel_description_t * kernel
    );

OVXLIB_API vx_node vsi_nn_RegisterClientKernelAndNewNode
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_info_t * kernel_info
    );

OVXLIB_API vsi_status vsi_nn_ClientNodePassParameters
    (
    vx_node node,
    vx_reference * params,
    uint32_t num
    );

OVXLIB_API vsi_status VX_CALLBACK vsi_nn_KernelValidator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
    );

OVXLIB_API vsi_status VX_CALLBACK vsi_nn_KernelInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    );

OVXLIB_API vsi_status VX_CALLBACK vsi_nn_KernelDeinitializer
    (
    vx_node nodObj,
    const vx_reference *paraObj,
    uint32_t paraNum
    );

OVXLIB_API const char * vsi_nn_VxResourceGetPath();

OVXLIB_API void vsi_nn_VxResourceSetPath
    (
    char* path
    );

OVXLIB_API const uint8_t * vsi_nn_VxBinResourceGetResource
    (
    char* name,
    vx_size *len
    );

OVXLIB_API vx_kernel_type_e vsi_nn_GetVXKernelTypeForShader();

OVXLIB_API vx_bool vsi_nn_is_do_vx_op_pre_init
    (
    vx_kernel_type_e type
    );

#ifdef __cplusplus
}
#endif

#endif
