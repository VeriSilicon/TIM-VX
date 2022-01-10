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

#include "VX/vx.h"
#include "VX/vxu.h"
#include "VX/vx_ext_program.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vsi_nn_libnnext_resource.h"

static char s_vx_resource_path[VSI_NN_MAX_PATH] = "VX";

uint8_t * vsi_nn_LoadBinarySource
    (
    uint8_t * file,
    int32_t * sz
    )
{
    int32_t len;
    int32_t n;
    FILE *fp;
    uint8_t *buf;

    buf = NULL;

    fp = fopen( (char *)file, "rb" );

    VSILOGI( "Loading program from binary file." );
    if( NULL == fp )
    {
        VSILOGE( "Open program file fail." );
        return buf;
    }

    fseek( fp, 0, SEEK_END );
    len = ftell( fp );
    fseek( fp, 0, SEEK_SET );

    buf = (uint8_t *)malloc( len + 1 );
    n = (int32_t)fread( buf, 1, len, fp );
    fclose( fp );

    if( n != len )
    {
        VSILOGE( "Read source file error(%d/%d).", n, len );
    }

    buf[len] = 0;

    if( NULL != sz )
    {
        *sz = len;
    }
    return buf;
} /* vsi_nn_LoadBinarySource() */

static vsi_status vsi_nn_InitKernel
    (
    vx_kernel_description_t * kernel,
    vx_kernel obj
    )
{
    vsi_status status;
    uint32_t i;

    status = VSI_SUCCESS;
    for( i = 0; i < kernel->numParams; i ++ )
    {
        status = vxAddParameterToKernel(
            obj,
            i,
            kernel->parameters[i].direction,
            kernel->parameters[i].data_type,
            kernel->parameters[i].state
            );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Add parameter %d to kernel %s fail. with %d.",
                i, kernel->name, status );
            break;
        }
    }

    if( VSI_SUCCESS == status )
    {
        status = vxFinalizeKernel( obj );
    }

    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Finalize kernel %s fail with %d.",
            kernel->name, status );
        status = vxRemoveKernel( obj );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Remove kernel %s fail with %d.",
                kernel->name, status );
        }
    }
    return status;
}

static vsi_status vsi_nn_RegisterCPUKernel
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_status status;
    vx_kernel obj;
    vx_context ctx;
    vx_kernel_description_t * kernel = kernel_info->kernel[kernel_info->kernel_index];

    status = VSI_FAILURE;
    ctx = vxGetContext( (vx_reference)graph->g );

    obj = vxAddUserKernel(
        ctx,
        kernel->name,
        kernel->enumeration,
        kernel->function,
        kernel->numParams,
        kernel->validate,
        kernel->initialize,
        kernel->deinitialize
        );

    if( NULL != obj )
    {
        status = vsi_nn_InitKernel(kernel,obj);
        vxReleaseKernel( &obj );
    }
    else
    {
        VSILOGE( "Add kernel %s fail.", kernel->name );
    }
    return status;
} /* vsi_nn_RegisterCPUKernel() */

static const char * vsi_nn_LoadVxResourceFromFile
    (
    char * resource_name,
    vx_size * program_len
    )
{
    char resource_path[VSI_NN_MAX_PATH];
    const char * vx_resource_path = vsi_nn_VxResourceGetPath();

    if (strncmp(vx_resource_path, "", VSI_NN_MAX_PATH) == 0)
    {
        VSILOGE("No Valid VX Resource Path Error!\n");
    }
    snprintf(resource_path, VSI_NN_MAX_PATH, "%s/%s.vx", vx_resource_path, resource_name);
    return (char *)vsi_nn_LoadBinarySource((uint8_t *)resource_path, (int32_t *)program_len);
}

static vsi_status vsi_nn_RegisterVXKernel
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_status status = VSI_FAILURE;
    vx_kernel obj = NULL;
    vx_program program = NULL;
    vx_size * program_len = NULL;
    const char **program_src = NULL;
    vx_context ctx = NULL;
    vsi_nn_context_t context = NULL;
    vx_kernel_description_t * kernel = kernel_info->kernel[kernel_info->kernel_index];
    uint8_t i = 0;
    vsi_bool load_from_file = FALSE;

#define MAX_BUILDPROGRAM_LEN 128
    char cmd[MAX_BUILDPROGRAM_LEN] = {0};
    int32_t evis = 0;

    memset(cmd, 0, sizeof(char) * MAX_BUILDPROGRAM_LEN);
    status = VSI_FAILURE;
    ctx = vxGetContext( (vx_reference)graph->g );
    context = graph->ctx;
    evis = context->config.evis.ver;

    program_src = (const char**)malloc(kernel_info->resource_num * sizeof(char *));
    program_len = (vx_size*)malloc(kernel_info->resource_num * sizeof(vx_size));
    for (i = 0; i < kernel_info->resource_num; i++)
    {
        program_src[i] = vsi_nn_resource_load_source_code(
                kernel_info->resource_name[i], &program_len[i], VSI_NN_KERNEL_TYPE_EVIS);
        if (program_src[i] == NULL)
        {
            VSILOGI("Try to Load VX Resource from file...\n");

            program_src[i] = vsi_nn_LoadVxResourceFromFile(kernel_info->resource_name[i], &program_len[i]);
            load_from_file = TRUE;
        }
    }

    program = vxCreateProgramWithSource(ctx, kernel_info->resource_num, program_src, (vx_size *)program_len);
    status = vxGetStatus((vx_reference)program);
    if(VSI_SUCCESS != status)
    {
        VSILOGE("[%s : %d] vxCreateProgramWithSource() Error!\n", __FILE__, __LINE__);
        status = VSI_FAILURE;
        goto OnError;
    }

    if(evis == VSI_NN_HW_EVIS_NONE)
    {
        // set default evis version is 2
        sprintf(cmd, "-cl-viv-vx-extension -D VX_VERSION=2 -D USE_40BITS_VA=%d", context->config.use_40bits_va);
    }
    else
    {
        sprintf(cmd, "-cl-viv-vx-extension -D VX_VERSION=%d -D USE_40BITS_VA=%d", evis, context->config.use_40bits_va);
    }
    status = vxBuildProgram(program, cmd);

    if(VSI_SUCCESS != status)
    {
        VSILOGE("[%s : %d] vxBuildProgram() Error!\n", __FILE__, __LINE__);
    }

    obj = vxAddKernelInProgram(program,
        kernel->name,
        kernel->enumeration,
        kernel->numParams,
        kernel->validate,
        kernel->initialize,
        kernel->deinitialize
        );

    if( NULL != obj )
    {
        status = vsi_nn_InitKernel(kernel,obj);
        vxReleaseProgram(&program);
    }
    else
    {
        VSILOGE( "Add kernel %s fail.", kernel->name );
    }
OnError:
    for (i = 0; i < kernel_info->resource_num; i++)
    {
        if (program_src[i] && load_from_file)
        {
            free((char *)program_src[i]);
        }
    }
    if(program_src) free((char**)program_src);
    if(program_len) free(program_len);
    return status;
}

static vsi_status vsi_nn_RegisterBinKernel
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_status status;
    vx_kernel obj;
    vx_program program = NULL;
    vx_size program_len = 0;
    const uint8_t *program_ptr = NULL;
    vx_context ctx;
    vsi_nn_context_t context;
    vx_kernel_description_t * kernel = kernel_info->kernel[kernel_info->kernel_index];

#define MAX_BUILDPROGRAM_LEN 128
    char cmd[MAX_BUILDPROGRAM_LEN];
    int32_t evis;

    memset(cmd, 0, sizeof(char) * MAX_BUILDPROGRAM_LEN);
    status = VSI_FAILURE;

    ctx = vxGetContext( (vx_reference)graph->g );
    context = graph->ctx;
    evis = context->config.evis.ver;

    program_ptr = vsi_nn_VxBinResourceGetResource(
            kernel_info->resource_name[kernel_info->resource_num - 1], &program_len);
    program = vxCreateProgramWithBinary(ctx, (const vx_uint8 *)program_ptr, program_len);

    status = vxGetStatus((vx_reference)program);
    if(VSI_SUCCESS != status)
    {
        VSILOGE("[%s : %d] vxCreateProgramWithSource() Error!\n", __FILE__, __LINE__);
        return status;
    }

#if 1
    if(evis == VSI_NN_HW_EVIS_NONE)
    {
        // set default evis version is 2
        sprintf(cmd, "-cl-viv-vx-extension -D VX_VERSION=2 -D USE_40BITS_VA=%d", context->config.use_40bits_va);
    }
    else
    {
        sprintf(cmd, "-cl-viv-vx-extension -D VX_VERSION=%d -D USE_40BITS_VA=%d", evis, context->config.use_40bits_va);
    }
#else
    sprintf(cmd, "-cl-viv-vx-extension");
#endif
    status = vxBuildProgram(program, cmd);

    if(VSI_SUCCESS != status)
    {
        VSILOGE("[%s : %d] vxBuildProgram() Error!\n", __FILE__, __LINE__);
        return status;
    }

    obj = vxAddKernelInProgram(program,
        kernel->name,
        kernel->enumeration,
        kernel->numParams,
        kernel->validate,
        kernel->initialize,
        kernel->deinitialize
        );

    if( NULL != obj )
    {
        status = vsi_nn_InitKernel(kernel,obj);
        vxReleaseProgram(&program);
    }
    else
    {
        VSILOGE( "Add kernel %s fail.", kernel->name );
    }

    return status;
}

vsi_status vsi_nn_RegisterClientKernel
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_status status = VSI_SUCCESS;
    switch (kernel_info->type)
    {
    case VX_KERNEL_TYPE_VX:
        status = vsi_nn_RegisterVXKernel(graph, kernel_info);
        break;

    case VX_KERNEL_TYPE_CPU:
        status = vsi_nn_RegisterCPUKernel(graph, kernel_info);
        break;

    case VX_KERNEL_TYPE_BIN:
        status = vsi_nn_RegisterBinKernel(graph, kernel_info);
        break;

    default:
        status = VSI_FAILURE;
    }
    return status;
} /* vsi_nn_RegisterClientKernel() */

vx_node vsi_nn_RegisterClientKernelAndNewNode
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_status status;
    vx_context ctx;
    vx_kernel obj;
    vx_node node;
    vx_kernel_description_t * kernel = kernel_info->kernel[kernel_info->kernel_index];

    ctx = vxGetContext( (vx_reference)graph->g );

    /* Load kernel */
    obj = vxGetKernelByName( ctx, kernel->name );
    status = vxGetStatus( (vx_reference)obj );
    if (VSI_SUCCESS != status)
    {
        /* Register kernel */
        status = vsi_nn_RegisterClientKernel( graph, kernel_info);
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Register client kernel %s fail with %d.",
                kernel->name, status );
            return NULL;
        }
        else
        {
            VSILOGI( "Register client kernel %s successfully.",
                kernel->name );
        }

        /* Load kernel */
        obj = vxGetKernelByName( ctx, kernel->name );
        status = vxGetStatus( (vx_reference)obj );
    }

    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Load client kernel %s fail with %d.",
            kernel->name, status );
        return NULL;
    }

    /* Create node */
    node = vxCreateGenericNode( graph->g, obj );
    vxReleaseKernel( &obj );
    status = vxGetStatus( (vx_reference)node );
    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Load client node from kernel %s fail with %d.",
            kernel->name, status );
        return NULL;
    }
    return node;
} /* vsi_nn_RegisterClientKernelAndNewNode() */

vx_node vsi_nn_RegisterClientKernelAndCreateNode
    (
    vsi_nn_graph_t * graph,
    vx_kernel_description_t * kernel
    )
{
    /*
    * Deprecated: use vsi_nn_RegisterClientKernelAndNewNode() instead.
    */
    vsi_nn_kernel_info_t kernel_info;
    char *resource_name[1] = {NULL};

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    kernel_info.resource_name = resource_name;
    kernel_info.resource_name[0] = "old_client_interface";
    kernel_info.type = VX_KERNEL_TYPE_CPU;
    kernel_info.kernel = &kernel;
    kernel_info.kernel_index = 0;
    kernel_info.init_index = 0;

    return vsi_nn_RegisterClientKernelAndNewNode(graph, &kernel_info);
} /* vsi_nn_RegisterClientKernelAndCreateNode() */

vsi_status vsi_nn_ClientNodePassParameters
    (
    vx_node node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    uint8_t i;

    status = VSI_FAILURE;
    for( i = 0; i < num; i++ )
    {
        status = vxSetParameterByIndex( node, i, params[i] );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Set %d parameter fail.", i );
            break;
        }
    }
    return status;
} /* vsi_nn_ClientKernelPassParameters() */

vsi_status VX_CALLBACK vsi_nn_KernelValidator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
)
{
    return VSI_SUCCESS;
} /* vsi_nn_KernelValidator() */

vsi_status VX_CALLBACK vsi_nn_KernelInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    return VSI_SUCCESS;
} /* vsi_nn_KernelInitializer() */

vsi_status VX_CALLBACK vsi_nn_KernelDeinitializer
    (
    vx_node nodObj,
    const vx_reference *paraObj,
    uint32_t paraNum
    )
{
    return VSI_SUCCESS;
} /* vsi_nn_KernelDeinitializer() */

const char * vsi_nn_VxResourceGetPath()
{
    return s_vx_resource_path;
} /* vsi_nn_VxResourceGetPath() */

void vsi_nn_VxResourceSetPath
    (
    char* path
    )
{
    strncpy(s_vx_resource_path, path, VSI_NN_MAX_PATH - 1);
} /* vsi_nn_VxResourceSetPath() */

const uint8_t * vsi_nn_VxBinResourceGetResource
    (
    char* name,
    vx_size *len
    )
{
    return NULL;
} /* vsi_nn_VxResourceGetBinResource() */

vx_kernel_type_e vsi_nn_GetVXKernelTypeForShader()
{
#if VSI_USE_VXC_BINARY
    return VX_KERNEL_TYPE_BIN;
#else
    return VX_KERNEL_TYPE_VX;
#endif
}

vx_bool vsi_nn_is_do_vx_op_pre_init
    (
    vx_kernel_type_e type
    )
{
    return (vx_bool)(type == VX_KERNEL_TYPE_VX || type == VX_KERNEL_TYPE_BIN);
}
