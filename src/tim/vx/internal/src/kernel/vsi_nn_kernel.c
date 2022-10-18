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

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "vsi_nn_context.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_types.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_error.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_dtype_util.h"

#include "libnnext/vsi_nn_libnnext_resource.h"
#if VSI_USE_VXC_BINARY
/*this header can be only included once in all *.c files*/
#include "libnnext/vx_bin/vxc_binaries.h"
#endif

typedef struct
{
    size_t size;
    const void* data;
    void* reserve_mem;
} kernel_program_info_t;

static vsi_status _kernel_init_obj
    (
    vx_kernel_description_t* info,
    vx_kernel obj
    );

static vsi_status _cpu_register
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel
    );

static vsi_status _gpu_register
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel
    );

static vsi_status _gpu_register_ext
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel,
    const char** resources
    );

static vx_program _create_program_from_executable
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel
    );

static vx_program _create_program_from_code
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel
    );

static vx_program _create_program_from_code_ext
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel,
    const char** resources
    );

static const uint8_t* _load_internal_executable
    (
    const char* source_name,
    size_t* size,
    vsi_nn_kernel_type_e type
    );

static char* _load_source_code_from_file
    (
    const char* source_name,
    size_t* size
    );

static vx_program _create_program
    (
    vx_context ctx,
    kernel_program_info_t *program_info,
    size_t size
    );

static void _kernel_clear_source
    ( vsi_nn_kernel_t * kernel );

static vsi_bool _check_shader_support(vsi_nn_graph_t* graph);

static vsi_bool _check_stream_process_support
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t** inputs,
    size_t input_num
    );

vsi_bool vsi_nn_kernel_is_supported_types
    (
    vsi_nn_tensor_t** inputs,
    size_t input_num,
    vsi_nn_tensor_t** outputs,
    size_t output_num
    );

static vsi_status VX_CALLBACK _kernel_validator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
    )
{
    return VSI_SUCCESS;
} /* _kernel_validator() */

static vsi_status VX_CALLBACK _kernel_initializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    return VSI_SUCCESS;
} /* _kernel_initializer() */

static vsi_status VX_CALLBACK _kernel_deinitializer
    (
    vx_node nodObj,
    const vx_reference *paraObj,
    uint32_t paraNum
    )
{
    return VSI_SUCCESS;
} /* _kernel_deinitializer() */

static void _kernel_clear_build_option
    (
    vsi_nn_kernel_source_info_t * source
    )
{
    vsi_nn_kernel_build_option_t * option;
    if( !source )
    {
        return;
    }
    option = &source->build_option;
    if( option->data )
    {
        free( option->data );
        option->data = NULL;
    }
} /* _kernel_clear_build_option() */

static void _kernel_clear_source
    ( vsi_nn_kernel_t * kernel )
{
    size_t i;
    if( !kernel )
    {
        return;
    }
    for( i = 0; i < VSI_NN_GPU_SOURCE_FMT_NUM; i ++ )
    {
        vsi_nn_kernel_source_info_t* source = &(kernel->gpu.sources[i]);
        if( source->data )
        {
            size_t j;
            for( j = 0; j < source->num; j ++ )
            {
                if( source->data[j] )
                {
                    free( source->data[j] );
                    source->data[j] = NULL;
                }
            }
            free( source->data );
            source->data = NULL;
            _kernel_clear_build_option( source );
        }
    }
} /* _kernel_clear_source() */

static vsi_status _cpu_register
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_status status;
    vx_kernel_description_t* info;
    vx_kernel obj;

    status = VSI_FAILURE;
    info = &kernel->info;

    obj = vxAddUserKernel(
        graph->ctx->c,
        info->name,
        info->enumeration,
        info->function,
        info->numParams,
        info->validate,
        info->initialize,
        info->deinitialize
        );
    if( NULL != obj )
    {
        status = _kernel_init_obj( info, obj );
        //vxReleaseKernel( &obj );
    }
    else
    {
        VSILOGE( "Add kernel %s fail.", info->name );
    }
    return status;
} /* _cpu_register() */

#if VSI_USE_VXC_BINARY
static const uint8_t* _load_bin
    (
    const char* source_name,
    size_t* size,
    const vsi_nn_vx_bin_resource_item_type* source_map,
    size_t source_map_size,
    const char* tail
    )
{
    const uint8_t* source;
    char source_path[VSI_NN_MAX_PATH];
    size_t n;
    int i;
    source = NULL;
    n = snprintf( source_path, VSI_NN_MAX_PATH, "%s%s", source_name, tail );
    if( n == VSI_NN_MAX_PATH )
    {
        VSILOGE("Kernel source path overflow %d/%d", n, VSI_NN_MAX_PATH);
        *size = 0;
        return NULL;
    }
    for( i = 0; i < (int)source_map_size; i++ )
    {
        if( strncmp( source_map[i].name, source_path, VSI_NN_MAX_PATH ) == 0 )
        {
            source = source_map[i].data;
            *size = source_map[i].len;
            break;
        }
    }
    if( !source )
    {
        *size = 0;
    }
    return source;
} /* _load_bin() */
#endif

static const uint8_t* _load_internal_executable
    (
    const char* source_name,
    size_t* size,
    vsi_nn_kernel_type_e type
    )
{
#if VSI_USE_VXC_BINARY
    switch( type )
    {
        case VSI_NN_KERNEL_TYPE_EVIS:
            return _load_bin( source_name, size,
                vx_bin_resource_items_vx, vx_bin_resource_items_vx_cnt, "_vx" );
            break;
        case VSI_NN_KERNEL_TYPE_CL:
            return _load_bin( source_name, size,
                vx_bin_resource_items_cl, vx_bin_resource_items_cl_cnt, "_cl" );
    }
#endif
    return NULL;
} /* _load_internal_executable() */

static char* _load_source_code_from_file
    (
    const char* source_name,
    size_t* size
    )
{
    char* source;
    FILE* fp;
    size_t total_bytes;
    size_t read_bytes;
    source = NULL;
    //TODO: Pack new name
    fp = vsi_nn_fopen( source_name, "rb" );
    if( NULL == fp )
    {
        VSILOGE("Open program file %s fail.", source_name);
        *size = 0;
        goto final;
    }
    fseek( fp, 0, SEEK_END );
    total_bytes = ftell( fp );
    fseek( fp, 0, SEEK_SET );
    if( total_bytes == 0 )
    {
        VSILOGE("Program file %s is empty.", source_name);
        *size = 0;
        goto final;
    }
    source = (char*)malloc( total_bytes + 1 );
    if( source )
    {
        read_bytes = 0;
        while( total_bytes - read_bytes > 0 )
        {
            read_bytes += fread( &source[read_bytes], 1, total_bytes - read_bytes, fp );
        }
        source[read_bytes] = 0;
        *size = read_bytes;
    }
final:
    if (fp) fclose( fp );
    return source;
} /* _load_source_code_from_file() */

static vx_program _create_program
    (
    vx_context ctx,
    kernel_program_info_t *program_info,
    size_t num
    )
{
    vx_char** sources = NULL;
    vx_size* source_sizes = NULL;
    size_t i;
    vsi_status status;
    vx_program program;
    program = NULL;

    sources = (vx_char**)malloc( sizeof(vx_char*) * num );
    CHECK_PTR_FAIL_GOTO( sources, "Create buffer fail.", final );
    source_sizes = (vx_size*)malloc( sizeof(vx_size) * num );
    CHECK_PTR_FAIL_GOTO( source_sizes, "Create buffer fail.", final );

    for( i = 0; i < num; i ++ )
    {
        sources[i] = (vx_char*)program_info[i].data;
        source_sizes[i] = (vx_size)program_info[i].size;
    }
    program = vxCreateProgramWithSource( ctx, (vx_uint32)num,
            (const vx_char**)sources, source_sizes );
    status = vxGetStatus( (vx_reference)program );
    if(VSI_SUCCESS != status)
    {
        VSILOGE("Create program from source fail!");
    }

final:
    vsi_nn_safe_free( sources );
    vsi_nn_safe_free( source_sizes );

    return program;
} /* _create_program() */

static vx_program _create_program_from_code
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel
    )
{
    const vsi_nn_kernel_source_info_t* source_info;
    kernel_program_info_t* program_info;
    size_t i;
    vx_program program = NULL;
    source_info = &kernel->gpu.sources[VSI_NN_GPU_SOURCE_FMT_CODE];

    if( source_info->num == 0 )
    {
        VSILOGE("Not executable source found in kernel.");
        return NULL;
    }
    program_info = (kernel_program_info_t*)malloc(
            source_info->num * sizeof(kernel_program_info_t) );
    if( !program_info )
    {
        VSILOGE("Malloc program memory fail.");
        return NULL;
    }
    memset( program_info, 0, source_info->num * sizeof(kernel_program_info_t) );

    for( i = 0; i < source_info->num; i ++ )
    {
        program_info[i].data = (const void*)vsi_nn_resource_load_source_code(
                source_info->data[i], &program_info[i].size, kernel->type );
        if( !program_info[i].data )
        {
            program_info[i].reserve_mem = (void*)_load_source_code_from_file(
                    source_info->data[i], &program_info[i].size );
            program_info[i].data = (const void*)program_info[i].reserve_mem;
        }
    }
    program = _create_program( graph->ctx->c, program_info, source_info->num );
    if( program_info )
    {
        for( i = 0; i < source_info->num; i ++ )
        {
            if( program_info[i].reserve_mem )
            {
                free( program_info[i].reserve_mem );
            }
        }
        free( program_info );
    }
    return program;
} /* _create_program_from_code() */

static vx_program _create_program_from_code_ext
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel,
    const char** resources
    )
{
    const vsi_nn_kernel_source_info_t* source_info;
    kernel_program_info_t* program_info;
    size_t i;
    vx_program program = NULL;
    source_info = &kernel->gpu.sources[VSI_NN_GPU_SOURCE_FMT_CODE];

    if( source_info->num == 0 )
    {
        VSILOGE("Not executable source found in kernel.");
        return NULL;
    }
    program_info = (kernel_program_info_t*)malloc(
            source_info->num * sizeof(kernel_program_info_t) );
    if( !program_info )
    {
        VSILOGE("Malloc program memory fail.");
        return NULL;
    }
    memset( program_info, 0, source_info->num * sizeof(kernel_program_info_t) );

    for( i = 0; i < source_info->num; i ++ )
    {
        program_info[i].data = (const void*)(resources[i]);
        if( !program_info[i].data )
        {
            program_info[i].reserve_mem = (void*)_load_source_code_from_file(
                    source_info->data[i], &program_info[i].size );
            program_info[i].data = (const void*)program_info[i].reserve_mem;
        }
    }
    program = _create_program( graph->ctx->c, program_info, source_info->num );
    if( program_info )
    {
        for( i = 0; i < source_info->num; i ++ )
        {
            if( program_info[i].reserve_mem )
            {
                free( program_info[i].reserve_mem );
            }
        }
        free( program_info );
    }
    return program;
} /* _create_program_from_code_ext() */

static vx_program _create_program_from_executable
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel
    )
{
    const vsi_nn_kernel_source_info_t* source_info;
    kernel_program_info_t program_info;
    vx_program program = NULL;
    source_info = &kernel->gpu.sources[VSI_NN_GPU_SOURCE_FMT_EXECUTABLE];

    if( source_info->num == 0 )
    {
        VSILOGE("Not executable source found in kernel.");
        return NULL;
    }

    VSI_ASSERT( source_info->num == 1 );
    memset( &program_info, 0, sizeof( kernel_program_info_t ) );

    program_info.data = _load_internal_executable(
            source_info->data[0], &program_info.size, kernel->type);
    program = vxCreateProgramWithBinary( graph->ctx->c,
            (const vx_uint8 *)program_info.data, program_info.size );
    return program;
} /* _create_program_from_executable() */

static vsi_status _gpu_register
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_status status;
    vx_kernel_description_t* info;
    vx_kernel obj;
    vsi_nn_context_t context;
    vx_program program = NULL;
    const vsi_nn_gpu_source_fmt_e active_fmt = kernel->gpu.active_source_fmt;

#define MAX_BUILDPROGRAM_LEN 1024
    char cmd[MAX_BUILDPROGRAM_LEN] = { 0 };
    size_t cost_bytes = 0;

    memset( cmd, 0, sizeof(char) * MAX_BUILDPROGRAM_LEN );
    context = graph->ctx;

    status = VSI_FAILURE;
    info = &(kernel->info);

    switch( active_fmt )
    {
        case VSI_NN_GPU_SOURCE_FMT_CODE:
            program = _create_program_from_code( graph, kernel );
            break;
        case VSI_NN_GPU_SOURCE_FMT_EXECUTABLE:
            program = _create_program_from_executable( graph, kernel );
            break;
        default:
            VSILOGE("Unknown source format %d", kernel->gpu.active_source_fmt);
            break;
    }
    if( NULL == program )
    {
        return status;
    }

    if( context->config.evis.ver == VSI_NN_HW_EVIS_NONE )
    {
        // set default evis version is 2
        if( VSI_NN_KERNEL_TYPE_EVIS == kernel->type )
        {
            cost_bytes = snprintf( cmd, MAX_BUILDPROGRAM_LEN,
                    "-cl-viv-vx-extension -D VX_VERSION=2 -D USE_40BITS_VA=%d",
                    context->config.use_40bits_va );
        }
    }
    else
    {
        cost_bytes = snprintf( cmd, MAX_BUILDPROGRAM_LEN,
                "-cl-viv-vx-extension -D VX_VERSION=%d -D USE_40BITS_VA=%d",
                context->config.evis.ver, context->config.use_40bits_va );
    }
    // Pack build option
    if( kernel->gpu.sources[active_fmt].build_option.data )
    {
        vsi_nn_kernel_build_option_t * option = &kernel->gpu.sources[active_fmt].build_option;
        if( MAX_BUILDPROGRAM_LEN - cost_bytes > strlen( option->data ) + 1 )
        {
            snprintf( &cmd[cost_bytes], MAX_BUILDPROGRAM_LEN - cost_bytes,
                    " %s", option->data );
        }
        else
        {
            VSILOGE("Build option is too long!");
            VSI_ASSERT( FALSE );
        }
    }

    status = vxBuildProgram( program, cmd );

    if( VSI_SUCCESS != status )
    {
        VSILOGE("Build program fail.");
        return status;
    }

    obj = vxAddKernelInProgram(
        program,
        info->name,
        info->enumeration,
        info->numParams,
        info->validate,
        info->initialize,
        info->deinitialize
        );

    if( obj )
    {
        status = _kernel_init_obj( info, obj );
        //vxReleaseKernel( &obj );
    }
    else
    {
        VSILOGE( "Add kernel %s fail.", info->name );
    }
    if( program )
    {
        vxReleaseProgram( &program );
    }
    return status;
} /* _gpu_register() */

static vsi_status _gpu_register_ext
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel,
    const char** resources
    )
{
    vsi_status status;
    vx_kernel_description_t* info;
    vx_kernel obj;
    vsi_nn_context_t context;
    vx_program program = NULL;
    const vsi_nn_gpu_source_fmt_e active_fmt = kernel->gpu.active_source_fmt;

#define MAX_BUILDPROGRAM_LEN 1024
    char cmd[MAX_BUILDPROGRAM_LEN] = { 0 };
    size_t cost_bytes = 0;

    memset( cmd, 0, sizeof(char) * MAX_BUILDPROGRAM_LEN );
    context = graph->ctx;

    status = VSI_FAILURE;
    info = &(kernel->info);

    switch( active_fmt )
    {
        case VSI_NN_GPU_SOURCE_FMT_CODE:
            program = _create_program_from_code_ext( graph, kernel,resources );
            break;
        case VSI_NN_GPU_SOURCE_FMT_EXECUTABLE:
            program = _create_program_from_executable( graph, kernel );
            break;
        default:
            VSILOGE("Unknown source format %d", kernel->gpu.active_source_fmt);
            break;
    }
    if( NULL == program )
    {
        return status;
    }

    if( context->config.evis.ver == VSI_NN_HW_EVIS_NONE )
    {
        // set default evis version is 2
        if( VSI_NN_KERNEL_TYPE_EVIS == kernel->type )
        {
            cost_bytes = snprintf( cmd, MAX_BUILDPROGRAM_LEN,
                    "-cl-viv-vx-extension -D VX_VERSION=2 -D USE_40BITS_VA=%d",
                    context->config.use_40bits_va );
        }
    }
    else
    {
        cost_bytes = snprintf( cmd, MAX_BUILDPROGRAM_LEN,
                "-cl-viv-vx-extension -D VX_VERSION=%d -D USE_40BITS_VA=%d",
                context->config.evis.ver, context->config.use_40bits_va );
    }
    // Pack build option
    if( kernel->gpu.sources[active_fmt].build_option.data )
    {
        vsi_nn_kernel_build_option_t * option = &kernel->gpu.sources[active_fmt].build_option;
        if( MAX_BUILDPROGRAM_LEN - cost_bytes > strlen( option->data ) + 1 )
        {
            snprintf( &cmd[cost_bytes], MAX_BUILDPROGRAM_LEN - cost_bytes,
                    " %s", option->data );
        }
        else
        {
            VSILOGE("Build option is too long!");
            VSI_ASSERT( FALSE );
        }
    }

    status = vxBuildProgram( program, cmd );

    if( VSI_SUCCESS != status )
    {
        VSILOGE("Build program fail.");
        return status;
    }

    obj = vxAddKernelInProgram(
        program,
        info->name,
        info->enumeration,
        info->numParams,
        info->validate,
        info->initialize,
        info->deinitialize
        );

    if( obj )
    {
        status = _kernel_init_obj( info, obj );
        //vxReleaseKernel( &obj );
    }
    else
    {
        VSILOGE( "Add kernel %s fail.", info->name );
    }
    if( program )
    {
        vxReleaseProgram( &program );
    }
    return status;
} /* _gpu_register_ext() */

static vsi_status _kernel_init_obj
    (
    vx_kernel_description_t* info,
    vx_kernel obj
    )
{
    vsi_status status;
    uint32_t i;

    status = VSI_SUCCESS;
    for( i = 0; i < info->numParams; i ++ )
    {
        status = vxAddParameterToKernel(
            obj,
            i,
            info->parameters[i].direction,
            info->parameters[i].data_type,
            info->parameters[i].state
            );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Add parameter %d to kernel %s fail. with %d.",
                i, info->name, status );
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
            info->name, status );
        status = vxRemoveKernel( obj );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Remove kernel %s fail with %d.",
                info->name, status );
        }
    }
    return status;
} /* _kernel_init_obj() */

vsi_status vsi_nn_kernel_register
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_status status;
    status = VSI_FAILURE;
    switch( kernel->type )
    {
    case VSI_NN_KERNEL_TYPE_CPU:
        status = _cpu_register( graph, kernel );
        break;
    case VSI_NN_KERNEL_TYPE_EVIS:
    case VSI_NN_KERNEL_TYPE_CL:
        status = _gpu_register( graph, kernel );
        break;
    case VSI_NN_KERNEL_TYPE_VX:
        VSILOGE("Openvx node no need to register.");
        break;
    default:
        VSILOGE("Unknown kernel %d.", kernel->type);
        break;
    }
    return status;
} /* vsi_nn_kernel_register() */

vsi_status vsi_nn_kernel_register_ext
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_t * kernel,
    const char** resources
    )
{
    vsi_status status;
    status = VSI_FAILURE;
    status = _gpu_register_ext( graph, kernel,resources );
    return status;
} /* vsi_nn_kernel_register_ext */

vsi_nn_kernel_node_t  vsi_nn_kernel_create_node
    (
    vsi_nn_graph_t* graph,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_status status;
    vx_context ctx;
    vx_kernel obj;
    vx_node node;
    vx_kernel_description_t* info;

    info = &(kernel->info);
    // Validate kernel
    if( !info->initialize )
    {
        VSILOGE("Kernel %s initializer is NULL", info->name);
        return NULL;
    }
    if( !info->validate )
    {
        VSILOGE("Kernel %s validator is NULL", info->name);
        return NULL;
    }
    if( !info->deinitialize )
    {
        VSILOGE("Kernel %s deinitializer is NULL", info->name);
        return NULL;
    }
    if( info->enumeration == KERNEL_ID_PLACEHOLDER )
    {
        //VSILOGD("Kernel id: %#x, %#x", kernel->unique_id, info->enumeration);
        info->enumeration = (vx_enum)kernel->unique_id;
    }
    if( info->enumeration > KERNEL_ID_OVXLIB_RESERVED )
    {
        VSILOGE("Kernel id is invalid %#x(max: %#x)",
                info->enumeration, KERNEL_ID_OVXLIB_RESERVED);
        return NULL;
    }

    ctx = vxGetContext( (vx_reference)graph->g );

    obj = vxGetKernelByName( ctx, info->name );
    status = vxGetStatus( (vx_reference)obj );
    if (VSI_SUCCESS != status)
    {
        /* Register kernel */
        status = vsi_nn_kernel_register( graph, kernel );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Register client kernel %s fail with %d.",
                info->name, status );
            return NULL;
        }
        else
        {
            VSILOGD( "Register client kernel %s successfully.",
                info->name );
        }

        /* Load kernel */
        obj = vxGetKernelByName( ctx, info->name );
        status = vxGetStatus( (vx_reference)obj );
    }
    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Load client kernel %s fail with %d.",
            info->name, status );
        return NULL;
    }
    node = vxCreateGenericNode( graph->g, obj );
    vxReleaseKernel( &obj );
    status = vxGetStatus( (vx_reference)node );
    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Load client node from kernel %s fail with %d.",
            info->name, status );
        return NULL;
    }
    if( node )
    {
        // Set default border mode.
        vx_border_t border;
        border.mode = VX_BORDER_REPLICATE;
        border.constant_value.U32 = 0;
        status |= vxSetNodeAttribute( node, VX_NODE_BORDER, &border, sizeof(border) );
    }
    return (vsi_nn_kernel_node_t)node;
} /* vsi_nn_kernel_create_node() */

vsi_nn_kernel_node_t  vsi_nn_kernel_create_node_ext
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_t * kernel,
    const char** resources
    ){
    vsi_status status;
    vx_context ctx;
    vx_kernel obj;
    vx_node node;
    vx_kernel_description_t* info;

    info = &(kernel->info);
    // Validate kernel
    if( !info->initialize )
    {
        VSILOGE("Kernel %s initializer is NULL", info->name);
        return NULL;
    }
    if( !info->validate )
    {
        VSILOGE("Kernel %s validator is NULL", info->name);
        return NULL;
    }
    if( !info->deinitialize )
    {
        VSILOGE("Kernel %s deinitializer is NULL", info->name);
        return NULL;
    }
    if( info->enumeration == KERNEL_ID_PLACEHOLDER )
    {
        //VSILOGD("Kernel id: %#x, %#x", kernel->unique_id, info->enumeration);
        info->enumeration = (vx_enum)kernel->unique_id;
    }

    ctx = vxGetContext( (vx_reference)graph->g );

    obj = vxGetKernelByName( ctx, info->name );
    status = vxGetStatus( (vx_reference)obj );
    if (VSI_SUCCESS != status)
    {
        fprintf(stderr, "\n"); // TODO: This is a hack for driver msg
        /* Register kernel */
        status = vsi_nn_kernel_register_ext( graph, kernel,resources );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Register client kernel %s fail with %d.",
                info->name, status );
            return NULL;
        }
        else
        {
            VSILOGD( "Register client kernel %s successfully.",
                info->name );
        }

        /* Load kernel */
        obj = vxGetKernelByName( ctx, info->name );
        status = vxGetStatus( (vx_reference)obj );
    }
    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Load client kernel %s fail with %d.",
            info->name, status );
        return NULL;
    }
    node = vxCreateGenericNode( graph->g, obj );
    vxReleaseKernel( &obj );
    status = vxGetStatus( (vx_reference)node );
    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Load client node from kernel %s fail with %d.",
            info->name, status );
        return NULL;
    }
    if( node )
    {
        // Set default border mode.
        vx_border_t border;
        border.mode = VX_BORDER_REPLICATE;
        border.constant_value.U32 = 0;
        status |= vxSetNodeAttribute( node, VX_NODE_BORDER, &border, sizeof(border) );
    }
    return (vsi_nn_kernel_node_t)node;
} /* vsi_nn_kernel_create_node_ext() */

vsi_status vsi_nn_kernel_node_set_border
    (vsi_nn_kernel_node_t node,
    vx_border_t* border)
{
    vsi_status status = VSI_FAILURE;
    status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, border, sizeof(vx_border_t) );
    return status;
}

vsi_status vsi_nn_kernel_node_pass_param
    (
    vsi_nn_kernel_node_t node,
    vsi_nn_kernel_node_param_t * params,
    size_t num
    )
{
    vsi_status status;
    uint32_t i;

    status = VSI_FAILURE;
    for( i = 0; i < num; i++ )
    {
        status = vxSetParameterByIndex( (vx_node)node, i, (vx_reference)params[i] );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Set %d parameter fail.", i );
            break;
        }
    }
    return status;
} /* vsi_nn_kernel_node_pass_param() */

vsi_nn_kernel_tensor_t vsi_nn_kernel_tensor_reshape
    (
    vsi_nn_kernel_tensor_t tensor,
    vsi_size_t* shape,
    vsi_size_t rank
    )
{
    return (vsi_nn_kernel_tensor_t)vsi_nn_safe_reshape_tensor((vx_tensor)tensor,
            (void*)shape, (vsi_size_t)rank, sizeof(shape[0]));
} /* vsi_nn_kernel_tensor_reshape() */

void vsi_nn_kernel_tensor_release
    (
    vsi_nn_kernel_tensor_t* tensor
    )
{
    vxReleaseTensor( (vx_tensor*)tensor );
    *tensor = NULL;
} /* vsi_nn_kernel_tensor_release() */

void vsi_nn_kernel_add_source
    (
    vsi_nn_kernel_t* kernel,
    vsi_nn_gpu_source_fmt_e fmt,
    size_t source_num,
    ...
    )
{
    va_list args;
    va_start( args, source_num );
    vsi_nn_kernel_add_source_internal( kernel, fmt, source_num, args );
    va_end( args );
} /* vsi_nn_kernel_add_source() */

void vsi_nn_kernel_add_build_option
    (
    vsi_nn_kernel_t * kernel,
    const char * option
    )
{
    const vsi_nn_gpu_source_fmt_e fmt = VSI_NN_GPU_SOURCE_FMT_CODE;
    vsi_nn_kernel_build_option_t * build_option;
    size_t new_size;
    size_t item_size;
    size_t org_size;
    char * buf = NULL;
    if( !kernel || !option )
    {
        VSILOGW("Get NULL pointer.");
        return;
    }
    build_option = &kernel->gpu.sources[fmt].build_option;
    buf = build_option->data;
    item_size = strlen( option );
    org_size = 0;
    if( buf )
    {
        org_size = strlen( buf );
    }
    new_size = org_size + item_size;
    buf = (char*)realloc( buf, new_size + 2 ); // Space and terminator
    if( !buf )
    {
        VSILOGE("Out of memory");
        return;
    }
    snprintf( &buf[org_size], item_size + 2, " %s", option );
    build_option->data = buf;
} /* vsi_nn_kernel_add_build_option() */

void vsi_nn_kernel_release
    (
    vsi_nn_kernel_t ** kernel
    )
{
    if( kernel && *kernel )
    {
        _kernel_clear_source( *kernel );
        free( *kernel );
        *kernel = NULL;
    }
} /* vsi_nn_kernel_release() */

vsi_nn_kernel_t* vsi_nn_kernel_create
    (
    vsi_nn_kernel_type_e type
    )
{
    vsi_nn_kernel_t* kernel = (vsi_nn_kernel_t*)malloc( sizeof(vsi_nn_kernel_t) );
    if( !kernel )
    {
        VSILOGE("Out of memory, create kernel fail.");
        return NULL;
    }
    /*
     * !!!WARNING!!!
     * Here must reset memory to 0, or vsi_nn_kernel_reset will crash.
     */
    memset( kernel, 0, sizeof(vsi_nn_kernel_t) );
    vsi_nn_kernel_reset( kernel, type );
    return kernel;
} /* vsi_nn_kernel_create() */

void vsi_nn_kernel_reset
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_kernel_type_e type
    )
{
    if( kernel )
    {
        _kernel_clear_source( kernel );
        memset( kernel, 0, sizeof(vsi_nn_kernel_t) );
        kernel->type = type;
        // TODO: Choose binary
#if VSI_USE_VXC_BINARY
        kernel->gpu.active_source_fmt = VSI_NN_GPU_SOURCE_FMT_EXECUTABLE;
#else
        kernel->gpu.active_source_fmt = VSI_NN_GPU_SOURCE_FMT_CODE;
#endif
        // Set default functions.
        kernel->info.enumeration = KERNEL_ID_PLACEHOLDER;
        kernel->info.initialize = _kernel_initializer;
        kernel->info.validate = _kernel_validator;
        kernel->info.deinitialize = _kernel_deinitializer;
    }
} /* vsi_nn_kernel_reset() */

vsi_nn_kernel_node_t vsi_nn_kernel_selector
    (
    vsi_nn_graph_t* graph,
    const char* kernel_name,
    vsi_nn_tensor_t** inputs,
    size_t input_num,
    vsi_nn_tensor_t** outputs,
    size_t output_num,
    const vsi_nn_kernel_param_t* params
    )
{
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_t * kernel;
    const vsi_nn_kernel_backend_t* backend;
    vsi_nn_kernel_selector_t selector;
    vsi_status status = VSI_SUCCESS;
    if( !kernel_name )
    {
        VSI_ASSERT( FALSE );
        return NULL;
    }
    if( !graph )
    {
        VSI_ASSERT( FALSE );
        return NULL;
    }

    backend = vsi_nn_kernel_backend_get( kernel_name );
    if( !backend )
    {
        VSILOGW("Not found kernel \"%s\"", kernel_name);
        return NULL;
    }
    kernel = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_NONE );
    if( !kernel )
    {
        return NULL;
    }
    memset( &selector, 0, sizeof(vsi_nn_kernel_selector_t) );
    if( backend->select )
    {
        status = backend->select( graph, inputs, input_num, outputs, output_num,
                params, &selector );
        VSI_ASSERT( status == VSI_SUCCESS );
    }
    else
    {
        vsi_nn_kernel_pirority_t default_pirority[] = {
            { VSI_NN_KERNEL_TYPE_SP,    5 },
            { VSI_NN_KERNEL_TYPE_EVIS,  4 },
            { VSI_NN_KERNEL_TYPE_CL,    3 },
            { VSI_NN_KERNEL_TYPE_VX,    2 },
            { VSI_NN_KERNEL_TYPE_CPU,   1 },
            };
        vsi_nn_kernel_pirority_set( &selector,
                default_pirority, _cnt_of_array(default_pirority) );
    }
    /**
     * All kernels for one operation will share the same id.
     */

    {
        uint32_t i;
        vsi_nn_kernel_type_e type;
        vsi_nn_kernel_setup_func_t kernel_func = NULL;;
        for( i = 0; i < (uint32_t)selector.allow_kernel_num; i ++ )
        {
            type = selector.pirority[i].kernel_type;

            /* Skip evis and cl when disable shader */
            if ( (type == VSI_NN_KERNEL_TYPE_EVIS || type == VSI_NN_KERNEL_TYPE_CL)
                && ( _check_shader_support(graph) == FALSE ||
                vsi_nn_kernel_is_supported_types(inputs, input_num, outputs, output_num) == FALSE ) )
            {
                continue;
            }
            /* Skip evis if not support */
            if( type == VSI_NN_KERNEL_TYPE_EVIS
                    && graph->ctx->config.evis.ver == VSI_NN_HW_EVIS_NONE )
            {
                continue;
            }

            /* Skip StreamProcesor if not support */
            if( type == VSI_NN_KERNEL_TYPE_SP &&
                _check_stream_process_support(graph, inputs, input_num) == FALSE )
            {
                continue;
            }

            kernel_func = backend->setup[type];
            /* Skip no kernel func */
            if( NULL == kernel_func )
            {
                continue;
            }
            vsi_nn_kernel_reset( kernel, type );
            kernel->unique_id = KERNEL_ID_OVXLIB_START + backend->unique_id;
            node = kernel_func( graph, inputs, input_num,
                    outputs, output_num, params, kernel );
            /* If node created, break the loop */
            if( node )
            {
                VSILOGD("Instance %s node with kernel \"%s\" ",
                    vsi_nn_kernel_type_str(type), kernel_name);
                break;
            }
        }
    }
    if( !node )
    {
        VSILOGW("No valid kernel for %s", kernel_name);
    }
    vsi_nn_kernel_release( &kernel );

    return node;
} /* vsi_nn_kernel_selector() */

vsi_bool vsi_nn_kernel_gpu_check_shape
    ( const vsi_size_t * shape, vsi_size_t rank )
{
    vsi_size_t i;
    vsi_bool ret = TRUE;
    const vsi_size_t channel_dim = 2;
    for( i = 0; i < vsi_nn_min(rank, channel_dim); i++ )
    {
        if( shape[i] == 0
         || shape[i] >= GPU_TENSOR_MAX_WIDTH )
        {
            ret = FALSE;
            break;
        }
    }
    return ret;
} /* vsi_nn_kernel_gpu_check_shape() */

vsi_nn_kernel_scalar_t vsi_nn_kernel_scalar_create
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_dtype_e dtype,
    const void * data
    )
{
    vx_enum vxtype = VX_TYPE_FLOAT32;
    if( !graph || !data )
    {
        return NULL;
    }
    switch( dtype )
    {
        case I8:
            vxtype = VX_TYPE_INT8;
            break;
        case I16:
            vxtype = VX_TYPE_INT16;
            break;
        case I32:
            vxtype = VX_TYPE_INT32;
            break;
        case I64:
            vxtype = VX_TYPE_INT64;
            break;
        case U8:
            vxtype = VX_TYPE_UINT8;
            break;
        case U16:
            vxtype = VX_TYPE_UINT16;
            break;
        case U32:
            vxtype = VX_TYPE_UINT32;
            break;
        case U64:
            vxtype = VX_TYPE_UINT64;
            break;
        case F16:
            vxtype = VX_TYPE_FLOAT16;
            break;
        case F32:
            vxtype = VX_TYPE_FLOAT32;
            break;
        case BF16:
        case BOOL8:
        default:
            VSILOGW("Unsupport dtype %d", dtype);
            return NULL;
    }
    return vxCreateScalar( graph->ctx->c, vxtype, (void*)data );
}  /* vsi_nn_kernel_scalar_create() */

vsi_status vsi_nn_kernel_gpu_add_param
    (
    vsi_nn_kernel_node_t node,
    const char * param_key,
    void * data
    )
{
    return vxSetNodeUniform( (vx_node)node, param_key, 1, data );
} /* vsi_nn_kernel_gpu_add_param() */

vsi_status vsi_nn_kernel_gpu_config
    (
    vsi_nn_kernel_node_t node,
    const gpu_param_t * gpu_param
    )
{
    vsi_status status;
    vx_kernel_execution_parameters_t param;
    param.workDim = gpu_param->dim;
    memcpy( param.globalWorkOffset, gpu_param->global_offset,
            sizeof(size_t) * GPU_MAX_DIMENSION_SIZE );
    memcpy( param.globalWorkScale, gpu_param->global_scale,
            sizeof(size_t) * GPU_MAX_DIMENSION_SIZE );
    memcpy( param.localWorkSize, gpu_param->local_size,
            sizeof(size_t) * GPU_MAX_DIMENSION_SIZE );
    memcpy( param.globalWorkSize, gpu_param->global_size,
            sizeof(size_t) * GPU_MAX_DIMENSION_SIZE );
    status = vxSetNodeAttribute( (vx_node)node,
            VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
            &param, sizeof(vx_kernel_execution_parameters_t) );
    return status;
} /* vsi_nn_kernel_gpu_config() */

vsi_nn_kernel_tensor_attr_t * vsi_nn_kernel_tensor_attr_create
    ( vsi_nn_kernel_tensor_t tensor )
{
    vsi_nn_kernel_tensor_attr_t * attr;
    vsi_status status;
    vsi_size_t dim_num;
    vsi_nn_type_e dtype = VSI_NN_TYPE_FLOAT16;
    vsi_nn_qnt_type_e quant_type = VSI_NN_QNT_TYPE_NONE;
    attr = (vsi_nn_kernel_tensor_attr_t *)malloc(
            sizeof(vsi_nn_kernel_tensor_attr_t) );
    if( !attr )
    {
        VSILOGE("Out of memory, create tensor attr fail!");
        return NULL;
    }
    memset( attr, 0, sizeof(vsi_nn_kernel_tensor_attr_t) );

    status = vxQueryTensor( (vx_tensor)tensor, VX_TENSOR_NUM_OF_DIMS,
        &dim_num, sizeof(dim_num));
    CHECK_STATUS( status );
    if( status == VSI_SUCCESS )
    {
        vsi_size_array_t * shape = vsi_size_array_create( dim_num );
        if( shape )
        {
            status = vxQueryTensor( (vx_tensor)tensor, VX_TENSOR_DIMS,
                shape->data, sizeof(shape->data[0]) * dim_num);
            attr->shape = shape;
            CHECK_STATUS( status );
        }
    }
    status = vxQueryTensor( (vx_tensor)tensor, VX_TENSOR_DATA_TYPE,
        &dtype, sizeof(vsi_enum));
    CHECK_STATUS( status );
    attr->dtype = vsi_nn_kernel_map_dtype( dtype );

    status = vxQueryTensor( (vx_tensor)tensor, VX_TENSOR_QUANT_FORMAT,
        &quant_type, sizeof(uint32_t));
    CHECK_STATUS( status );
    attr->quant = vsi_nn_kernel_map_quant_type( quant_type );

    switch( attr->quant )
    {
    case VSI_NN_KERNEL_QUANT_DFP:
    {
        int8_t fl = 0;
        status = vxQueryTensor( (vx_tensor)tensor, VX_TENSOR_FIXED_POINT_POS,
            &fl, sizeof(int8_t));
        CHECK_STATUS( status );
        attr->dfp.fl = (int32_t)fl;
        if (fl >= 0) {
            attr->scale = 1.0f / ((float)((int64_t)1 << fl));
        } else {
            attr->scale = (float)((int64_t)1 << -fl);
        }
    } break;
    case VSI_NN_KERNEL_QUANT_ASYMM:
    {
        status = vxQueryTensor((vx_tensor)tensor,
                               VX_TENSOR_ZERO_POINT,
                               &(attr->asymm.zero_point),
                               sizeof(int32_t));
        CHECK_STATUS(status);
        status = vxQueryTensor((vx_tensor)tensor,
                               VX_TENSOR_SCALE,
                               &(attr->asymm.scale),
                               sizeof(float));
        CHECK_STATUS(status);
        // Reset scale to 1e-8
        if ((attr->asymm.scale - 0.f) < 1e-8)
        {
            attr->asymm.scale = (float)1e-8;
            attr->asymm.zero_point = 0;
        }
        attr->scale = attr->asymm.scale;
        attr->zero_point = attr->asymm.zero_point;
    }
    break;
    default:
        attr->scale = 1.0f;
        break;
    }
    return attr;
} /* vsi_nn_kernel_tensor_attr_create() */

void vsi_nn_kernel_tensor_attr_release
    ( vsi_nn_kernel_tensor_attr_t ** p_attr )
{
    if( p_attr && *p_attr )
    {
        vsi_nn_kernel_tensor_attr_t * attr = *p_attr;
        vsi_size_array_release( &attr->shape );
        if( attr->quant == VSI_NN_KERNEL_QUANT_ASYMM_PERCHANNEL )
        {
            vsi_float_array_release( &attr->asymm_v.scale );
            vsi_int_array_release( &attr->asymm_v.zero_point );
        }
        else if( attr->quant == VSI_NN_KERNEL_QUANT_SYMM_PERCHANNEL )
        {
            //TODO:
        }
        free( attr );
        *p_attr = NULL;
    }
} /* vsi_nn_kernel_tensor_attr_release() */

vsi_status vsi_nn_kernel_pirority_set
    (
    vsi_nn_kernel_selector_t * selector,
    const vsi_nn_kernel_pirority_t * pirority,
    size_t pirority_size
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_pirority_t tmp;
    uint32_t i, j, k;
    VSI_ASSERT( pirority_size <= VSI_NN_KERNEL_TYPE_NUM );
    VSI_ASSERT( pirority_size > 0 );
    VSI_ASSERT( pirority != NULL );
    VSI_ASSERT( selector != NULL );
    memcpy( selector->pirority, pirority,
            pirority_size * sizeof(vsi_nn_kernel_pirority_t) );
    selector->allow_kernel_num = (int32_t)pirority_size;

    for( i = 0; i < pirority_size; i ++ )
    {
        k = i;
        VSI_ASSERT( selector->pirority[k].fps <= VSI_NN_KERNEL_PIRORITY_NORMAL_LIMIT );
        for( j = i; j < pirority_size; j ++ )
        {
            if( selector->pirority[k].fps < selector->pirority[j].fps )
            {
                k = j;
            }
        }
        if( k != i )
        {
            memcpy( &tmp, &selector->pirority[i],
                    sizeof( vsi_nn_kernel_pirority_t ) );
            memcpy( &selector->pirority[i], &selector->pirority[k],
                    sizeof( vsi_nn_kernel_pirority_t ) );
            memcpy( &selector->pirority[k], &tmp,
                    sizeof( vsi_nn_kernel_pirority_t ) );
        }
    }
    return status;
} /* vsi_nn_kernel_pirority_set() */

vsi_nn_kernel_t * vsi_nn_KernelCreate(vsi_nn_kernel_type_e type)
{
    return vsi_nn_kernel_create(type);
}/* vsi_nn_KernelCreate() */

void vsi_nn_kernel_add_source_internal
    (
        vsi_nn_kernel_t * kernel,
        vsi_nn_gpu_source_fmt_e fmt,
        size_t source_num,
        va_list args
    )
{
    size_t i;
    vsi_nn_kernel_source_info_t* source;
    if( source_num == 0 )
    {
        return;
    }
    if( fmt >= VSI_NN_GPU_SOURCE_FMT_NUM )
    {
        VSILOGE("Unknown source type %d", fmt);
        return;
    }
    if( kernel->gpu.sources[fmt].data )
    {
        VSILOGE("Kernel source %d has been attached!", fmt);
        return;
    }
    source = &(kernel->gpu.sources[fmt]);
    if( source_num > 0 )
    {
        const size_t mem_size = sizeof(vsi_nn_kernel_source_t) * source_num;
        source->data = (vsi_nn_kernel_source_t*)malloc( mem_size );
        if( !source->data )
        {
            VSILOGE("Out of memory, create kernel source fail.");
            return;
        }
        memset( source->data, 0, mem_size );
    }
    for( i = 0; i < source_num; i ++ )
    {
        vsi_nn_kernel_source_t src = va_arg( args, vsi_nn_kernel_source_t );
        size_t size = strlen( src );
        source->data[i] = (vsi_nn_kernel_source_t)malloc( size * sizeof(char) + 1 );
        if( source->data[i] )
        {
            memcpy( source->data[i], src, size );
            source->data[i][size] = 0;
        }
        else
        {
            VSILOGE("Malloc source memory fail.");
            return;
        }
    }
    source->num = source_num;
}/* vsi_nn_kernel_add_source_internal() */

void vsi_nn_KernelAddSource
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_gpu_source_fmt_e fmt,
    size_t source_num,
    ...
    )
{
    va_list args;
    va_start( args, source_num);
    vsi_nn_kernel_add_source_internal( kernel, fmt, source_num, args );
    va_end( args );
}/* vsi_nn_KernelAddSource() */

void vsi_nn_KernelAddBuildOption
    (
    vsi_nn_kernel_t * kernel,
    const char * option
    )
{
    vsi_nn_kernel_add_build_option( kernel, option );
}/* vsi_nn_KernelAddBuildOption() */

vsi_status vsi_nn_KernelNodePassParam
    (
    vsi_nn_kernel_node_t node,
    vsi_nn_kernel_node_param_t * params,
    size_t num
    )
{
    return vsi_nn_kernel_node_pass_param( node, params, num );
}/* vsi_nn_KernelNodePassParam() */

vsi_nn_kernel_node_t  vsi_nn_KernelCreateNodeExt
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_t * kernel,
    const char** resources
    )
{
    return vsi_nn_kernel_create_node_ext( graph, kernel, resources );
}/* vsi_nn_KernelCreateNodeExt() */

vsi_nn_kernel_scalar_t vsi_nn_kernelScalarCreate
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_dtype_e dtype,
    const void * data
    )
{
    return vsi_nn_kernel_scalar_create( graph, dtype, data );
}/* vsi_nn_kernelScalarCreate() */

vsi_status vsi_nn_KernelGpuConfig
    (
    vsi_nn_kernel_node_t node,
    const gpu_param_t * gpu_param
    )
{
    return vsi_nn_kernel_gpu_config( node, gpu_param );
}/* vsi_nn_KernelGpuConfig() */

static vsi_bool _check_shader_support(vsi_nn_graph_t* graph)
{
    int32_t enableShader = graph->ctx->options.enable_shader;

#if VX_HARDWARE_CAPS_PARAMS_EXT_SUPPORT
    if ( graph->ctx->config.subGroupSize == 0 )
    {
        return FALSE;
    }
#endif

    if (enableShader >= 1)
    {
        return TRUE;
    }

    return FALSE;
}

vsi_bool vsi_nn_kernel_is_supported_types
    (
    vsi_nn_tensor_t** inputs,
    size_t input_num,
    vsi_nn_tensor_t** outputs,
    size_t output_num
    )
{
    size_t i = 0;

    for (i = 0; i < input_num; i++)
    {
        if ( inputs[i] && vsi_nn_TypeGetBits(inputs[i]->attr.dtype.vx_type) == 4 )
        {
            return FALSE;
        }
    }

    for (i = 0; i < output_num; i++)
    {
        if ( outputs[i] && vsi_nn_TypeGetBits(outputs[i]->attr.dtype.vx_type) == 4 )
        {
            return FALSE;
        }
    }

    return TRUE;
}

static vsi_bool _check_stream_process_support
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t** inputs,
    size_t input_num
    )
{
    if ( graph->ctx->config.support_stream_processor == 0 )
    {
        return FALSE;
    }

    if ( graph->ctx->config.sp_exec_count == 0 )
    {
        return FALSE;
    }

    if (inputs && input_num > 0 &&
        inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT32)
    {
        return FALSE;
    }

    return TRUE;
}