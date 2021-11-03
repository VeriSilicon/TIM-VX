/****************************************************************************
*
*    Copyright 2017 - 2020 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#ifndef _VX_EXT_PROGRAM_H_
#define _VX_EXT_PROGRAM_H_

#include <VX/vx.h>

/***********************************************************************************/

#define VX_512BITS_DISABLE      0
#define VX_512BITS_ADD          0x1
#define VX_512BITS_SUBTRACT     0x2
#define VX_512BITS_ACCUMULATOR  0x3

#define VX_512BITS_TYPE_FLOAT32     0x0
#define VX_512BITS_TYPE_FLOAT16     0x1
#define VX_512BITS_TYPE_SIGNED32    0x2
#define VX_512BITS_TYPE_SIGNED16    0x3
#define VX_512BITS_TYPE_SIGNED8     0x4
#define VX_512BITS_TYPE_UNSIGNED32  0x5
#define VX_512BITS_TYPE_UNSIGNED16  0x6
#define VX_512BITS_TYPE_UNSIGNED8   0x7

#define VX_512BITS_SELECT_SRC0      0
#define VX_512BITS_SELECT_SRC1      1
#define VX_512BITS_SELECT_CONSTANTS 2

typedef union _vx_512bits_bin_t
{
    vx_uint8  bin8[16];
    vx_uint16 bin16[8];
    vx_uint32 bin32[4];
}
vx_512bits_bin_t;

typedef union _vx_512bits_config_t
{
    struct
    {
        vx_uint32 flag0 :2;
        vx_uint32 flag1 :2;
        vx_uint32 flag2 :2;
        vx_uint32 flag3 :2;
        vx_uint32 flag4 :2;
        vx_uint32 flag5 :2;
        vx_uint32 flag6 :2;
        vx_uint32 flag7 :2;
        vx_uint32 flag8 :2;
        vx_uint32 flag9 :2;
        vx_uint32 flag10:2;
        vx_uint32 flag11:2;
        vx_uint32 flag12:2;
        vx_uint32 flag13:2;
        vx_uint32 flag14:2;
        vx_uint32 flag15:2;
    }
    bin2;
    
    struct
    {
        vx_uint32 flag0 :4;
        vx_uint32 flag1 :4;
        vx_uint32 flag2 :4;
        vx_uint32 flag3 :4;
        vx_uint32 flag4 :4;
        vx_uint32 flag5 :4;
        vx_uint32 flag6 :4;
        vx_uint32 flag7 :4;
    }
    bin4;
}
vx_512bits_config_t;

typedef struct _vx_512bits_miscconfig_t
{
    vx_uint32 post_shift    :5; /*[0:4]*/
    vx_uint32 resolve1      :3; /*[5:7]*/
    vx_uint32 constant_type :3; /*[8:10]*/
    vx_uint32 resolve2      :1; /*[11:11]*/
    vx_uint32 accu_type     :3; /*[12:14]*/
    vx_uint32 resolve3      :17;/*[15:31]*/
}
vx_512bits_miscconfig_t;

typedef struct _vx_512bits_t
{
    vx_512bits_config_t termConfig;
    vx_512bits_config_t aSelect;
    vx_512bits_config_t aBin[2];
    vx_512bits_config_t bSelect;
    vx_512bits_config_t bBin[2];
    vx_512bits_miscconfig_t miscConfig;
    vx_512bits_bin_t bins[2];
}
vx_512bits_t;

/***********************************************************************************/

typedef enum vx_ext_program_type_e
{
    VX_TYPE_PROGRAM = 0x900
}
vx_ext_program_type_e;

typedef enum vx_program_attribute_e
{
    VX_PROGRAM_ATTRIBUTE_BUILD_LOG  = VX_ATTRIBUTE_BASE(VX_ID_VIVANTE, VX_TYPE_PROGRAM) + 0x0,
}
vx_program_attribute_e;

typedef enum vx_ext_node_attribute_e
{
    VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS = VX_ATTRIBUTE_BASE(VX_ID_VIVANTE, VX_TYPE_NODE) + 0x0,
}
vx_ext_node_attribute_e;

#define VX_MAX_WORK_ITEM_DIMENSIONS        3

typedef struct _vx_kernel_execution_parameters {
    vx_uint32   workDim;
    vx_size     globalWorkOffset[VX_MAX_WORK_ITEM_DIMENSIONS];
    vx_size     globalWorkScale[VX_MAX_WORK_ITEM_DIMENSIONS];
    vx_size     localWorkSize[VX_MAX_WORK_ITEM_DIMENSIONS];
    vx_size     globalWorkSize[VX_MAX_WORK_ITEM_DIMENSIONS];
} vx_kernel_execution_parameters_t;

typedef struct _vx_program *  vx_program;

#define VX_BUILD_SUCCESS                    0
#define VX_BUILD_NONE                       -1
#define VX_BUILD_ERROR                      -2
#define VX_BUILD_IN_PROGRESS                -3

#if defined(__cplusplus)
extern "C" {
#endif


VX_API_ENTRY vx_program VX_API_CALL vxCreateProgramWithSource(
        vx_context context, vx_uint32 count, const vx_char *  strings[], vx_size lengths[]);

VX_API_ENTRY vx_program VX_API_CALL vxCreateProgramWithBinary(
        vx_context context, const vx_uint8 * binary, vx_size size);

VX_API_ENTRY vx_status VX_API_CALL vxReleaseProgram(vx_program *program);

VX_API_ENTRY vx_status VX_API_CALL vxBuildProgram(vx_program program, const vx_char *  options);


VX_API_ENTRY vx_status VX_API_CALL vxQueryProgram(vx_program program, vx_enum attribute, void *ptr, vx_size size);

VX_API_ENTRY vx_kernel VX_API_CALL vxAddKernelInProgram(
        vx_program program, vx_char name[VX_MAX_KERNEL_NAME], vx_enum enumeration, vx_uint32 num_params, vx_kernel_validate_f validate,
        vx_kernel_initialize_f initialize, vx_kernel_deinitialize_f deinitialize);

VX_API_ENTRY vx_status VX_API_CALL vxSetNodeUniform(vx_node node, const vx_char * name, vx_size count, void * value);

VX_API_ENTRY vx_status VX_API_CALL vxSetChildGraphOfNode(vx_node node, vx_graph graph);

VX_API_ENTRY vx_graph VX_API_CALL vxGetChildGraphOfNode(vx_node node);

VX_API_ENTRY vx_status VX_API_CALL vxSetArrayAttribute(vx_array array, vx_enum attribute, void *ptr, vx_size size);

VX_API_ENTRY vx_status VX_API_CALL vxSelectKernelSubname(vx_node node, const vx_char * subname);

#if defined(__cplusplus)
}
#endif

#endif /* __GC_VX_PROGRAM_H__ */
