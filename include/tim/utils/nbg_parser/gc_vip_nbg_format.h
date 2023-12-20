/****************************************************************************
*
*    Copyright (c) 2020-2023 Vivante Corporation
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
#ifndef _VIP_LITE_NBG_FORMAT_H
#define _VIP_LITE_NBG_FORMAT_H

#if defined(__cplusplus)
extern "C"{
#endif

typedef unsigned char       vip_uint8_t;
typedef unsigned short      vip_uint16_t;
typedef unsigned int        vip_uint32_t;
typedef unsigned long long  vip_uint64_t;
typedef signed char         vip_int8_t;
typedef signed short        vip_int16_t;
typedef signed int          vip_int32_t;
typedef signed long long    vip_int64_t;
typedef char                vip_char_t;
typedef float               vip_float_t;
typedef unsigned long long  vip_address_t;


#define NETWORK_NAME_SIZE           64
#define LAYER_NAME_SIZE             64
#define MAX_SW_LAYER_NAME_LENGTH    64
#define MAX_IO_NAME_LEGTH           64
#define NN_CMD_SIZE_128             128
#define NN_CMD_SIZE_192             192
#define TP_CMD_SIZE                 128
#define MAX_NUM_DIMS                6

/* The vip lite binary data (file) is composed as following layout:
1. Fixed section.
    Where fixed information is stored, such as header, pool info, and data entries.

2. Dynamic section.
    Where the real data is stored, or indexed.
*/
/********************** The fixed part of the binary file *************************/

typedef struct _gcvip_bin_feature_database
{
    vip_uint32_t hi_reorder_fix:1;  /* gcFEATURE_BIT_HI_REORDER_FIX */
    vip_uint32_t ocb_counter:1;     /* gcFEATURE_BIT_OCB_COUNTER */
    vip_uint32_t nn_command_size:2; /* the size of NN command, 0: 128bytes, 1: 192bytes */
    vip_uint32_t change_ppu_param:1;/* 1: the NBG supports change PPU param, 0: not supports */
    vip_uint32_t reserved:27;       /* reserved bits */

    vip_uint32_t num_pixel_pipes;   /* gcFEATURE_VALUE_NumPixelPipes */
    vip_uint8_t  core_count;        /* VIP core count */

    vip_uint8_t  reserved1;
    vip_uint8_t  reserved2;
    vip_uint8_t  reserved3;

    vip_uint32_t vsi_reserved[13];  /* reserved for  verisilicon */

    vip_uint32_t customer_reserved[48]; /* reserved for customer */
} gcvip_bin_feature_database_t;

typedef struct _gcvip_bin_header
{
    vip_char_t      magic[4];
    vip_uint32_t    version;
    vip_uint32_t    hw_target;
    vip_char_t      network_name[NETWORK_NAME_SIZE];
    vip_uint32_t    layer_count;
    vip_uint32_t    operation_count;
    vip_uint32_t    input_count;
    vip_uint32_t    output_count;
    gcvip_bin_feature_database_t feature_db;
} gcvip_bin_header_t;

typedef struct _gcvip_bin_pool
{
    vip_uint32_t    size;
    vip_uint32_t    alignment;
    vip_uint32_t    base;
} gcvip_bin_pool_t;

typedef struct _gcvip_bin_entry
{
    vip_uint32_t    offset;
    vip_uint32_t    size;
} gcvip_bin_entry_t;

typedef struct _gcvip_bin_fixed
{
    gcvip_bin_header_t      header;
    gcvip_bin_pool_t        pool;
    vip_uint32_t            axi_sram_base;
    vip_uint32_t            axi_sram_size;
    vip_uint32_t            vip_sram_base;
    vip_uint32_t            vip_sram_size;
    gcvip_bin_entry_t       layer_table;
    gcvip_bin_entry_t       opeartion_table;
    gcvip_bin_entry_t       LCD_table;
    gcvip_bin_entry_t       LCD; /* Loading Config Data */
    gcvip_bin_entry_t       nn_op_data_table;
    gcvip_bin_entry_t       tp_op_data_table;
    gcvip_bin_entry_t       sh_op_data_table;
    gcvip_bin_entry_t       input_table;
    gcvip_bin_entry_t       output_table;
    gcvip_bin_entry_t       patch_data_table;
    gcvip_bin_entry_t       layer_param_table;
    gcvip_bin_entry_t       sw_op_data_table;
    gcvip_bin_entry_t       hw_init_op_table;
    gcvip_bin_entry_t       ICD_table;
    gcvip_bin_entry_t       ICD; /* Initialize Config Data */
    gcvip_bin_entry_t       ppu_param_table;
} gcvip_bin_fixed_t;

typedef struct _gcvip_bin_inout_entry
{
    vip_uint32_t    dim_count;
    vip_uint32_t    dim_size[MAX_NUM_DIMS];
    vip_uint32_t    data_format;
    vip_uint32_t    data_type;
    vip_uint32_t    quan_format;
    vip_int32_t     fixed_pos;
    vip_float_t     tf_scale;
    vip_int32_t     tf_zerop;
    vip_char_t      name[MAX_IO_NAME_LEGTH];
} gcvip_bin_inout_entry_t;

typedef struct _gcvip_bin_layer
{
    vip_char_t      name[LAYER_NAME_SIZE];
    vip_uint32_t    id;
    vip_uint32_t    operation_count;
    vip_uint32_t    uid;
} gcvip_bin_layer_t;

typedef struct _gcvip_bin_operation
{
    vip_uint32_t    type;
    vip_uint32_t    index;
    vip_uint32_t    layer_id;
    vip_uint32_t    state_id;       /* States buffer index in LCDT. */
    vip_uint32_t    patch_index;
    vip_uint32_t    patch_count;
} gcvip_bin_operation_t;

typedef struct _gcvip_bin_nn_operation
{
    vip_uint8_t    cmd[NN_CMD_SIZE_128];
} gcvip_bin_nn_operation_t;

typedef struct _gcvip_bin_nn_operation_192bytes
{
    vip_uint8_t    cmd[NN_CMD_SIZE_192];
} gcvip_bin_nn_operation_192bytes_t;

typedef struct _gcvip_bin_tp_operation
{
    vip_uint8_t    cmd[TP_CMD_SIZE];
} gcvip_bin_tp_operation_t;

typedef struct _gcvip_bin_sh_operation
{
    vip_uint32_t    lcdt_index;
    vip_uint32_t    ppu_param_index;
} gcvip_bin_sh_operation_t;

typedef struct _gcvip_bin_patch_data_entry
{
    vip_uint32_t    type;
    vip_uint32_t    offset_in_states;
    vip_uint32_t    source_type;
    vip_int32_t     index;
    vip_int32_t     orig_base;
    vip_uint32_t    transformed;
    vip_uint32_t    name;
} gcvip_bin_patch_data_entry_t;

typedef struct _gcvip_bin_layer_parameter_entry
{
    vip_char_t      param_name[16];
    vip_uint32_t    dim_count;
    vip_uint32_t    dims[MAX_NUM_DIMS];
    vip_uint32_t    data_format;
    vip_uint32_t    data_type;
    vip_uint32_t    quant_format;
    vip_int32_t     fixpoint_zeropoint;
    vip_float_t     tfscale;
    vip_int32_t     index;
    vip_uint32_t    address_offset;
    vip_uint32_t    source_type;
} gcvip_bin_layer_parameter_t;

typedef struct _gcvip_bin_sw_operation_info_entry
{
    vip_uint32_t    sw_peration_type;
    vip_char_t      name[MAX_SW_LAYER_NAME_LENGTH];
} gcvip_bin_sw_operation_info_t;

typedef struct _gcvip_bin_hw_init_operation_info_entry
{
    vip_uint32_t    state_id;       /* States buffer index in LCDT. */
    vip_uint32_t    patch_index;    /* the first index in patch table */
    vip_uint32_t    patch_count;    /* the total patch count in patch table */
} gcvip_bin_hw_init_operation_info_entry_t;

typedef struct _gcvip_ppu_param_data
{
    vip_uint32_t    global_offset_x;
    vip_uint32_t    global_offset_y;
    vip_uint32_t    global_offset_z;
    vip_uint32_t    global_scale_x;
    vip_uint32_t    global_scale_y;
    vip_uint32_t    global_scale_z;
    vip_uint32_t    group_size_x;
    vip_uint32_t    group_size_y;
    vip_uint32_t    group_size_z;
    vip_uint32_t    group_count_x;
    vip_uint32_t    group_count_y;
    vip_uint32_t    group_count_z;
} gcvip_ppu_param_data_t;

#if defined(__cplusplus)
}
#endif
#endif
