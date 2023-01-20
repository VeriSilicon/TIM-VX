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

#ifndef _NBG_PARSER_IMPL_H
#define _NBG_PARSER_IMPL_H

#if defined(__cplusplus)
extern "C"{
#endif

#include "gc_vip_nbg_format.h"


/* NBG format version */
#define NBG_FORMAT_VERSION         0x00010014

enum nbg_nn_command_size_e
{
    NBG_NN_COMMAND_SIZE_128 = 0,
    NBG_NN_COMMAND_SIZE_192 = 1,
};

typedef struct _nbg_reader
{
    vip_uint32_t    offset;
    vip_uint32_t    total_size;
    vip_uint8_t     *data;
    vip_uint8_t     *current_data;
} nbg_reader_t;

typedef struct _nbg_parser_data
{
    /* Fixed part of the bin. */
    gcvip_bin_fixed_t              fixed;

    /* Dynamic data part of the bin. */
    gcvip_bin_inout_entry_t        *inputs;
    gcvip_bin_inout_entry_t        *outputs;
    gcvip_bin_layer_t              *orig_layers; /* original layers info, loading from binary graph */
    gcvip_bin_operation_t          *operations;
    gcvip_bin_entry_t              *LCDT;
    gcvip_bin_sh_operation_t       *sh_ops;
    void                           *nn_ops;
    gcvip_bin_tp_operation_t       *tp_ops;
    gcvip_bin_patch_data_entry_t   *pd_entries;
    gcvip_bin_hw_init_operation_info_entry_t *hw_init_ops;
    gcvip_bin_entry_t              *ICDT;
    void                           *LCD;

    vip_uint32_t                    n_inputs;
    vip_uint32_t                    n_outputs;
    vip_uint32_t                    n_orig_layers; /* the number of original layers */
    vip_uint32_t                    n_operations;
    vip_uint32_t                    n_LCDT;
    vip_uint32_t                    n_nn_ops;
    vip_uint32_t                    n_tp_ops;
    vip_uint32_t                    n_sh_ops;
    vip_uint32_t                    n_pd_entries;

    vip_uint32_t                    n_hw_init_ops;
    vip_uint32_t                    n_ICDT;

    nbg_reader_t                    reader;
} nbg_parser_data_t;

#if defined(__cplusplus)
}
#endif

#endif
