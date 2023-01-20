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

#include "tim/utils/nbg_parser/nbg_parser.h"
#include "tim/utils/nbg_parser/nbg_parser_impl.h"
#include "tim/utils/nbg_parser/nbg_parser_version.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Some functions, depends on runtime system, need to be modified on RTOS or DSP */

#define nbg_printf   printf

#define goOnError(func)  \
    status = func; \
    goto onError;

static const vip_char_t* dummy_name = "dummy_name";

static void* nbg_malloc(vip_uint32_t size)
{
    return malloc(size);
}

static nbg_status_e nbg_free(void *data)
{
    free(data);
    return NBG_SUCCESS;
}

static void* nbg_memcpy(void *dst, const void *src, vip_uint32_t size)
{
    return memcpy(dst, src, size);
}

static void* nbg_memset(void *dst, vip_uint32_t size)
{
    return memset(dst, 0, size);
}

static nbg_uint32_t nbg_strlen(const nbg_char_t *str)
{
    return strlen(str);
}

static nbg_char_t* nbg_strcpy(nbg_char_t *dst, const nbg_char_t *src)
{
    strcpy(dst, src);
    return dst;
}

/*********************** NBG parser internal functions ***********/

#define OLD_NBG_FORMAT_DIMS_NUM     4
static vip_int8_t read_byte(nbg_reader_t *reader)
{
    vip_int8_t data = 0;
    if (reader->offset <= reader->total_size) {
        data = *((vip_int8_t *)reader->current_data);

        reader->offset += sizeof(vip_int8_t);
        reader->current_data += sizeof(vip_int8_t);
    }
    else {
        nbg_printf("fail to read nbg data, out of buffer, offset=%d, total size=%d\n",
                    reader->offset, reader->total_size);
    }

    return data;
}

static vip_uint32_t read_uInt(nbg_reader_t *reader)
{
    vip_uint32_t data = 0;

    if (reader->offset <= reader->total_size) {
        data = *((vip_uint32_t *)reader->current_data);

        reader->offset += sizeof(vip_uint32_t);
        reader->current_data += sizeof(vip_uint32_t);
    }
    else {
        nbg_printf("fail to read nbg data, out of buffer, offset=%d, total size=%d\n",
                    reader->offset, reader->total_size);
    }

    return data;
}

static nbg_status_e read_data(nbg_reader_t *reader, void *dst, vip_uint32_t size)
{
    nbg_status_e status = NBG_SUCCESS;

    if (reader->offset <= reader->total_size) {
        nbg_memcpy(dst, reader->current_data, size);

        reader->offset += size;
        reader->current_data += size;
    }
    else {
        nbg_printf("fail to read nbg data, out of buffer, offset=%d, total size=%d\n",
                    reader->offset, reader->total_size);
        status = NBG_ERROR_FAILURE;
    }

    return status;
}

static nbg_status_e reader_locate(
    nbg_reader_t *reader,
    vip_uint32_t location
    )
{
    nbg_status_e status = NBG_SUCCESS;

    if (location > reader->total_size) {
        nbg_printf("failed to locate nbg buffer, Location > reader->size\n");
        status = NBG_ERROR_INVALID_ARGUMENTS;
    }
    else {
        reader->offset = location;
        reader->current_data = reader->data + reader->offset;
    }

    return status;
}

static nbg_status_e read_bin_header(nbg_reader_t *reader, gcvip_bin_header_t *header)
{
    nbg_status_e status = NBG_SUCCESS;
    vip_uint32_t i = 0;
    static const vip_char_t magic[4] = {'V', 'P', 'M', 'N'};

    /* Read magic. */
    for (i = 0; i < 4; i++) {
        header->magic[i] = read_byte(reader);
        if (header->magic[i] != magic[i]) {
            nbg_printf("binary magic not match\n");
            goOnError(NBG_ERROR_FAILURE);
        }
    }

    /* Read "Version". */
    header->version = read_uInt(reader);

    if (header->version > NBG_FORMAT_VERSION) {
        nbg_printf("NBG parser is forward compatible with NBG format, "
                   "please update NBG parser in time\n");
        goOnError(NBG_ERROR_FAILURE);
    }

    /* Read "hardware target". */
    header->hw_target = read_uInt(reader);

    /* Read "network_name". */
    read_data(reader, header->network_name, sizeof(header->network_name));

    /* Read "Layer_count". */
    header->layer_count = read_uInt(reader);

    /* Read "operation_count". */
    header->operation_count = read_uInt(reader);

    /* Read "input_count". */
    header->input_count = read_uInt(reader);

    /* Read "output_count". */
    header->output_count = read_uInt(reader);

    if (header->version >= 0x00010003) {
        read_data(reader, &header->feature_db, sizeof(gcvip_bin_feature_database_t));
    }

onError:
    return status;
}

static nbg_status_e  read_bin_entry(nbg_reader_t *reader, gcvip_bin_entry_t *entry
    )
{
    nbg_status_e status = NBG_SUCCESS;

    entry->offset = read_uInt(reader);
    entry->size = read_uInt(reader);

    return status;
}

static nbg_status_e read_nbg_dyn_data(nbg_parser_data_t *nbg)
{
    nbg_status_e status = NBG_SUCCESS;
    nbg_reader_t *reader = &nbg->reader;

    /* read input data */
    if (nbg->fixed.input_table.size > 0) {
        nbg->inputs = (gcvip_bin_inout_entry_t*)nbg_malloc(nbg->fixed.input_table.size);
        if (nbg->inputs == NBG_NULL) {
            nbg_printf("failed to malloc memory for inputs\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->inputs, nbg->fixed.input_table.size);

        reader_locate(reader, nbg->fixed.input_table.offset);
        read_data(reader, nbg->inputs, nbg->fixed.input_table.size);

        if (nbg->fixed.header.version >= 0x0001000B) {
            nbg->n_inputs = nbg->fixed.input_table.size / sizeof(gcvip_bin_inout_entry_t);
        }
        else if ((nbg->fixed.header.version >= 0x00010004) &&
                 (nbg->fixed.header.version < 0x0001000B)) {
            nbg_uint32_t size = 0;
            size = sizeof(gcvip_bin_inout_entry_t) - (MAX_NUM_DIMS - OLD_NBG_FORMAT_DIMS_NUM) * sizeof(nbg_uint32_t);
            nbg->n_inputs = nbg->fixed.input_table.size / size;
        }
        else {
            nbg_uint32_t size = 0;
            size = (sizeof(gcvip_bin_inout_entry_t) - sizeof(vip_char_t) * MAX_IO_NAME_LEGTH -
                                (MAX_NUM_DIMS - OLD_NBG_FORMAT_DIMS_NUM) * sizeof(nbg_uint32_t));
            nbg->n_inputs = nbg->fixed.input_table.size / size;
        }
    }

    /* read output data */
    if (nbg->fixed.output_table.size > 0) {
        nbg->outputs = (gcvip_bin_inout_entry_t*)nbg_malloc(nbg->fixed.output_table.size);
        if (nbg->outputs == NBG_NULL) {
            nbg_printf("failed to malloc memory for outputs\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->outputs, nbg->fixed.output_table.size);

        reader_locate(reader, nbg->fixed.output_table.offset);
        read_data(reader, nbg->outputs, nbg->fixed.output_table.size);

        if (nbg->fixed.header.version >= 0x0001000B) {
            nbg->n_outputs = nbg->fixed.output_table.size / sizeof(gcvip_bin_inout_entry_t);
        }
        else if ((nbg->fixed.header.version >= 0x00010004) &&
                 (nbg->fixed.header.version < 0x0001000B)) {
            nbg_uint32_t size = 0;
            size = sizeof(gcvip_bin_inout_entry_t) - (MAX_NUM_DIMS - OLD_NBG_FORMAT_DIMS_NUM) * sizeof(nbg_uint32_t);
            nbg->n_outputs = nbg->fixed.output_table.size / size;
        }
        else {
            nbg_uint32_t size = 0;
            size = sizeof(gcvip_bin_inout_entry_t) - sizeof(vip_char_t) * MAX_IO_NAME_LEGTH -
                                (MAX_NUM_DIMS - OLD_NBG_FORMAT_DIMS_NUM) * sizeof(nbg_uint32_t);
            nbg->n_outputs = nbg->fixed.output_table.size / size;
        }
    }

    /* read layer data */
    if (nbg->fixed.layer_table.size > 0) {
        nbg->orig_layers = (gcvip_bin_layer_t*)nbg_malloc(nbg->fixed.layer_table.size);
        if (nbg->orig_layers == NBG_NULL) {
            nbg_printf("failed to malloc memory for layer data\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->orig_layers, nbg->fixed.layer_table.size);

        reader_locate(reader, nbg->fixed.layer_table.offset);
        read_data(reader, nbg->orig_layers, nbg->fixed.layer_table.size);

        if (nbg->fixed.header.version >= 0x00010008) {
            nbg->n_orig_layers = nbg->fixed.layer_table.size / sizeof(gcvip_bin_layer_t);
        }
        else {
            nbg->n_orig_layers = nbg->fixed.layer_table.size / (sizeof(gcvip_bin_layer_t) - sizeof(vip_uint32_t));
        }
    }

    /* read operation data */
    if (nbg->fixed.opeartion_table.size > 0) {
        nbg->operations = (gcvip_bin_operation_t*)nbg_malloc(nbg->fixed.opeartion_table.size);
        if (nbg->operations == NBG_NULL) {
            nbg_printf("failed to malloc memory for operation data\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->operations, nbg->fixed.opeartion_table.size);

        reader_locate(reader, nbg->fixed.opeartion_table.offset);
        read_data(reader, nbg->operations, nbg->fixed.opeartion_table.size);

        nbg->n_operations = nbg->fixed.opeartion_table.size / sizeof(gcvip_bin_operation_t);
    }

    /* read nn operation */
    if (nbg->fixed.nn_op_data_table.size > 0) {
        nbg->nn_ops = (void*)nbg_malloc(nbg->fixed.nn_op_data_table.size);
        if (nbg->nn_ops == NBG_NULL) {
            nbg_printf("failed to malloc memory for nn operation data\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->nn_ops, nbg->fixed.nn_op_data_table.size);

        reader_locate(reader, nbg->fixed.nn_op_data_table.offset);
        read_data(reader, nbg->nn_ops, nbg->fixed.nn_op_data_table.size);
        if (NBG_NN_COMMAND_SIZE_192 == nbg->fixed.header.feature_db.nn_command_size) {
            nbg->n_nn_ops = nbg->fixed.nn_op_data_table.size / sizeof(gcvip_bin_nn_operation_192bytes_t);
        }
        else {
            nbg->n_nn_ops = nbg->fixed.nn_op_data_table.size / sizeof(gcvip_bin_nn_operation_t);
        }
    }

    /* read TP opeartion */
    if (nbg->fixed.tp_op_data_table.size > 0) {
        nbg->tp_ops = (void*)nbg_malloc(nbg->fixed.tp_op_data_table.size);
        if (nbg->tp_ops == NBG_NULL) {
            nbg_printf("failed to malloc memory for tp operation data\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->tp_ops, nbg->fixed.tp_op_data_table.size);

        reader_locate(reader, nbg->fixed.tp_op_data_table.offset);
        read_data(reader, nbg->tp_ops, nbg->fixed.tp_op_data_table.size);
        nbg->n_tp_ops = nbg->fixed.tp_op_data_table.size / sizeof(gcvip_bin_tp_operation_t);
    }

    /* read shader opeartion */
    if (nbg->fixed.sh_op_data_table.size > 0) {
        nbg->sh_ops = (gcvip_bin_sh_operation_t*)nbg_malloc(nbg->fixed.sh_op_data_table.size);
        if (nbg->sh_ops == NBG_NULL) {
            nbg_printf("failed to malloc memory for shader operation data\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->sh_ops, nbg->fixed.sh_op_data_table.size);

        reader_locate(reader, nbg->fixed.sh_op_data_table.offset);
        read_data(reader, nbg->sh_ops, nbg->fixed.sh_op_data_table.size);
        if (nbg->fixed.header.version >= 0x0001000E) {
            nbg->n_sh_ops = nbg->fixed.sh_op_data_table.size / sizeof(gcvip_bin_sh_operation_t);
        }
        else {
            nbg->n_sh_ops = nbg->fixed.sh_op_data_table.size /
                            (sizeof(gcvip_bin_sh_operation_t) - sizeof(vip_uint32_t));
        }
    }

    /* read patch data */
    if (nbg->fixed.patch_data_table.size > 0) {
        nbg->pd_entries = (gcvip_bin_patch_data_entry_t*)nbg_malloc(nbg->fixed.patch_data_table.size);
        if (nbg->pd_entries == NBG_NULL) {
            nbg_printf("failed to malloc memory for patch data data\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->pd_entries, nbg->fixed.patch_data_table.size);

        reader_locate(reader, nbg->fixed.patch_data_table.offset);
        read_data(reader, nbg->pd_entries, nbg->fixed.patch_data_table.size);

        nbg->n_pd_entries = nbg->fixed.patch_data_table.size / sizeof(gcvip_bin_patch_data_entry_t);
    }

    /* read lcd table */
    if (nbg->fixed.LCD_table.size > 0) {
        nbg->LCDT = (gcvip_bin_entry_t*)nbg_malloc(nbg->fixed.LCD_table.size);
        if (nbg->LCDT == NBG_NULL) {
            nbg_printf("failed to malloc memory for lcd table data\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->LCDT, nbg->fixed.LCD_table.size);

        reader_locate(reader, nbg->fixed.LCD_table.offset);
        read_data(reader, nbg->LCDT, nbg->fixed.LCD_table.size);

        nbg->n_LCDT = nbg->fixed.LCD_table.size / sizeof(gcvip_bin_entry_t);
    }

    /* read hw init operation table */
    if (nbg->fixed.hw_init_op_table.size > 0) {
        nbg->hw_init_ops = (gcvip_bin_hw_init_operation_info_entry_t*)nbg_malloc(nbg->fixed.hw_init_op_table.size);
        if (nbg->hw_init_ops == NBG_NULL) {
            nbg_printf("failed to malloc memory for hardware initialize operations\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->hw_init_ops, nbg->fixed.hw_init_op_table.size);

        reader_locate(reader, nbg->fixed.hw_init_op_table.offset);
        read_data(reader, nbg->hw_init_ops, nbg->fixed.hw_init_op_table.size);

        nbg->n_hw_init_ops = nbg->fixed.hw_init_op_table.size / sizeof(gcvip_bin_hw_init_operation_info_entry_t);
    }

    /* read ICD table */
    if (nbg->fixed.ICD_table.size > 0) {
        nbg->ICDT = (gcvip_bin_entry_t*)nbg_malloc(nbg->fixed.ICD_table.size);
        if (nbg->ICDT == NBG_NULL) {
            nbg_printf("failed to malloc memory for ICDT\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->ICDT, nbg->fixed.ICD_table.size);

        reader_locate(reader, nbg->fixed.ICD_table.offset);
        read_data(reader, nbg->ICDT, nbg->fixed.ICD_table.size);

        nbg->n_ICDT = nbg->fixed.ICD_table.size / sizeof(gcvip_bin_entry_t);
    }

    /* read LCD */
    if (nbg->fixed.LCD.size > 0) {
        nbg->LCD = (void*)nbg_malloc(nbg->fixed.LCD.size);
        if (nbg->LCD == NBG_NULL) {
            nbg_printf("failed to malloc memory for LCD operation data\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        nbg_memset(nbg->LCD, nbg->fixed.LCD.size);

        reader_locate(reader, nbg->fixed.LCD.offset);
        read_data(reader, nbg->LCD, nbg->fixed.LCD.size);
    }

    return status;

onError:
    if (nbg->inputs != NBG_NULL) {
        nbg_free(nbg->inputs);
        nbg->inputs = NBG_NULL;
    }
    if (nbg->outputs != NBG_NULL) {
        nbg_free(nbg->outputs);
        nbg->outputs = NBG_NULL;
    }
    if (nbg->orig_layers != NBG_NULL) {
        nbg_free(nbg->orig_layers);
        nbg->orig_layers = NBG_NULL;
    }
    if (nbg->operations != NBG_NULL) {
        nbg_free(nbg->operations);
        nbg->operations = NBG_NULL;
    }
    if (nbg->LCDT != NBG_NULL) {
        nbg_free(nbg->LCDT);
        nbg->LCDT = NBG_NULL;
    }
    if (nbg->pd_entries != NBG_NULL) {
        nbg_free(nbg->pd_entries);
        nbg->pd_entries = NBG_NULL;
    }
    if (nbg->sh_ops != NBG_NULL) {
        nbg_free(nbg->sh_ops);
        nbg->sh_ops = NBG_NULL;
    }
    if (nbg->hw_init_ops != NBG_NULL) {
        nbg_free(nbg->hw_init_ops);
        nbg->hw_init_ops = NBG_NULL;
    }
    if (nbg->ICDT != NBG_NULL) {
        nbg_free(nbg->ICDT);
        nbg->ICDT = NBG_NULL;
    }
    if (nbg->nn_ops != NBG_NULL) {
        nbg_free(nbg->nn_ops);
        nbg->nn_ops = NBG_NULL;
    }
    if (nbg->tp_ops != NBG_NULL) {
        nbg_free(nbg->tp_ops);
        nbg->tp_ops = NBG_NULL;
    }
    if (nbg->LCD != NBG_NULL) {
        nbg_free(nbg->LCD);
        nbg->LCD = NBG_NULL;
    }

    return status;
}

static nbg_status_e read_nbg_fix_data(nbg_parser_data_t *nbg)
{
    nbg_status_e status = NBG_SUCCESS;
    nbg_reader_t *reader = &nbg->reader;

    status = read_bin_header(reader, &nbg->fixed.header);
    if (status != NBG_SUCCESS) {
        nbg_printf("fail to read NBG header\n");
        goOnError(status);
    }

    /* read memory pool */
    nbg->fixed.pool.size = read_uInt(reader);
    nbg->fixed.pool.alignment = read_uInt(reader);
    nbg->fixed.pool.base = read_uInt(reader);

    /* read sram */
    nbg->fixed.axi_sram_base = read_uInt(reader);
    nbg->fixed.axi_sram_size = read_uInt(reader);
    if (nbg->fixed.header.version >= 0x00010008) {
        nbg->fixed.vip_sram_base = read_uInt(reader);
        nbg->fixed.vip_sram_size = read_uInt(reader);
    }

    /* read entry */
    read_bin_entry(reader, &nbg->fixed.input_table);
    read_bin_entry(reader, &nbg->fixed.output_table);
    read_bin_entry(reader, &nbg->fixed.layer_table);
    read_bin_entry(reader, &nbg->fixed.opeartion_table);
    read_bin_entry(reader, &nbg->fixed.LCD_table);
    read_bin_entry(reader, &nbg->fixed.LCD);
    read_bin_entry(reader, &nbg->fixed.nn_op_data_table);
    read_bin_entry(reader, &nbg->fixed.tp_op_data_table);
    read_bin_entry(reader, &nbg->fixed.sh_op_data_table);
    read_bin_entry(reader, &nbg->fixed.patch_data_table);
    read_bin_entry(reader, &nbg->fixed.layer_param_table);
    read_bin_entry(reader, &nbg->fixed.sw_op_data_table);
    if (nbg->fixed.header.version >= 0x0001000C) {
        read_bin_entry(reader, &nbg->fixed.hw_init_op_table);
        read_bin_entry(reader, &nbg->fixed.ICD_table);
        read_bin_entry(reader, &nbg->fixed.ICD);
    }
    else {
        nbg->fixed.hw_init_op_table.size = 0;
        nbg->fixed.ICD_table.size = 0;
        nbg->fixed.ICD.size = 0;
    }

    if (nbg->fixed.header.version >= 0x0001000E) {
        read_bin_entry(reader, &nbg->fixed.ppu_param_table);
    }
    else {
        nbg->fixed.ppu_param_table.size = 0;
    }

onError:
    return status;
}

static void *get_io_ptr_by_index(
    nbg_parser_data_t *nbg,
    gcvip_bin_inout_entry_t *io_ptr,
    vip_int32_t index
)
{
    void *ptr_io = NBG_NULL;
    vip_uint32_t size_04 = 0, size_0B = 0;

    size_04 = sizeof(gcvip_bin_inout_entry_t) - sizeof(vip_int8_t) * MAX_IO_NAME_LEGTH -
              sizeof (vip_uint32_t) * (MAX_NUM_DIMS - OLD_NBG_FORMAT_DIMS_NUM);
    size_0B = sizeof(gcvip_bin_inout_entry_t) - sizeof (vip_uint32_t) * (MAX_NUM_DIMS - OLD_NBG_FORMAT_DIMS_NUM);

    if (nbg->fixed.header.version >= 0x0001000B) {
        ptr_io = (void *)(io_ptr + index);
    }
    else if ((nbg->fixed.header.version >= 0x00010004) && (nbg->fixed.header.version < 0x0001000B)) {
         ptr_io = (void *)((vip_int8_t *)io_ptr + index * size_0B);
    }
    else {
        ptr_io = (void *)((vip_int8_t *)io_ptr + index * size_04);
    }

    return ptr_io;
}

static nbg_status_e query_input_output(
    nbg_parser_data nbg,
    void *entry,
    vip_uint32_t property,
    void *value,
    nbg_uint32_t size
    )
{
    nbg_status_e status = NBG_SUCCESS;
    nbg_parser_data_t *nbg_data = (nbg_parser_data_t*)nbg;
    vip_uint32_t *ptr_u32 = (vip_uint32_t *)value;
    vip_int32_t *ptr_32 = (vip_int32_t *)value;
    vip_float_t *ptr_fl = (vip_float_t *)value;
    vip_char_t *ptr_char = (vip_char_t *)value;
    vip_uint32_t i = 0;

    if (NBG_NULL == entry) {
        nbg_printf("buffer entry is NULL\n");
        goOnError(NBG_ERROR_FAILURE);
    }

    if (nbg_data->fixed.header.version >= 0x0001000B) {
        gcvip_bin_inout_entry_t *inout_entry = (gcvip_bin_inout_entry_t*)entry;
        switch (property) {
        case NBG_PARSER_BUFFER_PROP_QUANT_FORMAT:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = inout_entry->quan_format;
            }
            else {
                nbg_printf("failed to query quan format, value is a uint32 buffer and size is 4byte\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_NUM_OF_DIMENSION:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = inout_entry->dim_count;
            }
            else {
                nbg_printf("failed to query dims count, value is a uint32 buffer and size is 4byte\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_DIMENSIONS:
            if (size >= (sizeof(vip_uint32_t) * inout_entry->dim_count)) {
                for (i = 0; i < inout_entry->dim_count; i++) {
                    ptr_u32[i] = inout_entry->dim_size[i];
                }
            }
            else {
                nbg_printf("failed to query dims, value is 4 uint32 buffer and size is 16byte\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_DATA_FORMAT:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = inout_entry->data_format;
            }
            else {
                nbg_printf("failed to query data format, value is a uint32 buffer and size is 4byte\n\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_DATA_TYPE:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = inout_entry->data_type;
            }
            else {
                nbg_printf("failed to query data type, value is a uint32 buffer and size is 4byte\n\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_FIXED_POINT_POS:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = inout_entry->fixed_pos;
            }
            else {
                nbg_printf("failed to query fixed point pos, value is a uint32 buffer and size is 4byte\n\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_SCALE:
            if (size >= sizeof(vip_float_t)) {
                *ptr_fl = inout_entry->tf_scale;
            }
            else {
                nbg_printf("failed to query scale, value is a float buffer and size is 4byte\n\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_ZERO_POINT:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_32 = inout_entry->tf_zerop;
            }
            else {
                nbg_printf("failed to query zero point, value is a uint32 buffer and size is 4byte\n\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_NAME:
        {
            nbg_uint32_t len = nbg_strlen(inout_entry->name) + 1;
            if (size >= len) {
                nbg_strcpy(ptr_char, inout_entry->name);
            }
            else {
                nbg_printf("failed to query name, value size is %d\n\n", len);
                goOnError(NBG_ERROR_FAILURE);
            }
        }
        break;

        case NBG_PARSER_BUFFER_PROP_NAME_SIZE:
        {
            nbg_uint32_t len = nbg_strlen(inout_entry->name) + 1;
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = len;
            }
            else {
                nbg_printf("failed to query name size, value size if uint32 dat type\n");
                goOnError(NBG_ERROR_FAILURE);
            }
        }
        break;

        default:
            nbg_printf("NBG doesn't support this property=%d\n", property);
            status = NBG_ERROR_INVALID_ARGUMENTS;
            break;
        }
    }
    else {
        typedef struct _gcvip_bin_inout_entry_old
        {
            vip_uint32_t    dim_count;
            vip_uint32_t    dim_size[4];
            vip_uint32_t    data_format;
            vip_uint32_t    data_type;
            vip_uint32_t    quan_format;
            vip_uint32_t    fixed_pos;
            vip_float_t     tf_scale;
            vip_uint32_t    tf_zerop;
            vip_char_t      name[MAX_IO_NAME_LEGTH];
        } gcvip_bin_inout_entry_old_t;

        gcvip_bin_inout_entry_old_t *inout_entry = (gcvip_bin_inout_entry_old_t*)entry;

        switch (property) {
        case NBG_PARSER_BUFFER_PROP_QUANT_FORMAT:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = inout_entry->quan_format;
            }
            else {
                nbg_printf("failed to query quan format, value is a uint32 buffer and size is 4byte\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_NUM_OF_DIMENSION:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = inout_entry->dim_count;
            }
            else {
                nbg_printf("failed to query dims count, value is a uint32 buffer and size is 4byte\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_DIMENSIONS:
            if (size >= (sizeof(vip_uint32_t) * inout_entry->dim_count)) {
                for (i = 0; i < inout_entry->dim_count; i++) {
                    ptr_u32[i] = inout_entry->dim_size[i];
                }
            }
            else {
                nbg_printf("failed to query dims, value is 4 uint32 buffer and size is 16byte\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_DATA_FORMAT:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = inout_entry->data_format;
            }
            else {
                nbg_printf("failed to query data format, value is a uint32 buffer and size is 4byte\n\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_DATA_TYPE:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = inout_entry->data_type;
            }
            else {
                nbg_printf("failed to query data type, value is a uint32 buffer and size is 4byte\n\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_FIXED_POINT_POS:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_u32 = inout_entry->fixed_pos;
            }
            else {
                nbg_printf("failed to query fixed point pos, value is a uint32 buffer and size is 4byte\n\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_SCALE:
            if (size >= sizeof(vip_float_t)) {
                *ptr_fl = inout_entry->tf_scale;
            }
            else {
                nbg_printf("failed to query scale, value is a float buffer and size is 4byte\n\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_ZERO_POINT:
            if (size >= sizeof(vip_uint32_t)) {
                *ptr_32 = inout_entry->tf_zerop;
            }
            else {
                nbg_printf("failed to query zero point, value is a uint32 buffer and size is 4byte\n\n");
                goOnError(NBG_ERROR_FAILURE);
            }
            break;

        case NBG_PARSER_BUFFER_PROP_NAME:
        {
            nbg_uint32_t len = nbg_strlen(inout_entry->name) + 1;
            if (nbg_data->fixed.header.version >= 0x00010004) {
                if (size >= len) {
                    nbg_strcpy(ptr_char, inout_entry->name);
                }
                else {
                    nbg_printf("failed to query name, value size is %d\n\n", len);
                    goOnError(NBG_ERROR_FAILURE);
                }
            }
            else {
                nbg_strcpy(ptr_char, dummy_name);
            }
        }
        break;

        case NBG_PARSER_BUFFER_PROP_NAME_SIZE:
        {
            if (nbg_data->fixed.header.version < 0x00010004) {
                *ptr_u32 = nbg_strlen(dummy_name) + 1;
            }
            else {
                nbg_uint32_t len = nbg_strlen(inout_entry->name) + 1;
                if (size >= sizeof(vip_uint32_t)) {
                    *ptr_u32 = len;
                }
                else {
                    nbg_printf("failed to query name size, value size if uint32 dat type\n");
                    goOnError(NBG_ERROR_FAILURE);
                }
            }
        }
        break;

        default:
            nbg_printf("NBG doesn't support this property=%d\n", property);
            status = NBG_ERROR_INVALID_ARGUMENTS;
            break;
        }
    }

onError:
    return status;
}

/*********************** expose APIs ****************************/

/*
@brief, Query NBG parser library version
*/
nbg_uint32_t nbg_parser_version(void)
{
    nbg_uint32_t version = (VERSION_MAJOR << 16) | (VERSION_MINOR << 8) | (VERSION_SUB_MINOR);

    return version;
}

/*
@brief, Initialize NBG parser.
@param, buffer. a pointer to the start of the NBG data
@param size, the size of NBG data.
@param, nbg_t*, the nbg object created by NBG data.
*/
nbg_status_e nbg_parser_init(void *buffer, nbg_uint32_t size, nbg_parser_data *nbg)
{
    nbg_parser_data_t *nbg_data = NBG_NULL;
    nbg_status_e status = NBG_SUCCESS;

    nbg_data = (nbg_parser_data_t *)nbg_malloc(sizeof(nbg_parser_data_t));
    if (nbg_data != NBG_NULL) {
        nbg_memset(nbg_data, sizeof(nbg_parser_data_t));
        nbg_data->reader.current_data = (vip_uint8_t*)buffer;
        nbg_data->reader.data = (vip_uint8_t*)buffer;
        nbg_data->reader.total_size = size;
        nbg_data->reader.offset = 0;

        status  = read_nbg_fix_data(nbg_data);
        if (status != NBG_SUCCESS) {
            nbg_printf("failed to read fix section data\n");
            goOnError(status);
        }

        status = read_nbg_dyn_data(nbg_data);
        if (status != NBG_SUCCESS) {
            nbg_printf("failed to read dynmic section data\n");
            goOnError(status);
        }
    }
    else {
        nbg_printf("failed to malloc memory for nbg object\n");
        goOnError(status);
    }

    if (nbg != NBG_NULL) {
        *nbg = (nbg_parser_data)nbg_data;
    }

onError:
    return status;
}

/*
@brief, query the input info of network.
@param nbg, The nbg object created by NBG data.
@param index, The index of input.
@param property, The property of input.
       Quant format, dimension count, shape, data format and so on.
       see nbg_buffer_property_e enumeration.
@param size, the size of value buffer.
@param, return value.
*/
nbg_status_e nbg_parser_query_input(
    nbg_parser_data nbg,
    vip_uint32_t index,
    vip_uint32_t property,
    void *value,
    nbg_uint32_t size
    )
{
    nbg_status_e status = NBG_SUCCESS;
    gcvip_bin_inout_entry_t *input;
    nbg_parser_data_t *nbg_data = (nbg_parser_data_t*)nbg;

    if ((nbg_data == NBG_NULL) || (value == NBG_NULL) || (0 == size)) {
        nbg_printf("failed to query input, parameter is NULL, nbg=%p, "
                   "index=%d, property=%d, value=%p, size=%d\n",
                    nbg, index, property, value, size);
        return NBG_ERROR_FAILURE;
    }

    input = (gcvip_bin_inout_entry_t *)get_io_ptr_by_index(nbg_data, nbg_data->inputs, index);

    status = query_input_output(nbg, input, property, value, size);
    if (status != NBG_SUCCESS) {
        nbg_printf("failed to query input buffer information\n");
    }

    return status;
}

/*
@brief, query the output info of network.
@param nbg, The nbg object created by NBG data.
@param index, The nbg object created by NBG data.
@param property, The property of input.
       Quant format, dimension count, shape, data format and so on.
       see nbg_buffer_property_e enumeration.
@param size, The size of value buffer.
@param, return value.
*/
nbg_status_e nbg_parser_query_output(
    nbg_parser_data nbg,
    vip_uint32_t index,
    vip_uint32_t property,
    void *value,
    nbg_uint32_t size
    )
{
    nbg_status_e status = NBG_SUCCESS;
    gcvip_bin_inout_entry_t *output;
    nbg_parser_data_t *nbg_data = (nbg_parser_data_t*)nbg;

    if ((nbg_data == NBG_NULL) || (value == NBG_NULL) || (0 == size)) {
        nbg_printf("failed to query output, parameter is NULL, nbg=%p, index=%d, "
                   "property=%d, value=%p, size=%d\n",
                    nbg, index, property, value, size);
        return NBG_ERROR_FAILURE;
    }

    output = (gcvip_bin_inout_entry_t *)get_io_ptr_by_index(nbg_data, nbg_data->outputs, index);

    status = query_input_output(nbg, output, property, value, size);
    if (status != NBG_SUCCESS) {
        nbg_printf("failed to query input buffer information\n");
    }

    return status;
}

/*
@brief, query the network info.
@param property, The property of the network is queried.
       network name, input count, output count and so on.
       see nbg_network_propery_e enumeration.
@param size, the size of value buffer.
@param value, return value.
*/
nbg_status_e nbg_parser_query_network(
    nbg_parser_data nbg,
    vip_uint32_t property,
    void *value,
    nbg_uint32_t size
    )
{
    nbg_status_e status = NBG_SUCCESS;
    nbg_parser_data_t *nbg_data = (nbg_parser_data_t*)nbg;

    if ((nbg_data == NBG_NULL) || (value == NBG_NULL) || (0 == size)) {
        nbg_printf("failed to query network, parameter is NULL, nbg=%p, property=%d, "
                    "value=%p, size=%d\n",
                    nbg,  property, value, size);
        return NBG_ERROR_FAILURE;
    }

    switch (property) {
    case NBG_PARSER_NETWORK_INPUT_COUNT:
        if (size >= sizeof(vip_uint32_t)) {
            *((vip_uint32_t *)value) = nbg_data->fixed.header.input_count;
        }
        else {
            nbg_printf("failed to query network input count, value is a uint32 buffer "
                       "and size is 4byte\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        break;

    case NBG_PARSER_NETWORK_OUTPUT_COUNT:
        if (size >= sizeof(vip_uint32_t)) {
            *((vip_uint32_t *)value) = nbg_data->fixed.header.output_count;
        }
        else {
            nbg_printf("failed to query network output count, value is a uint32 buffer "
                       "and size is 4byte\n");
            goOnError(NBG_ERROR_FAILURE);
        }
        break;

    case NBG_PARSER_NETWORK_NAME:
    {
        nbg_uint32_t len = nbg_strlen(nbg_data->fixed.header.network_name) + 1;
        if (size >= len) {
            nbg_strcpy((nbg_char_t *)value, nbg_data->fixed.header.network_name);
        }
        else {
            nbg_printf("failed to query network name, the size of value buffer should be more than %dbyte\n",
                        len);
            goOnError(NBG_ERROR_FAILURE);
        }
    }
    break;

    case NBG_PARSER_NETWORK_NAME_SIZE:
    {
        nbg_uint32_t len = nbg_strlen(nbg_data->fixed.header.network_name) + 1;
        if (size >= sizeof(vip_uint32_t)) {
            *((vip_uint32_t *)value) = len;
        }
        else {
            nbg_printf("failed to query network name size, value is a uint32 buffer "
                       "and size is 4byte\n");
            goOnError(NBG_ERROR_FAILURE);
        }
    }
    break;

    case NBG_PARSER_NETWORK_CID:
    {
        *((vip_uint32_t *)value) = nbg_data->fixed.header.hw_target;
    }
    break;

    default:
        nbg_printf("not support this property=%d\n", property);
        status = NBG_ERROR_INVALID_ARGUMENTS;
        break;
    }

onError:
    return status;
}

/*
@brief, destroy nbg parser.
@param, the nbg object created by NBG data.
*/
nbg_status_e nbg_parser_destroy(
    nbg_parser_data nbg
    )
{
    nbg_status_e status = NBG_SUCCESS;
    nbg_parser_data_t *nbg_data = (nbg_parser_data_t*)nbg;

    if (nbg_data != NBG_NULL) {
        nbg_data->reader.total_size = 0;
        nbg_data->reader.offset = 0;

        if (nbg_data->inputs != NBG_NULL) {
            nbg_free(nbg_data->inputs);
            nbg_data->inputs = NBG_NULL;
        }
        if (nbg_data->outputs != NBG_NULL) {
            nbg_free(nbg_data->outputs);
            nbg_data->outputs = NBG_NULL;
        }
        if (nbg_data->orig_layers != NBG_NULL) {
            nbg_free(nbg_data->orig_layers);
            nbg_data->orig_layers = NBG_NULL;
        }
        if (nbg_data->operations != NBG_NULL) {
            nbg_free(nbg_data->operations);
            nbg_data->operations = NBG_NULL;
        }
        if (nbg_data->LCDT != NBG_NULL) {
            nbg_free(nbg_data->LCDT);
            nbg_data->LCDT = NBG_NULL;
        }
        if (nbg_data->pd_entries != NBG_NULL) {
            nbg_free(nbg_data->pd_entries);
            nbg_data->pd_entries = NBG_NULL;
        }
        if (nbg_data->sh_ops != NBG_NULL) {
            nbg_free(nbg_data->sh_ops);
            nbg_data->sh_ops = NBG_NULL;
        }
        if (nbg_data->hw_init_ops != NBG_NULL) {
            nbg_free(nbg_data->hw_init_ops);
            nbg_data->hw_init_ops = NBG_NULL;
        }
        if (nbg_data->ICDT != NBG_NULL) {
            nbg_free(nbg_data->ICDT);
            nbg_data->ICDT = NBG_NULL;
        }
        if (nbg_data->nn_ops != NBG_NULL) {
            nbg_free(nbg_data->nn_ops);
            nbg_data->nn_ops = NBG_NULL;
        }
        if (nbg_data->tp_ops != NBG_NULL) {
            nbg_free(nbg_data->tp_ops);
            nbg_data->tp_ops = NBG_NULL;
        }
        if (nbg_data->LCD != NBG_NULL) {
            nbg_free(nbg_data->LCD);
            nbg_data->LCD = NBG_NULL;
        }

        nbg_free(nbg);
    }

    return status;
}

