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
#include "utils/vsi_nn_constraint_check.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "vsi_nn_node.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"

typedef struct _node_io_signature_t {
    int count;
    vsi_nn_type_e types[1];
} node_io_signature_t;

static const char* _get_dtype_name(vsi_nn_type_e type)
{
    switch(type)
    {
        case D_NONE: return "Optional";
        case D_I8: return "INT8";
        case D_I16: return "INT16";
        case D_I32: return "INT32";
        case D_I64: return "INT64";
        case D_U8: return "UINT8";
        case D_U16: return "UINT16";
        case D_U32: return "UINT32";
        case D_U64: return "UINT64";
        case D_F16: return "FLOAT16";
        case D_F32: return "FLOAT32";
        case D_F64: return "FLOAT64";
        case D_BF16: return "BFLOAT16";
        case D_BOOL8: return "BOOL8";
        default:
            VSILOGE("Unknown data type: %d\n", type);
            break;
    }

    return NULL;
}

static const char* _get_qtype_name(vsi_nn_qnt_type_e type)
{
    switch(type)
    {
        case VSI_NN_QNT_TYPE_NONE: return "";
        case VSI_NN_QNT_TYPE_DFP: return "DFP";
        case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC: return "ASYM";
        case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC: return "SYMM PC";
        default:
            VSILOGE("Unknown quant type: %d\n", type);
            break;
    }

    return NULL;
}

static node_io_signature_t* _get_op_signature
    (
    vsi_nn_tensor_t** inputs,
    int inputs_num,
    vsi_nn_tensor_t** outputs,
    int outputs_num,
    const op_constraint_reg_type* op_constraint_reg
    )
{
    int i = 0;
    int reg_io_count = op_constraint_reg->reg_input_num +
            op_constraint_reg->reg_output_num;
    node_io_signature_t* item = NULL;

    if((inputs_num + outputs_num) > reg_io_count) {
        VSILOGW("Inputs/outputs count greater than registered inputs/outputs count: %d > %d",
                (inputs_num + outputs_num), reg_io_count);
    }

    item = malloc(sizeof(node_io_signature_t) + \
        (reg_io_count - 1) * sizeof(vsi_nn_type_e));
    item->count = inputs_num + outputs_num;
    memset(&item->types[0], 0x00, reg_io_count * sizeof(vsi_nn_type_e));

    inputs_num = vsi_nn_min(inputs_num, (int)op_constraint_reg->reg_input_num);
    for(i = 0; i < inputs_num; i++) {
        if(!inputs[i]) {
            item->types[i] = VSI_NN_TYPE_NONE \
                | VSI_NN_QNT_TYPE_NONE << Q_SHIFT;
            continue;
        }
        item->types[i] = inputs[i]->attr.dtype.vx_type \
            | inputs[i]->attr.dtype.qnt_type << Q_SHIFT;
    }

    outputs_num = vsi_nn_min(outputs_num, (int)op_constraint_reg->reg_output_num);
    for(i = 0; i < outputs_num; i++) {
        if(!outputs[i]) {
            item->types[op_constraint_reg->reg_input_num + i] = \
                    VSI_NN_TYPE_NONE | VSI_NN_QNT_TYPE_NONE << Q_SHIFT;
            continue;
        }
        item->types[op_constraint_reg->reg_input_num + i] = \
                outputs[i]->attr.dtype.vx_type |
                outputs[i]->attr.dtype.qnt_type << Q_SHIFT;
    }

    return item;
}

vsi_bool is_const_tensor
    (
    const vsi_nn_tensor_t* tensor
    )
{
    if(!tensor) {
        return FALSE;
    }

    return tensor->attr.is_const;
}

vsi_bool validate_op_io_types
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t** inputs,
    int inputs_num,
    vsi_nn_tensor_t** outputs,
    int outputs_num,
    const op_constraint_reg_type* op_constraint_reg,
    const char* name
    )
{
    vsi_bool matched = FALSE;

    if(self && self->attr.enable_op_constraint_check) {
        uint32_t i = 0;
        int32_t j = 0;
        int32_t reg_tensor_num = op_constraint_reg->reg_input_num + op_constraint_reg->reg_output_num;

        node_io_signature_t* sig = _get_op_signature(inputs, inputs_num,
                outputs, outputs_num, op_constraint_reg);

        VSILOGD("Validate [%s]", name);
        if(sig && op_constraint_reg && op_constraint_reg->types) {
            for(i = 0; i < op_constraint_reg->io_types_item_count; i++) {
                const uint8_t* curr = ((const uint8_t*)op_constraint_reg->types) \
                        + op_constraint_reg->io_types_item_size * i;
                vsi_nn_type_e *curr_type = (vsi_nn_type_e *)curr;

                for (j = 0; j < reg_tensor_num; j++)
                {
                    vsi_nn_type_e qnt_type = sig->types[j] >> Q_SHIFT;
                    vsi_nn_type_e data_type = sig->types[j] & ((1 << Q_SHIFT) - 1);
                    vsi_nn_type_e curr_qnt_type = curr_type[j] >> Q_SHIFT;
                    vsi_nn_type_e curr_data_type = curr_type[j] & ((1 << Q_SHIFT) - 1);
                   if ( (qnt_type != (vsi_nn_type_e)VSI_NN_QNT_TYPE_NONE && qnt_type != curr_qnt_type) ||
                       data_type != curr_data_type )
                   {
                       break;
                   }
                }
                if (j == reg_tensor_num)
                {
                    matched = TRUE;
                    break;
                }
            }
        }

        vsi_nn_safe_free(sig);
    } else {
        matched = TRUE;
    }

    return matched;
}

char* generate_op_io_types_desc
    (
    vsi_nn_tensor_t** inputs,
    int inputs_num,
    vsi_nn_tensor_t** outputs,
    int outputs_num
    )
{
    int i = 0;
    int total_sz = 0;
    int used_sz = 0;
    char* desc = NULL;

    for(i = 0; i < inputs_num; i++) {
        if(inputs[i]) {
            total_sz += snprintf(NULL, 0, "%s %s, ",
                    _get_qtype_name(inputs[i]->attr.dtype.qnt_type),
                    _get_dtype_name(inputs[i]->attr.dtype.vx_type));
        }
    }
    for(i = 0; i < outputs_num; i++) {
        if(outputs[i]) {
            total_sz += snprintf(NULL, 0, "%s %s, ",
                    _get_qtype_name(outputs[i]->attr.dtype.qnt_type),
                    _get_dtype_name(outputs[i]->attr.dtype.vx_type));
        }
    }

    total_sz += 1; /* terminator */
    desc = (char*)malloc(sizeof(char) * total_sz);
    memset(desc, 0x00, sizeof(char) * total_sz);

    for(i = 0; i < inputs_num; i++) {
        if(inputs[i]) {
            used_sz += snprintf(desc + used_sz, total_sz - used_sz, "%s %s, ",
                    _get_qtype_name(inputs[i]->attr.dtype.qnt_type),
                    _get_dtype_name(inputs[i]->attr.dtype.vx_type));
        }
    }
    for(i = 0; i < outputs_num; i++) {
        if(outputs[i]) {
            used_sz += snprintf(desc + used_sz, total_sz - used_sz, "%s %s, ",
                    _get_qtype_name(outputs[i]->attr.dtype.qnt_type),
                    _get_dtype_name(outputs[i]->attr.dtype.vx_type));
        }
    }

    if(used_sz >= 2) {
        desc[used_sz - 2] = '\0';
    }

    return desc;
}

void destroy_op_io_types_desc
    (
    char* desc
    )
{
    if(desc) {
        free(desc);
    }
}

void print_op_io_types
    (
    const char* name,
    const op_constraint_reg_type* op_constraint_reg
    )
{
    /* print supported types for statistics use */
    VSILOGI("Operation: %s", name);
    (void)op_constraint_reg;
    (void)_get_dtype_name;
    (void)_get_qtype_name;
}

vsi_bool is_item_in_array
    (
    const void* item,
    const void* items,
    int item_size,
    int item_count
    )
{
    int i = 0;

    if (item && items) {
        for (;i < item_count; i++) {
            if(0 == memcmp(item, (uint8_t*)items + i * item_size, item_size)) {
                return TRUE;
            }
        }
    }

    return FALSE;
}
