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

/* WARNING! AUTO-GENERATED, DO NOT MODIFY MANUALLY */

#ifndef __VXC_BINARIES_H__
#define __VXC_BINARIES_H__

#ifndef _cnt_of_array
#define _cnt_of_array( arr )            (sizeof( arr )/sizeof( arr[0] ))
#endif

typedef struct _vsi_nn_vx_bin_resource_item_type
{
    char const* name;
    uint8_t const* data;
    uint32_t len;
} vsi_nn_vx_bin_resource_item_type;

const vsi_nn_vx_bin_resource_item_type vx_bin_resource_items[] =
{
    {NULL, NULL, 0},
};

const int vx_bin_resource_items_cnt = _cnt_of_array(vx_bin_resource_items);

#endif
