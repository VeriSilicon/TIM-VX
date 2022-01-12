#!/bin/bash

if [ -z $4 ]; then
echo
echo usage:
echo "    $0 VIV_SDK_PATH VCCOMPILER OVXLIB GPU_CONFIG_FILE"
echo
echo "   VIV_SDK_PATH: VIVANTE SDK path"
echo "   VCCOMPILER: vcCompiler path"
echo "   OVXLIB: ovxlib path"
echo "   GPU_CONFIG_FILE: gpu config file path"
echo
echo "e.g."
echo "    ./ovxlib_bin_build.sh VIV_SDK_PATH VCCOMPILER OVXLIB vip8000.config"
echo
exit 1
fi

export VIV_SDK_PATH=$1
export VCCOMPILER=$2
export OVXLIB=$3
export GPU_CONFIG_FILE=$4

function convert_vxc_shader()
{
    echo "== convert VXC shader to header files ..."

    VX_BIN_PATH=$OVXLIB/include/libnnext/vx_bin
    if [ ! -e "$VX_BIN_PATH" ]; then
       mkdir -p $VX_BIN_PATH
    fi
    rm -f $VX_BIN_PATH/*.h

    (
cat<<EOF
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

EOF
    )>$VX_BIN_PATH/vxc_binaries.h

    cd $OVXLIB/src/libnnext/ops/vx
    rm -f *.gcPGM *.vxgcSL

    echo "== generating $VX_BIN_PATH/vxc_binaries.h ..."

    for vxFile in `ls *.vx | sed "s/\.vx//"`
    do
    {
        if [ "${vxFile}" != "vsi_nn_kernel_header" ]; then
            echo $VCCOMPILER -f${GPU_CONFIG_FILE} -allkernel -cl-viv-gcsl-driver-image \
              -o${vxFile}_vx -m vsi_nn_kernel_header.vx ${vxFile}.vx
            $VCCOMPILER -f${GPU_CONFIG_FILE} -allkernel -cl-viv-gcsl-driver-image \
              -o${vxFile}_vx -m vsi_nn_kernel_header.vx ${vxFile}.vx || exit 1
            echo "python $OVXLIB/ConvertPGMToH.py -i ${vxFile}_vx_all.gcPGM \
              -o $VX_BIN_PATH/vxc_bin_${vxFile}_vx.h"
            python $OVXLIB/ConvertPGMToH.py -i ${vxFile}_vx_all.gcPGM \
              -o $VX_BIN_PATH/vxc_bin_${vxFile}_vx.h || exit 1
            echo "#include \"vxc_bin_${vxFile}_vx.h\"" >> $VX_BIN_PATH/vxc_binaries.h
        fi
    } &
    done

    wait
    echo "== convert VXC shader to header files: success!"
    echo "== convert VXC shader to header files: success!"
    echo "== convert VXC shader to header files: success!"
    rm -f *.gcPGM *.vxgcSL

    cd $OVXLIB/src/libnnext/ops/cl
    rm -f *.gcPGM *.vxgcSL *.clgcSL

    for vxFile in `ls *.cl | sed "s/\.cl//"`
    do
    {
        if [ "${vxFile}" != "eltwise_ops_helper" ]; then
            cp ${vxFile}.cl ${vxFile}.vx
            echo $VCCOMPILER -f${GPU_CONFIG_FILE} -allkernel -cl-viv-gcsl-driver-image \
              -o${vxFile}_cl -m eltwise_ops_helper.cl ${vxFile}.vx
            $VCCOMPILER -f${GPU_CONFIG_FILE} -allkernel -cl-viv-gcsl-driver-image \
              -o${vxFile}_cl -m eltwise_ops_helper.cl ${vxFile}.vx || exit 1
            echo "python $OVXLIB/ConvertPGMToH.py -i ${vxFile}_cl_all.gcPGM \
              -o $VX_BIN_PATH/vxc_bin_${vxFile}_cl.h"
            python $OVXLIB/ConvertPGMToH.py -i ${vxFile}_cl_all.gcPGM \
              -o $VX_BIN_PATH/vxc_bin_${vxFile}_cl.h || exit 1
            echo "#include \"vxc_bin_${vxFile}_cl.h\"" >> $VX_BIN_PATH/vxc_binaries.h
            rm ${vxFile}.vx
        fi
    } &
    done

    wait
    echo "== convert CL shader to header files: success!"
    echo "== convert CL shader to header files: success!"
    echo "== convert CL shader to header files: success!"
    rm -f *.gcPGM *.vxgcSL *.clgcSL

    (
cat<<EOF

#ifndef _cnt_of_array
#define _cnt_of_array( arr )            (sizeof( arr )/sizeof( arr[0] ))
#endif

typedef struct _vsi_nn_vx_bin_resource_item_type
{
    char const* name;
    uint8_t const* data;
    uint32_t len;
} vsi_nn_vx_bin_resource_item_type;

const vsi_nn_vx_bin_resource_item_type vx_bin_resource_items_vx[] =
{
EOF
    )>>$VX_BIN_PATH/vxc_binaries.h

    cd $OVXLIB/src/libnnext/ops/vx
    for vxFile in `ls *.vx | sed "s/\.vx//"`
    do
        vxFileUpper=`echo ${vxFile} | tr 'a-z' 'A-Z'`
        if [ "${vxFile}" != "vsi_nn_kernel_header" ]; then
            echo "    {\"${vxFile}_vx\", vxcBin${vxFile}_vx, VXC_BIN_${vxFileUpper}_VX_LEN}," \
            >> $VX_BIN_PATH/vxc_binaries.h
        fi
    done

    (
cat<<EOF
};

const int vx_bin_resource_items_vx_cnt = _cnt_of_array(vx_bin_resource_items_vx);

const vsi_nn_vx_bin_resource_item_type vx_bin_resource_items_cl[] =
{
EOF
    )>>$VX_BIN_PATH/vxc_binaries.h


    cd $OVXLIB/src/libnnext/ops/cl
    for vxFile in `ls *.cl | sed "s/\.cl//"`
    do
        vxFileUpper=`echo ${vxFile} | tr 'a-z' 'A-Z'`
        if [ "${vxFile}" != "eltwise_ops_helper" ]; then
            echo "    {\"${vxFile}_cl\", vxcBin${vxFile}_cl, VXC_BIN_${vxFileUpper}_CL_LEN}," \
            >> $VX_BIN_PATH/vxc_binaries.h
        fi
    done
    (
cat<<EOF
};

const int vx_bin_resource_items_cl_cnt = _cnt_of_array(vx_bin_resource_items_cl);

#endif
EOF
    )>>$VX_BIN_PATH/vxc_binaries.h

    exit 0
}

export VIVANTE_SDK_DIR=$VIV_SDK_PATH
convert_vxc_shader

exit 0
