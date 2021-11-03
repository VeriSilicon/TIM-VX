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
#ifndef _VSI_NN_PRV_H_
#define _VSI_NN_PRV_H_

#if defined(__cplusplus)
extern "C"{
#endif

#define VSI_NN_MAX_PATH 256

#ifdef __linux__
#define VSI_NN_STD_CALL
#else
#define VSI_NN_STD_CALL __stdcall
#endif

typedef enum _vsi_nn_broad_cast_bits_e
{
    VSI_NN_BROAD_CAST_BITS_0 = 0x01,
    VSI_NN_BROAD_CAST_BITS_1 = 0x02,
    VSI_NN_BROAD_CAST_BITS_2 = 0x04,
    VSI_NN_BROAD_CAST_BITS_4 = 0x08,
} vsi_nn_broad_cast_bits_e;

#define REQUIRED_IO( _IOPORT ) ( (_IOPORT) != NULL ? (_IOPORT)->t : \
    ( VSILOGE("Required IO port: %s", #_IOPORT), (_IOPORT)->t ) )
#define OPTIONAL_IO( _IOPORT ) ( (_IOPORT) != NULL ? (_IOPORT)->t : NULL)

#ifndef __BEGIN_DECLS
    #if defined(__cplusplus)
    #define __BEGIN_DECLS extern "C" {
    #define __END_DECLS }
    #else
    #define __BEGIN_DECLS
    #define __END_DECLS
    #endif
#endif

#if defined(__cplusplus)
}
#endif

#endif
