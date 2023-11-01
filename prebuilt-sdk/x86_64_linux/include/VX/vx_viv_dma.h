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

#ifndef _VX_VIV_DMA_H_
#define _VX_VIV_DMA_H_

#include <VX/vx.h>
#include <VX/vx_khr_compatible.h>

#ifdef  __cplusplus
extern "C" {
#endif

typedef enum _vx_viv_bayer_format_type_e
{
    VX_VIV_BAYER_MODE_RGGB = 0,
    VX_VIV_BAYER_MODE_GRBG = 1,
    VX_VIV_BAYER_MODE_GBRG = 2,
    VX_VIV_BAYER_MODE_BGGR = 3
} vx_viv_bayer_format_type_e;

typedef enum _vx_viv_dma_rgb2raw_format_type_e
{
    VX_VIV_DMA_RGB2RAW_FORMAT_BGGR = 0,
    VX_VIV_DMA_RGB2RAW_FORMAT_RGGB = 1,
    VX_VIV_DMA_RGB2RAW_FORMAT_GRBG = 2,
    VX_VIV_DMA_RGB2RAW_FORMAT_GBRG = 3,
} vx_viv_dma_rgb2raw_format_type_e;

typedef enum _vx_viv_dma_compress_input_source_type_e
{
    VX_VIV_DMA_COMPRESS_INPUT_SOURCE_DISABLE,
    VX_VIV_DMA_COMPRESS_INPUT_SOURCE_DPP,
    VX_VIV_DMA_COMPRESS_INPUT_SOURCE_REF,
    VX_VIV_DMA_COMPRESS_INPUT_SOURCE_UNUSED
} vx_viv_dma_compress_input_source_type_e;

typedef enum _vx_viv_dma_ref_input_source_type_e
{
    VX_VIV_DMA_REF_INPUT_SOURCE_DISABLE,
    VX_VIV_DMA_REF_INPUT_SOURCE_DDR,
    VX_VIV_DMA_REF_INPUT_SOURCE_2ND_SBI
} vx_viv_dma_ref_input_source_type_e;

typedef struct _vx_viv_dma_compressed_args
{
    vx_uint32 ratio[2];
    vx_uint32 bayerMode;
}vx_viv_dma_compressed_args;

typedef struct _vx_viv_dma_psi_struct
{
    vx_uint32 dataElement;
    vx_uint32 length;
    vx_uint32 start;
    vx_uint32 end;
    vx_int32  leftPad;
    /*extrator only*/
    vx_int32  rightPad;
    vx_uint32 borderMode;
    vx_uint32 borderConst;
    vx_uint32 ignoreInput;
}vx_viv_dma_psi_struct;

typedef struct _vx_viv_dma_isp_pattern_args
{
    vx_int32 initRed;
    vx_int32 incXRed;
    vx_int32 incYRed;
    vx_int32 initGreen;
    vx_int32 incXGreen;
    vx_int32 incYGreen;
    vx_int32 initBlue;
    vx_int32 incXBlue;
    vx_int32 incYBlue;
    vx_int32 checkSumExpectedRed;
    vx_int32 checkSumExpectedGreen;
    vx_int32 checkSumExpectedBlue;
} vx_viv_dma_isp_pattern_args;

typedef struct _vx_viv_dma_denoise_post_proc_args
{
    vx_uint32 S0;
    vx_uint32 C0;
    vx_uint32 C1;
    vx_uint32 C2;
    vx_uint32 C3;
    vx_uint32 clampMin;
    vx_uint32 clampMax;
#if VX_GRAPH_V500_DMA_SUPPOPRT
    vx_uint32  oneSubMaskBypass;
    vx_uint32  dmaSourceMask;
#if VX_GRAPH_V500_NEW_DPP_SUPPORT
    vx_uint32 C4;
    vx_uint32 C5;
    vx_uint32 C6;
    vx_uint32 C7;
    vx_uint32 resMaskClip;
#endif
#endif
} vx_viv_dma_dpp_args;


#if VX_GRAPH_V500_DMA_SUPPOPRT
typedef struct _vx_viv_dma_data_flow_args
{
    vx_viv_dma_compress_input_source_type_e compressInputSrc;
    vx_viv_dma_ref_input_source_type_e refInputSrc;
} vx_viv_dma_data_flow_args;

typedef struct _vx_viv_dma_rgb2raw_args
{
    vx_viv_dma_rgb2raw_format_type_e format;    /*data format before convert to raw data*/
    vx_uint32                        rowLength; /*data length in a row*/
} vx_viv_dma_rgb2raw_args;

typedef struct _vx_viv_dma_ref_out_crop_args
{
    vx_uint32        totalLength;  /* total pixel length of ref data in NHWC*/
    vx_uint32        start;        /*crop pixel start in NHWC*/
    vx_uint32        length;       /*crop pixel length in NHWC*/
} vx_viv_dma_ref_out_crop_args;

typedef struct _vx_viv_dma_base_crop_args
{
    vx_int32 left;    /*left cropped pixels*/
    vx_int32 right;   /*right cropped pixels*/
} vx_viv_dma_base_crop_args;

typedef struct _vx_viv_dma_extractor_crop_args
{
    vx_viv_dma_base_crop_args in;
    vx_viv_dma_base_crop_args ref;
}vx_viv_dma_extractor_crop_args;

typedef struct _vx_viv_dma_filler_crop_args
{
    vx_viv_dma_base_crop_args in;
    vx_viv_dma_base_crop_args ref;
    vx_viv_dma_base_crop_args nnOut;
} vx_viv_dma_filler_crop_args;

typedef struct _vx_viv_dma_v500_args
{
    vx_viv_dma_data_flow_args dmaDataFlowArgs;
#if VX_GRAPH_V500_RGB2RAW_REFOUTPUTCROP_SUPPOPRT
    vx_viv_dma_rgb2raw_args dmaRGB2RawArgs;          /*added for RGB2Raw module*/
    vx_viv_dma_ref_out_crop_args  dmaRefOutCropArgs; /*added for PSIExtractorForRefOut module*/
    vx_uint32 stripeIndex;  /*save stripe index to read refInput_Fram%d.bin */
#if VX_GRAPH_V500_EXTRACTOR_FILLER_CROP_SUPPORT
    vx_viv_dma_extractor_crop_args dmaExtractorCropArgs;
    vx_viv_dma_filler_crop_args dmaFillerCropArgs;
#endif
#endif
#if VX_GRAPH_V500_UNALIGNED_HEIGHT_SUPPORT
    vx_int32   dmaExtraPadded[5]; /* to reduce dma in/out height for unaligned height input */
#endif
} vx_viv_dma_v500_args;

#endif

#ifdef  __cplusplus
}
#endif

#endif /*_VX_VIV_DMA_H_*/

