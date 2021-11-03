/* 

 * Copyright (c) 2012-2017 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _OPENVX_EXT_DEBUG_H_
#define _OPENVX_EXT_DEBUG_H_

#include <VX/vx.h>

/*!
 * \file
 * \brief The OpenVX Debugging Extension.
 * \defgroup group_debug_ext Debugging Extension
 * \defgroup group_vision_function_copy_image Kernel: Copy Image
 * \defgroup group_vision_function_copy_array Kernel: Copy Array
 * \defgroup group_vision_function_fwrite_image Kernel: File Write Image
 * \defgroup group_vision_function_fwrite_array Kernel: File Write Array
 * \defgroup group_vision_function_plus1 Kernel: Plus One Image
 * \defgroup group_vision_function_fill_image Kernel: Fill Image
 * \defgroup group_vision_function_check_image Kernel: Check Image
 * \defgroup group_vision_function_check_array Kernel: Check Array
 * \defgroup group_vision_function_compare_images Kernel: Compare Images
 */

/*! \brief The maximum filepath name length.
 * \ingroup group_debug_ext
 */
#define VX_MAX_FILE_NAME    (256)

/*! \brief The library value for the extension
 * \ingroup group_debug_ext
 */
#define VX_LIBRARY_KHR_DEBUG (0xFF)

/*! \brief The list of extensions to OpenVX from the Sample Implementation.
 * \ingroup group_debug_ext
 */
enum vx_kernel_debug_ext_e {

    /*!
     * \brief The Copy kernel. Output = Input.
     * \param  [in] vx_image The input image.
     * \param [out] vx_image The output image.
     * \see group_vision_function_copy_image
     */
    VX_KERNEL_DEBUG_COPY_IMAGE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0x0,

    /*!
     * \brief The Copy Kernel, Output = Input.
     * \param [in] vx_array The input array.
     * \param [out] vx_array The output array.
     * \see group_vision_function_copy_array
     */
     VX_KERNEL_DEBUG_COPY_ARRAY = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0x1,

    /*!
     * \brief The File Writing Kernel for Images.
     * \param [in] vx_image The input image.
     * \param [in] vx_array The name of the file.
     * \see group_vision_function_fwrite_image
     */
    VX_KERNEL_DEBUG_FWRITE_IMAGE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0x2,

    /*!
     * \brief The File Writing Kernel for Arrays
     * \param [in] vx_array The input array.
     * \param [in] vx_array The name of the file.
     * \see group_vision_function_fwrite_array
     */
     VX_KERNEL_DEBUG_FWRITE_ARRAY = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0x3,

     /*!
      * \brief The File Reading Kernel for images.
      * \param [in] vx_array The name of the file to read.
      * \param [out] vx_image The output image.
      * \see group_vision_function_fread_image
      */
     VX_KERNEL_DEBUG_FREAD_IMAGE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0x4,

     /*!
      * \brief The File Reading Kernel for Arrays.
      * \param [in] vx_array The name of the file to read.
      * \param [out] vx_image The output image.
      * \see group_vision_function_fread_array
      */
     VX_KERNEL_DEBUG_FREAD_ARRAY = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0x5,

     /*!
      * \brief Fills the image with a given value.
      * \param [in] vx_uint32
      * \param [out] vx_image
      * \ingroup group_vision_function_fill_image
      */
     VX_KERNEL_FILL_IMAGE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0x6,

     /*!
      * \brief Checks an image against a known value and returns a number of
      * errors.
      * \param [in] vx_image
      * \param [in] vx_uint32
      * \param [out] vx_scalar
      * \ingroup group_vision_function_check_image
      */
     VX_KERNEL_CHECK_IMAGE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0x7,

     /*!
      * \brief Checks an array against a known value and returns a number of
      * errors.
      * \param [in] vx_array
      * \param [in] vx_uint8
      * \param [out] vx_scalar
      * \ingroup group_vision_function_check_array
      */
     VX_KERNEL_CHECK_ARRAY = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0x8,

     /*!
      * \brief Compares two images and returns the number of differences.
      * \param [in] vx_image
      * \param [in] vx_image
      * \param [out] vx_scalar
      * \ingroup group_vision_function_compare_image
      */
     VX_KERNEL_COMPARE_IMAGE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0x9,

     /*!
      * \brief Copies an image from a memory area.
      * \param [in] void *
      * \param [out] vx_image
      * \see group_vision_function_copy_ptr
      */
     VX_KERNEL_COPY_IMAGE_FROM_PTR = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_DEBUG) + 0xA,
};

/******************************************************************************/
// GRAPH MODE FUNCTIONS
/******************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif
/*!
 * \brief [Graph] Creates a Copy Image Node.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input image.
 * \param [out] output The output image.
 * \see VX_KERNEL_COPY_IMAGE
 * \note Graph Mode Function.
 * \ingroup group_vision_function_copy_image
 */
vx_node vxCopyImageNode(vx_graph graph, vx_image input, vx_image output);

/*!
 * \brief [Graph] Creates a Copy Array Node.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input array.
 * \param [out] output The output array.
 * \see VX_KERNEL_COPY_ARRAY
 * \note Graph Mode Function.
 * \ingroup group_vision_function_copy_array
 */
vx_node vxCopyArrayNode(vx_graph graph, vx_array input, vx_array output);

/*! \brief [Graph] Writes the source image to the file.
 * \param [in] graph The handle to the graph.
 * \param [in] image The input array.
 * \param [in] name The name of the file.
 * \note Graph Mode Function.
 * \ingroup group_vision_function_fwrite_image
 */
vx_node vxFWriteImageNode(vx_graph graph, vx_image image, vx_char name[VX_MAX_FILE_NAME]);

/*! \brief [Graph] Writes the source array to the file.
 * \param [in] graph The handle to the graph.
 * \param [in] array The input array.
 * \param [in] name The name of the file.
 * \note Graph Mode Function.
 * \ingroup group_vision_function_fwrite_array
 */
vx_node vxFWriteArrayNode(vx_graph graph, vx_array array, vx_char name[VX_MAX_FILE_NAME]);

/*! \brief [Graph] Writes the source image to the file.
 * \param [in] graph The handle to the graph.
 * \param [in] name The name of the file.
 * \param [out] image The output image.
 * \note Graph Mode Function.
 * \ingroup group_vision_function_fread_image
 */
vx_node vxFReadImageNode(vx_graph graph, vx_char name[VX_MAX_FILE_NAME], vx_image image);

/*! \brief [Graph] Writes the source array to the file.
 * \param [in] graph The handle to the graph.
 * \param [in] name The name of the file.
 * \param [out] array The output array.
 * \note Graph Mode Function.
 * \ingroup group_vision_function_fread_array
 */
vx_node vxFReadArrayNode(vx_graph graph, vx_char name[VX_MAX_FILE_NAME], vx_array array);

/*! \brief [Graph] Adds 1 to each uint8 pixel. This will clamp at 255.
 * \param [in] graph The handle to the graph.
 * \param [in,out] image The image to increment.
 * \note Graph Mode Function
 * \ingroup group_vision_function_plus1
 */
vx_node vxPlusOneNode(vx_graph graph, vx_image image);

/*!
 * \brief [Graph] Fills an image with a known value.
 * \param [in] graph The handle to the graph.
 * \param [in] value The known value to fill the image with.
 * \param [out] output The image to fill.
 * \note Graph Mode Function
 * \ingroup group_vision_function_fill_image
 */
vx_node vxFillImageNode(vx_graph graph, vx_uint32 value, vx_image output);

/*!
 * \brief [Graph] Checks an image against a known value.
 * \param [in] graph The handle to the graph.
 * \param [in] input The image to check.
 * \param [in] value The known value to check the image against.
 * \param [out] errs The handle to the number of errors found.
 * \note Graph Mode Function
 * \ingroup group_vision_function_check_image
 */
vx_node vxCheckImageNode(vx_graph graph, vx_image input, vx_uint32 value, vx_scalar errs);

/*!
 * \brief [Graph] Checks a array for a known value.
 * \param [in] graph The handle to the graph.
 * \param [in] input The array to check.
 * \param [in] value The known value to check against.
 * \param [out] errs An output of the number of errors.
 * \note Graph Mode Function
 * \ingroup group_vision_function_check_array
 */
vx_node vxCheckArrayNode(vx_graph graph, vx_array input, vx_uint8 value, vx_scalar errs);

/*!
 * \brief [Graph] Compares two images and returns the number of pixel sub-channels
 * which are different.
 * \param [in] graph The handle to the graph.
 * \param [in] a The first image.
 * \param [in] b The second image.
 * \param [out] diffs The handle to scalar to hold the number of differences.
 * \note Graph Mode Function
 * \ingroup group_vision_function_compare_image
 */
vx_node vxCompareImagesNode(vx_graph graph, vx_image a, vx_image b, vx_scalar diffs);

/*! \brief [Graph] Copies a HOST memory area into an image.
 * \param [in] graph The handle to the graph.
 * \param [in] ptr The input pointer to the memory area to copy.
 * \param [out] output The output image.
 * \note Graph Mode Function
 * \ingroup group_vision_function_copy_ptr
 */
vx_node vxCopyImageFromPtrNode(vx_graph graph, void *ptr, vx_image output);

/******************************************************************************/
// IMMEDIATE MODE FUNCTION
/******************************************************************************/

/*! \brief [Immediate] Copies the source image to the destination image.
 * \param [in] src The input image.
 * \param [in] dst The output image.
 * \note Immediate Mode Function.
 * \ingroup group_vision_function_copy_image
 */
vx_status vxuCopyImage(vx_context context, vx_image src, vx_image dst);

/*! \brief [Immediate] Copies the source array to the destination array.
 * \param [in] src The input array.
 * \param [in] dst The output array.
 * \note Immediate Mode Function.
 * \ingroup group_vision_function_copy_array
 */
vx_status vxuCopyArray(vx_context context, vx_array src, vx_array dst);

/*! \brief [Immediate] Writes the source image to the file.
 * \param [in] image The input array.
 * \param [in] name The name of the file.
 * \note Immediate Mode Function.
 * \ingroup group_vision_function_fwrite_image
 */
vx_status vxuFWriteImage(vx_context context, vx_image image, vx_char name[VX_MAX_FILE_NAME]);

/*! \brief [Immediate] Writes the source array to the file.
 * \param [in] array The input array.
 * \param [in] name The name of the file.
 * \note Immediate Mode Function.
 * \ingroup group_vision_function_fwrite_array
 */
vx_status vxuFWriteArray(vx_context context, vx_array array, vx_char name[VX_MAX_FILE_NAME]);

/*! \brief [Immediate] Reads the source image from the file.
 * \param [in] name The name of the file.
 * \param [out] image The output image.
  * \note Immediate Mode Function.
 * \ingroup group_vision_function_fread_image
 */
vx_status vxuFReadImage(vx_context context, vx_char name[VX_MAX_FILE_NAME], vx_image image);

/*! \brief [Immediate] Reads the source array from the file.
 * \param [in] name The name of the file.
 * \param [out] array The output array.
 * \note Immediate Mode Function.
 * \ingroup group_vision_function_fread_array
 */
vx_status vxuFReadArray(vx_context context, vx_char name[VX_MAX_FILE_NAME], vx_array array);

/*! \brief [Immediate] Adds 1 to each uint8 pixel. This will clamp at 255.
 * \param [in,out] image The image to increment.
 * \note Immediate Mode Function
 * \ingroup group_vision_function_plus1
 */
vx_node vxuPlusOneNode(vx_context context, vx_image image);

/*!
 * \brief [Immediate] Fills an image with a known value.
 * \param [in] value The known value to fill the image with.
 * \param [out] output The image to fill.
 * \note Immediate Mode Function
 * \ingroup group_vision_function_fill_image
 */
vx_status vxuFillImage(vx_context context, vx_uint32 value, vx_image output);

/*!
 * \brief [Immediate] Checks an image against a known value.
 * \param [in] output The image to check.
 * \param [in] value The known value to check the image against.
 * \param [out] numErrors The handle to the number of errors found.
 * \note Immediate Mode Function
 * \ingroup group_vision_function_check_image
 */
vx_status vxuCheckImage(vx_context context, vx_image input, vx_uint32 value, vx_uint32 *numErrors);

/*!
 * \brief [Immediate] Checks a array for a known value.
 * \param [in] input The array to check.
 * \param [in] value The known value to check against.
 * \param [out] numErrors An output of the number of errors.
 * \note Immediate Mode Function
 * \ingroup group_vision_function_check_array
 */
vx_status vxuCheckArray(vx_context context, vx_array input, vx_uint8 value, vx_uint32 *numErrors);

/*!
 * \brief [Immediate] Compares two images and returns the number of pixel sub-channels
 * which are different.
 * \param [in] a The first image.
 * \param [in] b The second image.
 * \param [out] numDiffs The handle to scalar to hold the number of differences.
 * \note Immediate Mode Function
 * \ingroup group_vision_function_compare_image
 */
vx_status vxuCompareImages(vx_context context, vx_image a, vx_image b, vx_uint32 *numDiffs);

/*! \brief [Immediate] Copies a HOST memory area into an image.
 * \param [in] ptr The input pointer to the memory area to copy.
 * \param [out] output The output image.
 * \note Immediate Mode Function
 * \ingroup group_vision_function_copy_ptr
 */
vx_status vxuCopyImageFromPtr(vx_context context, void *ptr, vx_image output);

#ifdef __cplusplus
}
#endif

#endif

