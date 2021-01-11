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

#ifndef _VX_KHR_OPENCL_H_
#define _VX_KHR_OPENCL_H_

#include <VX/vx.h>
#include <VX/vx_compatibility.h>

/*! \file
 * \brief The OpenVX to OpenCL Inter-op Extension Header.
 *
 * \defgroup group_cl_api API
 * \brief The API used by Clients to add OpenCL Kernels as <tt>vx_kernel</tt>.
 * \details
 *
 * \defgroup group_cl_def Extension Defines
 * \brief The Extension defines and constants.
 *
 * \defgroup group_cl_image Images
 * \brief OpenVX Images
 * \details Depending on whether the OpenCL implementation supports images, <tt>vx_image</tt>
 * may map to an <tt>image2d_t</tt> or a OpenCL buffer.
 *
 * \defgroup group_cl_array Arrays
 * \brief OpenVX Arrays
 *
 * \defgroup group_cl_convolution Convolutions
 * \brief OpenVX Convolutions
 *
 * \defgroup group_cl_distribution Distributions
 * \brief OpenVX Distributions
 *
 * \defgroup group_cl_matrix Matricies
 * \brief OpenVX Matrix
 *
 * \defgroup group_cl_types OpenVX to OpenCL Atomic Types
 * \brief Atomic Types
 * \details OpenVX types map to OpenCL types through this table:
 * | VX       | OpenCL|
 * |:---------|:------|
 * |<tt>vx_uint8</tt>  |<tt>uchar</tt>  |
 * |<tt>vx_int8</tt>   |<tt>char</tt>   |
 * |<tt>vx_uint16</tt> |<tt>ushort</tt> |
 * |<tt>vx_int16</tt>  |<tt>short</tt>  |
 * |<tt>vx_uint32</tt> |<tt>uint</tt>   |
 * |<tt>vx_int32</tt>  |<tt>int</tt>    |
 * |<tt>vx_uint64</tt> |<tt>ulong</tt>  |
 * |<tt>vx_int64</tt>  |<tt>long</tt>   |
 * |<tt>vx_float32</tt>|<tt>float</tt>  |
 * |<tt>vx_float64</tt>|<tt>double</tt> |
 * |<tt>vx_size</tt>   |<tt>size_t</tt> |
 *
 * \note <tt>size_t</tt> can not be used as a parameter to a <tt>__kernel</tt>.
 */

#ifndef VX_SCALE_UNITY
#define VX_SCALE_UNITY  (1024)
#endif

/*!\brief The maximum number of planes an image may have which is compatible across both
 * API.
 * \ingroup group_cl_def
 */
#define VX_CL_MAX_PLANES (4)

#if defined(VX_CL_DOCUMENTATION) || !defined(VX_CL_KERNEL)

#if defined(__APPLE__) || defined(DARWIN)
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

#if (!defined(__APPLE__)) && defined(CL_USE_LUMINANCE)
#define CL_USE_IMAGES
#endif

/*! \brief The string name of this extension to match for in the extensions list
 * \ingroup group_cl_def
 */
#define OPENVX_KHR_OPENCL "vx_khr_opencl"

/*! \brief Adds an OpenCL Kernel as source code into the OpenVX implementation.
 * \param [in] context The OpenVX Context.
 * \param [in] name The name of the kernel in OpenVX nomenclature.
 * \param [in] enumeration The OpenVX kernel enumeration used to identify this kernel.
 * \param [in] source The array of source line pointers.
 * \param [in] line_lengths The array of lines lengths for each line of source.
 * \param [in] num_lines the number of lines in both the sources array and line_lengths array.
 * \param [in] symbol_name The name of the kernel to call in the program.
 * \param [in] numParams The number of parameters to the OpenVX kernel.
 * \param [in] input The input validator.
 * \param [in] output The output validator.
 * \see <tt>vxAddParameterToKernel</tt> to configure the specific parameter attributes.
 * \ingroup group_cl_api
 */
VX_API_ENTRY vx_kernel VX_API_CALL vxAddOpenCLAsSourceKernel(vx_context context,
                                    vx_char name[VX_MAX_KERNEL_NAME],
                                    vx_enum enumeration,
                                    char *source[],
                                    size_t line_lengths[],
                                    size_t num_lines,
                                    char symbol_name[],
                                    vx_uint32 numParams,
                                    vx_kernel_input_validate_f input,
                                    vx_kernel_output_validate_f output);

/*! \brief Adds an OpenCL Kernel as binary program into the OpenVX implementation.
 * \param [in] context The OpenVX Context.
 * \param [in] name The name of the kernel in OpenVX nomenclature.
 * \param [in] enumeration The OpenVX kernel enumeration used to identify this kernel.
 * \param [in] program The OpenCL Program which contains the kernel (either pre-compiled or compiled by user).
 * \param [in] symbol_name The name of the kernel to call in the program.
 * \param [in] numParams The number of parameters to the OpenVX kernel.
 * \param [in] input The input validator.
 * \param [in] output The output validator.
 * \see <tt>vxAddParameterToKernel</tt> to configure the specific parameter attributes.
 * \ingroup group_cl_api
 */
VX_API_ENTRY vx_kernel VX_API_CALL vxAddOpenCLAsBinaryKernel(vx_context context,
                                    vx_char name[VX_MAX_KERNEL_NAME],
                                    vx_enum enumeration,
                                    cl_program program,
                                    char symbol_name[],
                                    vx_uint32 numParams,
                                    vx_kernel_input_validate_f input,
                                    vx_kernel_output_validate_f output);

#endif // External API

#if defined(VX_CL_DOCUMENTATION) || defined(VX_CL_KERNEL)

#if defined(__IMAGE_SUPPORT__) && defined(CL_USE_LUMINANCE)
#define CL_USE_IMAGES
#endif

/*! \brief Allows access to an image pixel as a typecast pointer deference.
 * \param type The OpenCL single element type
 * \param ptr The <tt>__global</tt> pointer to the base of the image.
 * \param x The x coordinate.
 * \param y The y coordinate.
 * \param sx The x stride.
 * \param sy The y stride.
 * \ingroup group_cl_image
 */
#define vxImagePixel(type, ptr, x, y, sx, sy) \
    (*(type *)(&((uchar *)ptr)[((y) * sy) + ((x) * sx)]))

/*!
 * \brief Allows access to an array item as a typecast pointer deference.
 * \param type The OpenCL single element type or structure type.
 * \param ptr The <tt>__global</tt> pointer to the base of the array.
 * \param index The index of the element to access.
 * \param stride The stride in bytes between two adjacent elements.
 * \ingroup group_cl_array
 */
#define vxArrayItem(type, ptr, index, stride) \
    (*(type *)(&((uchar *)ptr)[index*stride]))

/*! \brief Allows access to a matrix element \f$ M_{ij} \f$ where i is the column and j is the row.
 * \param type The OpenCL single element type of the matrix.
 * \param ptr The <tt>__global</tt> pointer to the base of the array.
 * \param columns The number of columns in the matrix.
 * \param i The column index
 * \param j The row index
 * \ingroup group_cl_matrix
 */
#define vxMatrixElement(type, ptr, columns, i, j) (((type *)ptr)[columns*j + i])

/*! \brief Allows access to a convolution element \f$ C_{ij} \f$ where i is the column and j is the row.
 * \note Convolution elements are always of type <tt>short</tt>.
 * \param ptr The <tt>__global</tt> pointer to the base of the array.
 * \param columns The number of columns in the matrix.
 * \param i The column index
 * \param j The row index
 * \ingroup group_cl_convolution
 */
#define vxConvolveElement(ptr, columns, i, j) (((short *)ptr)[columns*j + i])

/*! \brief Allows access to a distribution frequency counter.
 * \param ptr The <tt>__global</tt> pointer to the base of the distribution.
 * \param value The value to retrive the frequency count for.
 * \param offset The offset within the input domain.
 * \param range The total range within the domain starting from offset.
 * \param window_size The window size of the bin.
 * \ingroup group_cl_distribution
 */
#define vxGetFrequency(ptr, value, offset, range, window_size) \
    ((offset <= value) && (value <= (range+offset)) ? ptr[(value-offset)/window_size] : 0)

/*! \brief Increments a distribution frequency counter for a value.
 * \param ptr The <tt>__global</tt> pointer to the base of the distribution.
 * \param value The value to increment the frequency count for.
 * \param offset The offset within the input domain.
 * \param range The total range within the domain starting from offset.
 * \param window_size The window size of the bin.
 * \ingroup group_cl_distribution
 */
#define vxIncFrequency(ptr, value, offset, range, window_size) \
    ((offset <= value) && (value <= (range+offset)) ? ++ptr[(value-offset)/window_size] : 0)

/*! \brief Decrements a distribution frequency counter for a value.
 * \param ptr The <tt>__global</tt> pointer to the base of the distribution.
 * \param value The value to decrement the frequency count for.
 * \param offset The offset within the input domain.
 * \param range The total range within the domain starting from offset.
 * \param window_size The window size of the bin.
 * \ingroup group_cl_distribution
 */
#define vxDecFrequency(ptr, value, offset, range, window_size) \
    ((offset <= value) && (value <= (range+offset)) ? --ptr[(value-offset)/window_size] : 0)

#if defined(VX_VERSION_1_1) && (VX_VERSION >= VX_VERSION_1_1)

/*! \brief Allows access to a distribution frequency counter.
 * \param ptr The <tt>__global</tt> pointer to the base of the distribution.
 * \param value The value to retrive the frequency count for.
 * \param offset The offset within the input domain.
 * \param range The total range within the domain starting from offset.
 * \param num_bins The number of bins in the domain range.
 * \ingroup group_cl_distribution
 */
#define vxGetFrequency2(ptr, value, offset, range, num_bins) \
    ((offset <= value) && (value <= (range+offset)) ? ptr[(value-offset)*num_bins/range] : 0)

/*! \brief Increments a distribution frequency counter for a value.
 * \param ptr The <tt>__global</tt> pointer to the base of the distribution.
 * \param value The value to increment the frequency count for.
 * \param offset The offset within the input domain.
 * \param range The total range within the domain starting from offset.
 * \param num_bins The number of bins in the domain range.
 * \ingroup group_cl_distribution
 */
#define vxIncFrequency2(ptr, value, offset, range, num_bins) \
    ((offset <= value) && (value <= (range+offset)) ? ++ptr[(value-offset)*num_bins/range] : 0)

/*! \brief Decrements a distribution frequency counter for a value.
 * \param ptr The <tt>__global</tt> pointer to the base of the distribution.
 * \param value The value to decrement the frequency count for.
 * \param offset The offset within the input domain.
 * \param range The total range within the domain starting from offset.
 * \param num_bins The number of bins in the domain range.
 * \ingroup group_cl_distribution
 */
#define vxDecFrequency2(ptr, value, offset, range, num_bins) \
    ((offset <= value) && (value <= (range+offset)) ? --ptr[(value-offset)*num_bins/range] : 0)

#endif /*VX_VERSION_1_1*/

#endif

#endif

