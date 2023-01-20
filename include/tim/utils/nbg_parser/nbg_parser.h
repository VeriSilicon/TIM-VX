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

#ifndef _NBG_PARSER_H
#define _NBG_PARSER_H

#if defined(__cplusplus)
extern "C"{
#endif

#ifndef NBG_NULL
#define NBG_NULL                   0
#endif

typedef struct nbg_parser_data_t *nbg_parser_data;

typedef unsigned char            nbg_uint8_t;
typedef unsigned short           nbg_uint16_t;
typedef unsigned int             nbg_uint32_t;
typedef unsigned long long       nbg_uint64_t;
typedef signed char              nbg_int8_t;
typedef signed short             nbg_int16_t;
typedef signed int               nbg_int32_t;
typedef signed long long         nbg_int64_t;
typedef char                     nbg_char_t;
typedef float                    nbg_float_t;
typedef unsigned long long       nbg_address_t;

typedef enum _nbg_status
{
    NBG_ERROR_OUT_OF_MEMORY         = -5,
    NBG_ERROR_NOT_SUPPORT           = -4,
    NBG_ERROR_INVALID_ARGUMENTS     = -3,
    NBG_ERROR_FORMAT                = -2,
    NBG_ERROR_FAILURE               = -1,
    NBG_SUCCESS                     =  0,
} nbg_status_e;

typedef enum _nbg_buffer_quantize_format_e
{
    /*! \brief Not quantized format */
    NBG_BUFFER_QUANTIZE_NONE                    = 0,
    /*! \brief The data is quantized with dynamic fixed point */
    NBG_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT     = 1,
    /*! \brief The data is quantized with TF asymmetric format */
    NBG_BUFFER_QUANTIZE_AFFINE_ASYMMETRIC       = 2
}   nbg_buffer_quantize_format_e;

typedef enum _nbg_buffer_format_e
{
    /*! \brief A float type of buffer data */
    NBG_BUFFER_FORMAT_FP32       = 0,
    /*! \brief A half float type of buffer data */
    NBG_BUFFER_FORMAT_FP16       = 1,
    /*! \brief A 8 bit unsigned integer type of buffer data */
    NBG_BUFFER_FORMAT_UINT8      = 2,
    /*! \brief A 8 bit signed integer type of buffer data */
    NBG_BUFFER_FORMAT_INT8       = 3,
    /*! \brief A 16 bit unsigned integer type of buffer data */
    NBG_BUFFER_FORMAT_UINT16     = 4,
    /*! \brief A 16 signed integer type of buffer data */
    NBG_BUFFER_FORMAT_INT16      = 5,
    /*! \brief A char type of data */
    NBG_BUFFER_FORMAT_CHAR       = 6,
    /*! \brief A bfloat 16 type of data */
    NBG_BUFFER_FORMAT_BFP16      = 7,
    /*! \brief A 32 bit integer type of data */
    NBG_BUFFER_FORMAT_INT32      = 8,
    /*! \brief A 32 bit unsigned signed integer type of buffer */
    NBG_BUFFER_FORMAT_UINT32     = 9,
    /*! \brief A 64 bit signed integer type of data */
    NBG_BUFFER_FORMAT_INT64      = 10,
    /*! \brief A 64 bit unsigned integer type of data */
    NBG_BUFFER_FORMAT_UINT64     = 11,
    /*! \brief A 64 bit float type of buffer data */
    NBG_BUFFER_FORMAT_FP64       = 12,
}   nbg_buffer_format_e;

typedef enum _nbg_buffer_type_e
{
    /*! \brief A tensor type of buffer data */
    NBG_BUFFER_TYPE_TENSOR      = 0,
    /*! \brief A image type of buffer data */
    NBG_BUFFER_TYPE_IMAGE       = 1,
    /*! \brief A array type of buffer data */
    NBG_BUFFER_TYPE_ARRAY       = 2,
    /*! \brief A scalar type of buffer data */
    NBG_BUFFER_TYPE_SCALAR      = 3,
}   nbg_buffer_type_e;

typedef enum _nbg_buffer_property
{
    /*!< \brief The quantization format, the returned value is nbg_buffer_quantize_format_e */
    NBG_PARSER_BUFFER_PROP_QUANT_FORMAT          = 0,
    /*!< \brief The number of dimension for this input, the returned value is nbg_uint32_t*/
    NBG_PARSER_BUFFER_PROP_NUM_OF_DIMENSION      = 1,
    /*!< \brief The size of each dimension for this input,
         the returned value is nbg_uint32_t * num_of_dimension */
    NBG_PARSER_BUFFER_PROP_DIMENSIONS            = 2,
    /*!< \brief The data format for this input, the returned value is nbg_buffer_format_e */
    NBG_PARSER_BUFFER_PROP_DATA_FORMAT           = 3,
    /*!< \brief The position of fixed point for dynamic fixed point, the returned value is nbg_uint32_t */
    NBG_PARSER_BUFFER_PROP_FIXED_POINT_POS       = 4,
    /*!< \brief The scale value for TF quantization format, the returned value is nbg_float_t */
    NBG_PARSER_BUFFER_PROP_SCALE                 = 5,
    /*!< \brief The zero point for TF quantization format, the returned value is nbg_uint32_t */
    NBG_PARSER_BUFFER_PROP_ZERO_POINT            = 6,
    /*!< \brief The name for network's inputs and outputs,
         the returned value size is queried by NBG_PARSER_BUFFER_PROP_NAME_SIZE */
    NBG_PARSER_BUFFER_PROP_NAME                  = 7,
    /*!< \brief The data type for input/output buffer, the returned value is nbg_buffer_type_e */
    NBG_PARSER_BUFFER_PROP_DATA_TYPE             = 8,

    /*!< \brief The size of input/output name. the returned value is nbg_uint32_t */
    NBG_PARSER_BUFFER_PROP_NAME_SIZE             = 128 + NBG_PARSER_BUFFER_PROP_NAME,
} nbg_buffer_property_e;

typedef enum _nbg_network_property_e
{
    /* !< \brief The name of network, the returned value size is queried by NBG_PARSET_NETWORK_SIZE_OF_NAME */
    NBG_PARSER_NETWORK_NAME            = 0,
    /* !< \brief The number of network input, the returned value is nbg_uint32_t */
    NBG_PARSER_NETWORK_INPUT_COUNT     = 1,
    /* !< \brief The number of network output, the returned value is nbg_uint32_t */
    NBG_PARSER_NETWORK_OUTPUT_COUNT    = 2,
    /* !< \brief The CID of this NBG, the returned value is nbg_uint32_t */
    NBG_PARSER_NETWORK_CID             = 3,

    /*!< \brief The size of network name. the returned value is nbg_uint32_t */
    NBG_PARSER_NETWORK_NAME_SIZE       = 128 + NBG_PARSER_NETWORK_NAME,
} nbg_network_property_e;

/*
@brief, Query NBG parser library version
*/
nbg_uint32_t nbg_parser_version(void);

/*
@brief, Initialize NBG parser. use NBG data to initialize nbg parser library.
@param IN buffer, NBG data in memory.
@param IN size, the size of NBG data.
@param OUT nbg, the NBG parser object which created by nbg_parser_init().
*/
nbg_status_e nbg_parser_init(
    void *buffer,
    nbg_uint32_t size,
    nbg_parser_data *nbg
    );

/*
@brief, query the input info of network.
@param IN nbg, the NBG parser object created by nbg_parser_init().
@param IN index, the index of network input.
@param IN property, property being queried.
       Quant format, dimension count, shape, data format and so on. see nbg_buffer_property_e enumeration.
@param IN size, the size of value buffer.
@param OUT value, The return value data.
       please refer nbg_buffer_property_e to know the size of  returned value data.
*/
nbg_status_e nbg_parser_query_input(
    nbg_parser_data nbg,
    nbg_uint32_t index,
    nbg_uint32_t property,
    void *value,
    nbg_uint32_t size
    );

/*
@brief, query the output info of network.
@param IN nbg, the NBG parser object created by nbg_parser_init().
@param IN index, the index of network input.
@param IN property, property being queried.
       Quant format, dimension count, shape, data format and so on.
       see nbg_buffer_property_e enumeration.
@param IN size, the size of value buffer.
@param OUT value, The return value data.
       please refer nbg_buffer_property_e to know the size of  returned value data.
*/
nbg_status_e nbg_parser_query_output(
    nbg_parser_data nbg,
    nbg_uint32_t index,
    nbg_uint32_t property,
    void *value,
    nbg_uint32_t size
    );

/*
@brief, query the network info.
@param IN nbg, the NBG parser object created by nbg_parser_init().
@param IN property, The property of the network is queried.
       network name, input count, output count and so on. see nbg_network_propery_e enumeration.
@param IN size, the size of value buffer.
@param OUT value, The return value data.
       please refer nbg_network_propery_e to know the size of  returned value data.
*/
nbg_status_e nbg_parser_query_network(
    nbg_parser_data nbg,
    nbg_uint32_t property,
    void *value,
    nbg_uint32_t size
    );

/*
@brief, destroy nbg parser.
@param, IN nbg, the NBG parser object created by nbg_parser_init().
*/
nbg_status_e nbg_parser_destroy(
    nbg_parser_data nbg
    );

#if defined(__cplusplus)
}
#endif
#endif
