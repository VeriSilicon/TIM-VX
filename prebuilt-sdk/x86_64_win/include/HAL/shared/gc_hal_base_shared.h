/******************************************************************************\
|*                                                                            *|
|* Copyright (c) 2007-2010 by Vivante Corp.  All rights reserved.             *|
|*                                                                            *|
|* The material in this file is confidential and contains trade secrets of    *|
|* Vivante Corporation.  This is proprietary information owned by Vivante     *|
|* Corporation.  No part of this work may be disclosed, reproduced, copied,   *|
|* transmitted, or used in any way for any purpose, without the express       *|
|* written permission of Vivante Corporation.                                 *|
|*                                                                            *|
\******************************************************************************/

#ifndef __gc_hal_base_shared_h_
#define __gc_hal_base_shared_h_

#ifdef __cplusplus
extern "C" {
#endif

#define gcdBINARY_TRACE_MESSAGE_SIZE 240

typedef struct _gcsBINARY_TRACE_MESSAGE * gcsBINARY_TRACE_MESSAGE_PTR;
typedef struct _gcsBINARY_TRACE_MESSAGE
{
    gctUINT32   signature;
    gctUINT32   pid;
    gctUINT32   tid;
    gctUINT32   line;
    gctUINT32   numArguments;
    gctUINT8    payload;
}
gcsBINARY_TRACE_MESSAGE;

/* gcsOBJECT object defintinon. */
typedef struct _gcsOBJECT
{
    /* Type of an object. */
    gceOBJECT_TYPE              type;
}
gcsOBJECT;

#ifdef __cplusplus
}
#endif

#endif /* __gc_hal_base_shared_h_ */

