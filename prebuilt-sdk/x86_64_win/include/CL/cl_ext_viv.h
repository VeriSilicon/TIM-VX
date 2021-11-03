#ifndef __CL_EXT_VIV_H
#define __CL_EXT_VIV_H

#include <CL/cl.h>
#include <CL/cl_platform.h>

#ifdef __cplusplus
extern "C" {
#endif


/********************************************************************
* cl_vivante_device_attribute_query
********************************************************************/
#define CL_MEM_USE_UNCACHED_HOST_MEMORY_VIV         (1 << 28)

/* for CL_MEM_USE_HOST_PHYSICAL_ADDR_VIV, application must make 
   sure the physical address passed in is a 40 bit address
*/
#define CL_MEM_USE_HOST_PHYSICAL_ADDR_VIV           (1 << 29)


#ifdef __cplusplus
}
#endif

#endif
