#include "vsi_nn_log.h"
#include "utils/vsi_nn_dlfcn.h"

#if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
void * vsi_nn_dlopen( const char *file, int mode )
{
    return NULL;
}

int vsi_nn_dlclose( void *handle )
{
    return -1;
}

__declspec(noinline)
void* vsi_nn_dlsym( void *handle, const char *name )
{
    return NULL;
}

char *vsi_nn_dlerror( void )
{
    return "\0";
}
#else

void* vsi_nn_dlsym
    (
    void *handle,
    const char *name
    )
{
    return dlsym( handle, name );
}

int vsi_nn_dlclose
    (
    void *handle
    )
{
    return dlclose( handle );
}

void* vsi_nn_dlopen
    (
    const char *file,
    int mode
    )
{
    return dlopen( file, mode );
}

char * vsi_nn_dlerror(void)
{
    return dlerror();
}

#endif

