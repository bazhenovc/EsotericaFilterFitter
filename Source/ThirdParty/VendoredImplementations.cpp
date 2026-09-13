
#include <cstdio>
#include <cstdlib>

#define FF_STB_ASSERT( expression )                                                       \
    do                                                                                    \
    {                                                                                     \
        if( !( expression ) )                                                             \
        {                                                                                 \
            std::fprintf( stderr, "vendored assertion failed: %s (%s:%d)\n",              \
                          #expression, __FILE__, __LINE__ );                              \
            std::abort();                                                                 \
        }                                                                                 \
    } while( 0 )

#define STBI_ASSERT( expression )  FF_STB_ASSERT( expression )
#define STBIW_ASSERT( expression ) FF_STB_ASSERT( expression )
#define STBIR_ASSERT( expression ) FF_STB_ASSERT( expression )

//-------------------------------------------------------------------------

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION 1
#include "TinyEXR/tinyexr.h"
