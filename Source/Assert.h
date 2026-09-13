#pragma once

#include <cstdio>
#include <cstdlib>

namespace FilterFitter
{
    [[noreturn]] inline void FailAssertion( char const* pCondition, char const* pFile, int line )
    {
        std::fprintf( stderr, "assertion failed: %s\n  %s(%d)\n", pCondition, pFile, line );
        std::fflush( stderr );
        std::abort();
    }
}

//-------------------------------------------------------------------------

#define FF_ASSERT( condition ) \
    ( ( condition ) ? ( void ) 0 : FilterFitter::FailAssertion( #condition, __FILE__, __LINE__ ) )
