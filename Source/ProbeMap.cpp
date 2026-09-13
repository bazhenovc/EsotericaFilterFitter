#include "ProbeMap.h"

#include <cstring>

#include "MapProjection.h"
#include "TetrahedralProjection.h"

//-------------------------------------------------------------------------
//  FilterFitter - the map table, and the asserts that keep it true
//-------------------------------------------------------------------------
//  The counts here are the projections' own, asserted rather than copied: a
//  number that silently disagreed with the map it names would key a cache
//  wrongly and size a buffer wrongly, and both failures are quiet.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    static_assert( static_cast<uint32_t>( ProbeMap::Cube ) == 0, "the cube's on-disk value is part of a format" );
    static_assert( static_cast<uint32_t>( ProbeMap::Tetrahedron ) == 1, "the tetrahedron's on-disk value is part of a format" );

    //-------------------------------------------------------------------------

    char const* GetProbeMapName( ProbeMap map )
    {
        switch( map )
        {
            case ProbeMap::Cube:        return "cube";
            case ProbeMap::Tetrahedron: return "tetrahedron";
        }

        // A value that came out of a file and names no map. Named rather than
        // asserting: the caller is expected to have rejected it, and a readable
        // string makes the rejection message useful.
        return "?";
    }

    //-------------------------------------------------------------------------

    uint32_t GetProbeMapAxisCount( ProbeMap map )
    {
        switch( map )
        {
            case ProbeMap::Cube:        return CubeProjection::NumFrames;
            case ProbeMap::Tetrahedron: return TetrahedralProjection::NumFrames;
        }

        return 0;
    }

    //-------------------------------------------------------------------------

    uint32_t GetProbeMapSliceCount( ProbeMap map )
    {
        switch( map )
        {
            case ProbeMap::Cube:        return CubeProjection::NumSlices;
            case ProbeMap::Tetrahedron: return TetrahedralProjection::NumSlices;
        }

        return 0;
    }

    //-------------------------------------------------------------------------

    bool IsProbeMapValue( uint32_t value )
    {
        return ( value == static_cast<uint32_t>( ProbeMap::Cube ) )
            || ( value == static_cast<uint32_t>( ProbeMap::Tetrahedron ) );
    }

    //-------------------------------------------------------------------------

    bool TryGetProbeMap( uint32_t value, ProbeMap& map )
    {
        if( !IsProbeMapValue( value ) )
        {
            return false;
        }

        map = static_cast<ProbeMap>( value );

        return true;
    }

    //-------------------------------------------------------------------------

    bool TryGetProbeMapFromName( char const* pName, ProbeMap& map )
    {
        if( pName == nullptr )
        {
            return false;
        }

        // Compared against the same table GetProbeMapName prints, so a name the
        // usage text offers is a name this accepts. The values are contiguous
        // from zero, which the static_asserts above and the validator pin down.
        uint32_t const numMaps = static_cast<uint32_t>( ProbeMap::Tetrahedron ) + 1;

        for( uint32_t value = 0; value < numMaps; ++value )
        {
            if( std::strcmp( pName, GetProbeMapName( static_cast<ProbeMap>( value ) ) ) == 0 )
            {
                map = static_cast<ProbeMap>( value );
                return true;
            }
        }

        return false;
    }
}
