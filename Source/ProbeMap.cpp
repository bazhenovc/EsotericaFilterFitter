#include "ProbeMap.h"

#include <cstring>

#include "MapProjection.h"
#include "OctahedralProjection.h"
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
    // The octahedral level correction's diagnostic scale, defined where the map's other values are read from and left at 1 unless --octahedral-correction says otherwise. See OctahedralProjection.h.
    double g_octahedralLevelCorrectionScale = 1.0;

    static_assert( static_cast<uint32_t>( ProbeMap::Cube ) == 0, "the cube's on-disk value is part of a format" );
    static_assert( static_cast<uint32_t>( ProbeMap::Tetrahedron ) == 1, "the tetrahedron's on-disk value is part of a format" );
    static_assert( static_cast<uint32_t>( ProbeMap::Octahedral ) == 2, "the octahedron's on-disk value is part of a format" );

    //-------------------------------------------------------------------------

    char const* GetProbeMapName( ProbeMap map )
    {
        switch( map )
        {
            case ProbeMap::Cube:        return "cube";
            case ProbeMap::Tetrahedron: return "tetrahedron";
            case ProbeMap::Octahedral:  return "octahedral";
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

            // Three, the same as the cube's, because the taps are organised by the three axial frames and only the texel layout differs.
            // The octahedron's eight faces are a property of the texture, not of the filter; see OctahedralProjection.h.
            case ProbeMap::Octahedral:  return OctahedralProjection::NumFrames;
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
            case ProbeMap::Octahedral:  return OctahedralProjection::NumSlices;
        }

        return 0;
    }

    //-------------------------------------------------------------------------

    uint32_t GetProbeMapBaseResolution( ProbeMap map )
    {
        switch( map )
        {
            // The paper's resolution, the base the published tables are data for, and the base of every cubemap measurement in this project.
            case ProbeMap::Cube:        return 128;

            // Completeness and validation only, at the cubemap's base so a tetrahedral and a cubemap fit are directly comparable.
            case ProbeMap::Tetrahedron: return 128;

            // What the engine will run. 256 is the closest power of two to the 313 that matches a 128 cubemap texel for texel, and a 256 square has about two thirds of that cubemap's texels, so it is the same quality band for less work per probe.
            case ProbeMap::Octahedral:  return 256;
        }

        return 0;
    }

    //-------------------------------------------------------------------------

    bool IsProbeMapValue( uint32_t value )
    {
        return ( value == static_cast<uint32_t>( ProbeMap::Cube ) )
            || ( value == static_cast<uint32_t>( ProbeMap::Tetrahedron ) )
            || ( value == static_cast<uint32_t>( ProbeMap::Octahedral ) );
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
        uint32_t const numMaps = static_cast<uint32_t>( ProbeMap::Octahedral ) + 1;

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
