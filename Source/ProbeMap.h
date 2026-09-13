#pragma once

#include <cstdint>

//  A probe level, a mip chain and a coefficient table all belong to one base map, and the map is not a detail of how they were produced: the same environment at the same width is six textures for a cubemap and one for a single-slice tetrahedral map, and a table's axis count, tap count and index range follow from it. 
// So it is recorded everywhere a reader could otherwise guess wrong - the artifact file name, the table binary, and the stage key of a cached chain - and a mismatch is rejected rather than interpreted.
//
//  THE VALUES ARE ON-DISK VALUES. 
// 
// Cube is 0 and Tetrahedron is 1 because those numbers are written into a table binary that a runtime reads, so they are part of a format rather than of this enum and must not be renumbered.
// Adding a map appends.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    enum class ProbeMap : uint8_t
    {
        Cube = 0,
        Tetrahedron = 1,
    };

    // "cube" or "tetrahedron", for a file name, a directory key or a manifest line.
    // An out-of-range value returns "?" rather than reading past the table, because the caller may be holding a value that came out of a file.
    char const* GetProbeMapName( ProbeMap map );

    // The map a name names, for a command line or a manifest. 
    // Returns false and leaves map untouched when the name is not one this build knows, so a caller validating an option can report the text it was given.
    bool TryGetProbeMapFromName( char const* pName, ProbeMap& map );

    // How many blend frames, and therefore how many axes a table for this map has.
    uint32_t GetProbeMapAxisCount( ProbeMap map );

    // How many textures a level of this map occupies.
    uint32_t GetProbeMapSliceCount( ProbeMap map );

    // Whether a value read from a file names a map. 
    // The on-disk field is a uint32, so a hostile or corrupted file can hold anything; this is what lets a reader reject it before it becomes an enum.
    bool IsProbeMapValue( uint32_t value );

    // The map, from a value read from a file. Returns false and leaves map untouched when the value names no map.
    bool TryGetProbeMap( uint32_t value, ProbeMap& map );

    //  Which map a projection type is
    //-------------------------------------------------------------------------
    //  Declared here and specialised beside each projection, so a function templated on the map can name its own projection instead of being told it and hoping the two agree. 
    // Anything that writes a table's projection, a  stage key or an artifact name uses this.
    //-------------------------------------------------------------------------

    template< typename TMap >
    struct ProbeMapOf;
}
