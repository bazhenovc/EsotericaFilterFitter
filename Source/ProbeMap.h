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

        // The sphere mapped to an octahedron and unwrapped into one square, which is what the engine's point light shadows already use.
        // Appended, so the cube's and the tetrahedron's on-disk values are untouched.
        Octahedral = 2,
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

    // The resolution a fit for this map targets: the base a level 0 texture has, which is what sets the lobe-to-texel ratio every level's coefficients are fitted against.
    //
    // It is a property of the map rather than one number for the whole tool, because the three maps are for different things.
    // 128 is a cubemap's: it is the paper's resolution, it is what the published tables are data for, and it is the base every cubemap comparison in this project has been measured at.
    // 256 is an octahedral map's, because that is what the runtime will actually allocate - the closest power of two to the 313 that would match a 128 cubemap texel for texel, so the runtime gets comparable quality for about two thirds of the work per probe.
    // A tetrahedral map is here for completeness and cross-validation, and stays at the cubemap's base so the two remain directly comparable.
    uint32_t GetProbeMapBaseResolution( ProbeMap map );

    // The training grid a fit of this map uses, a FLOOR under the sampling policy in TableHarness.h rather than a count of directions.
    // Per map because it is spent against the map's own base resolution and the two are chosen together: a grid of 8 at base 128 samples one texel in 256, which is what every cubemap number was taken at, while the octahedral map is fitted at base 256 where a floor of 8 would leave every level but the first trained on a quarter of the directions it could afford.
    // The tetrahedral map keeps 8 so its measurements do not move, and at base 128 the sampling policy's fraction is at most 8, so for both of those maps this value is the only thing the grid has ever been.
    inline uint32_t GetProbeMapFitGridSize( ProbeMap map )
    {
        if( map == ProbeMap::Octahedral )
        {
            return 16;
        }

        return 8;
    }

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
