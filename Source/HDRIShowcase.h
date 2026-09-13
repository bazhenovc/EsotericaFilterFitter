#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "HDRIImage.h"

// EXR that shows a whole probe at once
//-------------------------------------------------------------------------
// The validation writes one file per slice per level, which is what a cache wants and what a measurement reads. 
// It is not what a human wants: the answer to "does this table blur the way it should" is the source beside the filtered version beside the table's version, at every level, in one picture.
//
//  This composes that picture. Two layouts, chosen by how many textures a level
//  is:
//
//      many slices (a cubemap)      one row per level, three results across it
//          row 0                        the source, its slices across the first block,
//                                       the rest black
//          rows 1..N                    level 0 first, then down to level N-1: the
//                                       brute-force block, then the table's, then the
//                                       naive mip chain's, the map's slices across each
//
//      one slice                    four rows, one level per column
//          row 0, column 0              the source, and the rest of the row black
//          row 1                        the brute-force chain
//          row 2                        the table's chain
//          row 3                        the naive mip chain
//
// The naive result is the source chain's own level, unsampled by any lobe: it is what selecting a mip by roughness gives, and the method exists because it is wrong. 
// It sits beside the two filtered results so that what the filtering buys is visible rather than asserted. 
// At level 0 it is the source itself, which is the one level where the three agree.
//
// Comparing the results at one level is reading across a row; comparing levels is reading down. 
// That is why the source's row leaves its other blocks black rather than repeating the source in them.
//
// Every level below 0 is POINT upsampled to the source resolution: a level that is 8 texels across becomes 8x8 blocks of one colour.
//
// The values are written as they are, linear and in 32-bit float, so the bright end of an environment is visible rather than clipped
//-------------------------------------------------------------------------

namespace FilterFitter
{
    class ShowcaseImage
    {
    public:

        // Black, so an uncovered tile is unmistakably not a value
        void Create( uint32_t width, uint32_t height );

        inline bool     IsInitialized() const { return ( m_width != 0 ) && ( m_height != 0 ); }
        inline uint32_t GetWidth() const { return m_width; }
        inline uint32_t GetHeight() const { return m_height; }

        // One slice of a level into one tile, nearest-neighbour. 
        // The tile is tileWidth x tileWidth at ( tileX * tileWidth, tileY * tileWidth ), so a level coarser than the source repeats each of its texels.
        void BlitSlice( HDRIImage const& image, uint32_t slice, uint32_t tileX, uint32_t tileY, uint32_t tileWidth );

        float*       Pixel( uint32_t x, uint32_t y );
        float const* Pixel( uint32_t x, uint32_t y ) const;

        bool Save( std::string const& path, std::string& message ) const;

        // Reads the file back and checks that it is this image: the dimensions and a sample of the pixels. 
        // The writer is the one place a showcase can be silently wrong - every viewer would show a plausible picture of the wrong thing - so it verifies what it produced rather than assuming it.
        bool Verify( std::string const& path, std::string& message ) const;

    private:

        static constexpr uint32_t NumChannels = 3;

        uint32_t            m_width = 0;
        uint32_t            m_height = 0;
        std::vector<float>  m_rgb;
    };

    //  What a showcase picture is made of
    //-------------------------------------------------------------------------

    struct ShowcaseSource
    {
        // The ingested base level, which is the environment as filtered
        HDRIImage const*                m_pSource = nullptr;

        // The brute-force chain and the table's chain, level 0 first
        std::vector<HDRIImage> const*   m_pReference = nullptr;
        std::vector<HDRIImage> const*   m_pFast = nullptr;

        // The source's own chain, level 0 first: the naive result, which is the mip a roughness selects and the thing the method improves on
        std::vector<HDRIImage> const*   m_pNaive = nullptr;
    };

    // Composes the layout for the map whose slices the images carry
    template< typename TMap >
    void ComposeShowcaseImage( ShowcaseSource const& source, ShowcaseImage& image );

    //-------------------------------------------------------------------------

    extern template void ComposeShowcaseImage< CubeProjection >( ShowcaseSource const&, ShowcaseImage& );
    extern template void ComposeShowcaseImage< TetrahedralProjection >( ShowcaseSource const&, ShowcaseImage& );
}
