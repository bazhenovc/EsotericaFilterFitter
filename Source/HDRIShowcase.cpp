#include "Assert.h"
#include "HDRIShowcase.h"

#include "MapProjection.h"
#include "TetrahedralProjection.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "TinyEXR/tinyexr.h"

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    void ShowcaseImage::Create( uint32_t width, uint32_t height )
    {
        FF_ASSERT( width > 0 );
        FF_ASSERT( height > 0 );

        m_width = width;
        m_height = height;

        m_rgb.assign( static_cast<size_t>( width ) * height * NumChannels, 0.0f );
    }

    //-------------------------------------------------------------------------

    float* ShowcaseImage::Pixel( uint32_t x, uint32_t y )
    {
        FF_ASSERT( x < m_width );
        FF_ASSERT( y < m_height );

        return &m_rgb[( ( static_cast<size_t>( y ) * m_width ) + x ) * NumChannels];
    }

    float const* ShowcaseImage::Pixel( uint32_t x, uint32_t y ) const
    {
        FF_ASSERT( x < m_width );
        FF_ASSERT( y < m_height );

        return &m_rgb[( ( static_cast<size_t>( y ) * m_width ) + x ) * NumChannels];
    }

    //-------------------------------------------------------------------------

    void ShowcaseImage::BlitSlice( HDRIImage const& image, uint32_t slice, uint32_t tileX, uint32_t tileY, uint32_t tileWidth )
    {
        FF_ASSERT( image.IsInitialized() );
        FF_ASSERT( slice < image.GetNumSlices() );

        uint32_t const sourceResolution = image.GetResolution();
        uint32_t const destinationX = tileX * tileWidth;
        uint32_t const destinationY = tileY * tileWidth;

        FF_ASSERT( ( destinationX + tileWidth ) <= m_width );
        FF_ASSERT( ( destinationY + tileWidth ) <= m_height );

        for ( uint32_t y = 0; y < tileWidth; ++y )
        {
            // Integer division rather than a rounded float: a destination pixel takes the source texel that contains it, so a level at 1/8 resolution gives 8x8 flat blocks with no sampling filter of any kind.
            uint32_t const sourceY = ( y * sourceResolution ) / tileWidth;

            for ( uint32_t x = 0; x < tileWidth; ++x )
            {
                uint32_t const sourceX = ( x * sourceResolution ) / tileWidth;

                float const* const pTexel = image.Texel( slice, sourceX, sourceY );
                float* const       pPixel = Pixel( destinationX + x, destinationY + y );

                pPixel[0] = pTexel[0];
                pPixel[1] = pTexel[1];
                pPixel[2] = pTexel[2];
            }
        }
    }

    //-------------------------------------------------------------------------

    bool ShowcaseImage::Save( std::string const& path, std::string& message ) const
    {
        FF_ASSERT( IsInitialized() );

        char const* pError = nullptr;

        // 32-bit float and three channels: what the pipeline holds, so nothing is clipped and nothing is rounded on the way out
        int const result = SaveEXR( m_rgb.data(), static_cast<int>( m_width ), static_cast<int>( m_height ), static_cast<int>( NumChannels ), 0, path.c_str(), &pError );

        if ( result == TINYEXR_SUCCESS )
        {
            return true;
        }

        message = "cannot write " + path;

        if ( pError != nullptr )
        {
            message += ": ";
            message += pError;
            FreeEXRErrorMessage( pError );
        }

        return false;
    }

    //-------------------------------------------------------------------------

    bool ShowcaseImage::Verify( std::string const& path, std::string& message ) const
    {
        FF_ASSERT( IsInitialized() );

        float* pRgba = nullptr;
        int    width = 0;
        int    height = 0;

        char const* pError = nullptr;

        if ( LoadEXR( &pRgba, &width, &height, path.c_str(), &pError ) != TINYEXR_SUCCESS )
        {
            message = "cannot read back " + path;

            if ( pError != nullptr )
            {
                message += ": ";
                message += pError;
                FreeEXRErrorMessage( pError );
            }

            return false;
        }

        bool ok = ( width == static_cast<int>( m_width ) ) && ( height == static_cast<int>( m_height ) );

        if ( !ok )
        {
            char buffer[160] = {};
            std::snprintf( buffer, sizeof( buffer ), "%s came back as %dx%d, not %ux%u", path.c_str(), width, height, m_width, m_height );
            message = buffer;
        }

        // Four corners and the middle of every tile boundary region would be thorough; the corners and the centre are enough to catch a transposed, offset or empty image, which are the ways this can go wrong.
        uint32_t const sampleX[] = { 0, m_width / 2, m_width - 1 };
        uint32_t const sampleY[] = { 0, m_height / 2, m_height - 1 };

        for ( uint32_t y = 0; ok && ( y < 3 ); ++y )
        {
            for ( uint32_t x = 0; ok && ( x < 3 ); ++x )
            {
                float const* const pExpected = Pixel( sampleX[x], sampleY[y] );
                float const* const pActual = &pRgba[( ( static_cast<size_t>( sampleY[y] ) * m_width ) + sampleX[x] ) * 4];

                for ( uint32_t channel = 0; channel < NumChannels; ++channel )
                {
                    // Written as 32-bit float and compressed losslessly, so this is an equality and not a tolerance
                    if ( pActual[channel] != pExpected[channel] )
                    {
                        char buffer[192] = {};
                        std::snprintf( buffer, sizeof( buffer ), "%s differs at (%u, %u) channel %u: %.9g against %.9g", path.c_str(), sampleX[x], sampleY[y], channel, pActual[channel], pExpected[channel] );
                        message = buffer;
                        ok = false;
                        break;
                    }
                }
            }
        }

        std::free( pRgba );

        return ok;
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void ComposeShowcaseImage( ShowcaseSource const& source, ShowcaseImage& image )
    {
        FF_ASSERT( source.m_pSource != nullptr );
        FF_ASSERT( source.m_pReference != nullptr );
        FF_ASSERT( source.m_pFast != nullptr );
        FF_ASSERT( source.m_pNaive != nullptr );
        FF_ASSERT( source.m_pSource->IsInitialized() );

        uint32_t const baseWidth = source.m_pSource->GetResolution();
        uint32_t const numLevels = static_cast<uint32_t>( source.m_pReference->size() );

        FF_ASSERT( numLevels > 0 );
        FF_ASSERT( source.m_pFast->size() == numLevels );
        FF_ASSERT( source.m_pNaive->size() >= numLevels );

        if ( TMap::NumSlices == 1 )
        {
            // A single-slice map has one texture per level, so the levels go across the picture and each result gets a row. 
            // The source takes the first tile of its row and the rest stays black: there is nothing else at the source's level to put beside it.
            image.Create( numLevels * baseWidth, 4 * baseWidth );

            image.BlitSlice( *source.m_pSource, 0, 0, 0, baseWidth );

            for ( uint32_t level = 0; level < numLevels; ++level )
            {
                image.BlitSlice( ( *source.m_pReference )[level], 0, level, 1, baseWidth );
                image.BlitSlice( ( *source.m_pFast )[level], 0, level, 2, baseWidth );
                image.BlitSlice( ( *source.m_pNaive )[level], 0, level, 3, baseWidth );
            }

            return;
        }

        // One row per level, the three results across it - brute force, then the table's, then the naive mip - each with the map's slices across its own block: comparing the results at a level is reading across, and comparing levels is reading down.
        // The source has no filtered counterpart to sit beside, so its row leaves the other two blocks black rather than repeating itself in them.
        image.Create( 3 * TMap::NumSlices * baseWidth, ( 1 + numLevels ) * baseWidth );

        for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
        {
            image.BlitSlice( *source.m_pSource, slice, slice, 0, baseWidth );
        }

        for ( uint32_t level = 0; level < numLevels; ++level )
        {
            uint32_t const row = 1 + level;

            for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
            {
                image.BlitSlice( ( *source.m_pReference )[level], slice, slice, row, baseWidth );
                image.BlitSlice( ( *source.m_pFast )[level], slice, TMap::NumSlices + slice, row, baseWidth );
                image.BlitSlice( ( *source.m_pNaive )[level], slice, ( 2 * TMap::NumSlices ) + slice, row, baseWidth );
            }
        }
    }

    //-------------------------------------------------------------------------

    template void ComposeShowcaseImage< CubeProjection >( ShowcaseSource const&, ShowcaseImage& );
    template void ComposeShowcaseImage< TetrahedralProjection >( ShowcaseSource const&, ShowcaseImage& );
}
