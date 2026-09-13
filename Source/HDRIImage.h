#pragma once

#include <cstdint>
#include <vector>

#include "BsplineRecurrence.h"

// HDRI probe levels for radiance-domain validation
//-------------------------------------------------------------------------
// The fit is scored on a preimage, which says how well the taps reproduce the lobe and says nothing about what that does to an environment.
// This is the other half: run real HDRIs through both the reference convolution and the  table-driven one and measure the difference in radiance.
//
// Layout is SLICE-major, then row, then column, then channel:
//
//      index = ( ( slice * resolution + y ) * resolution + x ) * 3 + channel
//
// RGB, no alpha, no padding. 
// Float rather than double because the source is half-float EXR and because a validation buffer is not where precision is scarce - the convolution is dominated by its own sampling error.
//
// A slice is one texture, and how many a level has is the MAP's business: a cubemap is six, one face each, and a single-slice tetrahedral map is one  holding all four faces.
// The container therefore stores a slice count and knows nothing else about the map; everything that needs to go from a texel to a direction is templated on the map and lives in the functions below.
//
//  EQUIRECT CONVENTION
//
// The engine has no equirect path, so this fixes one rather than inheriting it.
// For texel ( x, y ) of a width x height image:
//
//      phi   = 2 * pi * ( x + 0.5 ) / width          [0, 2pi)
//      theta =     pi * ( y + 0.5 ) / height         [0, pi], 0 is +Y
//
//      dir   = ( sin(theta) * sin(phi), cos(theta), sin(theta) * cos(phi) )
//
// so phi = 0 looks down +Z - the centre of the image is the centre of the +Z face - and phi increases towards +X.
// Texel solid angle is sin(theta) * dtheta * dphi, which is what makes a plain average over the image wrong and a solid-angle weighted one right.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    struct HDRIImage
    {
        static constexpr uint32_t NumChannels = 3;

        // The engine's source chain is 128 down to 1, which is eight levels.
        // The binary format stores the level table inline, so this is also its bound.
        static constexpr uint32_t MaxLevels = 8;

        uint32_t                m_resolution = 0;
        uint32_t                m_numSlices = 0;
        std::vector<float>      m_texels;

        void Create( uint32_t resolution, uint32_t numSlices );

        inline bool     IsInitialized() const { return m_resolution != 0; }
        inline uint32_t GetResolution() const { return m_resolution; }
        inline uint32_t GetNumSlices() const { return m_numSlices; }
        inline uint32_t GetNumTexels() const { return m_numSlices * m_resolution * m_resolution; }
        inline size_t   GetNumFloats() const { return static_cast<size_t>( GetNumTexels() ) * NumChannels; }

        inline size_t GetTexelOffset( uint32_t slice, uint32_t texelX, uint32_t texelY ) const
        {
            return ( ( ( static_cast<size_t>( slice ) * m_resolution ) + texelY ) * m_resolution + texelX ) * NumChannels;
        }

        inline float* Texel( uint32_t slice, uint32_t texelX, uint32_t texelY )
        {
            return &m_texels[GetTexelOffset( slice, texelX, texelY )];
        }

        inline float const* Texel( uint32_t slice, uint32_t texelX, uint32_t texelY ) const
        {
            return &m_texels[GetTexelOffset( slice, texelX, texelY )];
        }
    };

    // A chart's texel, addressed the way the map addresses it
    //-------------------------------------------------------------------------
    // A chart is a face for a cubemap and a wedge of the single tile for a tetrahedral map.
    // The two coincide only because a cubemap gives every face its own slice: on a single-slice map all four charts share one slice, and  using the chart as a slice index reads three slices past the end of the image. 
    // Everything that resolves a sample to a chart goes through here.
    //-------------------------------------------------------------------------

    template< typename TMap >
    inline float const* TexelInMap( HDRIImage const& image, uint32_t face, uint32_t texelX, uint32_t texelY )
    {
        size_t const texelIndex = TMap::GetTexelIndex( face, texelX, texelY, image.GetResolution() );

        return &image.m_texels[texelIndex * HDRIImage::NumChannels];
    }

    //-------------------------------------------------------------------------

    // Resolution a level's slice has, for a chain whose level 0 is baseResolution.
    // Levels are 128, 64, 32, 16, 8, 4, 2, 1 - the engine's chain, so a tap level means the same thing here as it does there.
    inline uint32_t GetChainResolution( uint32_t baseResolution, uint32_t level )
    {
        uint32_t const resolution = baseResolution >> level;

        return ( resolution > 0 ) ? resolution : 1;
    }

    // Equirect to a probe level, on whichever map
    //-------------------------------------------------------------------------
    // A scatter rather than a gather, and that choice is the whole quality argument. 
    // A 4K equirect is 8.4M texels against 98k in a 128 cube, so a gather would need about 85 well-placed samples per cube texel to avoid aliasing the source, and a fixed supersample grid cannot know how many.
    // Scattering visits every source texel exactly once with its own solid angle as the weight, so nothing is dropped and nothing is counted twice.
    //
    // The cost is a bilinear spread of at most one probe texel, which is below the resolution of anything measured here.
    //
    //  Each worker accumulates into its own buffers and the results are summed in lane order, so the output does not depend on the scheduler.
    //-------------------------------------------------------------------------

    template< typename TMap >
    void ResampleEquirectToMap
    (
        float const* pEquirect,
        uint32_t width,
        uint32_t height,
        uint32_t numSourceChannels,
        uint32_t resolution,
        HDRIImage& out
    );

    // Source mip chain
    //-------------------------------------------------------------------------
    // Pass 1 as the tool models it, applied per channel: the paper's 4-tap quadratic b-spline with the Jacobian weighting, including cross-face taps.
    // This is what the fitted tables assume their taps read, so it is what the table-driven convolution has to sample.
    //-------------------------------------------------------------------------

    template< typename TMap >
    void BuildSourceChain
    (
        HDRIImage const& base,
        uint32_t numLevels,
        JacobianWeighting weighting,
        std::vector<HDRIImage>& chain
    );

    // Sampling
    //-------------------------------------------------------------------------
    // Hardware-style bilinear on a map: four taps around the sample, each resolved through direction space so a footprint that leaves its chart lands on whichever texel actually holds that direction. 
    // level is a fractional source mip, clamped to the chain, and the two adjacent levels are blended.
    //-------------------------------------------------------------------------

    template< typename TMap >
    void SampleChain( std::vector<HDRIImage> const& chain, double level, double const* pDirection, double* pOutRgb );

    // One level, bilinear, no mip blend.
    template< typename TMap >
    void SampleImage( HDRIImage const& image, double const* pDirection, double* pOutRgb );

    //  Statistics, for sanity rather than for scoring
    //-------------------------------------------------------------------------

    // Solid-angle weighted mean luminance, and the fraction of non-finite texels.
    // A probe that is 80% NaN still convolves to something, and it is better to find that out at ingest than to explain a validation number later.
    template< typename TMap >
    void GetImageStatistics( HDRIImage const& image, double& meanLuminance, double& numNonFinite );

    //-------------------------------------------------------------------------

    extern template void ResampleEquirectToMap< CubeProjection >( float const*, uint32_t, uint32_t, uint32_t, uint32_t, HDRIImage& );
    extern template void ResampleEquirectToMap< TetrahedralProjection >( float const*, uint32_t, uint32_t, uint32_t, uint32_t, HDRIImage& );

    extern template void BuildSourceChain< CubeProjection >( HDRIImage const&, uint32_t, JacobianWeighting, std::vector<HDRIImage>& );
    extern template void BuildSourceChain< TetrahedralProjection >( HDRIImage const&, uint32_t, JacobianWeighting, std::vector<HDRIImage>& );

    extern template void SampleChain< CubeProjection >( std::vector<HDRIImage> const&, double, double const*, double* );
    extern template void SampleChain< TetrahedralProjection >( std::vector<HDRIImage> const&, double, double const*, double* );

    extern template void SampleImage< CubeProjection >( HDRIImage const&, double const*, double* );
    extern template void SampleImage< TetrahedralProjection >( HDRIImage const&, double const*, double* );

    extern template void GetImageStatistics< CubeProjection >( HDRIImage const&, double&, double& );
    extern template void GetImageStatistics< TetrahedralProjection >( HDRIImage const&, double&, double& );
}
