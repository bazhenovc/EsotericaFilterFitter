#include "Assert.h"
#include "HDRIImage.h"

#include "MapReferenceFrame.h"
#include "ParallelFor.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace FilterFitter
{
    namespace
    {
        constexpr double kTwoPi = 2.0 * std::numbers::pi_v<double>;

        // The 2x2 hardware footprint around a direction.
        //
        // A tap that stays inside its chart is that chart's neighbour; one that leaves it is resolved through direction space onto whichever texel owns the direction.
        // This is what a hardware cube sampler does, and it is the same resolution the tap splat and the pass-1 footprint use, so a sample means the same thing in all three places. 
        // Nothing here is a cube: a chart is a face for a cubemap and a wedge of the tile for a single-slice tetrahedral map, and the map is what says which.
        //---------------------------------------------------------------------

        struct ProbeFootprint
        {
            static constexpr uint32_t NumTaps = 4;

            uint32_t    m_face[NumTaps] = {};
            uint32_t    m_texelX[NumTaps] = {};
            uint32_t    m_texelY[NumTaps] = {};
            double      m_weight[NumTaps] = {};
            uint32_t    m_count = 0;
        };

        //---------------------------------------------------------------------

        template< typename TMap >
        void ComputeFootprint( double const* pDirection, uint32_t resolution, ProbeFootprint& footprint )
        {
            FF_ASSERT( resolution > 0 );

            uint32_t face = 0;
            double   u = 0.0;
            double   v = 0.0;

            TMap::GetFaceAndUVFromDirection( pDirection, face, u, v );

            // Continuous coordinate whose integer values are texel centres
            double sampleX = 0.0;
            double sampleY = 0.0;
            TMap::GetTexelCoordinateFromUV( &sampleX, &sampleY, u, v, resolution );

            int32_t const floorX = static_cast<int32_t>( std::floor( sampleX ) );
            int32_t const floorY = static_cast<int32_t>( std::floor( sampleY ) );

            double const fractionX = sampleX - static_cast<double>( floorX );
            double const fractionY = sampleY - static_cast<double>( floorY );

            int32_t const lastTexel = static_cast<int32_t>( resolution ) - 1;

            // The common case, and the reason this is fast: an interior footprint is four same-chart reads with no direction round trip. 
            // Only a footprint touching a border needs the round trip, which is the outermost ring of every chart.
            bool const interior = ( floorX >= 0 ) && ( floorX < lastTexel ) && ( floorY >= 0 ) && ( floorY < lastTexel );

            footprint.m_count = 0;

            for ( int32_t stepY = 0; stepY < 2; ++stepY )
            {
                double const weightY = ( stepY == 0 ) ? ( 1.0 - fractionY ) : fractionY;

                for ( int32_t stepX = 0; stepX < 2; ++stepX )
                {
                    double const weightX = ( stepX == 0 ) ? ( 1.0 - fractionX ) : fractionX;
                    double const weight = weightX * weightY;

                    if ( weight == 0.0 )
                    {
                        continue;
                    }

                    uint32_t resolvedFace = face;
                    int32_t  resolvedX = floorX + stepX;
                    int32_t  resolvedY = floorY + stepY;

                    if ( !interior )
                    {
                        // Phantom texel centre, in the sample's own chart and allowed past its edge, then resolved by direction. 
                        // The chart is the one the sample landed in, not whichever chart holds the phantom's coordinate: on a map whose seams are discontinuous those are different directions, and the continuation of this chart is the right one.
                        double phantomU = 0.0;
                        double phantomV = 0.0;
                        TMap::GetTexelCentreUVOutside( &phantomU, &phantomV, floorX + stepX, floorY + stepY, resolution );

                        double phantomDirection[3];
                        TMap::GetDirectionFromUV( phantomDirection, phantomU, phantomV, face );

                        double resolvedU = 0.0;
                        double resolvedV = 0.0;
                        TMap::GetFaceAndUVFromDirection( phantomDirection, resolvedFace, resolvedU, resolvedV );

                        double resolvedContinuousX = 0.0;
                        double resolvedContinuousY = 0.0;
                        TMap::GetTexelCoordinateFromUV( &resolvedContinuousX, &resolvedContinuousY, resolvedU, resolvedV, resolution );

                        // Rounded to nearest rather than floored, for the reason pass 1 rounds: a cube face reads its own coordinate back exactly, a tetrahedral tile goes through an affine map onto irrational vertices and back, and a few ulps below an integer must not become a different texel.
                        resolvedX = static_cast<int32_t>( std::lround( resolvedContinuousX ) );
                        resolvedY = static_cast<int32_t>( std::lround( resolvedContinuousY ) );

                        resolvedX = std::clamp( resolvedX, 0, lastTexel );
                        resolvedY = std::clamp( resolvedY, 0, lastTexel );
                    }

                    uint32_t const tapIndex = footprint.m_count++;

                    footprint.m_face[tapIndex] = resolvedFace;
                    footprint.m_texelX[tapIndex] = static_cast<uint32_t>( resolvedX );
                    footprint.m_texelY[tapIndex] = static_cast<uint32_t>( resolvedY );
                    footprint.m_weight[tapIndex] = weight;
                }
            }
        }
    }

    //-------------------------------------------------------------------------

    void HDRIImage::Create( uint32_t resolution, uint32_t numSlices )
    {
        FF_ASSERT( numSlices > 0 );

        m_resolution = resolution;
        m_numSlices = numSlices;
        m_texels.assign( static_cast<size_t>( numSlices ) * resolution * resolution * NumChannels, 0.0f );
    }

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
    )
    {
        FF_ASSERT( pEquirect != nullptr );
        FF_ASSERT( width > 0 );
        FF_ASSERT( height > 0 );
        FF_ASSERT( numSourceChannels >= 3 );

        out.Create( resolution, TMap::NumSlices );

        // Solid angle of one source texel, before the sin(theta) of its own row
        double const deltaTheta = std::numbers::pi_v<double> / static_cast<double>( height );
        double const deltaPhi = kTwoPi / static_cast<double>( width );

        uint32_t const numWorkers = GetWorkerCount( height );

        // Per-worker scatter targets, then a fixed-order reduction. 
        // Atomics would avoid the memory but the accumulation is float and the sum would then depend on the scheduler, which is exactly what this tool refuses everywhere else.
        std::vector<std::vector<double>> workerTexels( numWorkers );
        std::vector<std::vector<double>> workerWeights( numWorkers );

        for ( uint32_t workerIndex = 0; workerIndex < numWorkers; ++workerIndex )
        {
            workerTexels[workerIndex].assign( out.GetNumFloats(), 0.0 );
            workerWeights[workerIndex].assign( out.GetNumTexels(), 0.0 );
        }

        auto resampleRows = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            std::vector<double>& texels = workerTexels[workerIndex];
            std::vector<double>& weights = workerWeights[workerIndex];

            for ( uint32_t y = begin; y < end; ++y )
            {
                double const theta = ( static_cast<double>( y ) + 0.5 ) * deltaTheta;
                double const sinTheta = std::sin( theta );
                double const cosTheta = std::cos( theta );

                // sin(theta) is zero at the poles, so those rows carry no measure
                double const texelSolidAngle = sinTheta * deltaTheta * deltaPhi;

                if ( texelSolidAngle <= 0.0 )
                {
                    continue;
                }

                for ( uint32_t x = 0; x < width; ++x )
                {
                    double const phi = ( static_cast<double>( x ) + 0.5 ) * deltaPhi;

                    double direction[3];
                    direction[0] = sinTheta * std::sin( phi );
                    direction[1] = cosTheta;
                    direction[2] = sinTheta * std::cos( phi );

                    float const* const pSource = pEquirect + ( ( static_cast<size_t>( y ) * width ) + x ) * numSourceChannels;

                    // A non-finite source texel would poison the whole cube through the normalisation, so it contributes nothing instead
                    if ( !std::isfinite( pSource[0] ) || !std::isfinite( pSource[1] ) || !std::isfinite( pSource[2] ) )
                    {
                        continue;
                    }

                    ProbeFootprint footprint;
                    ComputeFootprint< TMap >( direction, resolution, footprint );

                    for ( uint32_t tap = 0; tap < footprint.m_count; ++tap )
                    {
                        // The map owns the storage index, so a slice-major level of one face per slice and one of four faces in a single slice are addressed by the same line
                        size_t const texelIndex = TMap::GetTexelIndex( footprint.m_face[tap], footprint.m_texelX[tap], footprint.m_texelY[tap], resolution );
                        size_t const texelOffset = texelIndex * HDRIImage::NumChannels;
                        double const weight = footprint.m_weight[tap] * texelSolidAngle;

                        texels[texelOffset + 0] += weight * pSource[0];
                        texels[texelOffset + 1] += weight * pSource[1];
                        texels[texelOffset + 2] += weight * pSource[2];

                        weights[texelIndex] += weight;
                    }
                }
            }
        };

        ParallelForRanges( height, resampleRows );

        auto resolveRange = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            (void) workerIndex;

            for ( uint32_t texelIndex = begin; texelIndex < end; ++texelIndex )
            {
                double weight = 0.0;
                for ( std::vector<double> const& workerWeight : workerWeights )
                {
                    weight += workerWeight[texelIndex];
                }

                float* const pTarget = &out.m_texels[static_cast<size_t>( texelIndex ) * HDRIImage::NumChannels];

                if ( weight <= 0.0 )
                {
                    // No source texel landed here. Only reachable if the source is smaller than the cube, which the caller refuses, so leave it black rather than inventing a value.
                    continue;
                }

                double const inverseWeight = 1.0 / weight;

                for ( uint32_t channel = 0; channel < HDRIImage::NumChannels; ++channel )
                {
                    double sum = 0.0;
                    for ( std::vector<double> const& workerTexel : workerTexels )
                    {
                        sum += workerTexel[( static_cast<size_t>( texelIndex ) * HDRIImage::NumChannels ) + channel];
                    }

                    pTarget[channel] = static_cast<float>( sum * inverseWeight );
                }
            }
        };

        ParallelForRanges( out.GetNumTexels(), resolveRange );
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void BuildSourceChain
    (
        HDRIImage const& base,
        uint32_t numLevels,
        JacobianWeighting weighting,
        std::vector<HDRIImage>& chain
    )
    {
        FF_ASSERT( base.IsInitialized() );
        FF_ASSERT( numLevels >= 1 );

        chain.resize( numLevels );
        chain[0] = base;

        // One channel at a time through the fitted pass 1, so the chain is built by exactly the operator the tables were fitted against.
        // Reimplementing it for three channels at once would be a second thing to keep in step.
        std::vector<double> finer;
        std::vector<double> coarser;

        for ( uint32_t level = 1; level < numLevels; ++level )
        {
            HDRIImage const& fineImage = chain[level - 1];

            uint32_t const coarseResolution = GetChainResolution( base.GetResolution(), level );
            uint32_t const fineResolution = fineImage.GetResolution();

            FF_ASSERT( fineResolution == ( coarseResolution * 2 ) );

            HDRIImage& coarseImage = chain[level];
            coarseImage.Create( coarseResolution, TMap::NumSlices );

            finer.resize( fineImage.GetNumTexels() );
            coarser.resize( coarseImage.GetNumTexels() );

            for ( uint32_t channel = 0; channel < HDRIImage::NumChannels; ++channel )
            {
                for ( uint32_t texelIndex = 0; texelIndex < fineImage.GetNumTexels(); ++texelIndex )
                {
                    finer[texelIndex] = fineImage.m_texels[( static_cast<size_t>( texelIndex ) * HDRIImage::NumChannels ) + channel];
                }

                DownsampleLevel< TMap >( finer, coarseResolution, weighting, coarser );

                for ( uint32_t texelIndex = 0; texelIndex < coarseImage.GetNumTexels(); ++texelIndex )
                {
                    coarseImage.m_texels[( static_cast<size_t>( texelIndex ) * HDRIImage::NumChannels ) + channel] = static_cast<float>( coarser[texelIndex] );
                }
            }
        }
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void SampleImage( HDRIImage const& image, double const* pDirection, double* pOutRgb )
    {
        FF_ASSERT( image.IsInitialized() );

        ProbeFootprint footprint;
        ComputeFootprint< TMap >( pDirection, image.GetResolution(), footprint );

        double accumulated[3] = { 0.0, 0.0, 0.0 };

        for ( uint32_t tap = 0; tap < footprint.m_count; ++tap )
        {
            float const* const pTexel = TexelInMap< TMap >( image, footprint.m_face[tap], footprint.m_texelX[tap], footprint.m_texelY[tap] );
            double const weight = footprint.m_weight[tap];

            accumulated[0] += weight * pTexel[0];
            accumulated[1] += weight * pTexel[1];
            accumulated[2] += weight * pTexel[2];
        }

        pOutRgb[0] = accumulated[0];
        pOutRgb[1] = accumulated[1];
        pOutRgb[2] = accumulated[2];
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void SampleChain( std::vector<HDRIImage> const& chain, double level, double const* pDirection, double* pOutRgb )
    {
        FF_ASSERT( !chain.empty() );
        FF_ASSERT( chain[0].IsInitialized() );

        double const maximumLevel = static_cast<double>( chain.size() - 1 );
        double const clampedLevel = std::clamp( level, 0.0, maximumLevel );

        uint32_t const level0 = static_cast<uint32_t>( std::floor( clampedLevel ) );
        uint32_t const level1 = ( ( level0 + 1 ) < chain.size() ) ? ( level0 + 1 ) : level0;

        double const fraction1 = clampedLevel - static_cast<double>( level0 );
        double const fraction0 = 1.0 - fraction1;

        double sample0[3] = { 0.0, 0.0, 0.0 };
        double sample1[3] = { 0.0, 0.0, 0.0 };

        SampleImage< TMap >( chain[level0], pDirection, sample0 );

        if ( level1 != level0 )
        {
            SampleImage< TMap >( chain[level1], pDirection, sample1 );
        }

        for ( uint32_t channel = 0; channel < 3; ++channel )
        {
            pOutRgb[channel] = ( sample0[channel] * fraction0 ) + ( sample1[channel] * fraction1 );
        }
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void GetImageStatistics( HDRIImage const& image, double& meanLuminance, double& numNonFinite )
    {
        FF_ASSERT( image.IsInitialized() );

        MapReferenceFrame< TMap > frame;
        frame.Initialize( image.GetResolution() );

        double weightedSum = 0.0;
        double totalWeight = 0.0;
        double nonFinite = 0.0;

        for ( uint32_t texelIndex = 0; texelIndex < image.GetNumTexels(); ++texelIndex )
        {
            float const* const pTexel = &image.m_texels[static_cast<size_t>( texelIndex ) * HDRIImage::NumChannels];

            if ( !std::isfinite( pTexel[0] ) || !std::isfinite( pTexel[1] ) || !std::isfinite( pTexel[2] ) )
            {
                nonFinite += 1.0;
                continue;
            }

            double const solidAngle = frame.GetTexel( texelIndex ).m_solidAngle;
            double const luminance = ( 0.2126 * pTexel[0] ) + ( 0.7152 * pTexel[1] ) + ( 0.0722 * pTexel[2] );

            weightedSum += luminance * solidAngle;
            totalWeight += solidAngle;
        }

        meanLuminance = ( totalWeight > 0.0 ) ? ( weightedSum / totalWeight ) : 0.0;
        numNonFinite = nonFinite / static_cast<double>( image.GetNumTexels() );
    }

    //-------------------------------------------------------------------------

    template void ResampleEquirectToMap< CubeProjection >( float const*, uint32_t, uint32_t, uint32_t, uint32_t, HDRIImage& );
    template void ResampleEquirectToMap< TetrahedralProjection >( float const*, uint32_t, uint32_t, uint32_t, uint32_t, HDRIImage& );

    template void BuildSourceChain< CubeProjection >( HDRIImage const&, uint32_t, JacobianWeighting, std::vector<HDRIImage>& );
    template void BuildSourceChain< TetrahedralProjection >( HDRIImage const&, uint32_t, JacobianWeighting, std::vector<HDRIImage>& );

    template void SampleChain< CubeProjection >( std::vector<HDRIImage> const&, double, double const*, double* );
    template void SampleChain< TetrahedralProjection >( std::vector<HDRIImage> const&, double, double const*, double* );

    template void SampleImage< CubeProjection >( HDRIImage const&, double const*, double* );
    template void SampleImage< TetrahedralProjection >( HDRIImage const&, double const*, double* );

    template void GetImageStatistics< CubeProjection >( HDRIImage const&, double&, double& );
    template void GetImageStatistics< TetrahedralProjection >( HDRIImage const&, double&, double& );
}
