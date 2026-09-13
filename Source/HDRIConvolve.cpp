#include "Assert.h"
#include "HDRIConvolve.h"

#include "MapProjection.h"
#include "MapReferenceFrame.h"
#include "TetrahedralProjection.h"
#include "ParallelFor.h"
#include "ProfileGGX.h"
#include "TableHarness.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace FilterFitter
{
    namespace
    {
        // The engine's Hammersley sequence, from PBR.esh.
        //
        // Matched rather than replaced with a better sequence, because the point of the reference is to be the number the runtime would have produced.
        // A different low-discrepancy sequence would converge to the same value and no longer be comparable sample for sample.
        //---------------------------------------------------------------------

        double RadicalInverseVdc( uint32_t bits )
        {
            bits = ( bits << 16 ) | ( bits >> 16 );
            bits = ( ( bits & 0x55555555u ) << 1 ) | ( ( bits & 0xAAAAAAAAu ) >> 1 );
            bits = ( ( bits & 0x33333333u ) << 2 ) | ( ( bits & 0xCCCCCCCCu ) >> 2 );
            bits = ( ( bits & 0x0F0F0F0Fu ) << 4 ) | ( ( bits & 0xF0F0F0F0u ) >> 4 );
            bits = ( ( bits & 0x00FF00FFu ) << 8 ) | ( ( bits & 0xFF00FF00u ) >> 8 );

            return static_cast<double>( bits ) * 2.3283064365386963e-10;
        }

        //---------------------------------------------------------------------

        double Dot3( double const* pLeft, double const* pRight )
        {
            return ( pLeft[0] * pRight[0] ) + ( pLeft[1] * pRight[1] ) + ( pLeft[2] * pRight[2] );
        }

        void Normalize3( double* pVector )
        {
            double const length = std::sqrt( Dot3( pVector, pVector ) );

            if ( length > 0.0 )
            {
                double const inverseLength = 1.0 / length;

                pVector[0] *= inverseLength;
                pVector[1] *= inverseLength;
                pVector[2] *= inverseLength;
            }
        }

        // GGX half-vector importance sampling.
        //
        // PBR.esh's ImportanceSampleGGX, with the roughness parameterization collapsed: the engine takes roughness and computes a = roughness^2 internally, and the profile's width is that same a, so the engine's a * a is this alpha * alpha.
        // Passing the width directly removes a square root at both ends and one place for the two to disagree.
        //
        // This is the one part of the reference that is not profile-generic.
        // Importance sampling needs the NDF's own half-vector distribution, and a Beckmann reference would need Beckmann's formula here rather than a branch. Only GGX ships, so only GGX is implemented, and the driver refuses the combination it cannot do rather than approximating it.
        //---------------------------------------------------------------------

        void SampleGGXHalfVector( double alpha, double xi0, double xi1, double const* pNormal, double* pOutHalf )
        {
            double const a = alpha;
            double const phi = 2.0 * std::numbers::pi_v<double> *xi0;
            double const cosTheta = std::sqrt( ( 1.0 - xi1 ) / ( 1.0 + ( ( a * a ) - 1.0 ) * xi1 ) );
            double const sinTheta = std::sqrt( std::max( 0.0, 1.0 - ( cosTheta * cosTheta ) ) );

            double const h[3] = { std::cos( phi ) * sinTheta, std::sin( phi ) * sinTheta, cosTheta };

            // The pole the tangent frame is built against, as a float3 literal in the shader: (0,0,1) unless the normal is already that, in which case the cross product would vanish and (1,0,0) is used instead.
            double axis[3] = { 0.0, 0.0, 1.0 };
            if ( std::fabs( pNormal[2] ) >= 0.999 )
            {
                axis[0] = 1.0;
                axis[2] = 0.0;
            }

            double tangent[3];
            tangent[0] = ( axis[1] * pNormal[2] ) - ( axis[2] * pNormal[1] );
            tangent[1] = ( axis[2] * pNormal[0] ) - ( axis[0] * pNormal[2] );
            tangent[2] = ( axis[0] * pNormal[1] ) - ( axis[1] * pNormal[0] );
            Normalize3( tangent );

            double bitangent[3];
            bitangent[0] = ( pNormal[1] * tangent[2] ) - ( pNormal[2] * tangent[1] );
            bitangent[1] = ( pNormal[2] * tangent[0] ) - ( pNormal[0] * tangent[2] );
            bitangent[2] = ( pNormal[0] * tangent[1] ) - ( pNormal[1] * tangent[0] );
            Normalize3( bitangent );

            pOutHalf[0] = ( tangent[0] * h[0] ) + ( bitangent[0] * h[1] ) + ( pNormal[0] * h[2] );
            pOutHalf[1] = ( tangent[1] * h[0] ) + ( bitangent[1] * h[1] ) + ( pNormal[1] * h[2] );
            pOutHalf[2] = ( tangent[2] * h[0] ) + ( bitangent[2] * h[1] ) + ( pNormal[2] * h[2] );

            Normalize3( pOutHalf );
        }

        //---------------------------------------------------------------------

        void GetTexelDirection( double* pOutDirection, uint32_t face, uint32_t texelX, uint32_t texelY, uint32_t resolution )
        {
            double const inverseResolution = 1.0 / static_cast<double>( resolution );

            double const u = ( ( 2.0 * static_cast<double>( texelX ) ) + 1.0 ) * inverseResolution - 1.0;
            double const v = -( ( 2.0 * static_cast<double>( texelY ) ) + 1.0 ) * inverseResolution + 1.0;

            GetCubeFaceDirection( pOutDirection, u, v, face );
            Normalize3( pOutDirection );
        }
    }

    //-------------------------------------------------------------------------

    double GetLuminance( float const* pRgb )
    {
        return ( 0.2126 * pRgb[0] ) + ( 0.7152 * pRgb[1] ) + ( 0.0722 * pRgb[2] );
    }

    //-------------------------------------------------------------------------

    template< typename TMap, typename TProfile >
    void ConvolveReference
    (
        std::vector<HDRIImage> const& chain,
        TProfile const& profile,
        uint32_t level,
        uint32_t sampleCount,
        HDRIImage& out
    )
    {
        FF_ASSERT( !chain.empty() );
        FF_ASSERT( sampleCount > 0 );

        uint32_t const baseWidth = chain[0].GetResolution();
        uint32_t const resolution = GetChainResolution( baseWidth, level );

        out.Create( resolution, TMap::NumSlices );

        double const alpha = profile.GetWidth( level );

        double const inverseSampleCount = 1.0 / static_cast<double>( sampleCount );

        // The output is walked by SLICE, which is what a level is made of - six slices for a cubemap and one for a single-slice tetrahedral map.
        // A slice is a face only on a cubemap, where the two happen to coincide.
        uint32_t const texelsPerSlice = resolution * resolution;

        auto convolveRange = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            (void) workerIndex;

            for ( uint32_t texelIndex = begin; texelIndex < end; ++texelIndex )
            {
                uint32_t const slice = texelIndex / texelsPerSlice;
                uint32_t const within = texelIndex % texelsPerSlice;
                uint32_t const texelY = within / resolution;
                uint32_t const texelX = within % resolution;

                double normal[3];
                TMap::GetDirection( normal, slice, texelX, texelY, resolution );

                // A map's texel -> direction answer is the UNNORMALIZED face  coordinate vector, and its scale is the map's own: a cube face's vector has a component of exactly 1, so its length runs from 1 at a face centre to sqrt(3) at a corner, and a tetrahedral face's runs from 1 at a corner down to 1/3 at a face centre. 
                // This is a surface normal, and the engine normalizes the one it passes, so this has to as well - the NDF is evaluated at dot( n, h ), and an unnormalized n evaluates it at the wrong place. 
                // A cube survives it by accident (a longer n pushes the dot past 1, and the clamp then lands on the NDF's peak, which is sharp rather than wrong);
                // a tetrahedron, whose vectors are SHORTER than 1, does not.
                Normalize3( normal );

                float* const pOut = out.Texel( slice, texelX, texelY );

                // Zero width is a mirror, which is the unfiltered source under the output direction. 
                // The engine reaches the same place by skipping mip selection entirely at roughness zero.
                if ( alpha <= 0.0 )
                {
                    double rgb[3] = { 0.0, 0.0, 0.0 };
                    SampleChain< TMap >( chain, 0.0, normal, rgb );

                    pOut[0] = static_cast<float>( rgb[0] );
                    pOut[1] = static_cast<float>( rgb[1] );
                    pOut[2] = static_cast<float>( rgb[2] );

                    continue;
                }

                double accumulated[3] = { 0.0, 0.0, 0.0 };
                double weightSum = 0.0;

                for ( uint32_t sample = 0; sample < sampleCount; ++sample )
                {
                    double const xi0 = static_cast<double>( sample ) * inverseSampleCount;
                    double const xi1 = RadicalInverseVdc( sample );

                    double halfVector[3];
                    SampleGGXHalfVector( alpha, xi0, xi1, normal, halfVector );

                    double const dotNH = std::clamp( Dot3( normal, halfVector ), 0.0, 1.0 );

                    // L = normalize( 2 * dot( view, H ) * H - view ), with view = normal
                    double light[3];
                    light[0] = ( 2.0 * dotNH * halfVector[0] ) - normal[0];
                    light[1] = ( 2.0 * dotNH * halfVector[1] ) - normal[1];
                    light[2] = ( 2.0 * dotNH * halfVector[2] ) - normal[2];
                    Normalize3( light );

                    double const dotNL = std::clamp( Dot3( normal, light ), 0.0, 1.0 );

                    if ( dotNL <= 0.0 )
                    {
                        continue;
                    }

                    double const ndf = TProfile::EvaluateNDF( dotNH, alpha );

                    // view == normal, so dotVH is dotNH
                    double const pdf = ( ndf * dotNH ) / std::max( 4.0 * dotNH, 0.0001 );

                    double const sourceSample = 1.0 / std::max( static_cast<double>( sampleCount ) * pdf, 0.0001 );

                    // The sample's own footprint expressed as a source mip, which is what stops a narrow lobe from aliasing the source instead of averaging it.
                    // The map answers what one texel covers, because the two do not agree: a cube texel is 4J/R^2 and a single-slice tetrahedral texel is J/R^2,
                    // and the two Jacobians vary over 5.2:1 against 27:1.
                    double const inverseSourceTexelSolidAngle = TMap::GetInverseTexelSolidAngle( light, baseWidth );

                    double const sourceMipLevel = 0.5 * std::log2( sourceSample * inverseSourceTexelSolidAngle );

                    double radiance[3] = { 0.0, 0.0, 0.0 };
                    SampleChain< TMap >( chain, sourceMipLevel, light, radiance );

                    accumulated[0] += dotNL * radiance[0];
                    accumulated[1] += dotNL * radiance[1];
                    accumulated[2] += dotNL * radiance[2];

                    weightSum += dotNL;
                }

                if ( weightSum > 0.0 )
                {
                    double const inverseWeight = 1.0 / weightSum;

                    pOut[0] = static_cast<float>( accumulated[0] * inverseWeight );
                    pOut[1] = static_cast<float>( accumulated[1] * inverseWeight );
                    pOut[2] = static_cast<float>( accumulated[2] * inverseWeight );
                }
            }
        };

        ParallelForRanges( out.GetNumTexels(), convolveRange );
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void ConvolveTable
    (
        std::vector<HDRIImage> const& chain,
        CoefficientTable const& table,
        uint32_t level,
        uint32_t baseWidth,
        HDRIImage& out
    )
    {
        FF_ASSERT( !chain.empty() );

        uint32_t const resolution = GetChainResolution( baseWidth, level );

        out.Create( resolution, TMap::NumSlices );

        // The table's own tap count is the number of TableTap entries one output texel produces - it is what the parameter array is sized from - so it is read rather than counted, and the two cannot drift.
        uint32_t const numTaps = table.GetTapCount();

        // One tap list per worker, and the same per-worker index on every call, so no two lanes write to the same vector.
        uint32_t const numWorkers = GetWorkerCount( resolution * resolution * TMap::NumSlices );

        std::vector<std::vector<TableTap>> workerTaps( numWorkers );

        for ( uint32_t workerIndex = 0; workerIndex < numWorkers; ++workerIndex )
        {
            workerTaps[workerIndex].resize( numTaps );
        }

        uint32_t const texelsPerSlice = resolution * resolution;

        auto convolveRange = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            std::vector<TableTap>& taps = workerTaps[workerIndex];

            for ( uint32_t texelIndex = begin; texelIndex < end; ++texelIndex )
            {
                uint32_t const slice = texelIndex / texelsPerSlice;
                uint32_t const within = texelIndex % texelsPerSlice;
                uint32_t const texelY = within / resolution;
                uint32_t const texelX = within % resolution;

                BuildTableTaps< TMap >( table, level, slice, texelX, texelY, baseWidth, taps );

                double accumulated[3] = { 0.0, 0.0, 0.0 };
                double weightSum = 0.0;

                for ( TableTap const& tap : taps )
                {
                    double direction[3] = { tap.m_direction[0], tap.m_direction[1], tap.m_direction[2] };
                    Normalize3( direction );

                    double radiance[3] = { 0.0, 0.0, 0.0 };
                    SampleChain< TMap >( chain, tap.m_level, direction, radiance );

                    accumulated[0] += tap.m_weight * radiance[0];
                    accumulated[1] += tap.m_weight * radiance[1];
                    accumulated[2] += tap.m_weight * radiance[2];

                    weightSum += tap.m_weight;
                }

                float* const pOut = out.Texel( slice, texelX, texelY );

                // The runtime divides by the weight sum, so a non-positive one is a degenerate row. 
                // Black rather than a division, and the comparison then reports it rather than propagating a NaN.
                if ( weightSum > 0.0 )
                {
                    double const inverseWeight = 1.0 / weightSum;

                    pOut[0] = static_cast<float>( accumulated[0] * inverseWeight );
                    pOut[1] = static_cast<float>( accumulated[1] * inverseWeight );
                    pOut[2] = static_cast<float>( accumulated[2] * inverseWeight );
                }
            }
        };

        ParallelForRanges( out.GetNumTexels(), convolveRange );
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void CompareConvolutions( HDRIImage const& reference, HDRIImage const& approximate, ConvolutionError& error )
    {
        error = ConvolutionError();

        FF_ASSERT( reference.IsInitialized() );
        FF_ASSERT( approximate.IsInitialized() );
        FF_ASSERT( reference.GetResolution() == approximate.GetResolution() );

        MapReferenceFrame< TMap > frame;
        frame.Initialize( reference.GetResolution() );

        double absoluteSum = 0.0;
        double referenceSum = 0.0;
        double squaredSum = 0.0;
        double referenceSquared = 0.0;
        double weightSum = 0.0;
        double approximateWeighted = 0.0;

        for ( uint32_t texelIndex = 0; texelIndex < reference.GetNumTexels(); ++texelIndex )
        {
            float const* const pReference = &reference.m_texels[static_cast<size_t>( texelIndex ) * HDRIImage::NumChannels];
            float const* const pApproximate = &approximate.m_texels[static_cast<size_t>( texelIndex ) * HDRIImage::NumChannels];

            double const solidAngle = frame.GetTexel( texelIndex ).m_solidAngle;

            double const referenceLuminance = GetLuminance( pReference );
            double const approximateLuminance = GetLuminance( pApproximate );

            double const difference = referenceLuminance - approximateLuminance;

            absoluteSum += std::fabs( difference ) * solidAngle;
            referenceSum += std::fabs( referenceLuminance ) * solidAngle;

            squaredSum += difference * difference * solidAngle;
            referenceSquared += referenceLuminance * referenceLuminance * solidAngle;

            approximateWeighted += approximateLuminance * solidAngle;
            weightSum += solidAngle;

            if ( std::fabs( difference ) > error.m_maxAbsolute )
            {
                error.m_maxAbsolute = std::fabs( difference );
            }

            if ( referenceLuminance > error.m_referencePeak )
            {
                error.m_referencePeak = referenceLuminance;
            }

            if ( approximateLuminance > error.m_approximatePeak )
            {
                error.m_approximatePeak = approximateLuminance;
            }
        }

        if ( referenceSum > 0.0 )
        {
            error.m_relativeL1 = absoluteSum / referenceSum;
        }

        if ( referenceSquared > 0.0 )
        {
            error.m_relativeRms = std::sqrt( squaredSum / referenceSquared );
        }

        if ( weightSum > 0.0 )
        {
            error.m_referenceMean = referenceSum / weightSum;
            error.m_approximateMean = approximateWeighted / weightSum;
        }
    }

    //-------------------------------------------------------------------------

    template void ConvolveReference< CubeProjection, ProfileGGX >( std::vector<HDRIImage> const&, ProfileGGX const&, uint32_t, uint32_t, HDRIImage& );
    template void ConvolveReference< TetrahedralProjection, ProfileGGX >( std::vector<HDRIImage> const&, ProfileGGX const&, uint32_t, uint32_t, HDRIImage& );

    template void ConvolveTable< CubeProjection >( std::vector<HDRIImage> const&, CoefficientTable const&, uint32_t, uint32_t, HDRIImage& );
    template void ConvolveTable< TetrahedralProjection >( std::vector<HDRIImage> const&, CoefficientTable const&, uint32_t, uint32_t, HDRIImage& );

    template void CompareConvolutions< CubeProjection >( HDRIImage const&, HDRIImage const&, ConvolutionError& );
    template void CompareConvolutions< TetrahedralProjection >( HDRIImage const&, HDRIImage const&, ConvolutionError& );
}
