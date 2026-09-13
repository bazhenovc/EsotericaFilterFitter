#include "Assert.h"
#include "PreimageAccumulator.h"

#include <cmath>

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    template< typename TMap >
    size_t PreimageAccumulator< TMap >::GetTexelIndex( uint32_t face, uint32_t texelX, uint32_t texelY, uint32_t resolution )
    {
        return TMap::GetTexelIndex( face, texelX, texelY, resolution );
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void PreimageAccumulator< TMap >::Initialize( uint32_t baseResolution, uint32_t numLevels, JacobianWeighting weighting )
    {
        FF_ASSERT( baseResolution > 0 );
        FF_ASSERT( numLevels > 0 );
        FF_ASSERT( ( baseResolution >> ( numLevels - 1 ) ) > 0 );

        m_baseResolution = baseResolution;
        m_numLevels = numLevels;
        m_weighting = weighting;

        m_levelWeights.assign( numLevels, std::vector<double>() );
        m_pUpsampleOperator = &GetUpsampleOperator< TMap >( baseResolution, numLevels, weighting );
        Reset();
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void PreimageAccumulator< TMap >::Reset()
    {
        for ( uint32_t level = 0; level < m_numLevels; ++level )
        {
            uint32_t const resolution = GetResolutionForLevel( level );
            m_levelWeights[level].assign( static_cast<size_t>( TMap::NumSlices ) * resolution * resolution, 0.0 );
        }

        m_resolved = false;
    }

    //  A tap is splatted in the chart it landed in, and a phantom texel past the tap's own footprint continues that chart - the same rule pass 1 uses, and for the same reason: a map need not be continuous across its seams, so asking which chart owns the phantom's coordinate would read the wrong one.
    //-------------------------------------------------------------------------

    template< typename TMap >
    void PreimageAccumulator< TMap >::SplatBilinear( uint32_t face, double u, double v, uint32_t level, double weight )
    {
        uint32_t const resolution = GetResolutionForLevel( level );

        // Continuous texel coordinate of the sample on this level
        double texelX = 0.0;
        double texelY = 0.0;
        TMap::GetTexelCoordinateFromUV( &texelX, &texelY, u, v, resolution );

        int32_t const floorX = static_cast<int32_t>( std::floor( texelX ) );
        int32_t const floorY = static_cast<int32_t>( std::floor( texelY ) );

        double const fractionX = texelX - static_cast<double>( floorX );
        double const fractionY = texelY - static_cast<double>( floorY );

        int32_t const resolutionSigned = static_cast<int32_t>( resolution );
        std::vector<double>& levelWeights = m_levelWeights[level];

        for ( int32_t stepY = 0; stepY < 2; ++stepY )
        {
            double const weightY = ( stepY == 0 ) ? ( 1.0 - fractionY ) : fractionY;

            for ( int32_t stepX = 0; stepX < 2; ++stepX )
            {
                double const weightX = ( stepX == 0 ) ? ( 1.0 - fractionX ) : fractionX;
                double const bilinearWeight = weight * weightX * weightY;

                if ( bilinearWeight == 0.0 )
                {
                    continue;
                }

                // Phantom texel centre, expressed in this chart's parameterization and allowed to run past the edge, then resolved through direction space exactly as Pass 1's footprints are
                double phantomU = 0.0;
                double phantomV = 0.0;
                TMap::GetTexelCentreUVOutside( &phantomU, &phantomV, floorX + stepX, floorY + stepY, resolution );

                double direction[3];
                TMap::GetDirectionFromUV( direction, phantomU, phantomV, face );

                uint32_t resolvedFace = face;
                double resolvedU = 0.0;
                double resolvedV = 0.0;
                TMap::GetFaceAndUVFromDirection( direction, resolvedFace, resolvedU, resolvedV );

                double resolvedX = 0.0;
                double resolvedY = 0.0;
                TMap::GetTexelCoordinateFromUV( &resolvedX, &resolvedY, resolvedU, resolvedV, resolution );

                // Rounding to nearest, for the reason pass 1 does: a cube face reads its own coordinate back exactly and a tetrahedral tile goes through an affine map onto irrational vertices and back, and the difference must not become a texel.
                int32_t roundedX = static_cast<int32_t>( std::lround( resolvedX ) );
                int32_t roundedY = static_cast<int32_t>( std::lround( resolvedY ) );

                roundedX = ( roundedX < 0 ) ? 0 : ( ( roundedX >= resolutionSigned ) ? ( resolutionSigned - 1 ) : roundedX );
                roundedY = ( roundedY < 0 ) ? 0 : ( ( roundedY >= resolutionSigned ) ? ( resolutionSigned - 1 ) : roundedY );

                size_t const texelIndex = TMap::GetTexelIndex
                (
                    resolvedFace,
                    static_cast<uint32_t>( roundedX ),
                    static_cast<uint32_t>( roundedY ),
                    resolution
                );

                levelWeights[texelIndex] += bilinearWeight;
            }
        }
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void PreimageAccumulator< TMap >::AddSample( double const* pDirection, double level, double weight )
    {
        FF_ASSERT( pDirection != nullptr );
        FF_ASSERT( !m_resolved );

        uint32_t face = 0;
        double u = 0.0;
        double v = 0.0;
        TMap::GetFaceAndUVFromDirection( pDirection, face, u, v );

        // Trilinear reads two adjacent mip levels. Hardware clamps a level outside the chain rather than extrapolating.
        double const maximumLevel = static_cast<double>( m_numLevels - 1 );
        double const clampedLevel = ( level < 0.0 ) ? 0.0 : ( ( level > maximumLevel ) ? maximumLevel : level );

        uint32_t const level0 = static_cast<uint32_t>( std::floor( clampedLevel ) );
        uint32_t const level1 = ( ( level0 + 1 ) < m_numLevels ) ? ( level0 + 1 ) : level0;

        double const fraction1 = clampedLevel - static_cast<double>( level0 );
        double const fraction0 = 1.0 - fraction1;

        SplatBilinear( face, u, v, level0, weight * fraction0 );

        if ( level1 != level0 )
        {
            SplatBilinear( face, u, v, level1, weight * fraction1 );
        }
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    std::vector<double> const& PreimageAccumulator< TMap >::ResolveToBase()
    {
        FF_ASSERT( !m_resolved );
        FF_ASSERT( m_pUpsampleOperator != nullptr );

        // Walk from the coarsest level down, pushing each level's weights into the next finer one.
        // This is the paper's "successively higher resolutions of the mipmap".
        //
        // The transpose itself is cached and shared: rebuilding it from footprints costs about as much as a thousand of these sweeps, and this runs once per output texel per objective evaluation.
        for ( int32_t level = static_cast<int32_t>( m_numLevels ) - 1; level >= 1; --level )
        {
            m_pUpsampleOperator->ApplyAdd( static_cast<uint32_t>( level ), m_levelWeights[level], m_levelWeights[level - 1] );
        }

        m_resolved = true;
        return m_levelWeights[0];
    }

    //-------------------------------------------------------------------------

    template class PreimageAccumulator< CubeProjection >;
    template class PreimageAccumulator< TetrahedralProjection >;
}
