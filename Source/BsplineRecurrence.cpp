#include "Assert.h"
#include "BsplineRecurrence.h"

#include <cmath>
#include <deque>
#include <mutex>

namespace FilterFitter
{
    // Tap positions and weights
    //-------------------------------------------------------------------------
    // The reference computes, for a coarser texel (id.x, id.y) at resolution R:
    //
    //      u0 = ( 2x + 1 - 0.75 ) / R - 1        u1 = ( 2x + 1 + 0.75 ) / R - 1
    //      v0 = -( 2y + 1 - 0.75 ) / R + 1       v1 = -( 2y + 1 + 0.75 ) / R + 1
    //
    // and hands those four (u,v) pairs to SampleLevel on the finer level. 
    // The per-tap weight is
    //
    //      w_i = calcWeight( u_i, v_i ) * ( 0.5 / sum_j calcWeight_j ) + 0.125
    //
    // which sums to exactly 1. 
    // 
    // When the Jacobian term is constant this reduces to 0.25 per tap, i.e. the plain quadratic b-spline.
    //
    // calcWeight is the reciprocal of the sphere-to-face density, so the map's own GetJacobian is what this reads, and the two weightings are that quantity and its reciprocal.
    // They are one rounding apart from the vendored code's radius^3 for a cubemap, which is the same value computed in one step instead of two.
    //-------------------------------------------------------------------------

    template< typename TMap >
    static double ComputeTapWeight( double u, double v, JacobianWeighting weighting )
    {
        double const jacobian = TMap::GetJacobian( u, v );

        switch ( weighting )
        {
            case JacobianWeighting::None:            return 1.0;
            case JacobianWeighting::InverseJacobian: return 1.0 / jacobian;
            default:                                 return jacobian;
        }
    }

    char const* GetJacobianWeightingName( JacobianWeighting weighting )
    {
        switch ( weighting )
        {
            case JacobianWeighting::None:            return "none";
            case JacobianWeighting::InverseJacobian: return "1/J (vendored code)";
            default:                                 return "J (paper)";
        }
    }

    // A coarse texel is named by its SLICE, not its face.
    // A cubemap's slice is one face, so the two agree there; a single-slice map holds four faces in one storage grid, and the face a texel belongs to is derived from its coordinate rather than needed to address it
    static size_t GetSliceTexelIndex( uint32_t slice, uint32_t texelX, uint32_t texelY, uint32_t resolution )
    {
        return ( ( static_cast<size_t>( slice ) * resolution ) + texelY ) * resolution + texelX;
    }

    //-------------------------------------------------------------------------

    static void AccumulateFootprintEntry
    (
        DownsampleFootprint& footprint,
        uint32_t face,
        uint32_t texelX,
        uint32_t texelY,
        double weight
    )
    {
        for ( uint32_t entryIndex = 0; entryIndex < footprint.m_count; ++entryIndex )
        {
            if ( ( footprint.m_face[entryIndex] == face ) && ( footprint.m_texelX[entryIndex] == texelX ) && ( footprint.m_texelY[entryIndex] == texelY ) )
            {
                footprint.m_weight[entryIndex] += weight;
                return;
            }
        }

        FF_ASSERT( footprint.m_count < DownsampleFootprint::MaxTexels );

        uint32_t const entryIndex = footprint.m_count;
        footprint.m_face[entryIndex] = face;
        footprint.m_texelX[entryIndex] = texelX;
        footprint.m_texelY[entryIndex] = texelY;
        footprint.m_weight[entryIndex] = weight;
        ++footprint.m_count;
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void ComputeDownsampleFootprint
    (
        uint32_t coarseSlice,
        uint32_t coarseX,
        uint32_t coarseY,
        uint32_t coarseResolution,
        JacobianWeighting weighting,
        DownsampleFootprint& footprint
    )
    {
        FF_ASSERT( coarseResolution > 0 );

        footprint.m_count = 0;
        footprint.m_crossesFace = false;

        uint32_t const fineResolution = coarseResolution * 2;

        double centreU = 0.0;
        double centreV = 0.0;
        TMap::GetTexelCentreUV( &centreU, &centreV, coarseX, coarseY, coarseResolution );

        double stepU = 0.0;
        double stepV = 0.0;
        TMap::GetTexelStep( &stepU, &stepV, coarseResolution );

        // The chart this texel belongs to, which every phantom of this footprint extrapolates.
        // It is resolved from the texel's own centre rather than taken from the slice, because a slice need not hold one face, and it is the coarse texel's chart that a phantom continues - not whichever chart happens to hold the phantom's coordinate.
        double centreDirection[3];
        TMap::GetTexelDirection( centreDirection, coarseSlice, centreU, centreV );

        uint32_t coarseFace = 0;
        double coarseFaceU = 0.0;
        double coarseFaceV = 0.0;
        TMap::GetFaceAndUVFromDirection( centreDirection, coarseFace, coarseFaceU, coarseFaceV );

        // A tap sits 0.75 of a FINE texel from the coarse texel's centre, which is 0.375 of a coarse step: a fine texel is half a coarse one, so half a step is one fine texel and three quarters of that is the offset.
        double const tapOffsetU = 0.75 * ( 0.5 * std::fabs( stepU ) );
        double const tapOffsetV = 0.75 * ( 0.5 * std::fabs( stepV ) );

        double const tapU[4] =
        {
            centreU - tapOffsetU,
            centreU + tapOffsetU,
            centreU - tapOffsetU,
            centreU + tapOffsetU,
        };

        double const tapV[4] =
        {
            centreV + tapOffsetV,
            centreV + tapOffsetV,
            centreV - tapOffsetV,
            centreV - tapOffsetV,
        };

        double tapWeight[4];
        double totalTapWeight = 0.0;

        for ( uint32_t tapIndex = 0; tapIndex < 4; ++tapIndex )
        {
            tapWeight[tapIndex] = ComputeTapWeight< TMap >( tapU[tapIndex], tapV[tapIndex], weighting );
            totalTapWeight += tapWeight[tapIndex];
        }

        FF_ASSERT( totalTapWeight > 0.0 );

        double const jacobianScale = 0.5 / totalTapWeight;
        for ( uint32_t tapIndex = 0; tapIndex < 4; ++tapIndex )
        {
            tapWeight[tapIndex] = ( tapWeight[tapIndex] * jacobianScale ) + 0.125;
        }

        int32_t const fineResolutionSigned = static_cast<int32_t>( fineResolution );

        for ( uint32_t tapIndex = 0; tapIndex < 4; ++tapIndex )
        {
            // Continuous finer-level texel coordinate of this tap. The tap itself is always inside the coarse texel's own chart; its bilinear footprint may not be.
            double fineX = 0.0;
            double fineY = 0.0;
            TMap::GetTexelCoordinateFromUV( &fineX, &fineY, tapU[tapIndex], tapV[tapIndex], fineResolution );

            int32_t const floorX = static_cast<int32_t>( std::floor( fineX ) );
            int32_t const floorY = static_cast<int32_t>( std::floor( fineY ) );

            double const fractionX = fineX - static_cast<double>( floorX );
            double const fractionY = fineY - static_cast<double>( floorY );

            for ( int32_t stepY = 0; stepY < 2; ++stepY )
            {
                double const bilinearWeightY = ( stepY == 0 ) ? ( 1.0 - fractionY ) : fractionY;

                for ( int32_t stepX = 0; stepX < 2; ++stepX )
                {
                    double const bilinearWeightX = ( stepX == 0 ) ? ( 1.0 - fractionX ) : fractionX;
                    double const contribution = tapWeight[tapIndex] * bilinearWeightX * bilinearWeightY;

                    if ( contribution == 0.0 )
                    {
                        continue;
                    }

                    // Phantom texel centre, expressed in the coarse texel's own chart and deliberately allowed to run past its edge
                    double phantomCentreU = 0.0;
                    double phantomCentreV = 0.0;
                    TMap::GetTexelCentreUVOutside( &phantomCentreU, &phantomCentreV, floorX + stepX, floorY + stepY, fineResolution );

                    // Resolve through direction space, the way hardware does
                    double direction[3];
                    TMap::GetDirectionFromUV( direction, phantomCentreU, phantomCentreV, coarseFace );

                    uint32_t resolvedFace = coarseFace;
                    double resolvedU = 0.0;
                    double resolvedV = 0.0;
                    TMap::GetFaceAndUVFromDirection( direction, resolvedFace, resolvedU, resolvedV );

                    double resolvedX = 0.0;
                    double resolvedY = 0.0;
                    TMap::GetTexelCoordinateFromUV( &resolvedX, &resolvedY, resolvedU, resolvedV, fineResolution );

                    // Round to nearest rather than flooring an offset coordinate.
                    // The two agree for a texel that resolves exactly onto a centre, and only the direction round trip decides how exact that is: a cube face reads its own coordinate back exactly, a tetrahedral tile goes through an affine map onto irrational vertices and back, which leaves a few ulps.
                    // Rounding to nearest is what keeps those ulps from becoming a texel.
                    int32_t roundedX = static_cast<int32_t>( std::lround( resolvedX ) );
                    int32_t roundedY = static_cast<int32_t>( std::lround( resolvedY ) );

                    // Corner phantoms can land marginally outside; clamp rather
                    // than drop, so no weight is ever lost
                    roundedX = ( roundedX < 0 ) ? 0 : ( ( roundedX >= fineResolutionSigned ) ? ( fineResolutionSigned - 1 ) : roundedX );
                    roundedY = ( roundedY < 0 ) ? 0 : ( ( roundedY >= fineResolutionSigned ) ? ( fineResolutionSigned - 1 ) : roundedY );

                    if ( resolvedFace != coarseFace )
                    {
                        footprint.m_crossesFace = true;
                    }

                    AccumulateFootprintEntry
                    (
                        footprint,
                        resolvedFace,
                        static_cast<uint32_t>( roundedX ),
                        static_cast<uint32_t>( roundedY ),
                        contribution
                    );
                }
            }
        }
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void DownsampleLevel
    (
        std::vector<double> const& finer,
        uint32_t coarseResolution,
        JacobianWeighting weighting,
        std::vector<double>& coarser
    )
    {
        uint32_t const fineResolution = coarseResolution * 2;
        uint32_t const numSlices = TMap::NumSlices;

        FF_ASSERT( finer.size() == static_cast<size_t>( numSlices ) * fineResolution * fineResolution );

        coarser.assign( static_cast<size_t>( numSlices ) * coarseResolution * coarseResolution, 0.0 );

        for ( uint32_t slice = 0; slice < numSlices; ++slice )
        {
            for ( uint32_t coarseY = 0; coarseY < coarseResolution; ++coarseY )
            {
                for ( uint32_t coarseX = 0; coarseX < coarseResolution; ++coarseX )
                {
                    DownsampleFootprint footprint;
                    ComputeDownsampleFootprint< TMap >( slice, coarseX, coarseY, coarseResolution, weighting, footprint );

                    double sum = 0.0;
                    for ( uint32_t entryIndex = 0; entryIndex < footprint.m_count; ++entryIndex )
                    {
                        size_t const sourceIndex = TMap::GetTexelIndex
                        (
                            footprint.m_face[entryIndex],
                            footprint.m_texelX[entryIndex],
                            footprint.m_texelY[entryIndex],
                            fineResolution
                        );

                        sum += footprint.m_weight[entryIndex] * finer[sourceIndex];
                    }

                    coarser[GetSliceTexelIndex( slice, coarseX, coarseY, coarseResolution )] = sum;
                }
            }
        }
    }

    template< typename TMap >
    void UpsampleLevel
    (
        std::vector<double> const& coarser,
        uint32_t coarseResolution,
        JacobianWeighting weighting,
        std::vector<double>& finer
    )
    {
        uint32_t const fineResolution = coarseResolution * 2;
        uint32_t const numSlices = TMap::NumSlices;

        FF_ASSERT( coarser.size() == static_cast<size_t>( numSlices ) * coarseResolution * coarseResolution );

        finer.assign( static_cast<size_t>( numSlices ) * fineResolution * fineResolution, 0.0 );

        for ( uint32_t slice = 0; slice < numSlices; ++slice )
        {
            for ( uint32_t coarseY = 0; coarseY < coarseResolution; ++coarseY )
            {
                for ( uint32_t coarseX = 0; coarseX < coarseResolution; ++coarseX )
                {
                    double const coarseValue = coarser[GetSliceTexelIndex( slice, coarseX, coarseY, coarseResolution )];
                    if ( coarseValue == 0.0 )
                    {
                        continue;
                    }

                    DownsampleFootprint footprint;
                    ComputeDownsampleFootprint< TMap >( slice, coarseX, coarseY, coarseResolution, weighting, footprint );

                    for ( uint32_t entryIndex = 0; entryIndex < footprint.m_count; ++entryIndex )
                    {
                        size_t const targetIndex = TMap::GetTexelIndex
                        (
                            footprint.m_face[entryIndex],
                            footprint.m_texelX[entryIndex],
                            footprint.m_texelY[entryIndex],
                            fineResolution
                        );

                        finer[targetIndex] += footprint.m_weight[entryIndex] * coarseValue;
                    }
                }
            }
        }
    }

    //  Cached transpose
    //-------------------------------------------------------------------------

    template< typename TMap >
    void UpsampleOperator< TMap >::Build( uint32_t baseResolution, uint32_t numLevels, JacobianWeighting weighting )
    {
        FF_ASSERT( baseResolution > 0 );
        FF_ASSERT( numLevels > 1 );
        FF_ASSERT( ( baseResolution >> ( numLevels - 1 ) ) > 0 );

        m_baseResolution = baseResolution;
        m_numLevels = numLevels;
        m_numEntries = 0;
        m_levels.assign( numLevels - 1, LevelOperator() );

        for ( uint32_t coarseLevel = 1; coarseLevel < numLevels; ++coarseLevel )
        {
            LevelOperator& levelOperator = m_levels[coarseLevel - 1];

            uint32_t const coarseResolution = baseResolution >> coarseLevel;
            uint32_t const fineResolution = coarseResolution * 2;
            uint32_t const numFineTexels = TMap::NumSlices * fineResolution * fineResolution;

            // Row lengths. Counted by walking every coarse texel's footprint, not just the non-zero ones as the field-driven version does, because the operator has to be valid for every field.
            levelOperator.m_offsets.assign( static_cast<size_t>( numFineTexels ) + 1, 0 );

            DownsampleFootprint footprint;

            for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
            {
                for ( uint32_t coarseY = 0; coarseY < coarseResolution; ++coarseY )
                {
                    for ( uint32_t coarseX = 0; coarseX < coarseResolution; ++coarseX )
                    {
                        ComputeDownsampleFootprint< TMap >( slice, coarseX, coarseY, coarseResolution, weighting, footprint );

                        for ( uint32_t entryIndex = 0; entryIndex < footprint.m_count; ++entryIndex )
                        {
                            size_t const fineIndex = TMap::GetTexelIndex
                            (
                                footprint.m_face[entryIndex],
                                footprint.m_texelX[entryIndex],
                                footprint.m_texelY[entryIndex],
                                fineResolution
                            );

                            ++levelOperator.m_offsets[fineIndex + 1];
                        }
                    }
                }
            }

            for ( uint32_t fineIndex = 0; fineIndex < numFineTexels; ++fineIndex )
            {
                levelOperator.m_offsets[fineIndex + 1] += levelOperator.m_offsets[fineIndex];
            }

            uint32_t const numEntries = levelOperator.m_offsets[numFineTexels];
            levelOperator.m_sources.assign( numEntries, 0 );
            levelOperator.m_weights.assign( numEntries, 0.0 );

            // Write cursors, one per row
            std::vector<uint32_t> cursor( levelOperator.m_offsets.begin(), levelOperator.m_offsets.end() - 1 );

            for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
            {
                for ( uint32_t coarseY = 0; coarseY < coarseResolution; ++coarseY )
                {
                    for ( uint32_t coarseX = 0; coarseX < coarseResolution; ++coarseX )
                    {
                        uint32_t const coarseIndex = static_cast<uint32_t>( GetSliceTexelIndex( slice, coarseX, coarseY, coarseResolution ) );

                        ComputeDownsampleFootprint< TMap >( slice, coarseX, coarseY, coarseResolution, weighting, footprint );

                        for ( uint32_t entryIndex = 0; entryIndex < footprint.m_count; ++entryIndex )
                        {
                            size_t const fineIndex = TMap::GetTexelIndex
                            (
                                footprint.m_face[entryIndex],
                                footprint.m_texelX[entryIndex],
                                footprint.m_texelY[entryIndex],
                                fineResolution
                            );

                            uint32_t const slot = cursor[fineIndex]++;
                            levelOperator.m_sources[slot] = coarseIndex;
                            levelOperator.m_weights[slot] = footprint.m_weight[entryIndex];
                        }
                    }
                }
            }

            m_numEntries += numEntries;
        }
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    uint32_t UpsampleOperator< TMap >::GetFineTexelCount( uint32_t coarseLevel ) const
    {
        FF_ASSERT( coarseLevel > 0 );
        FF_ASSERT( coarseLevel < m_numLevels );

        LevelOperator const& levelOperator = m_levels[coarseLevel - 1];

        FF_ASSERT( !levelOperator.m_offsets.empty() );

        return static_cast<uint32_t>( levelOperator.m_offsets.size() - 1 );
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void UpsampleOperator< TMap >::Apply( uint32_t coarseLevel, std::vector<double> const& coarser, std::vector<double>& finer ) const
    {
        finer.assign( GetFineTexelCount( coarseLevel ), 0.0 );

        ApplyAdd( coarseLevel, coarser, finer );
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void UpsampleOperator< TMap >::ApplyAdd( uint32_t coarseLevel, std::vector<double> const& coarser, std::vector<double>& target ) const
    {
        FF_ASSERT( IsInitialized() );
        FF_ASSERT( coarseLevel > 0 );
        FF_ASSERT( coarseLevel < m_numLevels );

        LevelOperator const& levelOperator = m_levels[coarseLevel - 1];

        uint32_t const numFineTexels = static_cast<uint32_t>( levelOperator.m_offsets.size() - 1 );

        FF_ASSERT( target.size() == numFineTexels );
        FF_ASSERT( coarser.size() == ( target.size() / 4 ) );

        uint32_t const* const pOffsets = levelOperator.m_offsets.data();
        uint32_t const* const pSources = levelOperator.m_sources.data();
        double const* const   pWeights = levelOperator.m_weights.data();

        for ( uint32_t fineIndex = 0; fineIndex < numFineTexels; ++fineIndex )
        {
            double sum = 0.0;

            for ( uint32_t entryIndex = pOffsets[fineIndex]; entryIndex < pOffsets[fineIndex + 1]; ++entryIndex )
            {
                sum += pWeights[entryIndex] * coarser[pSources[entryIndex]];
            }

            target[fineIndex] += sum;
        }
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    size_t UpsampleOperator< TMap >::GetNumBytes() const
    {
        size_t total = 0;

        for ( LevelOperator const& levelOperator : m_levels )
        {
            total += levelOperator.m_offsets.size() * sizeof( uint32_t );
            total += levelOperator.m_sources.size() * sizeof( uint32_t );
            total += levelOperator.m_weights.size() * sizeof( double );
        }

        return total;
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    UpsampleOperator< TMap > const& GetUpsampleOperator( uint32_t baseResolution, uint32_t numLevels, JacobianWeighting weighting )
    {
        struct Entry
        {
            uint32_t                    m_baseResolution = {};
            uint32_t                    m_numLevels = {};
            JacobianWeighting           m_weighting = {};
            UpsampleOperator< TMap >    m_operator = {};
        };

        static std::mutex           cacheMutex;
        static std::deque<Entry>    cache;

        std::lock_guard<std::mutex> lock( cacheMutex );

        for ( Entry const& entry : cache )
        {
            if ( ( entry.m_baseResolution == baseResolution ) && ( entry.m_numLevels == numLevels ) && ( entry.m_weighting == weighting ) )
            {
                return entry.m_operator;
            }
        }

        // A deque, not a vector: references to existing entries have to survive this push_back, because accumulators hold pointers into them.
        cache.emplace_back();

        Entry& entry = cache.back();
        entry.m_baseResolution = baseResolution;
        entry.m_numLevels = numLevels;
        entry.m_weighting = weighting;
        entry.m_operator.Build( baseResolution, numLevels, weighting );

        return entry.m_operator;
    }

    //-------------------------------------------------------------------------

    template void ComputeDownsampleFootprint< CubeProjection >( uint32_t, uint32_t, uint32_t, uint32_t, JacobianWeighting, DownsampleFootprint& );
    template void ComputeDownsampleFootprint< TetrahedralProjection >( uint32_t, uint32_t, uint32_t, uint32_t, JacobianWeighting, DownsampleFootprint& );

    template void DownsampleLevel< CubeProjection >( std::vector<double> const&, uint32_t, JacobianWeighting, std::vector<double>& );
    template void DownsampleLevel< TetrahedralProjection >( std::vector<double> const&, uint32_t, JacobianWeighting, std::vector<double>& );

    template void UpsampleLevel< CubeProjection >( std::vector<double> const&, uint32_t, JacobianWeighting, std::vector<double>& );
    template void UpsampleLevel< TetrahedralProjection >( std::vector<double> const&, uint32_t, JacobianWeighting, std::vector<double>& );

    template class UpsampleOperator< CubeProjection >;
    template class UpsampleOperator< TetrahedralProjection >;

    template UpsampleOperator< CubeProjection > const& GetUpsampleOperator< CubeProjection >( uint32_t, uint32_t, JacobianWeighting );
    template UpsampleOperator< TetrahedralProjection > const& GetUpsampleOperator< TetrahedralProjection >( uint32_t, uint32_t, JacobianWeighting );
}
