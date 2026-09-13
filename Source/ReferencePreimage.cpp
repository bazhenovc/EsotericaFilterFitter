#include "Assert.h"
#include "ReferencePreimage.h"

#include "CubeReferenceFrame.h"
#include "ProfileBeckmann.h"
#include "ProfileGGX.h"
#include "TetrahedralProjection.h"

#include <algorithm>
#include <cmath>

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    template< typename TFrame >
    void ReferencePreimage< TFrame >::Initialize( TFrame const* pBaseFrame, uint32_t level, uint32_t supersampleRate )
    {
        FF_ASSERT( pBaseFrame != nullptr );
        FF_ASSERT( pBaseFrame->IsInitialized() );
        FF_ASSERT( level < 32 );
        FF_ASSERT( supersampleRate >= 1 );

        m_pBaseFrame = pBaseFrame;
        m_level = level;
        m_supersampleRate = supersampleRate;
        m_values.assign( pBaseFrame->GetNumTexels(), 0.0 );
    }

    //-------------------------------------------------------------------------

    template< typename TFrame >
    template< typename TProfile >
    void ReferencePreimage< TFrame >::Evaluate( TProfile const& profile, double const* pOutputDirection )
    {
        FF_ASSERT( m_pBaseFrame != nullptr );
        FF_ASSERT( pOutputDirection != nullptr );

        double const outputLength = std::sqrt( ( pOutputDirection[0] * pOutputDirection[0] ) + ( pOutputDirection[1] * pOutputDirection[1] ) + ( pOutputDirection[2] * pOutputDirection[2] ) );
        FF_ASSERT( outputLength > 0.0 );

        m_outputDirection[0] = pOutputDirection[0] / outputLength;
        m_outputDirection[1] = pOutputDirection[1] / outputLength;
        m_outputDirection[2] = pOutputDirection[2] / outputLength;

        // A zero width is a mirror, so the filter is the identity and the preimage is the output direction's own base texel and nothing else.
        // That is the limit of the point-sampled lobe as the width goes to zero, not a special case, and it is what a roughness of exactly zero means at runtime - the prefilter samples source mip 0 directly rather than choosing a mip.
        // A continuous NDF cannot express it, which is why it is handled here and not in a profile.
        if ( profile.GetWidth( m_level ) <= 0.0 )
        {
            std::fill( m_values.begin(), m_values.end(), 0.0 );
            m_values[FindNearestTexel()] = 1.0;

            return;
        }

        double const inverseRate = 1.0 / static_cast<double>( m_supersampleRate );
        double const inverseSampleCount = 1.0 / static_cast<double>( m_supersampleRate * m_supersampleRate );

        for ( uint32_t baseTexelIndex = 0; baseTexelIndex < GetNumTexels(); ++baseTexelIndex )
        {
            // Rate 1 uses the stored texel-centre direction directly
            if ( m_supersampleRate == 1 )
            {
                double const* const pTexelDir = m_pBaseFrame->GetTexel( baseTexelIndex ).m_dir;

                double const cosTheta = ( m_outputDirection[0] * pTexelDir[0] )
                    + ( m_outputDirection[1] * pTexelDir[1] )
                    + ( m_outputDirection[2] * pTexelDir[2] );

                m_values[baseTexelIndex] = profile.EvaluateZonal( cosTheta, m_level );
                continue;
            }

            // Supersampled: average the lobe over the texel's own footprint.
            // The frame supplies the sample directions, so this knows nothing about how the frame is laid out.
            double sum = 0.0;

            for ( uint32_t sampleY = 0; sampleY < m_supersampleRate; ++sampleY )
            {
                double const offsetY = ( ( static_cast<double>( sampleY ) + 0.5 ) * inverseRate ) - 0.5;

                for ( uint32_t sampleX = 0; sampleX < m_supersampleRate; ++sampleX )
                {
                    double const offsetX = ( ( static_cast<double>( sampleX ) + 0.5 ) * inverseRate ) - 0.5;

                    double dir[3];
                    m_pBaseFrame->GetTexelSampleDirection( dir, baseTexelIndex, offsetX, offsetY );

                    double const dirLength = std::sqrt( ( dir[0] * dir[0] ) + ( dir[1] * dir[1] ) + ( dir[2] * dir[2] ) );

                    double const cosTheta = ( ( m_outputDirection[0] * dir[0] ) + ( m_outputDirection[1] * dir[1] ) + ( m_outputDirection[2] * dir[2] ) ) / dirLength;

                    sum += profile.EvaluateZonal( cosTheta, m_level );
                }
            }

            m_values[baseTexelIndex] = sum * inverseSampleCount;
        }
    }

    //-------------------------------------------------------------------------

    template< typename TFrame >
    double ReferencePreimage< TFrame >::ComputeIntegral( ErrorMeasure measure ) const
    {
        FF_ASSERT( m_pBaseFrame != nullptr );

        double total = 0.0;
        for ( uint32_t baseTexelIndex = 0; baseTexelIndex < GetNumTexels(); ++baseTexelIndex )
        {
            double const texelMeasure = m_pBaseFrame->GetTexelMeasure( m_pBaseFrame->GetTexel( baseTexelIndex ), measure );
            total += m_values[baseTexelIndex] * texelMeasure;
        }

        return total;
    }

    //-------------------------------------------------------------------------

    template< typename TFrame >
    void ReferencePreimage< TFrame >::Smooth()
    {
        FF_ASSERT( m_pBaseFrame != nullptr );

        m_pBaseFrame->SmoothInPlace( m_values );
    }

    //-------------------------------------------------------------------------

    template< typename TFrame >
    void ReferencePreimage< TFrame >::Normalize( ErrorMeasure measure )
    {
        double const total = ComputeIntegral( measure );
        FF_ASSERT( total > 0.0 );

        double const scale = 1.0 / total;
        for ( double& value : m_values )
        {
            value *= scale;
        }
    }

    //-------------------------------------------------------------------------

    template< typename TFrame >
    double ReferencePreimage< TFrame >::GetPeakValue() const
    {
        double peak = 0.0;
        for ( double const value : m_values )
        {
            peak = ( value > peak ) ? value : peak;
        }

        return peak;
    }

    //-------------------------------------------------------------------------

    template< typename TFrame >
    uint32_t ReferencePreimage< TFrame >::CountSupport( double relativeThreshold ) const
    {
        double const threshold = GetPeakValue() * relativeThreshold;

        uint32_t count = 0;
        for ( double const value : m_values )
        {
            if ( value >= threshold )
            {
                ++count;
            }
        }

        return count;
    }

    //-------------------------------------------------------------------------

    template< typename TFrame >
    uint32_t ReferencePreimage< TFrame >::FindNearestTexel() const
    {
        FF_ASSERT( m_pBaseFrame != nullptr );

        uint32_t nearestIndex = 0;
        double bestCosTheta = -2.0;

        for ( uint32_t baseTexelIndex = 0; baseTexelIndex < GetNumTexels(); ++baseTexelIndex )
        {
            double const* const pTexelDir = m_pBaseFrame->GetTexel( baseTexelIndex ).m_dir;

            double const cosTheta = ( m_outputDirection[0] * pTexelDir[0] )
                + ( m_outputDirection[1] * pTexelDir[1] )
                + ( m_outputDirection[2] * pTexelDir[2] );

            if ( cosTheta > bestCosTheta )
            {
                bestCosTheta = cosTheta;
                nearestIndex = baseTexelIndex;
            }
        }

        return nearestIndex;
    }

    //-------------------------------------------------------------------------

    template< typename TFrame >
    bool ReferencePreimage< TFrame >::IsMonotonicInAngle() const
    {
        FF_ASSERT( m_pBaseFrame != nullptr );

        // Bucket by angle from the output axis rather than by cosine: cos is almost flat near the axis, so cosine buckets would put the first several texel rings in a single bucket and make this check vacuous.
        constexpr uint32_t numBuckets = 2048;

        double bucketSum[numBuckets] = {};
        uint32_t bucketCount[numBuckets] = {};

        for ( uint32_t baseTexelIndex = 0; baseTexelIndex < GetNumTexels(); ++baseTexelIndex )
        {
            double const* const pTexelDir = m_pBaseFrame->GetTexel( baseTexelIndex ).m_dir;

            double const cosTheta = ( m_outputDirection[0] * pTexelDir[0] )
                + ( m_outputDirection[1] * pTexelDir[1] )
                + ( m_outputDirection[2] * pTexelDir[2] );

            double const clampedCos = ( cosTheta < -1.0 ) ? -1.0 : ( ( cosTheta > 1.0 ) ? 1.0 : cosTheta );
            double const angleFromAxis = std::acos( clampedCos );

            uint32_t bucketIndex = static_cast<uint32_t>( ( angleFromAxis / std::numbers::pi_v<double> ) * static_cast<double>( numBuckets ) );
            bucketIndex = ( bucketIndex >= numBuckets ) ? ( numBuckets - 1 ) : bucketIndex;

            bucketSum[bucketIndex] += m_values[baseTexelIndex];
            ++bucketCount[bucketIndex];
        }

        // Walk outward from the axis, requiring the mean of each populated bucket not to exceed the previous one.
        // The tolerance is relative and loose because this is a shape sanity check, not a proof.
        double previousMean = 0.0;
        bool seenAny = false;

        for ( uint32_t bucketIndex = 0; bucketIndex < numBuckets; ++bucketIndex )
        {
            if ( bucketCount[bucketIndex] == 0 )
            {
                continue;
            }

            double const mean = bucketSum[bucketIndex] / static_cast<double>( bucketCount[bucketIndex] );

            if ( seenAny && ( mean > ( previousMean * ( 1.0 + 1.0e-6 ) ) ) )
            {
                return false;
            }

            previousMean = mean;
            seenAny = true;
        }

        return true;
    }

    //-------------------------------------------------------------------------

    template class ReferencePreimage< CubeReferenceFrame >;
    template void ReferencePreimage< CubeReferenceFrame >::Evaluate< ProfileGGX >( ProfileGGX const& profile, double const* pOutputDirection );
    template void ReferencePreimage< CubeReferenceFrame >::Evaluate< ProfileBeckmann >( ProfileBeckmann const& profile, double const* pOutputDirection );

    template class ReferencePreimage< MapReferenceFrame< TetrahedralProjection > >;
    template void ReferencePreimage< MapReferenceFrame< TetrahedralProjection > >::Evaluate< ProfileGGX >( ProfileGGX const& profile, double const* pOutputDirection );
    template void ReferencePreimage< MapReferenceFrame< TetrahedralProjection > >::Evaluate< ProfileBeckmann >( ProfileBeckmann const& profile, double const* pOutputDirection );
}
