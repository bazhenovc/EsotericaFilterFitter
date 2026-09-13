#include "Assert.h"
#include "PreimageError.h"

#include "CubeReferenceFrame.h"
#include "TetrahedralProjection.h"

#include <cmath>

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    template< typename TFrame >
    PreimageComparison ComparePreimages
    (
        TFrame const& baseFrame,
        std::vector<double> const& referenceValues,
        std::vector<double> const& approximationValues,
        double approximationWeightSum,
        ErrorMeasure measure
    )
    {
        FF_ASSERT( baseFrame.IsInitialized() );
        FF_ASSERT( referenceValues.size() == approximationValues.size() );
        FF_ASSERT( baseFrame.GetNumTexels() == referenceValues.size() );
        FF_ASSERT( std::fabs( approximationWeightSum ) > 0.0 );

        uint32_t const numTexels = baseFrame.GetNumTexels();

        PreimageComparison comparison;
        comparison.m_numTexels = numTexels;

        for ( uint32_t texelIndex = 0; texelIndex < numTexels; ++texelIndex )
        {
            double const texelMeasure = baseFrame.GetTexelMeasure( baseFrame.GetTexel( texelIndex ), measure );

            double const referenceValue = referenceValues[texelIndex];
            double const approximationCoefficient = approximationValues[texelIndex];

            // Convert the coefficient into the same density convention as the reference
            double const approximationDensity = approximationCoefficient / ( approximationWeightSum * texelMeasure );

            double const difference = std::fabs( referenceValue - approximationDensity );

            comparison.m_l1 += difference * texelMeasure;
            comparison.m_l1Unweighted += difference;
            comparison.m_referenceIntegral += referenceValue * texelMeasure;
            comparison.m_approximationIntegral += approximationDensity * texelMeasure;

            comparison.m_referencePeak = ( referenceValue > comparison.m_referencePeak ) ? referenceValue : comparison.m_referencePeak;

            double const approximationMagnitude = std::fabs( approximationDensity );
            comparison.m_approximationPeak = ( approximationMagnitude > comparison.m_approximationPeak ) ? approximationMagnitude : comparison.m_approximationPeak;
        }

        return comparison;
    }

    //-------------------------------------------------------------------------

    template PreimageComparison ComparePreimages< CubeReferenceFrame >( CubeReferenceFrame const& baseFrame, std::vector<double> const& referenceValues, std::vector<double> const& approximationValues, double approximationWeightSum, ErrorMeasure measure );

    template PreimageComparison ComparePreimages< MapReferenceFrame< TetrahedralProjection > >( MapReferenceFrame< TetrahedralProjection > const& baseFrame, std::vector<double> const& referenceValues, std::vector<double> const& approximationValues, double approximationWeightSum, ErrorMeasure measure );
}
