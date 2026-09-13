#pragma once

#include <cstdint>
#include <vector>

#include "CubeReferenceFrame.h"
#include "ReferenceFrame.h"
#include "ReferencePreimage.h"

//  scoring an approximation against the reference
//-------------------------------------------------------------------------
// The objective is the paper's, from Section 6:
//
//      integral | B(x) - A(x) | dx
//
// B and A are expressed in different units:
//
//    * B is the lobe averaged over each base texel, normalised so the discretised integral is 1. It is a density.
//    * A is a coefficient field whose SUM equals the sum of the tap weights. It is not a density and carries no measure - the runtime computes sum_x L_x A(x) / sum_taps w with no per-texel weight.
//
// So A is converted with
//
//      A_density(x) = A(x) / ( sum_taps w * measure(x) )
//
// after which both sides integrate to 1 and are directly comparable.
// Both reproduce a constant environment either way, since the runtime returns sum_taps w / sum_taps w = 1 by construction.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    struct PreimageComparison
    {
        // The objective: sum | B - A_density | * measure
        double      m_l1 = 0.0;

        // Unweighted sum | B - A_density |, for reference
        double      m_l1Unweighted = 0.0;

        double      m_referenceIntegral = 0.0;
        double      m_approximationIntegral = 0.0;

        double      m_referencePeak = 0.0;
        double      m_approximationPeak = 0.0;

        uint32_t    m_numTexels = 0;
    };

    // referenceValues must already be normalised by ReferencePreimage::Normalize.
    // approximationValues are the raw coefficients from PreimageAccumulator::ResolveToBase, and approximationWeightSum is the sum of the tap weights that produced them.
    //
    // TFrame is the reference frame contract in ReferenceFrame.h.
    // This needs only the texel count and the per-texel measure, so it is independent of the frame's topology.
    // The definition is in the translation unit; the instantiation the tool uses is at the bottom of it.
    template< typename TFrame >
    PreimageComparison ComparePreimages
    (
        TFrame const& baseFrame,
        std::vector<double> const& referenceValues,
        std::vector<double> const& approximationValues,
        double approximationWeightSum,
        ErrorMeasure measure
    );
}
