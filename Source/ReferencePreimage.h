#pragma once

#include <cstdint>
#include <vector>

#include "ReferenceFrame.h"

//  reference preimage B(x)
//-------------------------------------------------------------------------
// For one output texel, B(x) is that texel's filter projected onto the base-resolution reference frame.
// It is the target the fit is scored against, and it depends on no fit variable - which is what makes the paper's objective data-independent and cacheable (Section 6).
//
// The paper calls B "the projection of the filter associated with the texel onto the cubemap" without specifying whether the lobe is point-sampled at each base texel centre or averaged over its footprint.
// At level 0 the lobe is 2.8x narrower than a base texel and point sampling retains 52% of the total.
// SupersampleRate selects between them: rate 1 point-samples, rate 8 averages 64 subsamples per texel. The published tables select rate 1.
//
// TFrame is the reference frame contract in ReferenceFrame.h. This class reads a texel's direction and asks the frame for everything else, so it holds no knowledge of the frame's topology and works for any model of the contract.
// The definition is in the translation unit; the instantiations the tool uses are at the bottom of it.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    template< typename TFrame >
    class ReferencePreimage
    {
    public:

        void Initialize( TFrame const* pBaseFrame, uint32_t level, uint32_t supersampleRate );

        // Project the lobe for the given output direction. Values are left un-normalised.
        //
        // TProfile is the profile contract from Profile.h, taken as a template parameter so EvaluateZonal inlines into the loop below.
        template< typename TProfile >
        void Evaluate( TProfile const& profile, double const* pOutputDirection );

        // Scale so the discretised integral over the frame is 1
        void Normalize( ErrorMeasure measure );

        // One smoothing pass over the frame's neighbour structure
        void Smooth();

        inline uint32_t                     GetLevel() const { return m_level; }
        inline uint32_t                     GetSupersampleRate() const { return m_supersampleRate; }
        inline uint32_t                     GetNumTexels() const { return static_cast<uint32_t>( m_values.size() ); }
        inline double                       GetValue( uint32_t baseTexelIndex ) const { return m_values[baseTexelIndex]; }
        inline std::vector<double> const&   GetValues() const { return m_values; }
        inline double const*                GetOutputDirection() const { return m_outputDirection; }

        double ComputeIntegral( ErrorMeasure measure ) const;
        double GetPeakValue() const;

        // Texels holding at least the given fraction of the peak
        uint32_t CountSupport( double relativeThreshold ) const;

        // Any isotropic lobe must be non-increasing with angle from the axis. Used as a shape check.
        bool IsMonotonicInAngle() const;

        uint32_t FindNearestTexel() const;

    private:

        TFrame const*       m_pBaseFrame = nullptr;
        uint32_t            m_level = 0;
        uint32_t            m_supersampleRate = 1;
        double              m_outputDirection[3] = { 0.0, 0.0, 0.0 };
        std::vector<double> m_values;
    };
}
