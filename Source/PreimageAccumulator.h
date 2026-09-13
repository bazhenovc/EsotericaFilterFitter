#pragma once

#include <cstdint>
#include <vector>

#include "BsplineRecurrence.h"

//  preimage of the approximation, A(x)
//-------------------------------------------------------------------------
// A tap is one trilinear sample: a direction, a fractional mip level and a weight.
// Splatting it deposits the weight into the eight texels trilinear filtering reads, four at each of two adjacent mip levels. 
// Each level's weight field is then pushed down to base resolution and accumulated.
//
// The down-sweep uses the TRANSPOSE of pass 1, not its inverse. 
// Pass 1 makes a coarser texel an average of finer ones, T_k = sum_b D_kb L_b, and a tap contributes w * sum_k alpha_k T_k; substituting gives a preimage weight of sum_k alpha_k D_kb on finer texel b, which is D^T applied to the splatted field. 
// This mirrors the paper's "successively higher resolutions" sweep in Section 6.
//
// That construction satisfies an exact identity:
//
//      sum over all base texels of A(x)  ==  sum of the tap weights
//
// Every downsample step has row sums of 1, so sum_b D_kb == 1 for any mip texel k, and every tap's trilinear coefficients sum to 1. Equivalently: a constant  environment reproduces a constant. Substituting the downsample for its transpose fails this.
//
// Written against the map contract rather than against a cubemap, so the mip chain it splats into is the map's own: its slice count sizes every level, its texel convention places the sample, and a phantom texel past a texel's footprint is resolved through direction space exactly as pass 1's are.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    template< typename TMap >
    class PreimageAccumulator
    {
    public:

        void Initialize( uint32_t baseResolution, uint32_t numLevels, JacobianWeighting weighting );

        // Clears every level so the accumulator can be reused
        void Reset();

        // Splat one trilinear sample.
        // The direction need not be normalized; face selection is scale invariant.
        // The level must already include any Jacobian level offset the caller applied.
        void AddSample( double const* pDirection, double level, double weight );

        // Push every level down to base resolution.
        // Destructive - the level fields are consumed - so call once per accumulation.
        std::vector<double> const& ResolveToBase();

        inline uint32_t                     GetBaseResolution() const { return m_baseResolution; }
        inline uint32_t                     GetNumLevels() const { return m_numLevels; }
        inline JacobianWeighting            GetWeighting() const { return m_weighting; }
        inline std::vector<double> const&   GetLevelWeights( uint32_t level ) const { return m_levelWeights[level]; }
        inline uint32_t                     GetResolutionForLevel( uint32_t level ) const { return m_baseResolution >> level; }

        static size_t GetTexelIndex( uint32_t face, uint32_t texelX, uint32_t texelY, uint32_t resolution );

    private:

        void SplatBilinear( uint32_t face, double u, double v, uint32_t level, double weight );

    private:

        uint32_t                            m_baseResolution = 0;
        uint32_t                            m_numLevels = 0;
        JacobianWeighting                   m_weighting = JacobianWeighting::Jacobian;
        bool                                m_resolved = false;

        // Borrowed from the process-wide cache, so it is shared by every accumulator with the same configuration rather than rebuilt per worker
        UpsampleOperator< TMap > const*     m_pUpsampleOperator = nullptr;

        std::vector<std::vector<double>>    m_levelWeights;
    };

    //-------------------------------------------------------------------------

    extern template class PreimageAccumulator< CubeProjection >;
    extern template class PreimageAccumulator< TetrahedralProjection >;
}
