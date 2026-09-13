#pragma once

#include <cstdint>
#include <vector>

#include "DFGIntegrand.h"

// Preintegrated split-sum DFG term
//-------------------------------------------------------------------------
// The grid is indexed by ( N dot V, roughness ), which is what the runtime samples:
//
//      dfg = dfgTexture.SampleLevel( sampler, float2( N.V, roughness ), 0.0 ).xy
//
// and both are the material's own numbers, not a level or a mip.
// What each texel holds is the integral written out in DFGIntegrand.h.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    // Grid
    //-------------------------------------------------------------------------
    // Square, and the same resolution on both axes.
    // The engine samples this table at 128, so 128 is the grid the runtime reads.
    //
    // A texel's coordinates are its centre: N dot V is ( x + 0.5 ) / resolution and roughness is ( y + 0.5 ) / resolution.
    // Neither axis reaches exactly zero or exactly one, which is what a clamped bilinear lookup sees at the edges.
    //-------------------------------------------------------------------------

    inline constexpr uint32_t DFGDefaultResolution = 128;

    // Samples per texel.
    // The engine's pass evaluates this table on the GPU at 1024 samples inside a frame, and this is the same integral evaluated offline, so what the count buys here is accuracy rather than a frame budget.
    // The table is stored as half floats, so the count only has to be high enough that the estimate's own noise is below the storage's precision everywhere the estimator is conditioned.
    inline constexpr uint32_t DFGDefaultSampleCount = 16384;

    // What the two counts are allowed to reach. The resolution sizes the texture the engine creates, and the sample counts are multiplied before they are used, so a value that was mistyped is refused rather than wrapped into a different table.
    inline constexpr uint32_t DFGMaximumResolution = 4096;
    inline constexpr uint32_t DFGMaximumSampleCount = 1U << 24;

    //-------------------------------------------------------------------------

    struct DFGOptions
    {
        uint32_t m_resolution = DFGDefaultResolution;
        uint32_t m_sampleCount = DFGDefaultSampleCount;
    };

    // One grid point. Two numbers, in this order.
    struct DFGTexel
    {
        double m_scale = 0.0;
        double m_bias = 0.0;
    };

    // The sequence a sample is taken from.
    // The table's and the reference's are different constructions of the same quadrature, which is what makes comparing them a statement about the quadrature rather than about the integrand.
    enum class DFGSampleSequence
    {
        Hammersley,
        Stratified
    };

    // The table and what it was evaluated at.
    // The axes are kept because a texel's coordinates are its centre, so a consumer that wants the roughness of a row needs the resolution to derive it.
    struct DFGTable
    {
        uint32_t                m_resolution = 0;
        uint32_t                m_sampleCount = 0;
        std::vector<DFGTexel>   m_texels;

        inline uint32_t GetResolution() const { return m_resolution; }
        inline uint32_t GetSampleCount() const { return m_sampleCount; }

        // N dot V at the centre of column x
        inline double GetNdotV( uint32_t x ) const
        {
            return ( static_cast<double>( x ) + 0.5 ) / static_cast<double>( m_resolution );
        }

        // Roughness at the centre of row y
        inline double GetRoughness( uint32_t y ) const
        {
            return ( static_cast<double>( y ) + 0.5 ) / static_cast<double>( m_resolution );
        }

        inline DFGTexel const& GetTexel( uint32_t column, uint32_t row ) const
        {
            return m_texels[( static_cast<size_t>( row ) * m_resolution ) + column];
        }

        inline void SetTexel( uint32_t column, uint32_t row, DFGTexel const& texel )
        {
            m_texels[( static_cast<size_t>( row ) * m_resolution ) + column] = texel;
        }
    };

    //  The estimator
    //-------------------------------------------------------------------------
    // A sample of the half-vector from the NDF the roughness selects, reflected about V to give L, weighted by the BRDF against the density of that sample.
    // DFGIntegrand.h has the expression and where it comes from.

    DFGTexel EvaluateDFGTexel( double ndotV, double roughness, uint32_t sampleCount, DFGSampleSequence sequence );

    // The whole grid, from the Hammersley sequence the engine's pass uses. Rows are independent, so the grid is filled a row at a time.
    void BuildDFGTable( DFGOptions const& options, DFGTable& table );

    //  Half floats
    //-------------------------------------------------------------------------

    uint16_t FloatToHalf( float const value );
    float    HalfToFloat( uint16_t const value );

    // One texel's two channels, in the payload's order.
    struct DFGHalfTexel
    {
        uint16_t m_scale = 0;
        uint16_t m_bias = 0;
    };

    void GetHalfTexel( DFGTable const& table, uint32_t column, uint32_t row, DFGHalfTexel& texel );
}
