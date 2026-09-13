#pragma once

#include <cstdint>

#include "DFGIntegrand.h"
#include "DFGTable.h"

// Validation of the preintegrated split-sum DFG term
//-------------------------------------------------------------------------
// The table is one estimator of the integral and this is another, over the same integrand and the same grid.
// Both weight a sample of the half-vector by the BRDF against the density of that sample; what differs is where the samples are:
//
//      table       the van der Corput radical inverse of the sample index, which is the sequence the engine's own pass uses
//      reference   a Cartesian stratification shifted by an irrational offset, at several times the table's sample count, so that most of the deviation between them is the table's error rather than half of each
//
// Sampling the NDF is what makes a table at a few thousand samples possible at all: a uniform quadrature of the hemisphere cannot resolve a lobe a thousandth of a radian wide without millions of points.
// The price is that both are Monte Carlo, so their disagreement is bounded below by the reference's own error, and this measures that error directly - the reference at a fraction of its sample count against itself - and prints it beside the table's deviation.
//
// Sampling the half-vector is also what an estimator gets wrong, and a second sampling sequence cannot catch that: two estimators that share a weight share its fault.
// So the two rows the estimator is best conditioned at are also evaluated by a deterministic quadrature of the hemisphere, from the BRDF written out longhand rather than through the sampled estimator's expression.
// That one shares nothing with the estimator, which is what makes it a check of it.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    // Samples per texel in the reference. Several times the table's own count, so that the deviation between them is mostly the table's error rather than half of each.
    // The cost is this many samples per texel over the whole grid, so it is the slow part of the mode.
    inline constexpr uint32_t DFGDefaultReferenceSampleCount = 65536;

    // The reference's own convergence is measured at this fraction of its sample count, which bounds the reference's error from the change the extra samples make.
    inline constexpr uint32_t DFGConvergenceDivisor = 4;

    // Cells per axis in the deterministic quadrature, over the polar angle and the azimuth. 
    // The integrand is smooth at the widths it is used at - the BRDF's 1 / ( N.L ) is cancelled by the cosine and its shadowing term vanishes with N.L - so midpoint cells converge quickly and this is generous.
    inline constexpr uint32_t DFGQuadratureThetaCells = 512;
    inline constexpr uint32_t DFGQuadraturePhiCells = 512;

    // The rows the quadrature runs at. Both are wide enough that a hemisphere quadrature resolves their lobe, which is a property of the width and the reason the smoothest rows are not among them.
    inline constexpr uint32_t DFGQuadratureRows[2] = { 77, 115 };

    //-------------------------------------------------------------------------

    struct DFGPointError
    {
        double  m_absolute = 0.0;
        double  m_tableScale = 0.0;
        double  m_tableBias = 0.0;
        double  m_referenceScale = 0.0;
        double  m_referenceBias = 0.0;
        uint32_t m_column = 0;
        uint32_t m_row = 0;
    };

    struct DFGRegion
    {
        uint32_t        m_columnBegin = 0;
        uint32_t        m_columnEnd = 0;
        uint32_t        m_rowBegin = 0;
        uint32_t        m_rowEnd = 0;
        uint32_t        m_numTexels = 0;

        double          m_meanAbsoluteScale = 0.0;
        double          m_meanAbsoluteBias = 0.0;
        double          m_worstAbsoluteScale = 0.0;
        double          m_worstAbsoluteBias = 0.0;
        double          m_signedMeanScale = 0.0;
        double          m_signedMeanBias = 0.0;
    };

    struct DFGComparison
    {
        static constexpr uint32_t NumBands = 4;

        uint32_t        m_resolution = 0;
        uint32_t        m_numTexels = 0;

        double          m_worstAbsoluteScale = 0.0;
        double          m_worstAbsoluteBias = 0.0;
        double          m_meanAbsoluteScale = 0.0;
        double          m_meanAbsoluteBias = 0.0;
        double          m_signedMeanScale = 0.0;
        double          m_signedMeanBias = 0.0;

        DFGPointError   m_worstScale;
        DFGPointError   m_worstBias;

        // Indexed [ row band ][ column band ], with roughness banding the rows and N dot V the columns.
        DFGRegion       m_regions[NumBands][NumBands];
    };

    // Every texel of one against the other. The reference is a full grid of its own, so the cost is the reference's.
    void CompareDFGTables( DFGTable const& table, DFGTable const& reference, DFGComparison& comparison );

    //  The deterministic quadrature
    //-------------------------------------------------------------------------
    // Midpoint cells over the hemisphere, with the specular BRDF written longhand:
    //
    //      D  = alpha^2 / ( PI ( ( N.H )^2 ( alpha^2 - 1 ) + 1 )^2 )
    //      G  = G1( N.V ) G1( N.L ), Schlick with k = roughness / 2
    //      f  = D G F / ( 4 ( N.V ) ( N.L ) ),  the F above being the split's own term
    //
    // and the cosine folded in, so what is summed is D G ( 1 - F ) and D G F, each over 4 ( N.V ), against the solid angle of its cell. 
    // Nothing here is sampled from an NDF and nothing here divides by ( N.H ), which is the point of it.

    DFGTexel EvaluateDFGTexelQuadrature( double ndotV, double roughness, uint32_t numThetaCells, uint32_t numPhiCells );

    //-------------------------------------------------------------------------

    // Runs the whole mode: the table, the reference, the comparison, the degenerate cases, and the output files. Returns the process exit code.
    int RunDFGValidation
    (
        char const* pBinaryPath,
        char const* pHeaderPath,
        uint32_t resolution,
        uint32_t sampleCount,
        uint32_t referenceSampleCount
    );
}
