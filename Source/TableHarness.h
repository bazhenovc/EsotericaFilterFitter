#pragma once

#include <cstdint>
#include <vector>

#include "BsplineRecurrence.h"
#include "CoefficientTable.h"
#include "CubeReferenceFrame.h"
#include "Profile.h"
#include "ReferencePreimage.h"

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    struct TableTap
    {
        double  m_direction[3];
        double  m_level;
        double  m_weight;

        // Index of this tap in the PARAMETER layout, which is not its position in the tap list.
        // Frames whose blend weight is zero are skipped, and the tangent frame degenerates on the axis it is built from, so a level normally contributes two of its three axes - 16 taps rather than 24 for const_8. 
        // Anything addressing a tap's coefficients by list position is therefore addressing the wrong tap, which is silent and produces a gradient of the right magnitude pointing the wrong way.
        uint32_t m_tapIndex;

        // The blend weight of the axial frame this tap belongs to, kept separate from m_weight because m_weight already has it folded in. 
        // The weight gradient needs the frame weight on its own: the normalising sum that the runtime divides by is the frame-weighted sum, so the two enter the derivative separately.
        double  m_frameWeight;

        // The polynomial arguments at the output texel, per axis. 
        // Constant across a tap's sub-taps but not across axes, so they are carried on the tap.
        double  m_theta2;
        double  m_phi2;
    };

    // Reconstruct the trilinear samples a table prescribes for one output texel.
    // Mirrors Reference/filter_using_table_128.txt.
    //
    // Templated on the base map's projection, so the construction carries no assumption about what the environment is stored as.
    // The map supplies the six answers listed in MapProjection.h and nothing else.
    // The first coordinate is a SLICE and not a face: a level is made of slices, and the two coincide only on a map with one face per slice.
    template< typename TMap >
    void BuildTableTaps
    (
        CoefficientTable const& table,
        uint32_t level,
        uint32_t slice,
        uint32_t texelX,
        uint32_t texelY,
        uint32_t baseResolution,
        std::vector<TableTap>& taps
    );

    //-------------------------------------------------------------------------

    struct EvaluationSettings
    {
        uint32_t            m_baseResolution = 128;

        // 1 = point sample the kernel at each base texel centre, which is the definitionally correct target: the runtime's tap weights reconstruct a discrete measure on the base cube, so the target is that same discrete measure.
        // Values above 1 average the kernel over each texel instead, which is the better quadrature of the continuous kernel but a different function.
        // The paper does not state which it used; its level-0 table selects point sampling.
        uint32_t            m_supersampleRate = 1;

        // Sampling grid over each face, per axis
        uint32_t            m_gridSize = 4;

        // Levels in the intermediate mip chain the taps read.
        // The coefficient table indexes 7 OUTPUT levels, but spec_params_for_mip.txt defines PROBE_REFLECTION_MIPS_FULL = 8, so the chain the taps read carries an 8th level at resolution 1. 
        // Measured: adding it cuts quad_32's level-6 excess from 0.0207 to 0.0073.
        uint32_t            m_sampleLevelCount = 8;

        // Separable (0.25, 0.5, 0.25) passes applied to B before normalisation.
        // The paper states its fit target accounts for the trilinear reconstruction convolution; this measures whether that is the case.
        uint32_t            m_referenceSmoothing = 0;

        ErrorMeasure        m_measure = ErrorMeasure::SolidAngle;
        JacobianWeighting   m_weighting = JacobianWeighting::Jacobian;

        bool                m_verbose = false;
    };

    //-------------------------------------------------------------------------

    struct TableEvaluationLevel
    {
        double      m_averageL1 = 0.0;
        double      m_minL1 = 0.0;
        double      m_maxL1 = 0.0;
        double      m_averageTapWeight = 0.0;
        uint32_t    m_numTexels = 0;
        uint32_t    m_numTaps = 0;

        // Mean L1 bucketed by max(|u|,|v|) of the output texel, which is what selects how many axial frames are active.
        // Bands are 0-0.5 (the paper's 36 centre tiles), 0.5-0.75 and 0.75-1.0 (the 28 edge tiles).
        static constexpr uint32_t NumPositionBands = 3;
        double      m_bandAverageL1[NumPositionBands] = {};
        uint32_t    m_bandNumTexels[NumPositionBands] = {};
    };

    struct TableEvaluation
    {
        TableEvaluationLevel    m_levels[CoefficientTable::NumLevels];

        // The paper's Table 1 averages over levels 1-6, excluding level 0
        double                  m_averageL1 = 0.0;
        uint32_t                m_numTexels = 0;
    };

    // Splat a table's taps for a sample of output texels and score them against the reference preimage.
    template< typename TFrame, typename TProfile >
    TableEvaluation EvaluatePublishedTable
    (
        ReferenceTable table,
        TFrame const& baseFrame,
        TProfile const& profile,
        EvaluationSettings const& settings
    );

    // One level of the above, for scans that perturb a per-level input.
    template< typename TFrame, typename TProfile >
    TableEvaluationLevel EvaluatePublishedTableLevel
    (
        CoefficientTable const& table,
        TFrame const& baseFrame,
        TProfile const& profile,
        EvaluationSettings const& settings,
        uint32_t level
    );

    // Lobe angular width at a level, divided by the texel angular size at that level.
    // Below 1 the lobe is finer than one texel, which is where the conformance gap concentrates.
    //
    // This is a CUBE-FACE statistic and is only used by the gap diagnostics, which run against the paper's published cubemap tables: the texel size here is taken at a cube face centre.
    // It is not a general map property, and a map whose texels vary by 27:1 has no single answer to it.
    template< typename TProfile >
    double GetLobeToTexelRatio( TProfile const& profile, uint32_t level, uint32_t baseResolution );

    // Position band index for a face-space coordinate, matching the band boundaries in TableEvaluationLevel
    uint32_t GetPositionBand( double u, double v );

    // The output texels an evaluation samples, as flat ( slice, texelX, texelY ) triples, in a fixed order. 
    // Shared so that anything caching per-output-texel data - the fit's reference preimages in particular - cannot disagree with the evaluation about which texel is which.
    //
    // By SLICE and not by face: a level is a slice of a map, and the two are the same thing only on a cubemap, whose six faces are six slices. 
    // A map that holds four faces in one slice has one slice here and four faces inside it.
    template< typename TMap >
    void BuildOutputTexelList( uint32_t resolution, uint32_t gridSize, std::vector<uint32_t>& texels );

    // The clamped sampling grid per face for a level: min( gridSize, resolution )
    uint32_t GetFaceGridSize( uint32_t resolution, uint32_t gridSize );

    //-------------------------------------------------------------------------

    extern template void BuildTableTaps< CubeProjection >( CoefficientTable const&, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, std::vector<TableTap>& );

    extern template void BuildTableTaps< TetrahedralProjection >( CoefficientTable const&, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, std::vector<TableTap>& );
}
