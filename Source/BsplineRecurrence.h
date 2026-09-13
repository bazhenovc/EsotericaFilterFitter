#pragma once

#include <cstdint>
#include <vector>

#include "MapProjection.h"
#include "TetrahedralProjection.h"

// Quadratic b-spline recurrence, and its transpose
//-------------------------------------------------------------------------
// One coarser texel is generated from a 4x4 block of finer texels through  bilinear taps, positioned at +/-0.75 of a fine texel from its centre.
// Tap weights are shifted by the sphere-to-map Jacobian, blended 50/50 with a constant.
//
// The code and the reference excerpt disagree on the direction of that shift, by a reciprocal:
//
//      calcWeight( u, v )    = ( u^2 + v^2 + 1 )^(3/2)
//      J( x, y, z )          = 1 / ( x^2 + y^2 + z^2 )^(3/2)
//
// For a face whose direction is (u, v, 1) those are reciprocal, so the code weights towards face corners and the reference weights away from them.
// Evaluating the published tables gives 0.138 for J against 0.231 for 1/J, so the reference is what produced the tables and its calcWeight is inverted.
// J here is the map's own GetJacobian, which is the same quantity for either map.
//
//  CROSS-FACE TAPS
//
// A coarser texel's taps stay inside its own face, but their bilinear footprints do not: the leftmost tap at R = 64 lands at finer  -0.25, so its 2x2 block reaches texel -1 on the neighbouring face.
// Dropping those breaks constant reproduction along every edge.
// They are resolved through direction space as hardware does - extrapolate the coarse texel's own chart to the phantom texel's coordinate, giving a direction, then re-select the face by dominant axis and take the nearest texel - which is why a footprint stores an absolute ( face, x, y ) and both directions run over every slice.
//
//  THE CHART MATTERS FOR THE EXTRAPOLATION
//
// A phantom texel is placed by continuing the COARSE TEXEL'S OWN chart past the edge, not by asking the map which face owns the phantom's coordinate. 
// For a cube the two agree, because neighbouring faces meet with matching grids.
// For a single-slice tetrahedral map they do not: the tile's four triangles meet at a point and their images are not continuous there, so asking the coordinate would read the wrong chart's value.
// The face of the coarse texel is therefore resolved from its centre and used for every phantom of that footprint.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    enum class JacobianWeighting : uint8_t
    {
        None,               // Plain b-spline: the (1/8, 3/8, 3/8, 1/8) tensor product
        InverseJacobian,    // calcWeight, i.e. 1/J - what the vendored code computes
        Jacobian,           // J - what the reference describes, and what scores better
    };

    //-------------------------------------------------------------------------

    struct DownsampleFootprint
    {
        static constexpr uint32_t MaxTexels = 16;

        // Absolute finer-level texel, on whichever face it resolved to. 
        // Entries landing on the same texel are merged, so fewer than 16 is normal near a map's seams.
        uint32_t    m_face[MaxTexels] = {};
        uint32_t    m_texelX[MaxTexels] = {};
        uint32_t    m_texelY[MaxTexels] = {};
        double      m_weight[MaxTexels] = {};
        uint32_t    m_count = 0;

        bool        m_crossesFace = false;
    };

    //-------------------------------------------------------------------------

    // Footprint of one coarser texel on the finer level, which is always twice the coarser resolution.
    // coarseSlice is a slice index, not a face index: a map that holds several faces in one slice resolves the face itself.
    template< typename TMap >
    void ComputeDownsampleFootprint
    (
        uint32_t coarseSlice,
        uint32_t coarseX,
        uint32_t coarseY,
        uint32_t coarseResolution,
        JacobianWeighting weighting,
        DownsampleFootprint& footprint
    );

    // One level, every slice. Sizes are NumSlices * ( 2 * coarseResolution )^2 and NumSlices * coarseResolution^2.
    template< typename TMap >
    void DownsampleLevel
    (
        std::vector<double> const& finer,
        uint32_t coarseResolution,
        JacobianWeighting weighting,
        std::vector<double>& coarser
    );

    // The exact transpose of DownsampleLevel
    template< typename TMap >
    void UpsampleLevel
    (
        std::vector<double> const& coarser,
        uint32_t coarseResolution,
        JacobianWeighting weighting,
        std::vector<double>& finer
    );

    // Cached transpose
    //-------------------------------------------------------------------------
    // UpsampleLevel recomputes ComputeDownsampleFootprint for every non-zero coarser texel on every call.
    // In the fit that recomputation is the whole cost of an objective evaluation - measured at 96% of it at level 6 - and the footprint depends only on geometry: the slice, the texel coordinate, the coarse resolution and the weighting. Never on the field.
    //
    // This builds the transpose once, as a sparse matrix in compressed row form indexed by the FINER texel. Apply is then a pure gather, so it needs no zeroed output, no atomics, and gives a fixed summation order per element, which is what makes a parallel result reproducible.
    //
    // The memory cost is bounded by the footprint: at most 16 entries per finer texel, and merging near the seams brings the measured average to about 10.
    // For base 128 and 8 levels that is roughly 330k entries for a cubemap, or 4 MB of index and 2.6 MB of weight.
    //-------------------------------------------------------------------------

    template< typename TMap >
    class UpsampleOperator
    {
    public:

        // Levels are baseResolution >> level, so numLevels must satisfy baseResolution >> ( numLevels - 1 ) > 0.
        void Build( uint32_t baseResolution, uint32_t numLevels, JacobianWeighting weighting );

        // Overwrites finer with the transpose of the downsample from coarseLevel to coarseLevel - 1. coarseLevel is in [1, GetNumLevels()).
        void Apply( uint32_t coarseLevel, std::vector<double> const& coarser, std::vector<double>& finer ) const;

        // The same, accumulated into a caller-owned target that is already the right size. This is what ResolveToBase uses, so it needs no scratch.
        void ApplyAdd( uint32_t coarseLevel, std::vector<double> const& coarser, std::vector<double>& target ) const;

        // Finer texel count for the level pair coarseLevel - 1, i.e. the size Apply writes
        uint32_t GetFineTexelCount( uint32_t coarseLevel ) const;

        bool        IsInitialized() const { return m_numLevels > 1; }
        uint32_t    GetNumLevels() const { return m_numLevels; }
        uint32_t    GetNumEntries() const { return m_numEntries; }
        size_t      GetNumBytes() const;

    private:

        struct LevelOperator
        {
            // Row f of the transpose is sources[offsets[f] .. offsets[f+1])
            std::vector<uint32_t>   m_offsets;
            std::vector<uint32_t>   m_sources;
            std::vector<double>     m_weights;
        };

        uint32_t                    m_baseResolution = 0;
        uint32_t                    m_numLevels = 0;
        uint32_t                    m_numEntries = 0;
        std::vector<LevelOperator>  m_levels;   // m_levels[level - 1] maps level -> level - 1
    };

    // Process-wide operator cache
    //-------------------------------------------------------------------------
    // Building the operator costs two full passes over every footprint - about 15 ms and 6.5 MB at base 128 and 8 levels.
    // That is far too much to pay per worker or per level evaluation, and a run only ever has a handful of distinct ( map, resolution, level count, weighting ) keys.
    //
    // Entries are never evicted, so a returned reference stays valid for the lifetime of the process and holders can keep a pointer to it.
    //-------------------------------------------------------------------------

    template< typename TMap >
    UpsampleOperator< TMap > const& GetUpsampleOperator
    (
        uint32_t baseResolution,
        uint32_t numLevels,
        JacobianWeighting weighting
    );

    char const* GetJacobianWeightingName( JacobianWeighting weighting );

    //-------------------------------------------------------------------------

    extern template void ComputeDownsampleFootprint< CubeProjection >( uint32_t, uint32_t, uint32_t, uint32_t, JacobianWeighting, DownsampleFootprint& );
    extern template void ComputeDownsampleFootprint< TetrahedralProjection >( uint32_t, uint32_t, uint32_t, uint32_t, JacobianWeighting, DownsampleFootprint& );

    extern template void DownsampleLevel< CubeProjection >( std::vector<double> const&, uint32_t, JacobianWeighting, std::vector<double>& );
    extern template void DownsampleLevel< TetrahedralProjection >( std::vector<double> const&, uint32_t, JacobianWeighting, std::vector<double>& );

    extern template void UpsampleLevel< CubeProjection >( std::vector<double> const&, uint32_t, JacobianWeighting, std::vector<double>& );
    extern template void UpsampleLevel< TetrahedralProjection >( std::vector<double> const&, uint32_t, JacobianWeighting, std::vector<double>& );

    extern template class UpsampleOperator< CubeProjection >;
    extern template class UpsampleOperator< TetrahedralProjection >;

    extern template UpsampleOperator< CubeProjection > const& GetUpsampleOperator< CubeProjection >( uint32_t, uint32_t, JacobianWeighting );
    extern template UpsampleOperator< TetrahedralProjection > const& GetUpsampleOperator< TetrahedralProjection >( uint32_t, uint32_t, JacobianWeighting );
}
