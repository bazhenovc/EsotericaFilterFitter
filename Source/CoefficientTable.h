#pragma once

#include <cstdint>
#include <vector>

#include "ProbeMap.h"
#include "Profile.h"

// The vendored files come in two shapes:
//
//      const   float4 coeffs[7][5][3 * NUM_TAPS / 4]
//      quad    float4 coeffs[7][5][3][3 * NUM_TAPS / 4]
//
// A constant table is the degenerate quadratic with a1 = a2 = 0, so both are widened here to [level][parameter][coefficient][index] and evaluated through one code path.
//
// The leading 3 in those extents is the AXIS COUNT, and it is a property of the table rather than of the layout.
// A cubemap's frame has three axes; a regular tetrahedron's has four, so its table is
//
//      quad    float4 coeffs[7][5][3][4 * NUM_TAPS / 4]
//
// and the two are one layout with a different number in the last extent.
// 
// The published files fix theirs at three because they are cube data; Create takes the frame's count. 
// Nothing else in the layout depends on it, and GetNumIndices and GetTapCount are the only two places it appears.
//
// The table also records WHICH MAP it is for, not merely how many axes that map has.
// The two are not the same statement: the axis count says how to index the array, the map says what the indices mean, and a reader that is handed the wrong one of either is being handed a plausible wrong answer.
// 
// Both are written into the binary and both are checked against what the reader already knows.
//
// parameter is 0..4 for dir0, dir1, dir2, level, weight. 
// coefficient is 0..2 for the 1, theta^2 and phi^2 terms of the per-texel polynomial.
// Each stored element is a float4 holding four sub-taps, so index addresses a group of four and subTap selects within it; index is laid out as ( NUM_TAPS / 4 ) * axis + superTap.
//
// Storage is double, not float. The published literals are float and widen exactly, so reading them loses nothing; but the fit writes this same table, and rounding its iterates to float every evaluation would cap the accuracy of the result at the format's precision rather than at the objective's.
//
// The same type is therefore both the conformance test's input and the optimizer's output, which is what guarantees the fit optimizes the function the conformance test measures.
// 
// Create + SetLevelParameters builds one from scratch; Load fills one from a vendored file.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    enum class ReferenceTable : uint8_t
    {
        Const8,
        Const16,
        Const32,
        Quad32,
    };

    // The shape of a table, as a value
    //-------------------------------------------------------------------------
    // Everything that identifies a table's LAYOUT and its meaning, and nothing about its contents: how many taps along one axis, how many axes (and therefore which map), how many coefficient terms, and a human-readable name.
    //
    // It exists because three separate things have to agree on it - the table itself, the fit that produces one, and the checkpoint that records which  fit produced one - and before this they each named a published shape by its  enum, which cannot express a map the paper did not publish.
    //
    // The implicit constructor from ReferenceTable is deliberate: a published shape IS a table shape, and every existing call site that says "const_8" should keep saying it and mean exactly that.
    //-------------------------------------------------------------------------

    struct TableShape
    {
        char        m_name[32] = {};
        uint32_t    m_numTapsPerAxis = 0;
        uint32_t    m_numAxes = 0;
        uint32_t    m_numActiveCoefficients = 0;
        ProbeMap    m_projection = ProbeMap::Cube;

        // No published table has this shape, so the checkpoint's published-shape field says so rather than naming a table that is not this one.
        static constexpr uint32_t PublishedShapeNone = 0xFFFFFFFFu;

        TableShape() = default;

        // A published shape, which is cubemap data
        TableShape( ReferenceTable table );

        // Any shape, for a map the paper did not publish. The axis count is taken from the map rather than passed, so the two cannot disagree.
        static TableShape ForMap( ProbeMap map, uint32_t numTapsPerAxis, uint32_t numActiveCoefficients, char const* pName );

        // The published table this shape is, or PublishedShapeNone
        uint32_t GetPublishedShape() const;

        bool operator==( TableShape const& other ) const;
        bool operator!=( TableShape const& other ) const { return !( *this == other ); }
    };

    //-------------------------------------------------------------------------

    class CoefficientTable
    {
    public:

        static constexpr uint32_t NumLevels = 7;  // PROBE_REFLECTION_MIPS
        static constexpr uint32_t NumParameters = 5;
        static constexpr uint32_t NumCoefficients = 3;

        // The axis count of every PUBLISHED table, because the four shapes in Reference/ are cubemap data.
        // It is not the axis count of the layout: a table carries its own, and a regular tetrahedron's frame has four axes.
        static constexpr uint32_t NumCubeAxes = 3;

        // Parameter indices, so callers do not hardcode them
        static constexpr uint32_t ParameterDir0 = 0;
        static constexpr uint32_t ParameterDir1 = 1;
        static constexpr uint32_t ParameterDir2 = 2;
        static constexpr uint32_t ParameterLevel = 3;
        static constexpr uint32_t ParameterWeight = 4;

    public:

        // Fills from the vendored file
        void Load( ReferenceTable table );

        // Allocates the shape of the named published table and zeroes it, so the fit has a starting point with no reference to the published coefficients
        void Create( ReferenceTable table );

        // Allocates a shape and zeroes it. The shape carries the map, the axis count and the tap count, and those three have to agree.
        // TableShape is what makes them one value rather than three arguments.
        void Create( TableShape const& shape );

        // The shape this table has, for anything that has to name it: the writer, the fit's fingerprint and the checkpoint.
        inline TableShape const& GetShape() const { return m_shape; }

        inline uint32_t     GetNumTaps() const { return m_numTaps; }
        inline uint32_t     GetNumAxes() const { return m_numAxes; }
        inline ProbeMap     GetProjection() const { return m_shape.m_projection; }
        inline uint32_t     GetNumIndices() const { return ( m_numAxes * m_numTaps ) / 4; }
        inline char const*  GetName() const { return m_pName; }

        // 1 for a constant table, 3 for a quadratic one
        inline uint32_t     GetActiveCoefficientCount() const { return m_numActiveCoefficients; }

        // Sub-taps in a level: axes * ( NUM_TAPS / 4 ) supertaps * 4. Tap t corresponds to stored ( index, subTap ) = ( t / 4, t % 4 ).
        inline uint32_t     GetTapCount() const { return m_numAxes * m_numTaps; }

        // Unknowns in one level: tap count * 5 parameters * active coefficients
        inline uint32_t     GetLevelParameterCount() const { return GetTapCount() * NumParameters * m_numActiveCoefficients; }

        double GetCoefficient( uint32_t level, uint32_t parameter, uint32_t coefficient, uint32_t index, uint32_t subTap ) const;
        void   SetCoefficient( uint32_t level, uint32_t parameter, uint32_t coefficient, uint32_t index, uint32_t subTap, double value );

        // The same, addressed by tap rather than by ( index, subTap )
        double GetTapCoefficient( uint32_t level, uint32_t tap, uint32_t parameter, uint32_t coefficient ) const;
        void   SetTapCoefficient( uint32_t level, uint32_t tap, uint32_t parameter, uint32_t coefficient, double value );

        // One level's unknowns as a flat vector, laid out [tap][parameter][coefficient]
        void GetLevelParameters( uint32_t level, std::vector<double>& values ) const;
        void SetLevelParameters( uint32_t level, std::vector<double> const& values );

        // Verifies internal shape consistency. Returns false if a stored value would be addressed out of range, so a bad Create or Load is caught before it produces a plausible-looking number.
        bool ValidateShape() const;

    private:

        // Storage is [level][parameter][coefficient][index][subTap], always with all NumCoefficients slots allocated even when only one is used
        size_t GetStorageSize() const;
        size_t GetStorageOffset( uint32_t level, uint32_t parameter, uint32_t coefficient, uint32_t index, uint32_t subTap ) const;

    private:

        uint32_t            m_numTaps = 0;
        uint32_t            m_numAxes = NumCubeAxes;
        uint32_t            m_numActiveCoefficients = 0;
        TableShape          m_shape;
        char const*         m_pName = "";
        std::vector<double> m_values;
    };

    //-------------------------------------------------------------------------

    char const* GetReferenceTableName( ReferenceTable table );
}
