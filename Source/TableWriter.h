#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "CoefficientTable.h"
#include "LevelWidthCurve.h"

// writing a fitted table as a C header
//-------------------------------------------------------------------------
// The emitted file is shaped like the vendored reference tables in Reference/:  a static const float4 coeffs[...] array using HLSL constructor syntax.
// That is deliberate rather than incidental. The format exists so that a test translation unit can include ours in place of a vendored table and compare the two as ordinary arrays, with no adapter between them.
//
// It follows that the file is NOT self-contained: it needs float4 to be defined by the includer, exactly as the vendored files do. CoefficientTable.cpp holds the shim. 
// A header for the runtime wants the engine's own types and a different layout, and that is a later job rather than this one.
//
// Shape-agnostic by construction. Tap count, index count, active coefficient count and the array's rank all come from the table, because all four published shapes are in scope: const_8, const_16, const_32 and quad_32 differ in exactly those dimensions, and NUM_TAPS is per axis with the tap count the axis count times it.
// The AXIS COUNT comes from the table too, which is what lets a tetrahedral table be written by the same code as a cubemap's rather than by a variant of it.
//
// Values are emitted as float, not double, because the runtime stores float4.
// The fit stays double so that rounding never caps the objective; the narrowing happens once, here, at the format boundary. %.9g is FLT_DECIMAL_DIG, the smallest digit count that round-trips every float.
// %.8g is enough for the published tables, whose literals are already eight-digit decimals, and not enough for a fitted one.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    // The binary the engine's DataEmbed tool consumes
    //-------------------------------------------------------------------------
    // A fixed 144-byte header followed by the float4 payload, so the payload is 16-byte aligned within the file.
    // Every field is a uint32, a float or a char array, so there is no padding and sizeof is the on-disk size; the assert below holds that true.
    //
    // The payload is the same float4 array the text header declares, in the same order, so the runtime indexes both identically.
    //
    //  WHAT THE HEADER HAS TO CARRY, AND WHY
    //
    // A table is only valid against the profile and the width curve it was fitted with.
    // Selecting with a different curve is a silent quality loss rather than an error, so the runtime needs enough to detect the mismatch itself:
    //
    //      shape     the coefficient layout, because a constant table has no
    //                coefficient dimension where quad_32 has three
    //      profile   the NDF, because two profiles can share a width curve and produce
    //                different tables
    //      widths    the width at each level, which is the authority on what the curve
    //                was. Names are not carried: an explicit curve has no name beyond
    //                "explicit", and seven numbers identify it exactly.
    //
    // Little endian, matching the only platform this builds on.
    //
    // The runtime needs this layout when the gather is integrated and cannot include this header.
    // If this struct changes, the engine's copy changes with it, and the version field is what lets a stale blob be rejected rather than misread.
    //-------------------------------------------------------------------------

    // 1 had no profile or curve identity, so a table fitted for one curve could be loaded by a runtime selecting with another and nothing would notice.
    // Version 2 added that identity.
    //
    // 3 added the axis count.
    // Version 2's header could only describe a cubemap's three axes, because its index count was fixed at 3 * taps / 4, so a table over a tetrahedron's four had no way to say so and would have been read as a cube's.
    // A v2 blob is rejected rather than misread.
    //
    // 4 added the map the table is FOR, which is a different statement from the axis count even though one follows from the other today: the count sizes the array, the map says what the array means.
    // A v3 blob says how to index the coefficients but not which map they are for, so it is rejected as well - this tool can rewrite one, a runtime cannot guess it.
    static constexpr uint32_t TableBinaryVersion = 4;

    static constexpr uint32_t TableBinaryMagic = 0x4C425446;  // reads as FTBL in a dump

    // EVERY FIELD IS UNTRUSTED WHEN READ
    //-------------------------------------------------------------------------
    // A blob can be replaced by a fork, corrupted by a bad copy, or authored by someone hostile, and the runtime that loads it is the last place that can refuse.
    // So the reader checks, before it uses anything:
    //
    //      the magic, the version, and that the map is a map
    //      every shape field against the table the caller already has
    //      that the shape fields agree with EACH OTHER, because a file that lies
    //      consistently is still a lie
    //      the payload offset, alignment and size against the actual file length,
    //      before allocating anything sized by a field from the file
    //      that the names are terminated inside their arrays, and the widths finite
    //
    // Nothing here is a hardening gesture: every one of these has a plausible wrong answer rather than a crash, which is why the checks exist.
    //-------------------------------------------------------------------------

    struct TableBinaryHeader
    {
        uint32_t    m_magic;
        uint32_t    m_version;
        uint32_t    m_numLevels;
        uint32_t    m_numParameters;
        uint32_t    m_numActiveCoefficients;    // 1 for a constant table, 3 for a quadratic
        uint32_t    m_numTapsPerAxis;
        uint32_t    m_projection;               // the ProbeMap value, an on-disk number
        uint32_t    m_numAxes;                  // 3 for a cube, 4 for a tetrahedron
        uint32_t    m_numIndices;               // m_numAxes * taps / 4
        uint32_t    m_numFloat4;                // float4 in the payload
        uint32_t    m_payloadOffset;            // bytes, 16-byte aligned
        uint32_t    m_payloadBytes;             // m_numFloat4 * 16

        char        m_shapeName[16];
        char        m_profileName[48];
        float       m_widths[CoefficientTable::NumLevels];

        uint32_t    m_reserved[1];
    };

    static_assert( sizeof( TableBinaryHeader ) == 144, "the binary header is a file layout and must not gain padding" );

    // Formats the table. text is replaced rather than appended to.
    void FormatTableHeader( CoefficientTable const& table, std::string& text );

    // Formats and writes. False if the file could not be opened or a write failed, so a caller can tell a written table from a silently missing one.
    bool WriteTableHeader( CoefficientTable const& table, char const* pPath );

    // The same table as the binary above. Exposed separately from the file write so the self-check can parse the result back without touching the filesystem.
    void FormatTableBinary
    (
        CoefficientTable const& table,
        char const* pProfileName,
        LevelWidthCurve const& curve,
        std::vector<uint8_t>& bytes
    );

    bool WriteTableBinary
    (
        CoefficientTable const& table,
        char const* pProfileName,
        LevelWidthCurve const& curve,
        char const* pPath
    );

    // Formats every published shape both ways, parses each result back, and requires every value to survive exactly, with the brace nesting the declared rank implies and a binary header that matches what it was written from.
    // Run with the rest of the startup self-checks.
    bool RunTableWriterSelfCheck();
}
