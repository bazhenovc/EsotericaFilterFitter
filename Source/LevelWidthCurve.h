#pragma once

#include <cstdint>
#include <vector>

// Lobe width at each level
//-------------------------------------------------------------------------
// A profile supplies a lobe shape; this supplies its width at each level.
// They are separate because a renderer changes them separately: an NDF, and as roughness-to-mip mapping.
//
// Width is whatever the profile interprets it as. 
// For an NDF written in terms of a GGX-style alpha it is alpha, i.e. roughness squared.
//
// Two named curves exist, and a third constructor takes data:
//
//      paper      the reference's gloss curve at spec power 18, which the published tables were fitted against. 
//                 Geometric in roughness: 0.0526, 0.0884, 0.1487, 0.2498, 0.4190, 0.6866, 1.0
//      esoterica  roughness linear across the levels, which is what this project's runtime does: 0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0.
//                 A deliberate choice rather than the reference's, and the reason the two are separable here at all.
//      explicit   one width per level, for a fork whose curve is neither
//
// The choice decides which seven filter widths the table's seven levels encode, so the runtime's roughness-to-level mapping has to invert whichever curve the table was fitted with.
// Fitting one curve and selecting with another is a silent quality loss, not an error.
//
// Esoterica starts at zero width, which is a mirror rather than a narrow lobe.
// Zero is legal here and the reference preimage handles it; see Profile.h.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    class LevelWidthCurve
    {
    public:

        // roughness = ( 2 / ( 1 + 2^( gloss * maxSpecPower ) ) )^0.25 with gloss falling
        // from 1 to 0 across the levels; the width is roughness squared. 
        // Spec power 18 reproduces the published tables.
        static LevelWidthCurve MakePaperGloss( uint32_t numLevels, double maxSpecPower );

        // roughness = level / ( numLevels - 1 ), squared. Level 0 is a mirror.
        static LevelWidthCurve MakeEsotericaLinear( uint32_t numLevels );

        static LevelWidthCurve MakeExplicit( std::vector<double> const& widths );

        uint32_t    GetNumLevels() const { return static_cast<uint32_t>( m_widths.size() ); }
        double      GetWidth( uint32_t level ) const;
        char const* GetName() const { return m_name; }

        std::vector<double> const& GetWidths() const { return m_widths; }

        // Non-decreasing, which any usable curve is. Zero is allowed at the start.
        bool IsNonDecreasing() const;

        // True when level 0 is a mirror, so the preimage for it is the identity
        bool StartsAtMirror() const { return ( !m_widths.empty() ) && ( m_widths[0] <= 0.0 ); }

    private:

        std::vector<double> m_widths;
        char                m_name[32] = {};
    };
}
