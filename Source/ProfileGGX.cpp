#include "Assert.h"
#include "ProfileGGX.h"

#include <cmath>

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    double ProfileGGX::GetGloss( uint32_t level )
    {
        FF_ASSERT( level < NumLevels );

        return static_cast<double>( ( NumLevels - 1 ) - level ) / static_cast<double>( NumLevels - 1 );
    }

    //-------------------------------------------------------------------------

    double ProfileGGX::GlossToRoughness( double gloss )
    {
        double const exponent = std::pow( 2.0, gloss * MaxSpecPower );

        return std::pow( 2.0 / ( 1.0 + exponent ), 0.25 );
    }

    //-------------------------------------------------------------------------

    double ProfileGGX::GetAlpha( uint32_t level )
    {
        // GlossToRoughness returns sqrt(alpha)
        double const roughness = GlossToRoughness( GetGloss( level ) );

        return roughness * roughness;
    }

    //-------------------------------------------------------------------------

    ProfileGGX::ProfileGGX( LobeConvention convention, double alphaScale )
        : ProfileGGX( LevelWidthCurve::MakePaperGloss( NumLevels, MaxSpecPower ), convention, alphaScale )
    {}

    //-------------------------------------------------------------------------

    ProfileGGX::ProfileGGX( LevelWidthCurve const& curve, LobeConvention convention, double alphaScale )
        : m_convention( convention )
        , m_alphaScale( alphaScale )
        , m_curve( curve )
    {
        FF_ASSERT( curve.GetNumLevels() == NumLevels );

        for ( uint32_t level = 0; level < NumLevels; ++level )
        {
            m_widths[level] = curve.GetWidth( level ) * m_alphaScale;
        }
    }

    //-------------------------------------------------------------------------

    char const* ProfileGGX::GetName() const
    {
        if ( m_convention == LobeConvention::NDFOnly )
        {
            return "GGX (D, full sphere)";
        }

        if ( m_convention == LobeConvention::NDFHemisphere )
        {
            return "GGX (D, hemisphere)";
        }

        return "GGX (D * cos, hemisphere)";
    }
}
