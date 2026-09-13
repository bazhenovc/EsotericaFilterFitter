#include "Assert.h"
#include "ProfileBeckmann.h"

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    ProfileBeckmann::ProfileBeckmann( LobeConvention convention, double alphaScale )
        : ProfileBeckmann( LevelWidthCurve::MakePaperGloss( NumLevels, MaxSpecPower ), convention, alphaScale )
    {
    }

    //-------------------------------------------------------------------------

    ProfileBeckmann::ProfileBeckmann( LevelWidthCurve const& curve, LobeConvention convention, double alphaScale )
        : m_convention( convention )
        , m_alphaScale( alphaScale )
        , m_curve( curve )
    {
        FF_ASSERT( curve.GetNumLevels() == NumLevels );

        for( uint32_t level = 0; level < NumLevels; ++level )
        {
            m_widths[level] = curve.GetWidth( level ) * m_alphaScale;
        }
    }

    //-------------------------------------------------------------------------

    char const* ProfileBeckmann::GetName() const
    {
        if( m_convention == LobeConvention::NDFOnly )
        {
            return "Beckmann (D, full sphere)";
        }

        if( m_convention == LobeConvention::NDFHemisphere )
        {
            return "Beckmann (D, hemisphere)";
        }

        return "Beckmann (D * cos, hemisphere)";
    }
}
