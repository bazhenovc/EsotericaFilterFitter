#include "Assert.h"
#include "LevelWidthCurve.h"

#include <cmath>
#include <cstdio>

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    LevelWidthCurve LevelWidthCurve::MakePaperGloss( uint32_t numLevels, double maxSpecPower )
    {
        FF_ASSERT( numLevels > 1 );

        LevelWidthCurve curve;
        curve.m_widths.resize( numLevels );

        for ( uint32_t level = 0; level < numLevels; ++level )
        {
            // The reference names the two 6s in its own expression separately (ENVMAP_MIPLEVEL_0_PB and a span that shares its value); here the span is derived from the level count, which reproduces them for seven levels.
            double const gloss = static_cast<double>( ( numLevels - 1 ) - level ) / static_cast<double>( numLevels - 1 );

            double const exponent = std::pow( 2.0, gloss * maxSpecPower );
            double const rootRoughness = std::pow( 2.0 / ( 1.0 + exponent ), 0.25 );

            // The curve returns sqrt( alpha )
            curve.m_widths[level] = rootRoughness * rootRoughness;
        }

        std::snprintf( curve.m_name, sizeof( curve.m_name ), "paper(%.4g)", maxSpecPower );

        return curve;
    }

    //-------------------------------------------------------------------------

    LevelWidthCurve LevelWidthCurve::MakeEsotericaLinear( uint32_t numLevels )
    {
        FF_ASSERT( numLevels > 1 );

        LevelWidthCurve curve;
        curve.m_widths.resize( numLevels );

        for ( uint32_t level = 0; level < numLevels; ++level )
        {
            double const roughness = static_cast<double>( level ) / static_cast<double>( numLevels - 1 );

            curve.m_widths[level] = roughness * roughness;
        }

        std::snprintf( curve.m_name, sizeof( curve.m_name ), "esoterica" );

        return curve;
    }

    //-------------------------------------------------------------------------

    LevelWidthCurve LevelWidthCurve::MakeExplicit( std::vector<double> const& widths )
    {
        FF_ASSERT( !widths.empty() );

        LevelWidthCurve curve;
        curve.m_widths = widths;

        std::snprintf( curve.m_name, sizeof( curve.m_name ), "explicit" );

        return curve;
    }

    //-------------------------------------------------------------------------

    double LevelWidthCurve::GetWidth( uint32_t level ) const
    {
        FF_ASSERT( level < m_widths.size() );

        return m_widths[level];
    }

    //-------------------------------------------------------------------------

    bool LevelWidthCurve::IsNonDecreasing() const
    {
        for ( size_t index = 1; index < m_widths.size(); ++index )
        {
            if ( m_widths[index] < m_widths[index - 1] )
            {
                return false;
            }
        }

        return true;
    }
}
