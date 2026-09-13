#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>

#include "Assert.h"
#include "Profile.h"
#include "LevelWidthCurve.h"

//  Beckmann profile
//-------------------------------------------------------------------------
// The second NDF
//
//      D(h) = exp( -tan^2(theta_h) / alpha^2 ) / ( PI alpha^2 cos^4(theta_h) )
//
// with theta_h the angle between n and h. 
// Normalised so that integral D(h) (n.h) dOmega_h = 1, which the profile self-check measures rather than assumes.
//
// Beckmann shares the width parameterisation with GGX: alpha is the same width, so the same LevelWidthCurve applies to either, and the same runtime roughness values select the same levels.
//
//  INLINING
//
// As for ProfileGGX: everything the fit calls per texel is inline here, because EvaluateZonal sits in the innermost loop of the reference preimage.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    class ProfileBeckmann final
    {
    public:

        static constexpr uint32_t NumLevels = 7;        // PROBE_REFLECTION_MIPS
        static constexpr double   MaxSpecPower = 18.0;  // GGX_MAX_SPEC_POWER, shared curve

    public:

        explicit ProfileBeckmann( LobeConvention convention = LobeConvention::NDFCosineHemisphere, double alphaScale = 1.0 );

        ProfileBeckmann( LevelWidthCurve const& curve, LobeConvention convention, double alphaScale = 1.0 );

        //  Profile contract
        //---------------------------------------------------------------------

        char const* GetName() const;

        uint32_t GetNumLevels() const
        {
            return NumLevels;
        }

        double GetWidth( uint32_t level ) const
        {
            FF_ASSERT( level < NumLevels );

            return m_widths[level];
        }

        // Raw NDF as a function of cos(theta_h).
        //
        // Written in terms of tan^2 rather than the textbook cos^4 form.
        // The two are identical, but the cos^4 form divides by a quantity that goes to zero at grazing angles and multiplies by an exponential going to zero, which evaluates as inf * 0 and yields NaN.
        // Rearranged, every term stays finite and the tail decays to zero the way it should.
        //
        //      exp( -T ) / ( PI a^2 cos^4 )  ==  exp( -T ) ( 1 + T )^2 / ( PI a^2 )
        //      with T = tan^2(theta_h)
        //---------------------------------------------------------------------

        static double EvaluateNDF( double cosHalfTheta, double alpha )
        {
            // A zero width is a mirror; the reference preimage substitutes the identity
            if ( alpha <= 0.0 )
            {
                return 0.0;
            }

            if ( cosHalfTheta <= 0.0 )
            {
                return 0.0;
            }

            double const cosSquared = cosHalfTheta * cosHalfTheta;
            double const tanSquared = ( 1.0 - cosSquared ) / cosSquared;

            // Beyond this the exponential has underflowed and ( 1 + tan^2 )^2 is on its way to overflowing, so the product would be 0 * inf
            if ( tanSquared > 1.0e30 )
            {
                return 0.0;
            }

            double const onePlusTanSquared = 1.0 + tanSquared;

            return std::exp( -tanSquared / ( alpha * alpha ) ) * onePlusTanSquared * onePlusTanSquared / ( std::numbers::pi_v<double> *alpha * alpha );
        }

        double EvaluateZonal( double cosTheta, uint32_t level ) const
        {
            double const cosEnv = ClampCosine( cosTheta );
            double const NDF = EvaluateNDF( GetCosHalfAngle( cosEnv ), m_widths[level] );

            return ApplyLobeConvention( m_convention, cosEnv, NDF );
        }

        //  Sampling the half-vector from the NDF
        //---------------------------------------------------------------------
        // The inverse of this profile's own distribution of ( H.N ).
        // It is not GGX's inverse and the two are not interchangeable; see the note on ProfileGGX's.
        //
        // In the surface frame, with phi measured from the x axis.
        //---------------------------------------------------------------------

        static void SampleHalfVector( double xiX, double xiY, double alphaSquared, double* pHalf )
        {
            double const phi = ( 2.0 * std::numbers::pi_v<double> ) * xiX;

            // The smallest positive xi keeps the logarithm finite, and the largest keeps
            // the sample off an exactly grazing cosine, where this NDF is zero anyway.
            double const xi = ( xiY < 1.0e-12 ) ? 1.0e-12 : ( ( xiY > ( 1.0 - 1.0e-12 ) ) ? ( 1.0 - 1.0e-12 ) : xiY );
            double const denominator = 1.0 - ( alphaSquared * std::log( xi ) );

            double const cosTheta = std::sqrt( 1.0 / denominator );
            double const sinTheta = std::sqrt( ( 1.0 - cosTheta ) * ( 1.0 + cosTheta ) );

            pHalf[0] = std::cos( phi ) * sinTheta;
            pHalf[1] = std::sin( phi ) * sinTheta;
            pHalf[2] = cosTheta;
        }

        inline LobeConvention GetConvention() const { return m_convention; }
        inline double GetAlphaScale() const { return m_alphaScale; }
        inline LevelWidthCurve const& GetCurve() const { return m_curve; }

    private:

        LobeConvention      m_convention;
        double              m_alphaScale;
        LevelWidthCurve     m_curve;
        double              m_widths[NumLevels];
    };
}
