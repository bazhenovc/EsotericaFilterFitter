#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>

#include "Assert.h"
#include "Profile.h"
#include "LevelWidthCurve.h"

// GGX (Trowbridge-Reitz) profile
//-------------------------------------------------------------------------
// D(h) = alpha^2 / ( PI * ( (n.h)^2 (alpha^2 - 1) + 1 )^2 )
//
// which matches DistributionGGX in Shaders/PBR/PBR.esh, where roughness4 == alpha^2:
//
//      d = dotNH^2 * ( roughness4 - 1 ) + 1
//      D = roughness4 / ( PI * d * d )
//
// So this profile's width is the runtime's m_roughness4 with no conversion.
//
// The roughness curve is a constructor argument.
// The default is the reference's gloss curve, because that is what the published tables were fitted against and what the conformance test measures; see LevelWidthCurve.h for the two named curves and what separates them.

namespace FilterFitter
{
    class ProfileGGX final
    {
    public:

        static constexpr uint32_t NumLevels = 7;        // PROBE_REFLECTION_MIPS
        static constexpr double   MaxSpecPower = 18.0;  // GGX_MAX_SPEC_POWER

    public:

        explicit ProfileGGX( LobeConvention convention = LobeConvention::NDFCosineHemisphere, double alphaScale = 1.0 );

        ProfileGGX( LevelWidthCurve const& curve, LobeConvention convention, double alphaScale = 1.0 );

        // The reference's curve, kept for the conformance test's width sweep. GlossToRoughness returns sqrt(alpha).
        static double GlossToRoughness( double gloss );
        static double GetGloss( uint32_t level );
        static double GetAlpha( uint32_t level );

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
        // Separate from EvaluateZonal because the NDF normalisation integral is defined in terms of the half-angle, so validation needs it directly.
        //---------------------------------------------------------------------

        static double EvaluateNDF( double cosHalfTheta, double alpha )
        {
            // A zero width is a mirror, which this continuous form cannot represent; the reference preimage substitutes the identity filter instead.
            if ( alpha <= 0.0 )
            {
                return 0.0;
            }

            double const alphaSquared = alpha * alpha;
            double const cosHalfSquared = cosHalfTheta * cosHalfTheta;
            double const denominator = ( cosHalfSquared * ( alphaSquared - 1.0 ) ) + 1.0;

            return alphaSquared / ( std::numbers::pi_v<double> *denominator * denominator );
        }

        //  Zonal lobe weight as a function of the cosine between the environment direction and the filter axis.
        //---------------------------------------------------------------------

        double EvaluateZonal( double cosTheta, uint32_t level ) const
        {
            double const cosEnv = ClampCosine( cosTheta );
            double const NDF = EvaluateNDF( GetCosHalfAngle( cosEnv ), m_widths[level] );

            return ApplyLobeConvention( m_convention, cosEnv, NDF );
        }

        //  Sampling the half-vector from the NDF
        //---------------------------------------------------------------------
        // The inverse of this profile's own distribution of ( H.N ), for an estimator that has to average against the NDF rather than sample the hemisphere uniformly.
        // It belongs here rather than beside the caller because it is a property of the distribution: another NDF's inverse used with this NDF's weight is a wrong answer that still averages to something plausible.
        //
        // In the surface frame, with phi measured from the x axis.

        static void SampleHalfVector( double xiX, double xiY, double alphaSquared, double* pHalf )
        {
            double const phi = ( 2.0 * std::numbers::pi_v<double> ) * xiX;
            double const denominator = 1.0 + ( ( alphaSquared - 1.0 ) * xiY );

            double cosTheta = 0.0;

            if ( denominator > 0.0 )
            {
                double const ratio = ( 1.0 - xiY ) / denominator;

                cosTheta = ( ratio > 0.0 ) ? std::sqrt( ratio ) : 0.0;
            }

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

        // Per-level widths, resolved once.
        // EvaluateZonal is called once per base texel per output direction, so a per-call curve evaluation would put two pow calls inside the innermost loop; the benchmark measures that as about 60% of the reference preimage cost.
        double          m_widths[NumLevels];
    };
}
