#pragma once

#include <cmath>
#include <cstdint>

// Profile contract
//-------------------------------------------------------------------------
// A profile supplies the shape of the isotropic zonal lobe the environment is convolved with: the D(h) of Eq. 3 in Manson & Sloan, "Fast Filtering of Reflection Probes" (EGSR 2016), Section 2.
//
// It supplies ONLY that shape, and its width comes from a LevelWidthCurve.
// The generator owns the projection onto the cubemap and the solid-angle measure, so that the measure cannot diverge between profiles and L1 results stay comparable across them.
//
// This is a concept, not a base class. 
// A profile is passed as a template parameter and its members are called directly, so EvaluateZonal is inlined into the innermost loop of the reference preimage - which runs once per base texel per output texel per level, and is the hottest call in the tool.
// A virtual interface here costs an indirect call in exactly that loop. The contract is enforced by instantiation rather than by inheritance, so a type that fails to provide a member fails to compile.
//
// A profile type provides, all const:
//
//      char const* GetName() const;
//      uint32_t    GetNumLevels() const;
//      double      GetWidth( uint32_t level ) const;
//      double      EvaluateZonal( double cosTheta, uint32_t level ) const;
//
// WIDTH AND ROUGHNESS CURVE
//
// GetWidth is the width the profile's NDF is evaluated at, taken from the profile's LevelWidthCurve (see LevelWidthCurve.h).
// Shape and curve are separate axes because they are what a renderer changes separately: an NDF, and a roughness-to-mip mapping.
//
// A width of exactly zero means a mirror.
// A continuous NDF cannot represent one, so profiles return zero there and the reference preimage substitutes the identity filter - which is what a runtime does at roughness zero, sampling the unfiltered environment rather than picking a mip.
// See ReferencePreimage::Evaluate.
//
//  ANGLE CONVENTION
//
// cosTheta is the cosine between the environment direction L and the filter axis N, NOT the half-angle.
// A profile whose BRDF is written in terms of the half-vector h = normalize(l + n) - GGX and Beckmann included - applies that transform itself, through GetCosHalfAngle below.
//
//  Absolute scale is not significant: the generator normalises so that a constant environment reproduces a constant.
//
//  SHARED MACHINERY
//
// The half-angle transform and the lobe convention are the same for any isotropic NDF written as D(h), so they live here rather than in each profile.
// A profile supplies its NDF and nothing else; anything else it supplies is somewhere for two  profiles to disagree about the measure.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    // Which integrand the profile represents.
    //
    // This is definitional rather than a runtime option: it is fixed by which split-sum convention the profile and the DFG LUT implement together.
    // Exposing it separately would permit a prefilter without the cosine to be paired with a DFG built for the cosine convention.
    //
    // It is a constructor argument so the conformance test can compare the variants. 
    // That test selects the cosine form, which is also what the engine's radiance prefilter and its DFG table compute.
    //-------------------------------------------------------------------------

    enum class LobeConvention : uint8_t
    {
        // D(h) over the whole sphere - the paper's Eq. 3 read literally
        NDFOnly,

        // D(h) restricted to the upper hemisphere
        NDFHemisphere,

        // D(h) * (n.l) over the upper hemisphere
        NDFCosineHemisphere,
    };

    //-------------------------------------------------------------------------

    inline double ClampCosine( double cosTheta )
    {
        return ( cosTheta < -1.0 ) ? -1.0 : ( ( cosTheta > 1.0 ) ? 1.0 : cosTheta );
    }

    // Half-angle transform.
    // With the filter axis at the surface normal and the split-sum assumption v = n, the half-vector h = normalize(l + n) sits at half the angle of the environment direction l:
    //
    //      cos(theta_h) = sqrt( ( 1 + cos(theta_l) ) / 2 )
    //
    // l and h follow the paper's notation: incoming direction and half-vector.
    //-------------------------------------------------------------------------

    inline double GetCosHalfAngle( double cosEnv )
    {
        return std::sqrt( ( 1.0 + cosEnv ) * 0.5 );
    }

    //-------------------------------------------------------------------------

    inline double ApplyLobeConvention( LobeConvention convention, double cosEnv, double NDF )
    {
        if ( ( convention != LobeConvention::NDFOnly ) && ( cosEnv <= 0.0 ) )
        {
            return 0.0;
        }

        return ( convention == LobeConvention::NDFCosineHemisphere ) ? ( NDF * cosEnv ) : NDF;
    }
}
