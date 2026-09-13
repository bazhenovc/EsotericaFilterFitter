#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>

// The preintegrated split-sum DFG integrand, as this engine evaluates it
//-------------------------------------------------------------------------
// The specular half of a split-sum environment lookup is
//
//      radiance * ( F0 * scale + bias )
//
// where scale and bias are the integral of the specular BRDF against a constant environment over the upper hemisphere, with the Fresnel term factored out.
//
// This is NOT the integral as it appears in the literature.
// It is the one the engine evaluates, down to the choice of shadowing term, and the engine is the only thing it has to agree with.
//
// For reference: Code/Engine/Render/Shaders/PBR/PBR.esh ImportanceSampleGGX, GeometrySmith, DistributionGGX, PBR::New, and the lookup in ComputeIBL
//
//  quantity            engine
//  ------------------  --------------------------------------------------------------
//  N                   ( 0, 0, 1 )
//  V                   ( sin acos( N.V ), 0, N.V )   so N.V is the grid's x axis
//  alpha               roughness * roughness         ImportanceSampleGGX's `a`
//  half-vector sample  cosTheta = sqrt( ( 1 - xi.y ) / ( 1 + ( alpha^2 - 1 ) xi.y ) )
//  shadowing k         roughness * 0.5               PBR::New's m_roughness05
//  G                   G1( N.V ) * G1( N.L ), Schlick
//  weight              1 / ( 4 ( N.V ) ) with the half-vector sample's density, which collapses to the expression VisTerm returns below
//  F                   ( 1 - V.H )^5
//
// Sampling the half-vector from the NDF turns the integral into an average of
//
//      ( 1 - F ) Vis,  F Vis        with   Vis = G1( N.L ) G1( N.V ) ( V.H ) / ( ( N.V ) ( N.H ) )
//
// because the density of sampling that half-vector against the cosine is D ( N.H ), and the change of variables from H to L carries a Jacobian of 1 / ( 4 ( V.H ) ).
//
// The asymmetry is worth stating because getting it wrong is invisible: ( N.H ) is the density's own factor and ( N.L ) is not, and an estimator that divides by ( N.L ) instead still averages to something smooth and plausible, and is wrong.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    namespace DFGIntegrand
    {
        // Below this a cosine is treated as the zero it is approaching. Only an exact zero reaches it, since the grid's coordinates are texel centres.
        inline constexpr double MinCosine = 1.0e-6;

        //-------------------------------------------------------------------------

        // The NDF width the engine's roughness maps to: PBR.esh's ImportanceSampleGGX squares the roughness before it uses it as `a`.
        inline double EngineAlpha( double roughness )
        {
            return roughness * roughness;
        }

        // The Schlick k the engine pairs with that width: PBR.esh's PBR::New builds m_roughness05 as roughness * 0.5, so k is linear in the roughness and not in the width. 
        // The distinction only shows up below roughness one.
        inline double EngineShadowingK( double roughness )
        {
            return roughness * 0.5;
        }

        //-------------------------------------------------------------------------

        // The engine's GeometrySchlickBeckmann, which floors the denominator rather than testing it.
        // The floor is above the denominator's own value everywhere on this grid, so both forms agree here; the test is used because it says what it means at an exact zero.
        inline double GeometrySchlick( double dot, double k )
        {
            double const denominator = ( dot * ( 1.0 - k ) ) + k;

            if ( denominator <= 0.0 )
            {
                return 0.0;
            }

            return dot / denominator;
        }

        // The part of Schlick's Fresnel that does not depend on F0.
        inline double FresnelBase( double vdotH )
        {
            double const oneMinus = 1.0 - vdotH;

            return oneMinus * oneMinus * oneMinus * oneMinus * oneMinus;
        }

        // The half-vector the engine's ImportanceSampleGGX samples, in the surface frame with phi measured from the x axis.
        // `alphaSquared` is alpha squared, which is the width squared because the estimator's denominator is written in terms of the NDF's own a.
        //
        // The engine builds H through its own tangent frame, which rotates phi by a quarter turn; every term below reads H.x and H.z only and is even in H.x, so the integrand is the same function of the sample and the rotation moves which sample a given xi selects, not what is being integrated.
        inline void SampleHalfVector( double xiX, double xiY, double alphaSquared, double* pHalf )
        {
            double const phi = ( 2.0 * std::numbers::pi_v<double> ) * xiX;
            double const denominator = 1.0 + ( ( alphaSquared - 1.0 ) * xiY );

            double cosTheta = 1.0;

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

        // The BRDF against the density of the sample, for one sample of the half-vector.
        // Zero wherever the sample contributes nothing: a half-vector behind the view, a half-vector behind the surface, or a reflected direction below the horizon.
        inline double VisTerm( double vdotH, double cosV, double dotNH, double dotNL, double g1V, double k )
        {
            if ( ( vdotH <= 0.0 ) || ( dotNH <= 0.0 ) || ( dotNL <= 0.0 ) )
            {
                return 0.0;
            }

            return ( g1V * GeometrySchlick( dotNL, k ) * vdotH ) / ( cosV * dotNH );
        }

        //  Sample sequences
        //-------------------------------------------------------------------------
        // Both are deterministic and both are low discrepancy, and they are different constructions: the table samples from the van der Corput radical inverse the engine's own pass uses, the reference from a Cartesian stratification shifted by an irrational offset.
        // Nothing is seeded from the clock, so a table written twice is the same table.
        //-------------------------------------------------------------------------

        inline double RadicalInverseBase2( uint32_t bits )
        {
            bits = ( bits << 16 ) | ( bits >> 16 );
            bits = ( ( bits & 0x55555555u ) << 1 ) | ( ( bits & 0xAAAAAAAAu ) >> 1 );
            bits = ( ( bits & 0x33333333u ) << 2 ) | ( ( bits & 0xCCCCCCCCu ) >> 2 );
            bits = ( ( bits & 0x0F0F0F0Fu ) << 4 ) | ( ( bits & 0xF0F0F0F0u ) >> 4 );
            bits = ( ( bits & 0x00FF00FFu ) << 8 ) | ( ( bits & 0xFF00FF00u ) >> 8 );

            return static_cast<double>( bits ) * 2.3283064365386963e-10;
        }

        inline void Hammersley( uint32_t sampleIndex, uint32_t sampleCount, double& xiX, double& xiY )
        {
            xiX = static_cast<double>( sampleIndex ) / static_cast<double>( sampleCount );
            xiY = RadicalInverseBase2( sampleIndex );
        }

        // Two irrational offsets from the plastic constant, applied to a Cartesian stratification so that it carries no axis-aligned structure of its own.
        inline constexpr double StratifiedShiftX = 0.754877666246692760;
        inline constexpr double StratifiedShiftY = 0.569840290998053266;

        inline void Stratified( uint32_t sampleIndex, uint32_t sampleCount, double& xiX, double& xiY )
        {
            uint32_t const root = static_cast<uint32_t>( std::sqrt( static_cast<double>( sampleCount ) ) );

            // Neither axis is ever one wide, so a sample count that is not a perfect square still uses every index exactly once.
            uint32_t const columns = ( root > 0 ) ? root : 1;
            uint32_t const rows = ( ( sampleCount + columns - 1 ) / columns );

            uint32_t const column = sampleIndex % columns;
            uint32_t const row = sampleIndex / columns;

            xiX = ( static_cast<double>( column ) + 0.5 ) / static_cast<double>( columns );
            xiY = ( ( ( row < rows ) ? static_cast<double>( row ) : static_cast<double>( rows - 1 ) ) + 0.5 ) / static_cast<double>( rows );

            xiX += StratifiedShiftX;
            xiY += StratifiedShiftY;

            xiX -= std::floor( xiX );
            xiY -= std::floor( xiY );
        }
    }
}
