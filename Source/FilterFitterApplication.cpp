#include "BsplineRecurrence.h"
#include "CubeReferenceFrame.h"
#include "FitDriver.h"
#include "HDRIShowcase.h"
#include "HDRIValidation.h"
#include "LevelObjective.h"
#include "MapProjection.h"
#include "TetrahedralProjection.h"
#include "Optimizer.h"
#include "ParallelFor.h"
#include "PreimageAccumulator.h"
#include "TableSeed.h"
#include "PreimageError.h"
#include "ProfileBeckmann.h"
#include "ProfileGGX.h"
#include "TableHarness.h"
#include "TableWriter.h"
#include "LevelWidthCurve.h"
#include "ReferencePreimage.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <type_traits>
#include <vector>

using namespace FilterFitter;

//  FilterFitter
//-------------------------------------------------------------------------
// Offline generator for prefiltered-radiance coefficient tables, following Manson & Sloan, "Fast Filtering of Reflection Probes" (EGSR 2016).
//
// CPU only, no GPU. Runs for hours and runs rarely: it is a fork-time tool, used when a project deviates from the engine's default BRDF, roughness curves or lighting model.
// Its output is a generated C header consumed by the table-driven gather shader.
//
// Not part of the runtime render path.
//
// Each stage below carries a self-check that runs on startup.
//-------------------------------------------------------------------------

// BASE_RESOLUTION, from Reference/spec_params_for_mip.txt
static constexpr uint32_t g_paperBaseResolution = 128;

// Subsamples per axis when averaging a lobe over a base texel footprint.
// This is used only by the B(x) self-check, which compares a point sample against a texel average to show where the two discretisations diverge; the conformance test and the fitting target use g_conformanceSupersampleRate, which is 1.
static constexpr uint32_t g_supersampleRate = 8;

// Published-table conformance sampling.
// Grid size is per face per axis, so a level costs 6 * gridSize^2 texels.
// The outermost cells land at |u| = 0.758, so the third axial frame is active there with a frame weight of about 0.03.
// Where the grid clamps to the face resolution (level 6, 2x2) it is never taken.
//
// The supersample rate is 1 because the runtime's tap weights reconstruct a discrete measure on the base cube, so the target is that same discrete measure - point samples of the kernel at texel centres, normalised.
// Averaging the kernel over a texel is a better quadrature of the continuous kernel but a different function, and only agrees from level 3 outward.
// The paper's level-0 table selects the point-sampled form.
static constexpr uint32_t g_conformanceGridSize = 4;
static constexpr uint32_t g_conformanceSupersampleRate = 1;

static constexpr double g_pi = std::numbers::pi_v<double>;

static constexpr double g_degreesToRadians = g_pi / 180.0;

static constexpr double g_fourPi = std::numbers::pi_v<double>*4.0;

// A map's reference frame
//-------------------------------------------------------------------------
// The frame is one implementation for every map, so this is one check for every map rather than a version of it per map.
// What varies between maps is the numbers, and one of them is why a tetrahedron is harder to prefilter than a  cube: the ratio of the largest texel solid angle to the smallest is 5.2 for a cube and 27 for a regular tetrahedron, which is the distortion the table has to absorb.
//
// Two of the checks are about the frame rather than the map: a sample taken at zero offset must land back on the texel it started from, which is what  validates the map's signed coordinate step, and the smoothing pass must preserve both a constant field and the total, which is what validates the neighbour structure it smooths over.

template< typename TMap >
static bool RunMapFrameCheck( uint32_t resolution )
{
    MapReferenceFrame< TMap > frame;
    frame.Initialize( resolution );

    double minSolidAngle = 1.0e30;
    double maxSolidAngle = -1.0e30;
    double minJacobian = 1.0e30;
    double maxJacobian = -1.0e30;
    double sumJacobian = 0.0;

    for ( MapTexel const& texel : frame.GetTexels() )
    {
        minSolidAngle = ( texel.m_solidAngle < minSolidAngle ) ? texel.m_solidAngle : minSolidAngle;
        maxSolidAngle = ( texel.m_solidAngle > maxSolidAngle ) ? texel.m_solidAngle : maxSolidAngle;
        minJacobian = ( texel.m_jacobian < minJacobian ) ? texel.m_jacobian : minJacobian;
        maxJacobian = ( texel.m_jacobian > maxJacobian ) ? texel.m_jacobian : maxJacobian;

        sumJacobian += texel.m_jacobian;
    }

    uint32_t const numTexels = frame.GetNumTexels();
    double const total = frame.GetTotalSolidAngle();
    double const relError = std::fabs( total - g_fourPi ) / g_fourPi;

    // The reference shader's constant-Jacobian approximation, for comparison: 4 * J / resolution^2 with J taken as the mean over the frame. 
    // It is the same expression for any map, and it is a property of the shader rather than of the map, so it is reported and not checked.
    double const meanJacobian = sumJacobian / static_cast<double>( numTexels );
    double const approxTotal = static_cast<double>( numTexels ) * 4.0 * meanJacobian / static_cast<double>( resolution * resolution );
    double const approxRelError = std::fabs( approxTotal - g_fourPi ) / g_fourPi;

    // A sample at zero offset has to land on the texel that produced it. 
    // This is the round trip through the map's own coordinate convention, including the sign of the step between texels, which is the one part of the convention a consumer cannot derive and has to be told.
    uint32_t numCentreFailures = 0;
    double worstCentreError = 0.0;

    for ( uint32_t texelIndex = 0; texelIndex < numTexels; ++texelIndex )
    {
        MapTexel const& texel = frame.GetTexel( texelIndex );

        double sampled[3];
        frame.GetTexelSampleDirection( sampled, texelIndex, 0.0, 0.0 );

        double const length = std::sqrt( ( sampled[0] * sampled[0] ) + ( sampled[1] * sampled[1] ) + ( sampled[2] * sampled[2] ) );
        FF_ASSERT( length > 0.0 );

        double const dot = ( ( sampled[0] * texel.m_dir[0] ) + ( sampled[1] * texel.m_dir[1] ) + ( sampled[2] * texel.m_dir[2] ) ) / length;
        double const angularError = std::fabs( 1.0 - dot );

        if ( angularError > 1.0e-12 )
        {
            ++numCentreFailures;
        }

        if ( angularError > worstCentreError )
        {
            worstCentreError = angularError;
        }
    }

    // The smoothing pass is one separable (0.25, 0.5, 0.25) kernel over each slice with the edges clamped, so it preserves a constant exactly and preserves the total exactly.
    std::vector<double> constantField( numTexels, 1.0 );
    frame.SmoothInPlace( constantField );

    double worstConstantError = 0.0;

    for ( uint32_t texelIndex = 0; texelIndex < numTexels; ++texelIndex )
    {
        double const error = std::fabs( constantField[texelIndex] - 1.0 );

        if ( error > worstConstantError )
        {
            worstConstantError = error;
        }
    }

    std::vector<double> varyingField( numTexels );

    double totalBefore = 0.0;

    for ( uint32_t texelIndex = 0; texelIndex < numTexels; ++texelIndex )
    {
        MapTexel const& texel = frame.GetTexel( texelIndex );

        varyingField[texelIndex] = 1.0 + ( 0.5 * texel.m_dir[1] ) + ( 0.25 * texel.m_dir[2] );
        totalBefore += varyingField[texelIndex];
    }

    frame.SmoothInPlace( varyingField );

    double totalAfter = 0.0;

    for ( uint32_t texelIndex = 0; texelIndex < numTexels; ++texelIndex )
    {
        totalAfter += varyingField[texelIndex];
    }

    double const smoothTotalError = std::fabs( totalAfter - totalBefore ) / totalBefore;

    std::printf( "\n" );
    std::printf( "reference frame\n" );
    std::printf( "  topology              : %u faces in %u slice%s, %u x %u texels per slice\n", TMap::NumFaces, TMap::NumSlices, ( TMap::NumSlices == 1 ) ? "" : "s", resolution, resolution );
    std::printf( "  total texels          : %u\n", numTexels );
    std::printf( "  total solid angle     : %.17g sr\n", total );
    std::printf( "  4*pi                  : %.17g sr\n", g_fourPi );
    std::printf( "  relative error        : %.3e\n", relError );
    std::printf( "  solid angle     min/max : %.17g / %.17g  (ratio %.4f)\n", minSolidAngle, maxSolidAngle, maxSolidAngle / minSolidAngle );
    std::printf( "  jacobian        min/max : %.17g / %.17g\n", minJacobian, maxJacobian );
    std::printf( "  jacobian        mean    : %.17g\n", meanJacobian );
    std::printf( "\n" );
    std::printf( "  for reference, the constant-jacobian approximation used by the\n" );
    std::printf( "  vendored shader gives a total of %.17g sr (%.3e relative error)\n", approxTotal, approxRelError );

    bool passed = ( relError < 1.0e-12 ) && ( numTexels == ( TMap::NumSlices * resolution * resolution ) );

    std::printf( "  %-42s : %s (%u of %u failed, worst %.3e)\n", "zero offset lands on the texel it started from", ( numCentreFailures == 0 ) ? "PASS" : "FAIL", numCentreFailures, numTexels, worstCentreError );

    bool const smoothOk = ( worstConstantError == 0.0 ) && ( smoothTotalError < 1.0e-14 );

    std::printf( "  %-42s : %s (constant drift %.3e, total drift %.3e)\n", "smoothing preserves the field's total", smoothOk ? "PASS" : "FAIL", worstConstantError, smoothTotalError );

    passed = passed && ( numCentreFailures == 0 ) && smoothOk;

    std::printf( "\n" );
    std::printf( "  RESULT                : %s\n", passed ? "PASS" : "FAIL" );

    return passed;
}


//  Profile shape checks
//-------------------------------------------------------------------------

template< typename TProfile >
static bool RunProfileShapeChecks( TProfile const& profile, LevelWidthCurve const& curve )
{
    bool passed = true;

    uint32_t const numLevels = profile.GetNumLevels();

    std::printf( "\n  %s, curve %s\n", profile.GetName(), curve.GetName() );
    std::printf( "  %-4s %-5s %-14s %-14s %-12s\n", "lvl", "res", "width", "roughness", "peak" );

    double previousWidth = -1.0;
    bool widthsNonDecreasing = true;

    for ( uint32_t level = 0; level < numLevels; ++level )
    {
        double const width = profile.GetWidth( level );
        double const peak = profile.EvaluateZonal( 1.0, level );
        uint32_t const levelWidth = g_paperBaseResolution >> level;

        std::printf( "  %-4u %-5u %-14.8f %-14.8f %-12.6f\n", level, levelWidth, width, std::sqrt( width ), peak );

        if ( width < previousWidth )
        {
            widthsNonDecreasing = false;
        }

        previousWidth = width;
    }

    std::printf( "  widths non-decreasing                   : %s\n", widthsNonDecreasing ? "PASS" : "FAIL" );
    passed = passed && widthsNonDecreasing;

    // Half-angle transform
    // Validated geometrically rather than by re-deriving the identity, which would be circular.
    // For a reflection setup with v = n, build a from theta_a, form the half-vector h = normalize(a + n), and compare the measured dot(n, h) against the closed form sqrt( (1 + cos(theta_a)) / 2 ).
    // The profile must then agree with the raw NDF evaluated at that geometrically-derived half-angle.
    {
        // Named NDFProfile rather than NDFOnly so it does not read as the LobeConvention::NDFOnly enumerator it is constructed from
        TProfile const NDFProfile( curve, LobeConvention::NDFOnly );

        double const degrees[] = { 5.0, 15.0, 30.0, 60.0, 85.0, 89.5 };
        uint32_t const numAngles = static_cast<uint32_t>( sizeof( degrees ) / sizeof( degrees[0] ) );

        bool halfAngleOk = true;
        double worstIdentity = 0.0;
        double worstProfile = 0.0;

        for ( uint32_t angleIndex = 0; angleIndex < numAngles; ++angleIndex )
        {
            double const thetaL = degrees[angleIndex] * g_degreesToRadians;
            double const cosEnv = std::cos( thetaL );

            // Explicit unit vectors, with the filter axis along +Z
            double const a[3] = { std::sin( thetaL ), 0.0, cosEnv };
            double const n[3] = { 0.0, 0.0, 1.0 };

            double h[3];
            h[0] = a[0] + n[0];
            h[1] = a[1] + n[1];
            h[2] = a[2] + n[2];

            double const halfVectorLength = std::sqrt( ( h[0] * h[0] ) + ( h[1] * h[1] ) + ( h[2] * h[2] ) );
            h[0] /= halfVectorLength;
            h[1] /= halfVectorLength;
            h[2] /= halfVectorLength;

            double const cosHalfGeometric = ( n[0] * h[0] ) + ( n[1] * h[1] ) + ( n[2] * h[2] );
            double const cosHalfClosed = std::sqrt( ( 1.0 + cosEnv ) * 0.5 );
            double const identityDelta = std::fabs( cosHalfGeometric - cosHalfClosed );

            worstIdentity = ( identityDelta > worstIdentity ) ? identityDelta : worstIdentity;

            if ( identityDelta > 1.0e-12 )
            {
                halfAngleOk = false;
            }

            for ( uint32_t level = 0; level < numLevels; ++level )
            {
                double const alpha = profile.GetWidth( level );
                double const fromZonal = NDFProfile.EvaluateZonal( cosEnv, level );
                double const fromHalfAngle = TProfile::EvaluateNDF( cosHalfGeometric, alpha );
                double const delta = std::fabs( fromZonal - fromHalfAngle ) / ( ( fromHalfAngle > 0.0 ) ? fromHalfAngle : 1.0 );

                worstProfile = ( delta > worstProfile ) ? delta : worstProfile;

                if ( delta > 1.0e-9 )
                {
                    halfAngleOk = false;
                }
            }
        }

        std::printf( "  half-angle transform (geometric check)  : %s (identity %.3e, profile %.3e)\n", halfAngleOk ? "PASS" : "FAIL", worstIdentity, worstProfile );
        passed = passed && halfAngleOk;
    }

    // NDF normalisation
    // 
    // The NDF must integrate to 1 over the hemisphere against the cosine:
    //
    //      integral D(h) * (n.h) dw_h == 1
    //
    // Integrated over the half-vector polar angle in [0, pi/2] with the sin(theta_h) measure.
    // The choice of variable matters: in cos(theta_h) the peak width scales as alpha^2 (7.6e-6 for the sharpest level), which a uniform grid cannot resolve without millions of samples and which aliases badly, whereas in theta_h the peak width scales as alpha.
    {
        uint32_t const intervals = 200000;
        double const stepSize = ( g_pi * 0.5 ) / static_cast<double>( intervals );
        bool normalizationOk = true;
        double worstError = 0.0;
        uint32_t checkedLevels = 0;

        for ( uint32_t level = 0; level < numLevels; ++level )
        {
            double const alpha = profile.GetWidth( level );

            // A zero width is a mirror, not an NDF, and the reference preimage substitutes the identity filter for it.
            // Integrating it would ask a delta to have unit integral, which no sampled grid can show.
            if ( alpha <= 0.0 )
            {
                continue;
            }

            ++checkedLevels;

            double sum = 0.0;

            for ( uint32_t sampleIndex = 0; sampleIndex <= intervals; ++sampleIndex )
            {
                // thetaH is the half-vector polar angle, not the paper's cube parameterization angle
                double const thetaH = static_cast<double>( sampleIndex ) * stepSize;
                double const cosThetaH = std::cos( thetaH );
                double const sinThetaH = std::sin( thetaH );
                double const integrand = TProfile::EvaluateNDF( cosThetaH, alpha ) * cosThetaH * sinThetaH;
                double const quadratureWeight = ( ( sampleIndex == 0 ) || ( sampleIndex == intervals ) ) ? 1.0 : ( ( ( sampleIndex & 1u ) == 0 ) ? 2.0 : 4.0 );

                sum += quadratureWeight * integrand;
            }

            double const total = 2.0 * g_pi * ( sum * stepSize / 3.0 );
            double const error = std::fabs( total - 1.0 );

            worstError = ( error > worstError ) ? error : worstError;

            if ( error > 1.0e-9 )
            {
                normalizationOk = false;
            }
        }

        std::printf( "  NDF normalisation (int D*(n.h) dw == 1) : %s (worst abs err %.3e, %u of %u levels)\n", normalizationOk ? "PASS" : "FAIL", worstError, checkedLevels, numLevels );
        passed = passed && normalizationOk;
    }

    // Finiteness and non-negativity

    {
        LobeConvention const conventions[] =
        {
            LobeConvention::NDFOnly,
            LobeConvention::NDFHemisphere,
            LobeConvention::NDFCosineHemisphere,
        };

        uint32_t const numConventions = static_cast<uint32_t>( sizeof( conventions ) / sizeof( conventions[0] ) );
        uint32_t const numSamples = 4096;

        for ( uint32_t conventionIndex = 0; conventionIndex < numConventions; ++conventionIndex )
        {
            TProfile const testProfile( curve, conventions[conventionIndex] );

            bool allValid = true;
            double maxValue = 0.0;

            for ( uint32_t level = 0; level < testProfile.GetNumLevels(); ++level )
            {
                for ( uint32_t sampleIndex = 0; sampleIndex <= numSamples; ++sampleIndex )
                {
                    double const cosEnv = -1.0 + ( 2.0 * static_cast<double>( sampleIndex ) / static_cast<double>( numSamples ) );
                    double const lobeWeight = testProfile.EvaluateZonal( cosEnv, level );

                    if ( ( !std::isfinite( lobeWeight ) ) || ( lobeWeight < 0.0 ) )
                    {
                        allValid = false;
                    }

                    maxValue = ( lobeWeight > maxValue ) ? lobeWeight : maxValue;
                }
            }

            std::printf( "  %-38s : %s (max %.6f)\n", testProfile.GetName(), allValid ? "PASS" : "FAIL", maxValue );
            passed = passed && allValid;
        }
    }

    return passed;
}

//  Every profile against every curve it can be paired with
//-------------------------------------------------------------------------

static bool RunProfileSelfCheck()
{
    bool passed = true;

    uint32_t const numLevels = ProfileGGX::NumLevels;

    LevelWidthCurve const paperCurve = LevelWidthCurve::MakePaperGloss( numLevels, ProfileGGX::MaxSpecPower );
    LevelWidthCurve const esotericaCurve = LevelWidthCurve::MakeEsotericaLinear( numLevels );

    std::printf( "\n" );
    std::printf( "profile\n" );
    std::printf( "  widths per level, and the roughness each corresponds to\n" );
    std::printf( "\n  %-4s %-14s %-14s %-14s\n", "lvl", "paper width", "roughness", "esoterica width" );

    for ( uint32_t level = 0; level < numLevels; ++level )
    {
        double const paperWidth = paperCurve.GetWidth( level );
        double const esotericaWidth = esotericaCurve.GetWidth( level );

        std::printf( "  %-4u %-14.8f %-14.8f %-14.8f\n", level, paperWidth, std::sqrt( paperWidth ), esotericaWidth );
    }

    std::printf( "\n  paper is geometric: each level is roughly 1.7x the previous roughness.\n" );
    std::printf( "  esoterica is linear in roughness, so level 0 is a mirror rather than a\n" );
    std::printf( "  narrow lobe. Both are legal. The table has to be fitted with whichever one\n" );
    std::printf( "  the runtime selects with, because that is what decides each level's width.\n" );

    // Both NDFs on the reference curve, then both on this project's own, so a profile is measured against a curve it will actually be fitted with.
    passed = RunProfileShapeChecks( ProfileGGX( paperCurve, LobeConvention::NDFCosineHemisphere ), paperCurve ) && passed;
    passed = RunProfileShapeChecks( ProfileBeckmann( paperCurve, LobeConvention::NDFCosineHemisphere ), paperCurve ) && passed;
    passed = RunProfileShapeChecks( ProfileGGX( esotericaCurve, LobeConvention::NDFCosineHemisphere ), esotericaCurve ) && passed;
    passed = RunProfileShapeChecks( ProfileBeckmann( esotericaCurve, LobeConvention::NDFCosineHemisphere ), esotericaCurve ) && passed;

    // The two NDFs have to be told apart
    // 
    // Nothing above separates them.
    // Both peak at 1 / ( PI alpha^2 ), both normalise to one, and the half-angle check compares each against its own NDF - so a Beckmann that was accidentally a copy of GGX would pass all of it.
    // This is the check that would catch that.
    //
    // At tan^2(theta_h) == alpha^2, which is the angle where the tail has decayed to its e-fold, both have a closed form:
    //
    //      GGX        ( 1 + alpha^2 )^2 / ( 4 PI alpha^2 )
    //      Beckmann   exp( -1 ) ( 1 + alpha^2 )^2 / ( PI alpha^2 )
    //
    // so their ratio there is exactly 4/e, independent of alpha.
    // Both the absolute values and that ratio are checked, because a shared scale error would survive the ratio alone.
    {
        double worstGGX = 0.0;
        double worstBeckmann = 0.0;
        double worstRatio = 0.0;

        for ( uint32_t level = 0; level < numLevels; ++level )
        {
            double const alpha = paperCurve.GetWidth( level );

            if ( alpha <= 0.0 )
            {
                continue;
            }

            double const alphaSquared = alpha * alpha;
            double const onePlusAlphaSquared = 1.0 + alphaSquared;

            // tan^2 == alpha^2 implies cos^2 == 1 / ( 1 + alpha^2 )
            double const cosHalfTheta = std::sqrt( 1.0 / onePlusAlphaSquared );

            double const ggx = ProfileGGX::EvaluateNDF( cosHalfTheta, alpha );
            double const beckmann = ProfileBeckmann::EvaluateNDF( cosHalfTheta, alpha );

            double const expectedGGX = ( onePlusAlphaSquared * onePlusAlphaSquared ) / ( 4.0 * g_pi * alphaSquared );
            double const expectedBeckmann = std::exp( -1.0 ) * ( onePlusAlphaSquared * onePlusAlphaSquared ) / ( g_pi * alphaSquared );
            double const expectedRatio = 4.0 * std::exp( -1.0 );

            double const ggxError = std::fabs( ggx - expectedGGX ) / expectedGGX;
            double const beckmannError = std::fabs( beckmann - expectedBeckmann ) / expectedBeckmann;
            double const ratioError = std::fabs( ( beckmann / ggx ) - expectedRatio ) / expectedRatio;

            worstGGX = ( ggxError > worstGGX ) ? ggxError : worstGGX;
            worstBeckmann = ( beckmannError > worstBeckmann ) ? beckmannError : worstBeckmann;
            worstRatio = ( ratioError > worstRatio ) ? ratioError : worstRatio;
        }

        // Tolerances come from the conditioning of the anchor angle, not from taste.
        // For the narrowest lobe alpha^2 is 7.6e-6, so the denominator (n.h)^2 ( alpha^2 - 1 ) + 1 evaluates as 1 - 0.999992: one uap in cos^2 propagates to about 3e-11 relative in the denominator, and D goes as its reciprocal squared.
        // The measured worst case is 1.5e-11. 1e-9 leaves an order of magnitude over that bound, and tightening it towards 1e-12 measures double precision rather than the profile.
        // The ratio is better conditioned because the shared ( 1 + alpha^2 )^2 / ( PI alpha^2 ) factor cancels, which is also why it still discriminates the two NDFs at 1e-10.
        constexpr double absoluteTolerance = 1.0e-9;
        constexpr double ratioTolerance = 1.0e-10;

        bool const ggxOk = ( worstGGX < absoluteTolerance );
        bool const beckmannOk = ( worstBeckmann < absoluteTolerance );
        bool const ratioOk = ( worstRatio < ratioTolerance );

        bool const anchorOk = ggxOk && beckmannOk && ratioOk;

        std::printf( "\n  analytic anchor at tan^2(theta_h) == alpha^2\n" );
        std::printf( "    GGX against its closed form            : %s (rel %.3e)\n", ggxOk ? "PASS" : "FAIL", worstGGX );
        std::printf( "    Beckmann against its closed form       : %s (rel %.3e)\n", beckmannOk ? "PASS" : "FAIL", worstBeckmann );
        std::printf( "    Beckmann/GGX against 4/e               : %s (rel %.3e)\n", ratioOk ? "PASS" : "FAIL", worstRatio );
        std::printf( "  Note the two peak at the same value, 1 / ( PI alpha^2 ), so the peak does\n" );
        std::printf( "  not distinguish them; only the tails do.\n" );

        passed = passed && anchorOk;
    }

    return passed;
}

//  Reference preimage B(x)
//-------------------------------------------------------------------------

static bool RunReferencePreimageSelfCheck()
{
    bool passed = true;

    CubeReferenceFrame baseFrame;
    baseFrame.Initialize( g_paperBaseResolution );

    ProfileGGX const profile( LobeConvention::NDFCosineHemisphere );

    // +Z is the centre of face 4
    double const outputDirection[3] = { 0.0, 0.0, 1.0 };

    std::printf( "\n" );
    std::printf( "reference preimage B(x)\n" );
    std::printf( "  base resolution %u, output direction +Z (face 4 centre)\n", g_paperBaseResolution );
    std::printf( "  convention      %s\n", profile.GetName() );
    std::printf( "\n" );
    std::printf( "  %-4s %-12s %-13s %-13s %-13s %-13s %-12s\n", "lvl", "alpha", "int 1x1", "int 8x8", "peak 1x1", "peak 8x8", "L1 1x1/8x8" );

    double levelSixRawIntegral = 0.0;

    for ( uint32_t level = 0; level < ProfileGGX::NumLevels; ++level )
    {
        double const alpha = ProfileGGX::GetAlpha( level );

        ReferencePreimage<CubeReferenceFrame> pointSampled;
        pointSampled.Initialize( &baseFrame, level, 1 );
        pointSampled.Evaluate( profile, outputDirection );

        ReferencePreimage<CubeReferenceFrame> averaged;
        averaged.Initialize( &baseFrame, level, g_supersampleRate );
        averaged.Evaluate( profile, outputDirection );

        double const rawPoint = pointSampled.ComputeIntegral( ErrorMeasure::SolidAngle );
        double const rawAveraged = averaged.ComputeIntegral( ErrorMeasure::SolidAngle );
        double const peakPoint = pointSampled.GetPeakValue();
        double const peakAveraged = averaged.GetPeakValue();

        // How wrong point-sampling is, as an L1 distance between the two normalised preimages over the whole cube
        double a1Difference = 0.0;
        for ( uint32_t texelIndex = 0; texelIndex < baseFrame.GetNumTexels(); ++texelIndex )
        {
            double const texelMeasure = baseFrame.GetTexelMeasure( baseFrame.GetTexel( texelIndex ), ErrorMeasure::SolidAngle );

            a1Difference += std::fabs( pointSampled.GetValue( texelIndex ) - averaged.GetValue( texelIndex ) ) * texelMeasure;
        }

        std::printf( "  %-4u %-12.8f %-13.6f %-13.6f %-13.6f %-13.6f %-12.6f\n", level, alpha, rawPoint, rawAveraged, peakPoint, peakAveraged, a1Difference );

        if ( !averaged.IsMonotonicInAngle() )
        {
            std::printf( "       level %u: not monotonic in angle - FAIL\n", level );
            passed = false;
        }

        // Relative tolerance.
        // The four base texels adjacent to a face centre are equidistant from the output axis and must carry identical values.
        // Their supersampled sums accumulate subsamples in different orders, so they differ by a few uap and an absolute 1e-15 threshold failed on some levels.
        uint32_t const nearest = averaged.FindNearestTexel();
        if ( averaged.GetValue( nearest ) < ( peakAveraged * ( 1.0 - 1.0e-9 ) ) )
        {
            std::printf( "       level %u: peak is not on the output axis - FAIL\n", level );
            passed = false;
        }

        if ( level == 6 )
        {
            levelSixRawIntegral = rawAveraged;
        }
    }

    // Analytic anchor. At level 6 alpha is exactly 1, so the NDF collapses to the constant 1/PI and the cosine convention gives
    //
    //      B = cos(theta_a) / PI
    //
    // whose integral over the hemisphere is exactly 1 (since the integral of cos over the hemisphere is PI).
    // This validates the profile, the cube measure and the texel averaging together, independently of the fit.
    double const anchorError = std::fabs( levelSixRawIntegral - 1.0 );
    bool const anchorOk = anchorError < 1.0e-3;

    std::printf( "\n" );
    std::printf( "  level 6 analytic anchor (alpha == 1, integral of B dw == 1) : %s\n", anchorOk ? "PASS" : "FAIL" );
    std::printf( "    measured %.10f, expected 1.0, absolute error %.3e\n", levelSixRawIntegral, anchorError );

    passed = passed && anchorOk;
    return passed;
}

//  B-spline recurrence
//-------------------------------------------------------------------------

static uint32_t g_randomState = 12345U;

static double NextRandom()
{
    g_randomState = ( g_randomState * 1664525U ) + 1013904223U;
    return static_cast<double>( g_randomState ) / 4294967296.0;
}

template< typename TMap >
static bool RunBsplineRecurrenceSelfCheck()
{
    bool passed = true;

    uint32_t const coarseResolution = 64;
    uint32_t const fineResolution = coarseResolution * 2;
    uint32_t const numSlices = TMap::NumSlices;
    size_t const   fineCount = static_cast<size_t>( numSlices ) * fineResolution * fineResolution;
    size_t const   coarseCount = static_cast<size_t>( numSlices ) * coarseResolution * coarseResolution;

    std::printf( "\n" );
    std::printf( "bspline recurrence (pass 1 downsample, and its transpose)\n" );
    std::printf( "  %u faces in %u slice%s: coarse %u x %u <- fine %u x %u, %zu coarser texels\n", TMap::NumFaces, numSlices, ( numSlices == 1 ) ? "" : "s", coarseResolution, coarseResolution, fineResolution, fineResolution, coarseCount );
    std::printf( "\n" );
    std::printf( "  %-22s %-9s %-12s %-10s %-13s %-9s %-9s\n", "weighting", "tap sum", "cross-face", "ds const", "us spread", "adjoint", "bspline" );

    JacobianWeighting const weightings[] =
    {
        JacobianWeighting::None,
        JacobianWeighting::InverseJacobian,
        JacobianWeighting::Jacobian,
    };

    for ( JacobianWeighting const weighting : weightings )
    {
        // Every coarser texel's tap weights must sum to 1, including texels on a face edge.
        double minTapSum = 1.0e30;
        double maxTapSum = -1.0e30;
        uint32_t crossingTexels = 0;
        uint32_t mergedTexels = 0;

        for ( uint32_t slice = 0; slice < numSlices; ++slice )
        {
            for ( uint32_t coarseY = 0; coarseY < coarseResolution; ++coarseY )
            {
                for ( uint32_t coarseX = 0; coarseX < coarseResolution; ++coarseX )
                {
                    DownsampleFootprint footprint;
                    ComputeDownsampleFootprint< TMap >( slice, coarseX, coarseY, coarseResolution, weighting, footprint );

                    double tapSum = 0.0;
                    for ( uint32_t entryIndex = 0; entryIndex < footprint.m_count; ++entryIndex )
                    {
                        tapSum += footprint.m_weight[entryIndex];
                    }

                    minTapSum = ( tapSum < minTapSum ) ? tapSum : minTapSum;
                    maxTapSum = ( tapSum > maxTapSum ) ? tapSum : maxTapSum;

                    if ( footprint.m_crossesFace )
                    {
                        ++crossingTexels;
                    }

                    if ( footprint.m_count != DownsampleFootprint::MaxTexels )
                    {
                        ++mergedTexels;
                    }
                }
            }
        }

        // Analytic anchor: with no Jacobian weighting an interior footprint is the tensor product of 1D ( 1/8, 3/8, 3/8, 1/8 ) at offsets -1..2
        double bsplineError = 0.0;
        if ( weighting == JacobianWeighting::None )
        {
            double const oneDimensional[4] = { 0.125, 0.375, 0.375, 0.125 };

            // Any texel whose 4x4 footprint stays inside one face. A quarter of the way down a face is far enough from every seam of either map.
            uint32_t const anchorSlice = 0;
            uint32_t const anchorX = coarseResolution / 2;
            uint32_t const anchorY = coarseResolution / 4;

            DownsampleFootprint footprint;
            ComputeDownsampleFootprint< TMap >( anchorSlice, anchorX, anchorY, coarseResolution, weighting, footprint );

            if ( ( footprint.m_count != DownsampleFootprint::MaxTexels ) || footprint.m_crossesFace )
            {
                bsplineError = 1.0e30;
            }
            else
            {
                for ( uint32_t entryIndex = 0; entryIndex < footprint.m_count; ++entryIndex )
                {
                    int32_t const offsetX = static_cast<int32_t>( footprint.m_texelX[entryIndex] ) - static_cast<int32_t>( 2u * anchorX ) + 1;
                    int32_t const offsetY = static_cast<int32_t>( footprint.m_texelY[entryIndex] ) - static_cast<int32_t>( 2u * anchorY ) + 1;

                    if ( ( offsetX < 0 ) || ( offsetX >= 4 ) || ( offsetY < 0 ) || ( offsetY >= 4 ) )
                    {
                        bsplineError = 1.0e30;
                        continue;
                    }

                    double const expected = oneDimensional[offsetX] * oneDimensional[offsetY];
                    double const delta = std::fabs( footprint.m_weight[entryIndex] - expected );

                    bsplineError = ( delta > bsplineError ) ? delta : bsplineError;
                }
            }
        }

        // Downsample of a constant must stay constant, everywhere
        std::vector<double> fineOnes( fineCount, 1.0 );
        std::vector<double> downsampledOnes;
        DownsampleLevel< TMap >( fineOnes, coarseResolution, weighting, downsampledOnes );

        double minDownsampledConstant = 1.0e30;
        double maxDownsampledConstant = -1.0e30;
        for ( double const value : downsampledOnes )
        {
            minDownsampledConstant = ( value < minDownsampledConstant ) ? value : minDownsampledConstant;
            maxDownsampledConstant = ( value > maxDownsampledConstant ) ? value : maxDownsampledConstant;
        }

        // Upsample of a constant.
        // Not expected to be exactly constant: any position-dependent weighting breaks the partition of unity the plain b-spline has, because finer texel 2j receives 0.75*w0(j) + 0.25*w1(j-1) while 2j+1 receives 0.75*w1(j) + 0.25*w0(j+1).
        // Reported as a diagnostic.
        std::vector<double> coarseOnes( coarseCount, 1.0 );
        std::vector<double> upsampledOnes;
        UpsampleLevel< TMap >( coarseOnes, coarseResolution, weighting, upsampledOnes );

        double minUpsampledConstant = 1.0e30;
        double maxUpsampledConstant = -1.0e30;
        for ( double const value : upsampledOnes )
        {
            minUpsampledConstant = ( value < minUpsampledConstant ) ? value : minUpsampledConstant;
            maxUpsampledConstant = ( value > maxUpsampledConstant ) ? value : maxUpsampledConstant;
        }

        // A deviation indicates a fault unless a cross-face footprint reaches the texel. 
        // The plain b-spline's partition of unity is exact wherever every contributing footprint lies inside one face; where a footprint had to be re-resolved onto another face, the two terms that would have cancelled need not, and a small deviation is expected.
        // The test is therefore that no deviation is reachable only from same-face footprints, which holds for either map and asks nothing about where the seams are or what shape they have.
        std::vector<uint8_t> reachedByCrossFace( fineCount, 0 );
        uint32_t unsupportedTexels = 0;
        uint32_t deviatingTexels = 0;
        uint32_t unexplainedTexels = 0;
        double maxDeviation = 0.0;

        if ( weighting == JacobianWeighting::None )
        {
            for ( uint32_t slice = 0; slice < numSlices; ++slice )
            {
                for ( uint32_t coarseY = 0; coarseY < coarseResolution; ++coarseY )
                {
                    for ( uint32_t coarseX = 0; coarseX < coarseResolution; ++coarseX )
                    {
                        DownsampleFootprint footprint;
                        ComputeDownsampleFootprint< TMap >( slice, coarseX, coarseY, coarseResolution, weighting, footprint );

                        if ( !footprint.m_crossesFace )
                        {
                            continue;
                        }

                        for ( uint32_t entryIndex = 0; entryIndex < footprint.m_count; ++entryIndex )
                        {
                            reachedByCrossFace[
                                TMap::GetTexelIndex
                                (
                                    footprint.m_face[entryIndex],
                                    footprint.m_texelX[entryIndex],
                                    footprint.m_texelY[entryIndex],
                                    fineResolution
                                )
                            ] = 1;
                        }
                    }
                }
            }

            for ( uint32_t slice = 0; slice < numSlices; ++slice )
            {
                for ( uint32_t fineY = 0; fineY < fineResolution; ++fineY )
                {
                    for ( uint32_t fineX = 0; fineX < fineResolution; ++fineX )
                    {
                        size_t const fineIndex = TMap::GetTexelIndex( slice, fineX, fineY, fineResolution );
                        double const value = upsampledOnes[fineIndex];

                        if ( value <= 0.0 )
                        {
                            ++unsupportedTexels;
                            continue;
                        }

                        double const deviation = std::fabs( value - 0.25 );

                        if ( deviation <= 1.0e-9 )
                        {
                            continue;
                        }

                        ++deviatingTexels;

                        maxDeviation = ( deviation > maxDeviation ) ? deviation : maxDeviation;

                        if ( reachedByCrossFace[fineIndex] == 0 )
                        {
                            ++unexplainedTexels;
                        }
                    }
                }
            }
        }

        // Adjoint identity: < D x, y > == < x, D^T y >
        std::vector<double> fineField( fineCount );
        std::vector<double> coarseField( coarseCount );
        for ( double& value : fineField )
        {
            value = NextRandom();
        }
        for ( double& value : coarseField )
        {
            value = NextRandom();
        }

        std::vector<double> downsampledField;
        std::vector<double> upsampledField;
        DownsampleLevel< TMap >( fineField, coarseResolution, weighting, downsampledField );
        UpsampleLevel< TMap >( coarseField, coarseResolution, weighting, upsampledField );

        double innerLeft = 0.0;
        for ( size_t index = 0; index < coarseCount; ++index )
        {
            innerLeft += downsampledField[index] * coarseField[index];
        }

        double innerRight = 0.0;
        for ( size_t index = 0; index < fineCount; ++index )
        {
            innerRight += fineField[index] * upsampledField[index];
        }

        double const innerScale = ( std::fabs( innerLeft ) > 1.0e-30 ) ? std::fabs( innerLeft ) : 1.0;
        double const adjointError = std::fabs( innerLeft - innerRight ) / innerScale;

        double const downsampledSpread = maxDownsampledConstant - minDownsampledConstant;
        double const upsampledSpread = maxUpsampledConstant - minUpsampledConstant;
        double const tapSumSpread = maxTapSum - minTapSum;

        bool const tapSumOk = ( tapSumSpread < 1.0e-12 );
        bool const downsampleOk = ( downsampledSpread < 1.0e-12 ) && ( std::fabs( minDownsampledConstant - 1.0 ) < 1.0e-12 );
        bool const adjointOk = ( adjointError < 1.0e-12 );
        bool const bsplineOk = ( ( weighting != JacobianWeighting::None ) || ( bsplineError < 1.0e-12 ) );

        std::printf
        (
            "  %-22s %-9s %-12u %-10s %-13.6f %-9s %-9s\n",
            GetJacobianWeightingName( weighting ),
            tapSumOk ? "PASS" : "FAIL",
            crossingTexels,
            downsampleOk ? "PASS" : "FAIL",
            upsampledSpread,
            adjointOk ? "PASS" : "FAIL",
            ( weighting == JacobianWeighting::None ) ? ( bsplineOk ? "PASS" : "FAIL" ) : "n/a"
        );

        std::printf
        (
            "      tap sum %.9f..%.9f | ds const %.6f..%.6f | us const %.6f..%.6f | adjoint %.3e | merged %u",
            minTapSum, maxTapSum, minDownsampledConstant, maxDownsampledConstant,
            minUpsampledConstant, maxUpsampledConstant, adjointError, mergedTexels
        );

        if ( weighting == JacobianWeighting::None )
        {
            std::printf( " | bspline %.3e", bsplineError );
        }

        std::printf( "\n" );

        if ( weighting == JacobianWeighting::None )
        {
            // A deviation away from any cross-face footprint is a bug: nothing else can break the partition of unity.
            // An unsupported texel is a different statement - no coarse footprint reads it at all - and it is reported rather than failed on, because on a map whose seams are lines rather than points the fold can leave texels unread.
            // How many is a property of the map, and it is what the number is for.
            bool const deviationOk = ( unexplainedTexels == 0 );
            std::printf
            (
                "      upsampled-constant deviation: %u of %zu texels, worst %.3e, %u not reached by a cross-face footprint : %s\n",
                deviatingTexels, fineCount, maxDeviation, unexplainedTexels,
                deviationOk ? "PASS (seam-local)" : "FAIL (not seam-local)"
            );
            std::printf
            (
                "      fine texels no coarse footprint reads: %u of %zu (%.3f%%)\n",
                unsupportedTexels, fineCount,
                100.0 * static_cast<double>( unsupportedTexels ) / static_cast<double>( fineCount )
            );

            passed = passed && deviationOk;
        }

        passed = passed && tapSumOk && downsampleOk && adjointOk && bsplineOk;
    }

    return passed;
}

//  Preimage accumulator
//-------------------------------------------------------------------------

struct PreimageTap
{
    double  m_direction[3];
    double  m_level;
    double  m_weight;
};

template< typename TMap >
static void MakeTap( PreimageTap& tap, uint32_t face, double u, double v, double level, double weight )
{
    // The map's own face coordinate -> direction, which is the unnormalized chart vector for both maps.
    // Nothing here needs it normalized: the accumulator resolves a tap through direction space, and a scale shared by every direction resolves to the same texels.
    //
    // The face is taken modulo the map's own count, so a case that names face 4 exercises the last face of a four-face map rather than asserting.
    TMap::GetDirectionFromUV( tap.m_direction, u, v, face % TMap::NumFaces );
    tap.m_level = level;
    tap.m_weight = weight;
}

template< typename TFrame >
static bool RunPreimageCase
(
    char const* pName,
    PreimageTap const* pTaps,
    uint32_t numTaps,
    JacobianWeighting weighting,
    TFrame const& baseFrame,
    uint32_t baseResolution,
    uint32_t numLevels,
    bool checkCentroid
)
{
    using TMap = typename TFrame::Map;

    PreimageAccumulator< TMap > accumulator;
    accumulator.Initialize( baseResolution, numLevels, weighting );

    double sumTapWeights = 0.0;
    for ( uint32_t tapIndex = 0; tapIndex < numTaps; ++tapIndex )
    {
        accumulator.AddSample( pTaps[tapIndex].m_direction, pTaps[tapIndex].m_level, pTaps[tapIndex].m_weight );
        sumTapWeights += pTaps[tapIndex].m_weight;
    }

    std::vector<double> const& preimage = accumulator.ResolveToBase();

    double sumPreimage = 0.0;
    uint32_t support = 0;
    for ( double const value : preimage )
    {
        sumPreimage += value;
        if ( std::fabs( value ) > 1.0e-15 )
        {
            ++support;
        }
    }

    double const scale = ( std::fabs( sumTapWeights ) > 1.0e-30 ) ? std::fabs( sumTapWeights ) : 1.0;
    double const relError = std::fabs( sumPreimage - sumTapWeights ) / scale;

    // Centroid of the preimage, as a direction. For a single tap the footprint is symmetric about the sample direction, so this must land on it.
    double centroidAngleDegrees = -1.0;
    if ( checkCentroid )
    {
        // The tap's own direction is the map's chart vector and is not unit, so it is normalized here rather than compared raw: on a cube the length is at least 1 and the dot happens to clamp to 1, which is a coincidence and not a property of the construction.
        double const tapLength = std::sqrt( ( pTaps[0].m_direction[0] * pTaps[0].m_direction[0] ) + ( pTaps[0].m_direction[1] * pTaps[0].m_direction[1] ) + ( pTaps[0].m_direction[2] * pTaps[0].m_direction[2] ) );

        double centroid[3] = { 0.0, 0.0, 0.0 };

        // By SLICE, which is what a level is made of: a cubemap's six faces are six slices and a single-slice map's four faces share one.
        for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
        {
            for ( uint32_t texelY = 0; texelY < baseResolution; ++texelY )
            {
                for ( uint32_t texelX = 0; texelX < baseResolution; ++texelX )
                {
                    size_t const texelIndex = TMap::GetTexelIndex( slice, texelX, texelY, baseResolution );
                    double const weight = preimage[texelIndex];
                    if ( weight == 0.0 )
                    {
                        continue;
                    }

                    MapTexel const& texel = baseFrame.GetTexel( static_cast<uint32_t>( texelIndex ) );
                    centroid[0] += weight * texel.m_dir[0];
                    centroid[1] += weight * texel.m_dir[1];
                    centroid[2] += weight * texel.m_dir[2];
                }
            }
        }

        double const centroidLength = std::sqrt( ( centroid[0] * centroid[0] ) + ( centroid[1] * centroid[1] ) + ( centroid[2] * centroid[2] ) );
        if ( ( centroidLength > 0.0 ) && ( tapLength > 0.0 ) )
        {
            double const cosAngle = ( ( centroid[0] / centroidLength ) * ( pTaps[0].m_direction[0] / tapLength ) ) + ( ( centroid[1] / centroidLength ) * ( pTaps[0].m_direction[1] / tapLength ) ) + ( ( centroid[2] / centroidLength ) * ( pTaps[0].m_direction[2] / tapLength ) );

            double const clampedCos = ( cosAngle < -1.0 ) ? -1.0 : ( ( cosAngle > 1.0 ) ? 1.0 : cosAngle );
            centroidAngleDegrees = std::acos( clampedCos ) * 180.0 / g_pi;
        }
    }

    // The footprint of one tap is centred on that tap only where a chart owns its own coordinate box.
    // A map whose slice holds several charts - a single-slice tetrahedral map, four faces in one tile - has directions that jump across the lines those charts meet on, so a footprint that crosses one is not a reflection of the tap and its centroid is somewhere else.
    // The sum invariant above is a property of the operator and holds for both; this one is a property of the folding.
    bool const chartOwnsItsBox = ( TMap::NumFaces == TMap::NumSlices );

    bool const sumOk = ( relError < 1.0e-12 );
    bool const centroidOk = ( !checkCentroid ) || ( !chartOwnsItsBox ) || ( centroidAngleDegrees < 1.0e-3 );

    char const* const pVerdict = sumOk ? ( ( !checkCentroid || centroidOk ) ? "PASS" : "FAIL" )
        : "FAIL";

    if ( checkCentroid )
    {
        std::printf
        (
            "  %-34s %-8u %-14.6f %-14.6f %-12.3e %-9u centroid %.2e deg %s\n",
            pName, numTaps, sumTapWeights, sumPreimage, relError, support, centroidAngleDegrees,
            chartOwnsItsBox ? pVerdict : ( sumOk ? "PASS*" : "FAIL" )
        );
    }
    else
    {
        std::printf
        (
            "  %-34s %-8u %-14.6f %-14.6f %-12.3e %-9u %s\n",
            pName, numTaps, sumTapWeights, sumPreimage, relError, support,
            sumOk ? "PASS" : "FAIL"
        );
    }

    return sumOk && centroidOk;
}

template< typename TFrame >
static bool RunPreimageAccumulatorSelfCheck()
{
    using TMap = typename TFrame::Map;

    bool passed = true;

    uint32_t const baseResolution = g_paperBaseResolution;
    uint32_t const numLevels = ProfileGGX::NumLevels;

    JacobianWeighting const weighting = JacobianWeighting::Jacobian;

    TFrame baseFrame;
    baseFrame.Initialize( baseResolution );

    std::printf( "\n" );
    std::printf( "preimage accumulator (tap splat + upsample to base)\n" );
    std::printf( "  projection %s, %u slice%s per level\n", GetProbeMapName( ProbeMapOf< TMap >::Value ), TMap::NumSlices, ( TMap::NumSlices == 1 ) ? "" : "s" );
    std::printf( "  base %u x %u per slice, %u levels, weighting %s\n", baseResolution, baseResolution, numLevels, GetJacobianWeightingName( weighting ) );
    std::printf( "  invariant: sum over base texels of A(x) == sum of tap weights\n" );
    std::printf( "\n" );

    if ( TMap::NumFaces != TMap::NumSlices )
    {
        std::printf( "  this map puts %u charts in one coordinate box, so their directions meet along\n", TMap::NumFaces );
        std::printf( "  lines the field jumps across. A footprint that crosses one is not centred on its\n" );
        std::printf( "  tap, so those rows report the centroid and assert the sum only: PASS* is the sum.\n" );
        std::printf( "\n" );
    }

    // A texel centre, on the map's own terms.
    // The face the single-tap cases name is taken modulo the map's face count, so the same case exercises a real face of whichever map is running.
    uint32_t const centreTexel = baseResolution / 2;
    double       texelCentreU = 0.0;
    double       texelCentreV = 0.0;
    TMap::GetTexelCentreUV( &texelCentreU, &texelCentreV, centreTexel, centreTexel, baseResolution );

    // Single taps
    {
        PreimageTap taps[1];

        MakeTap< TMap >( taps[0], 4, texelCentreU, texelCentreV, 0.0, 1.0 );
        passed = RunPreimageCase( "single tap, level 0, texel centre", taps, 1, weighting, baseFrame, baseResolution, numLevels, true ) && passed;

        MakeTap< TMap >( taps[0], 4, 0.0, 0.0, 0.0, 1.0 );
        passed = RunPreimageCase( "single tap, level 0, texel corner", taps, 1, weighting, baseFrame, baseResolution, numLevels, true ) && passed;

        MakeTap< TMap >( taps[0], 4, 0.0, 0.0, 1.0, 1.0 );
        passed = RunPreimageCase( "single tap, level 1", taps, 1, weighting, baseFrame, baseResolution, numLevels, true ) && passed;

        MakeTap< TMap >( taps[0], 4, 0.0, 0.0, 2.5, 1.0 );
        passed = RunPreimageCase( "single tap, level 2.5 (trilinear)", taps, 1, weighting, baseFrame, baseResolution, numLevels, true ) && passed;

        MakeTap< TMap >( taps[0], 4, 0.0, 0.0, 6.0, 1.0 );
        passed = RunPreimageCase( "single tap, level 6 (coarsest)", taps, 1, weighting, baseFrame, baseResolution, numLevels, true ) && passed;

        MakeTap< TMap >( taps[0], 0, 0.0, 0.0, 3.0, 1.0 );
        passed = RunPreimageCase( "single tap, level 3, face 0", taps, 1, weighting, baseFrame, baseResolution, numLevels, true ) && passed;

        // A tap sitting on a face edge, so its bilinear footprint crosses
        MakeTap< TMap >( taps[0], 4, 1.0 - ( 0.25 / static_cast<double>( baseResolution ) ), 0.0, 2.0, 1.0 );
        passed = RunPreimageCase( "single tap, level 2, on a face edge", taps, 1, weighting, baseFrame, baseResolution, numLevels, false ) && passed;
    }

    // Multiple taps
    {
        PreimageTap taps[4];
        MakeTap< TMap >( taps[0], 4, 0.0, 0.0, 0.0, 0.35 );
        MakeTap< TMap >( taps[1], 4, 0.25, -0.25, 2.0, 0.25 );
        MakeTap< TMap >( taps[2], 0, 0.0, 0.0, 4.0, 0.25 );
        MakeTap< TMap >( taps[3], 2, -0.5, 0.5, 2.75, 0.15 );

        passed = RunPreimageCase( "four taps, mixed levels and faces", taps, 4, weighting, baseFrame, baseResolution, numLevels, false ) && passed;
    }

    // A tap weight of zero must contribute nothing
    {
        PreimageTap taps[2];
        MakeTap< TMap >( taps[0], 4, 0.0, 0.0, 3.0, 1.0 );
        MakeTap< TMap >( taps[1], 4, 0.5, 0.5, 3.0, 0.0 );

        passed = RunPreimageCase( "zero-weight tap is a no-op", taps, 2, weighting, baseFrame, baseResolution, numLevels, false ) && passed;
    }

    return passed;
}

//  L1 preimage error
//-------------------------------------------------------------------------

static bool RunPreimageErrorSelfCheck()
{
    bool passed = true;

    uint32_t const baseResolution = g_paperBaseResolution;
    uint32_t const numLevels = ProfileGGX::NumLevels;

    CubeReferenceFrame baseFrame;
    baseFrame.Initialize( baseResolution );

    ProfileGGX const profile( LobeConvention::NDFCosineHemisphere );

    double const outputDirection[3] = { 0.0, 0.0, 1.0 };

    std::printf( "\n" );
    std::printf( "L1 preimage error\n" );

    // Closed form
    // Comparing B against a scaled copy k*B must give exactly |1 - k|, since the measure-weighted integral of B is 1 by construction.
    // This pins the unit conversion between a coefficient field and a density.
    {
        ReferencePreimage<CubeReferenceFrame> reference;
        reference.Initialize( &baseFrame, 3, g_supersampleRate );
        reference.Evaluate( profile, outputDirection );
        reference.Normalize( ErrorMeasure::SolidAngle );

        std::printf( "\n" );
        std::printf( "  closed form: B against a scaled copy k*B, expect L1 == |1 - k|\n" );
        std::printf( "  %-8s %-16s %-16s %-13s %s\n", "k", "expected L1", "measured L1", "rel error", "" );

        double const scales[] = { 0.5, 0.8, 1.0, 1.25, 2.0 };

        for ( double const scale : scales )
        {
            // A_s = k * B_s * weightSum * measure_s, so A_density == k * B
            std::vector<double> approximation( baseFrame.GetNumTexels() );
            for ( uint32_t texelIndex = 0; texelIndex < baseFrame.GetNumTexels(); ++texelIndex )
            {
                double const texelMeasure = baseFrame.GetTexelMeasure( baseFrame.GetTexel( texelIndex ), ErrorMeasure::SolidAngle );
                approximation[texelIndex] = scale * reference.GetValue( texelIndex ) * texelMeasure;
            }

            PreimageComparison const comparison = ComparePreimages( baseFrame, reference.GetValues(), approximation, 1.0, ErrorMeasure::SolidAngle );

            double const expected = std::fabs( 1.0 - scale );
            double const relError = ( expected > 1.0e-30 ) ? ( std::fabs( comparison.m_l1 - expected ) / expected ) : std::fabs( comparison.m_l1 );

            bool const caseOk = ( relError < 1.0e-12 )
                && ( std::fabs( comparison.m_referenceIntegral - 1.0 ) < 1.0e-12 )
                && ( std::fabs( comparison.m_approximationIntegral - scale ) < 1.0e-12 );

            std::printf( "  %-8.2f %-16.9f %-16.9f %-13.3e %s\n", scale, expected, comparison.m_l1, relError, caseOk ? "PASS" : "FAIL" );

            passed = passed && caseOk;
        }
    }

    // End to end
    // 
    // A single tap's basis function is about as wide as the texel at its own level, so against a level-L lobe the error should bottom out when the tap level matches.
    // This is the first check that exercises the whole chain - profile, measure, splat, recurrence, metric - together.
    {
        uint32_t const outputLevel = 3;

        ReferencePreimage<CubeReferenceFrame> reference;
        reference.Initialize( &baseFrame, outputLevel, g_supersampleRate );
        reference.Evaluate( profile, outputDirection );
        reference.Normalize( ErrorMeasure::SolidAngle );

        std::printf( "\n" );
        std::printf( "  single tap against the level %u lobe (expect minimum at tap level %u)\n", outputLevel, outputLevel );
        std::printf( "  %-10s %-16s %-16s %-14s %s\n", "tap level", "L1 (solid angle)", "L1 (cube area)", "ref integral", "" );

        double bestL1 = 1.0e30;
        uint32_t bestLevel = 0;

        for ( uint32_t tapLevel = 0; tapLevel < numLevels; ++tapLevel )
        {
            PreimageAccumulator< CubeProjection > accumulator;
            accumulator.Initialize( baseResolution, numLevels, JacobianWeighting::Jacobian );
            accumulator.AddSample( outputDirection, static_cast<double>( tapLevel ), 1.0 );

            std::vector<double> const& approximation = accumulator.ResolveToBase();

            PreimageComparison const solid = ComparePreimages( baseFrame, reference.GetValues(), approximation, 1.0, ErrorMeasure::SolidAngle );
            PreimageComparison const coordinateArea = ComparePreimages( baseFrame, reference.GetValues(), approximation, 1.0, ErrorMeasure::CoordinateArea );

            if ( solid.m_l1 < bestL1 )
            {
                bestL1 = solid.m_l1;
                bestLevel = tapLevel;
            }

            std::printf( "  %-10u %-16.6f %-16.6f %-14.6f\n", tapLevel, solid.m_l1, coordinateArea.m_l1, solid.m_referenceIntegral );
        }

        bool const trendOk = ( bestLevel == outputLevel );
        std::printf( "\n" );
        std::printf( "  best tap level %u, L1 %.6f                          : %s\n", bestLevel, bestL1, trendOk ? "PASS" : "FAIL" );

        passed = passed && trendOk;
    }

    return passed;
}

// Published table conformance test
//-------------------------------------------------------------------------
// Runs the PUBLISHED coefficient tables through our evaluator and reports the resulting L1 beside the paper's Table 1.
// This is the end-to-end check that the measure, lobe projection, splat, recurrence and metric agree with theirs - it exercises every other stage at once.
//
// It reports rather than asserts: there is no meaningful pass threshold while a known definitional difference in B remains at sub-texel lobes.

static void RunPublishedTableConformanceTest()
{
    std::printf( "\n" );
    std::printf( "=========================================================================\n" );
    std::printf( " PUBLISHED TABLE CONFORMANCE - the paper's tables, run through our evaluator\n" );
    std::printf( "=========================================================================\n" );
    std::printf( "  paper Table 1, average L1 excluding level 0:\n" );
    std::printf( "    const_8 0.1111    const_16 0.0815    const_32 0.0613    quad_32 0.0506\n" );

    ProfileGGX const profile( LobeConvention::NDFCosineHemisphere );

    EvaluationSettings settings;
    settings.m_baseResolution = g_paperBaseResolution;
    settings.m_gridSize = g_conformanceGridSize;
    settings.m_supersampleRate = g_conformanceSupersampleRate;

    // Only quad_32 has published per-level numbers, from the paper's Table 2
    double const quad32PaperPerLevel[CoefficientTable::NumLevels] = { 0.0495, 0.0461, 0.0516, 0.0410, 0.0394, 0.0502, 0.0758 };

    ReferenceTable const tables[] =
    {
        ReferenceTable::Const8,
        ReferenceTable::Const16,
        ReferenceTable::Const32,
        ReferenceTable::Quad32,
    };

    double const published[] = { 0.1111, 0.0815, 0.0613, 0.0506 };

    // Built once and shared: it is 98304 texels, and the fit will hold one for the whole run rather than rebuilding it per table.
    CubeReferenceFrame baseFrame;
    baseFrame.Initialize( settings.m_baseResolution );

    for ( uint32_t tableIndex = 0; tableIndex < 4; ++tableIndex )
    {
        TableEvaluation const result = EvaluatePublishedTable( tables[tableIndex], baseFrame, profile, settings );
        bool const hasPerLevel = ( tables[tableIndex] == ReferenceTable::Quad32 );

        std::printf( "\n  %s   (published %.4f)\n", GetReferenceTableName( tables[tableIndex] ), published[tableIndex] );
        std::printf( "    %-6s %-12s %-10s", "level", "lobe/texel", "ours" );

        if ( hasPerLevel )
        {
            std::printf( "%-10s %-10s", "paper", "excess" );
        }

        std::printf( "\n" );

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            double const ours = result.m_levels[level].m_averageL1;

            std::printf( "    %-6u %-12.2f %-10.4f", level, GetLobeToTexelRatio( profile, level, g_paperBaseResolution ), ours );

            if ( hasPerLevel )
            {
                std::printf( "%-10.4f %+-10.4f", quad32PaperPerLevel[level], ours - quad32PaperPerLevel[level] );
            }

            std::printf( "\n" );
        }

        std::printf( "    levels 1-6 average: %.4f   published %.4f   ratio %.3f\n",
                     result.m_averageL1, published[tableIndex], result.m_averageL1 / published[tableIndex] );
        std::fflush( stdout );
    }

    std::printf( "\n" );
}

//  Radial profile
//-------------------------------------------------------------------------
// Ring means of the normalised reference and of the table reconstruction, in rings half a base texel wide.
// Locates an L1 difference at the peak, in the near tail or in the far tail, which L1 alone does not distinguish.

static void PrintRadialProfile
(
    CoefficientTable const& coefficientTable,
    CubeReferenceFrame const& baseFrame,
    ProfileGGX const& profile,
    EvaluationSettings const& settings,
    char const* pLabel,
    uint32_t level,
    uint32_t face,
    uint32_t texelX,
    uint32_t texelY
)
{
    // 16 rings of half a base texel
    constexpr uint32_t numBuckets = 32;

    uint32_t const resolution = settings.m_baseResolution >> level;
    double const   inverseResolution = 1.0 / static_cast<double>( resolution );

    double const u = ( ( 2.0 * static_cast<double>( texelX ) ) + 1.0 ) * inverseResolution - 1.0;
    double const v = -( ( 2.0 * static_cast<double>( texelY ) ) + 1.0 ) * inverseResolution + 1.0;

    double outputDirection[3];
    GetCubeFaceDirection( outputDirection, u, v, face );

    ReferencePreimage<CubeReferenceFrame> reference;
    reference.Initialize( &baseFrame, level, settings.m_supersampleRate );
    reference.Evaluate( profile, outputDirection );
    reference.Normalize( settings.m_measure );

    std::vector<TableTap> taps;
    BuildTableTaps< CubeProjection >( coefficientTable, level, face, texelX, texelY, settings.m_baseResolution, taps );

    PreimageAccumulator< CubeProjection > accumulator;
    accumulator.Initialize( settings.m_baseResolution, settings.m_sampleLevelCount, settings.m_weighting );
    accumulator.Reset();

    double totalTapWeight = 0.0;
    for ( TableTap const& tap : taps )
    {
        accumulator.AddSample( tap.m_direction, tap.m_level, tap.m_weight );
        totalTapWeight += tap.m_weight;
    }

    std::vector<double> const& approximation = accumulator.ResolveToBase();

    double const texelAngle = std::numbers::pi_v<double> / ( 2.0 * static_cast<double>( settings.m_baseResolution ) );
    double const ringWidth = 0.5 * texelAngle;

    double referenceSum[numBuckets] = {};
    double approximationSum[numBuckets] = {};
    double measureSum[numBuckets] = {};
    uint32_t count[numBuckets] = {};

    for ( uint32_t texelIndex = 0; texelIndex < baseFrame.GetNumTexels(); ++texelIndex )
    {
        MapTexel const& texel = baseFrame.GetTexel( texelIndex );

        double const cosTheta = ( outputDirection[0] * texel.m_dir[0] ) + ( outputDirection[1] * texel.m_dir[1] ) + ( outputDirection[2] * texel.m_dir[2] );

        double const clampedCos = ( cosTheta < -1.0 ) ? -1.0 : ( ( cosTheta > 1.0 ) ? 1.0 : cosTheta );
        uint32_t const bucketIndex = static_cast<uint32_t>( std::acos( clampedCos ) / ringWidth );
        if ( bucketIndex >= numBuckets )
        {
            continue;
        }

        double const measure = baseFrame.GetTexelMeasure( texel, settings.m_measure );

        referenceSum[bucketIndex] += reference.GetValue( texelIndex ) * measure;
        approximationSum[bucketIndex] += ( approximation[texelIndex] / ( totalTapWeight * measure ) ) * measure;
        measureSum[bucketIndex] += measure;
        ++count[bucketIndex];
    }

    std::printf( "\n     %s  level %u, face %u texel (%u, %u)\n", pLabel, level, face, texelX, texelY );
    std::printf( "     %-10s %-10s %-13s %-13s %-13s %-8s\n", "ring", "texels", "reference", "approx", "difference", "count" );

    for ( uint32_t bucketIndex = 0; bucketIndex < numBuckets; ++bucketIndex )
    {
        if ( count[bucketIndex] == 0 )
        {
            continue;
        }

        double const referenceDensity = referenceSum[bucketIndex] / measureSum[bucketIndex];
        double const approximationDensity = approximationSum[bucketIndex] / measureSum[bucketIndex];

        std::printf
        (
            "     %-10u %-10.2f %-13.6f %-13.6f %+-13.6f %-8u\n", bucketIndex,
            ( static_cast<double>( bucketIndex ) + 0.5 ) * 0.5,
            referenceDensity, approximationDensity, approximationDensity - referenceDensity,
            count[bucketIndex]
        );
    }

    std::fflush( stdout );
}

// Gap diagnostics
//-------------------------------------------------------------------------
// Each experiment perturbs one input we chose rather than read off the paper, and reports the per-level L1 that results.
// Reports only; nothing here changes the pipeline.

static void RunGapDiagnostics()
{
    std::printf( "\n" );
    std::printf( "=========================================================================\n" );
    std::printf( " GAP DIAGNOSTICS - where the conformance difference comes from\n" );
    std::printf( "=========================================================================\n" );

    ProfileGGX const profile( LobeConvention::NDFCosineHemisphere );

    EvaluationSettings settings;
    settings.m_baseResolution = g_paperBaseResolution;
    settings.m_gridSize = g_conformanceGridSize;
    settings.m_supersampleRate = g_conformanceSupersampleRate;

    ReferenceTable const tables[] = { ReferenceTable::Const8, ReferenceTable::Quad32 };
    char const* const    names[] = { "const_8", "quad_32" };

    CubeReferenceFrame diagnosticFrame;
    diagnosticFrame.Initialize( settings.m_baseResolution );

    // Levels in the intermediate mip chain
    // The table indexes 7 output levels, but PROBE_REFLECTION_MIPS_FULL is 8, so a tap may be able to read an 8th level at resolution 1.
    std::printf( "\n  1. intermediate mip chain length (PROBE_REFLECTION_MIPS_FULL = 8)\n" );
    std::printf( "     %-10s %-8s %-10s %-10s %-10s\n", "table", "levels", "L5", "L6", "avg 1-6" );

    for ( uint32_t tableIndex = 0; tableIndex < 2; ++tableIndex )
    {
        for ( uint32_t levelCount : { 7u, 8u } )
        {
            EvaluationSettings trial = settings;
            trial.m_sampleLevelCount = levelCount;

            TableEvaluation const result = EvaluatePublishedTable( tables[tableIndex], diagnosticFrame, profile, trial );

            std::printf( "     %-10s %-8u %-10.4f %-10.4f %-10.4f\n", names[tableIndex], levelCount, result.m_levels[5].m_averageL1, result.m_levels[6].m_averageL1, result.m_averageL1 );
            std::fflush( stdout );
        }
    }

    // Reference smoothing
    // Broadens B toward what a fit that accounts for the trilinear reconstruction convolution would be targeting.
    // If the sub-texel excess is that difference, levels 1-2 should fall and levels 3-5 should not move.
    std::printf( "\n  2. reference smoothing passes (0.25, 0.5, 0.25) per pass\n" );
    std::printf( "     %-10s %-8s %-10s %-10s %-10s %-10s\n", "table", "passes", "L0", "L1", "L2", "L3-5 mean" );

    for ( uint32_t tableIndex = 0; tableIndex < 2; ++tableIndex )
    {
        for ( uint32_t smoothing : { 0u, 1u, 2u } )
        {
            EvaluationSettings trial = settings;
            trial.m_referenceSmoothing = smoothing;

            TableEvaluation const result = EvaluatePublishedTable( tables[tableIndex], diagnosticFrame, profile, trial );

            double const midMean = ( result.m_levels[3].m_averageL1 + result.m_levels[4].m_averageL1 + result.m_levels[5].m_averageL1 ) / 3.0;

            std::printf( "     %-10s %-8u %-10.4f %-10.4f %-10.4f %-10.4f\n", names[tableIndex], smoothing, result.m_levels[0].m_averageL1, result.m_levels[1].m_averageL1, result.m_levels[2].m_averageL1, midMean );
            std::fflush( stdout );
        }
    }

    // Reference supersampling
    // Rate 1 is a point sample at each base texel centre, which is much sharper than the texel average at sub-texel lobes.
    // If the paper's B is point sampled, this is where it would show.
    std::printf( "\n  3. reference supersampling rate (1 = point sample)\n" );
    std::printf( "     %-10s %-8s %-10s %-10s %-10s %-10s %-10s\n", "table", "rate", "L0", "L1", "L2", "L3-5", "avg 1-6" );

    for ( uint32_t tableIndex = 0; tableIndex < 2; ++tableIndex )
    {
        for ( uint32_t rate : { 1u, 2u, 4u } )
        {
            EvaluationSettings trial = settings;
            trial.m_supersampleRate = rate;

            TableEvaluation const result = EvaluatePublishedTable( tables[tableIndex], diagnosticFrame, profile, trial );

            double const midMean = ( result.m_levels[3].m_averageL1 + result.m_levels[4].m_averageL1 + result.m_levels[5].m_averageL1 ) / 3.0;

            std::printf
            (
                "     %-10s %-8u %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n", names[tableIndex], rate,
                result.m_levels[0].m_averageL1, result.m_levels[1].m_averageL1, result.m_levels[2].m_averageL1,
                midMean, result.m_averageL1
            );
            std::fflush( stdout );
        }
    }

    // Texel measure
    // The paper defines SAtexel as 4*J/R^2 in Eq. 4, which is not the exact solid angle.
    // If their objective used that weighting, this is the difference.
    std::printf( "\n  4. texel measure\n" );
    std::printf( "     %-10s %-20s %-10s %-10s %-10s\n", "table", "measure", "L3-5 mean", "L6", "avg 1-6" );

    ErrorMeasure const measures[] = { ErrorMeasure::SolidAngle, ErrorMeasure::JacobianTexelArea, ErrorMeasure::CoordinateArea };
    char const* const    measureNames[] = { "exact solid angle", "4J/R^2 (paper Eq. 4)", "uniform map area" };

    for ( uint32_t tableIndex = 0; tableIndex < 2; ++tableIndex )
    {
        for ( uint32_t measureIndex = 0; measureIndex < 3; ++measureIndex )
        {
            EvaluationSettings trial = settings;
            trial.m_measure = measures[measureIndex];

            TableEvaluation const result = EvaluatePublishedTable( tables[tableIndex], diagnosticFrame, profile, trial );

            double const midMean = ( result.m_levels[3].m_averageL1 + result.m_levels[4].m_averageL1 + result.m_levels[5].m_averageL1 ) / 3.0;

            std::printf( "     %-10s %-20s %-10.4f %-10.4f %-10.4f\n", names[tableIndex], measureNames[measureIndex], midMean, result.m_levels[6].m_averageL1, result.m_averageL1 );
            std::fflush( stdout );
        }
    }

    // Output texel position
    // If L1 depends on where a texel sits on its face, our average depends on the sampling distribution, which the paper does not specify.
    std::printf( "\n  5. L1 by output texel position, levels 3-5 (bands of max(|u|,|v|))\n" );
    std::printf( "     %-10s %-12s %-12s %-12s\n", "table", "centre<0.5", "mid<0.75", "edge<=1" );

    for ( uint32_t tableIndex = 0; tableIndex < 2; ++tableIndex )
    {
        TableEvaluation const result = EvaluatePublishedTable( tables[tableIndex], diagnosticFrame, profile, settings );

        double bandSum[TableEvaluationLevel::NumPositionBands] = {};
        uint32_t bandCount[TableEvaluationLevel::NumPositionBands] = {};

        for ( uint32_t level = 3; level <= 5; ++level )
        {
            for ( uint32_t band = 0; band < TableEvaluationLevel::NumPositionBands; ++band )
            {
                bandSum[band] += result.m_levels[level].m_bandAverageL1[band] * result.m_levels[level].m_bandNumTexels[band];
                bandCount[band] += result.m_levels[level].m_bandNumTexels[band];
            }
        }

        std::printf( "     %-10s", names[tableIndex] );

        for ( uint32_t band = 0; band < TableEvaluationLevel::NumPositionBands; ++band )
        {
            std::printf( " %-12.4f", ( bandCount[band] > 0 ) ? ( bandSum[band] / bandCount[band] ) : 0.0 );
        }

        std::printf( "\n" );
        std::fflush( stdout );
    }

    // Lobe convention
    // The zonal shape is a choice we made, not something the paper states as a formula.
    // This measures which of the three the published tables were fitted against.
    std::printf( "\n  6. lobe convention\n" );
    std::printf( "     %-10s %-26s %-10s %-10s %-10s %-10s %-10s %-10s\n", "table", "convention", "L0", "L1", "L2", "L3-5", "L6", "avg 1-6" );

    LobeConvention const conventions[] = { LobeConvention::NDFOnly, LobeConvention::NDFHemisphere, LobeConvention::NDFCosineHemisphere };

    for ( uint32_t tableIndex = 0; tableIndex < 2; ++tableIndex )
    {
        for ( LobeConvention convention : conventions )
        {
            ProfileGGX const trialProfile( convention );

            TableEvaluation const result = EvaluatePublishedTable( tables[tableIndex], diagnosticFrame, trialProfile, settings );

            double const midMean = ( result.m_levels[3].m_averageL1 + result.m_levels[4].m_averageL1 + result.m_levels[5].m_averageL1 ) / 3.0;

            std::printf
            (
                "     %-10s %-26s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n", names[tableIndex], trialProfile.GetName(),
                result.m_levels[0].m_averageL1, result.m_levels[1].m_averageL1, result.m_levels[2].m_averageL1,
                midMean, result.m_levels[6].m_averageL1, result.m_averageL1
            );
            std::fflush( stdout );
        }
    }

    // Radial profile
    // Splits the per-level L1 into rings around the lobe axis, so a difference at the peak is separable from one in the tail.
    std::printf( "\n  7. radial profile, quad_32, lobe axis at the centre of face 4\n" );

    CubeReferenceFrame profileFrame;
    profileFrame.Initialize( settings.m_baseResolution );

    CoefficientTable quadTable;
    quadTable.Load( ReferenceTable::Quad32 );

    for ( uint32_t rate : { 1u, 4u } )
    {
        EvaluationSettings trial = settings;
        trial.m_supersampleRate = rate;

        for ( uint32_t level : { 0u, 1u, 2u, 3u } )
        {
            char label[32];
            std::snprintf( label, sizeof( label ), "rate %u", rate );

            // BuildTableTaps takes coordinates in the level's own resolution
            uint32_t const levelResolution = settings.m_baseResolution >> level;

            PrintRadialProfile
            (
                quadTable, profileFrame, profile, trial, label, level,
                4, levelResolution / 2, levelResolution / 2
            );
        }
    }

    // Lobe width
    // Scores each published level against a range of lobe widths.
    // If the minimum sits away from 1.0 the gap is a roughness curve difference, not something downstream of the profile.
    std::printf( "\n  8. L1 against a scaled lobe width (1.00 = spec_params_for_mip.txt)\n" );

    double const   scales[] = { 0.70, 0.85, 1.00, 1.20, 1.40 };
    uint32_t const numScales = 5;

    for ( uint32_t tableIndex = 0; tableIndex < 2; ++tableIndex )
    {
        CoefficientTable levelTable;
        levelTable.Load( tables[tableIndex] );

        std::printf( "\n     %s\n", names[tableIndex] );
        std::printf( "     %-8s", "level" );

        for ( uint32_t scaleIndex = 0; scaleIndex < numScales; ++scaleIndex )
        {
            std::printf( " %-10.2f", scales[scaleIndex] );
        }

        std::printf( " %-10s %-8s\n", "best", "at" );

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            std::printf( "     %-8u", level );

            double bestL1 = 1.0e30;
            double bestScale = 0.0;

            for ( uint32_t scaleIndex = 0; scaleIndex < numScales; ++scaleIndex )
            {
                ProfileGGX const trialProfile( LobeConvention::NDFCosineHemisphere, scales[scaleIndex] );

                TableEvaluationLevel const levelResult = EvaluatePublishedTableLevel( levelTable, profileFrame, trialProfile, settings, level );

                std::printf( " %-10.4f", levelResult.m_averageL1 );

                if ( levelResult.m_averageL1 < bestL1 )
                {
                    bestL1 = levelResult.m_averageL1;
                    bestScale = scales[scaleIndex];
                }
            }

            std::printf( " %-10.4f %-8.2f\n", bestL1, bestScale );
            std::fflush( stdout );
        }
    }

    // Level 0 versus the reference quadrature
    // Levels 1-6 prefer scale 1.00 sharply, so their roughness curve is confirmed.
    // Level 0 prefers a narrower lobe, which is what a point-sampled B would look like after normalisation: it under-counts the sub-texel core and so carries less tail mass than the texel average.
    // If the level-0 optimum moves to 1.00 at rate 1, that settles the paper's discretisation.
    std::printf( "\n  9. level 0 width optimum against supersampling rate, quad_32\n" );
    std::printf( "     %-8s", "rate" );

    for ( uint32_t scaleIndex = 0; scaleIndex < numScales; ++scaleIndex )
    {
        std::printf( " %-10.2f", scales[scaleIndex] );
    }

    std::printf( " %-10s %-8s\n", "best", "at" );

    CoefficientTable quadLevelTable;
    quadLevelTable.Load( ReferenceTable::Quad32 );

    for ( uint32_t rate : { 1u, 2u, 4u, 8u } )
    {
        EvaluationSettings trial = settings;
        trial.m_supersampleRate = rate;

        std::printf( "     %-8u", rate );

        double bestL1 = 1.0e30;
        double bestScale = 0.0;

        for ( uint32_t scaleIndex = 0; scaleIndex < numScales; ++scaleIndex )
        {
            ProfileGGX const trialProfile( LobeConvention::NDFCosineHemisphere, scales[scaleIndex] );

            TableEvaluationLevel const levelResult = EvaluatePublishedTableLevel( quadLevelTable, profileFrame, trialProfile, trial, 0 );

            std::printf( " %-10.4f", levelResult.m_averageL1 );

            if ( levelResult.m_averageL1 < bestL1 )
            {
                bestL1 = levelResult.m_averageL1;
                bestScale = scales[scaleIndex];
            }
        }

        std::printf( " %-10.4f %-8.2f\n", bestL1, bestScale );
        std::fflush( stdout );
    }

    std::printf( "\n" );
}

// Table round-trip check
//-------------------------------------------------------------------------
// A fit is worthless if it optimizes a function nobody measures.
// The guard is structural - CoefficientTable is both the conformance test's input and the optimizer's output, so there is only one evaluation path and no second implementation to drift - and this check proves the structural claim holds.
//
// It takes each published table, reads every level out as a flat parameter vector, writes that vector into a table built from scratch by Create, and requires the two tables to score identically.
// Any disagreement in the parameter layout, the tap addressing, the shape bookkeeping or the active coefficient count shows up as a different L1.
//
// The values it prints are the trusted ones: these are the same per-level numbers the conformance test reports for the vendored tables.

static bool RunTableRoundTripCheck()
{
    bool passed = true;

    ReferenceTable const tables[] = { ReferenceTable::Const8, ReferenceTable::Const16, ReferenceTable::Const32, ReferenceTable::Quad32 };

    ProfileGGX const profile( LobeConvention::NDFCosineHemisphere );

    EvaluationSettings settings;
    settings.m_baseResolution = g_paperBaseResolution;
    settings.m_gridSize = g_conformanceGridSize;
    settings.m_supersampleRate = g_conformanceSupersampleRate;

    CubeReferenceFrame baseFrame;
    baseFrame.Initialize( settings.m_baseResolution );

    std::printf( "\n" );
    std::printf( "optimizer objective over the published tables (gate 1)\n" );
    std::printf( "  each table rebuilt from its own parameter vectors, then both scored\n" );
    std::printf( "\n  %-10s %-7s %-7s %-8s %-12s %-12s %-12s %-6s\n", "table", "taps", "coeffs", "unknowns", "rebuilt L6", "loaded L6", "worst |d|", "" );

    for ( ReferenceTable tableId : tables )
    {
        CoefficientTable loaded;
        loaded.Load( tableId );

        if ( !loaded.ValidateShape() )
        {
            std::printf( "  %-10s shape invalid after Load - FAIL\n", loaded.GetName() );
            passed = false;
            continue;
        }

        CoefficientTable rebuilt;
        rebuilt.Create( tableId );

        if ( !rebuilt.ValidateShape() )
        {
            std::printf( "  %-10s shape invalid after Create - FAIL\n", rebuilt.GetName() );
            passed = false;
            continue;
        }

        if ( ( loaded.GetLevelParameterCount() != rebuilt.GetLevelParameterCount() )
         || ( loaded.GetTapCount() != rebuilt.GetTapCount() )
         || ( loaded.GetActiveCoefficientCount() != rebuilt.GetActiveCoefficientCount() ) )
        {
            std::printf( "  %-10s Create produced a different shape from Load - FAIL\n", loaded.GetName() );
            passed = false;
            continue;
        }

        std::vector<double> values;
        double worstDifference = 0.0;
        double loadedLevelSix = 0.0;
        double rebuiltLevelSix = 0.0;

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            loaded.GetLevelParameters( level, values );
            rebuilt.SetLevelParameters( level, values );

            TableEvaluationLevel const loadedResult = EvaluatePublishedTableLevel( loaded, baseFrame, profile, settings, level );
            TableEvaluationLevel const rebuiltResult = EvaluatePublishedTableLevel( rebuilt, baseFrame, profile, settings, level );

            double const difference = std::fabs( loadedResult.m_averageL1 - rebuiltResult.m_averageL1 );
            worstDifference = std::fmax( worstDifference, difference );

            if ( level == 6 )
            {
                loadedLevelSix = loadedResult.m_averageL1;
                rebuiltLevelSix = rebuiltResult.m_averageL1;
            }
        }

        bool const identical = worstDifference == 0.0;
        passed = passed && identical;

        std::printf
        (
            "  %-10s %-7u %-7u %-8u %-12.4f %-12.4f %-12.3e %-6s\n",
            loaded.GetName(), loaded.GetTapCount(), loaded.GetActiveCoefficientCount(),
            loaded.GetLevelParameterCount(), rebuiltLevelSix, loadedLevelSix,
            worstDifference, identical ? "ok" : "FAIL"
        );
        std::fflush( stdout );
    }

    std::printf( "\n  'unknowns' is the count the optimizer works in, per level. It is the\n" );
    std::printf( "  tap count times 5 parameters times the active coefficient count.\n" );

    return passed;
}

// Objective cross-check, and the recovery check
//-------------------------------------------------------------------------
// The cross-check confirms that LevelObjective reproduces the conformance numbers from the published parameter vectors.
// It is the same guard as the round-trip check one layer out: the objective caches B instead of recomputing it, so it is a second code path through the splat and the compare, and it has to be shown to agree.
//
// The recovery check then perturbs every unknown of a published level and requires the optimizer to drive the objective back to at most the published table's value, on the training sample and on a denser sample it never saw.
// That validates the objective, its gradient, the search and the convergence together, against a ground truth already established, without reference to any paper number.
//
// Level 6 first: cheapest, and its tap footprint is the widest so it is the least sensitive to sub-texel detail.

template< typename TProfile >
static bool RunObjectiveCrossCheck( CubeReferenceFrame const& baseFrame, TProfile const& profile, EvaluationSettings const& settings )
{
    bool passed = true;

    std::printf( "\n" );
    std::printf( "objective cross-check\n" );
    std::printf( "  cached-B objective against the harness, per level\n" );

    ReferenceTable const tables[] = { ReferenceTable::Const8, ReferenceTable::Quad32 };

    std::printf( "\n  %-10s %-7s %-8s %-14s %-14s %-14s %-6s\n", "table", "level", "texels", "objective", "harness", "difference", "" );
    for ( ReferenceTable tableId : tables )
    {
        CoefficientTable published;
        published.Load( tableId );

        for ( uint32_t level : { 0u, 3u, 6u } )
        {
            LevelObjective< CubeReferenceFrame, TProfile > objective;
            objective.Initialize( baseFrame, tableId, profile, settings, level );

            std::vector<double> parameters;
            published.GetLevelParameters( level, parameters );

            double const objectiveValue = objective.Evaluate( parameters );

            TableEvaluationLevel const harnessValue = EvaluatePublishedTableLevel( published, baseFrame, profile, settings, level );

            double const difference = std::fabs( objectiveValue - harnessValue.m_averageL1 );

            // Bit-identical is not achievable here and should not be required: the harness reduces per-worker partials in worker order, the objective sums per-texel values in texel order, and those two orders round differently. 
            // A few ULP, not a disagreement.
            double const tolerance = 1.0e-13 * ( ( objectiveValue > 0.0 ) ? objectiveValue : 1.0 );

            bool const identical = ( difference <= tolerance );
            passed = passed && identical;

            std::printf
            (
                "  %-10s %-7u %-8u %-14.8f %-14.8f %-14.3e %-6s\n",
                published.GetName(), level, objective.GetNumOutputTexels(),
                objectiveValue, harnessValue.m_averageL1, difference,
                identical ? "ok" : "FAIL"
            );
            std::fflush( stdout );
        }
    }

    return passed;
}

//  Showcase composition
//-------------------------------------------------------------------------

template< typename TMap >
static bool RunShowcaseLayoutCheck()
{
    bool passed = true;

    uint32_t const sourceResolution = 4;
    uint32_t const numLevels = 3;

    auto makeLevel = [&] ( uint32_t resolution, uint32_t level, HDRIImage& image )
    {
        image.Create( resolution, TMap::NumSlices );

        for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
        {
            for ( uint32_t y = 0; y < resolution; ++y )
            {
                for ( uint32_t x = 0; x < resolution; ++x )
                {
                    // Encodes the level, the slice and the texel, so a pixel can be traced back to exactly one source texel
                    float* const pTexel = image.Texel( slice, x, y );

                    pTexel[0] = static_cast<float>( ( level * 1000 ) + ( slice * 100 ) + ( y * 10 ) + x );
                    pTexel[1] = 0.0f;
                    pTexel[2] = 0.0f;
                }
            }
        }
    };

    HDRIImage source;
    makeLevel( sourceResolution, 0, source );

    std::vector<HDRIImage> reference( numLevels );
    std::vector<HDRIImage> fast( numLevels );
    std::vector<HDRIImage> naive( numLevels );

    for ( uint32_t level = 0; level < numLevels; ++level )
    {
        uint32_t const resolution = sourceResolution >> level;

        makeLevel( resolution, level, reference[level] );
        makeLevel( resolution, level + 100, fast[level] );
        makeLevel( resolution, level + 200, naive[level] );
    }

    ShowcaseSource showcase;
    showcase.m_pSource = &source;
    showcase.m_pReference = &reference;
    showcase.m_pFast = &fast;
    showcase.m_pNaive = &naive;

    ShowcaseImage image;
    ComposeShowcaseImage< TMap >( showcase, image );

    // What the layout says the dimensions are
    uint32_t expectedWidth = 0;
    uint32_t expectedHeight = 0;

    if ( TMap::NumSlices == 1 )
    {
        expectedWidth = numLevels * sourceResolution;
        expectedHeight = 4 * sourceResolution;
    }
    else
    {
        // Three blocks across - brute force, the table's, the naive mip - and one row per level under the source's
        expectedWidth = 3 * TMap::NumSlices * sourceResolution;
        expectedHeight = ( 1 + numLevels ) * sourceResolution;
    }

    bool const dimensionsOk = ( image.GetWidth() == expectedWidth ) && ( image.GetHeight() == expectedHeight );
    passed = passed && dimensionsOk;

    std::printf( "  %-22s %ux%u, %u levels  %s\n", GetProbeMapName( ProbeMapOf< TMap >::Value ), image.GetWidth(), image.GetHeight(), numLevels, dimensionsOk ? "ok" : "FAIL" );

    // Every pixel is checked against the source texel the layout says belongs there, by recomputing which tile it is in and which texel that tile's level puts at that offset.
    uint32_t numMismatches = 0;
    uint32_t firstBadTileX = 0;
    uint32_t firstBadTileY = 0;

    for ( uint32_t y = 0; y < image.GetHeight(); ++y )
    {
        for ( uint32_t x = 0; x < image.GetWidth(); ++x )
        {
            uint32_t const tileX = x / sourceResolution;
            uint32_t const tileY = y / sourceResolution;

            HDRIImage const* pExpected = nullptr;
            uint32_t         expectedSlice = 0;

            if ( TMap::NumSlices == 1 )
            {
                if ( tileY == 0 )
                {
                    // The source, in the first tile of its row and nowhere else
                    pExpected = ( tileX == 0 ) ? &source : nullptr;
                }
                else if ( tileY == 1 )
                {
                    pExpected = &reference[tileX];
                }
                else if ( tileY == 2 )
                {
                    pExpected = &fast[tileX];
                }
                else
                {
                    pExpected = &naive[tileX];
                }
            }
            else
            {
                // Three blocks across: brute force, then the table's, then the naive mip, the same level in all three.
                // The source's row has the first block only, and the other two are black.
                uint32_t const blockWidth = TMap::NumSlices * sourceResolution;

                uint32_t const block = x / blockWidth;
                uint32_t const blockX = x - ( block * blockWidth );

                expectedSlice = blockX / sourceResolution;

                if ( tileY == 0 )
                {
                    pExpected = ( block == 0 ) ? &source : nullptr;
                }
                else if ( block == 0 )
                {
                    pExpected = &reference[tileY - 1];
                }
                else if ( block == 1 )
                {
                    pExpected = &fast[tileY - 1];
                }
                else
                {
                    pExpected = &naive[tileY - 1];
                }
            }

            // An uncovered tile is black, and anything else is a bug
            float expectedValue = 0.0f;

            if ( pExpected != nullptr )
            {
                uint32_t const levelResolution = pExpected->GetResolution();

                uint32_t const sourceX = ( ( x % sourceResolution ) * levelResolution ) / sourceResolution;
                uint32_t const sourceY = ( ( y % sourceResolution ) * levelResolution ) / sourceResolution;

                expectedValue = pExpected->Texel( expectedSlice, sourceX, sourceY )[0];
            }

            if ( image.Pixel( x, y )[0] != expectedValue )
            {
                if ( numMismatches == 0 )
                {
                    firstBadTileX = tileX;
                    firstBadTileY = tileY;
                }

                ++numMismatches;
            }
        }
    }

    bool const valuesOk = ( numMismatches == 0 );
    passed = passed && valuesOk;

    uint32_t const numPixels = image.GetWidth() * image.GetHeight();

    if ( valuesOk )
    {
        std::printf( "  %-22s all %u pixels match the layout\n", "values", numPixels );
    }
    else
    {
        std::printf( "  %-22s %u of %u pixels differ, first in tile (%u, %u) : FAIL\n", "values", numMismatches, numPixels, firstBadTileX, firstBadTileY );
    }

    return passed;
}

//-------------------------------------------------------------------------

static bool RunShowcaseWriterCheck()
{
    HDRIImage source;
    source.Create( 4, 1 );

    for ( uint32_t y = 0; y < 4; ++y )
    {
        for ( uint32_t x = 0; x < 4; ++x )
        {
            float* const pTexel = source.Texel( 0, x, y );

            pTexel[0] = ( static_cast<float>( ( y * 4 ) + x ) * 0.5f ) - 3.0f;
            pTexel[1] = 1.0f;
            pTexel[2] = 2.0f;
        }
    }

    std::vector<HDRIImage> chain( 1 );
    chain[0].Create( 4, 1 );
    chain[0].Texel( 0, 0, 0 )[0] = -3000.0f;
    chain[0].Texel( 0, 3, 3 )[0] = 123456.0f;

    ShowcaseSource showcase;
    showcase.m_pSource = &source;
    showcase.m_pReference = &chain;
    showcase.m_pFast = &chain;
    showcase.m_pNaive = &chain;

    ShowcaseImage image;
    ComposeShowcaseImage< TetrahedralProjection >( showcase, image );

    // The tool always runs from the repository root, and this is removed again
    std::string const path = "FilterFitter_showcase_check.exr";

    std::string message;
    bool const saved = image.Save( path, message );
    bool const loaded = saved && image.Verify( path, message );

    if ( saved )
    {
        std::remove( path.c_str() );
    }

    std::printf( "  %-22s %s   %s\n", "written and read back", loaded ? "ok" : "FAIL", loaded ? "" : message.c_str() );

    return loaded;
}

static bool RunShowcaseSelfCheck()
{
    bool passed = true;

    std::printf( "\n" );
    std::printf( "showcase composition\n" );
    std::printf( "  every pixel against the source texel the layout puts there\n" );
    std::printf( "\n" );

    passed = RunShowcaseLayoutCheck< CubeProjection >() && passed;
    passed = RunShowcaseLayoutCheck< TetrahedralProjection >() && passed;

    passed = RunShowcaseWriterCheck() && passed;

    return passed;
}

// A width of zero is the identity, and it is the one level a search cannot reach: the objective is piecewise constant in the tap direction, so the gradient is zero almost everywhere and every seed scores the same.
// The closed form in BuildMirrorLevelParameters has to be exact, so this checks that it lands rather than merely that it looks reasonable.
// A level whose weight sits on one nominated tap, or whose level does not cancel the tap builder's cube-Jacobian term, scores 1e29 here rather than 0.
//-------------------------------------------------------------------------

template< typename TProfile >
static bool RunMirrorLevelCheck( CubeReferenceFrame const& baseFrame, TProfile const& profile, EvaluationSettings const& settings )
{
    if ( profile.GetWidth( 0 ) > 0.0 )
    {
        return true;
    }

    bool passed = true;

    std::printf( "\n" );
    std::printf( "mirror level construction\n" );
    std::printf( "  level 0 width is zero, so the filter is the identity and the closed form must score 0\n" );

    ReferenceTable const tables[] = { ReferenceTable::Const8, ReferenceTable::Quad32 };

    std::printf( "\n  %-10s %-8s %-14s %-6s %-6s\n", "table", "texels", "objective", "taps", "" );

    for ( ReferenceTable tableId : tables )
    {
        CoefficientTable named;
        named.Create( tableId );

        LevelObjective< CubeReferenceFrame, TProfile > objective;
        objective.Initialize( baseFrame, tableId, profile, settings, 0 );

        std::vector<double> parameters;
        BuildMirrorLevelParameters< CubeReferenceFrame::Map >( tableId, parameters );

        double const objectiveValue = objective.Evaluate( parameters );

        // Not a tolerance on a fit - the construction is exact, so anything above rounding is a defect
        bool const landed = ( objectiveValue <= 1.0e-12 );
        passed = passed && landed;

        std::printf
        (
            "  %-10s %-8u %-14.8f %-6u %-6s\n",
            named.GetName(), objective.GetNumOutputTexels(),
            objectiveValue, objective.GetTapCount(), landed ? "ok" : "FAIL"
        );
        std::fflush( stdout );
    }

    return passed;
}

//-------------------------------------------------------------------------

// Integral of a map's Jacobian over one triangle of a face's region.
//
// The seven-point degree-five rule, which is exact for a cubic and converges as the fourth power of the element size on a smooth integrand.
// A Jacobian is smooth on one chart - it is the chart's boundary that is not - so a rule with interior points is the right one, and a sum over the corners would converge only linearly.
//
// One subdivision level is applied first.
// Without it the tetrahedron's worst texel sits at 4e-4 against a 1e-3 bound, which leaves the check measuring its own quadrature as much as the map; the subdivision cuts that by about sixteen and the remaining error is the map's.

template< typename TMap >
static double IntegrateJacobianOverTriangle( double au, double av, double bU, double bV, double cU, double cV, uint32_t subdivisionDepth )
{
    static constexpr double kPoints[7][3] =
    {
        { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 },
        { 0.059715871789770, 0.470142064105115, 0.470142064105115 },
        { 0.470142064105115, 0.059715871789770, 0.470142064105115 },
        { 0.470142064105115, 0.470142064105115, 0.059715871789770 },
        { 0.797426985353087, 0.101286507323456, 0.101286507323456 },
        { 0.101286507323456, 0.797426985353087, 0.101286507323456 },
        { 0.101286507323456, 0.101286507323456, 0.797426985353087 },
    };

    static constexpr double kWeights[7] =
    {
        0.225,
        0.132394152788506, 0.132394152788506, 0.132394152788506,
        0.125939180544827, 0.125939180544827, 0.125939180544827,
    };

    if ( subdivisionDepth > 0 )
    {
        double const abU = 0.5 * ( au + bU );
        double const abV = 0.5 * ( av + bV );
        double const bcU = 0.5 * ( bU + cU );
        double const bcV = 0.5 * ( bV + cV );
        double const caU = 0.5 * ( cU + au );
        double const caV = 0.5 * ( cV + av );

        return IntegrateJacobianOverTriangle< TMap >( au, av, abU, abV, caU, caV, subdivisionDepth - 1 )
            + IntegrateJacobianOverTriangle< TMap >( abU, abV, bU, bV, bcU, bcV, subdivisionDepth - 1 )
            + IntegrateJacobianOverTriangle< TMap >( bcU, bcV, cU, cV, caU, caV, subdivisionDepth - 1 )
            + IntegrateJacobianOverTriangle< TMap >( abU, abV, bcU, bcV, caU, caV, subdivisionDepth - 1 );
    }

    double const doubleArea = std::fabs( ( ( bU - au ) * ( cV - av ) ) - ( ( bV - av ) * ( cU - au ) ) );

    if ( doubleArea <= 0.0 )
    {
        return 0.0;
    }

    double const area = 0.5 * doubleArea;
    double total = 0.0;

    for ( uint32_t point = 0; point < 7; ++point )
    {
        double const u = ( kPoints[point][0] * au ) + ( kPoints[point][1] * bU ) + ( kPoints[point][2] * cU );
        double const v = ( kPoints[point][0] * av ) + ( kPoints[point][1] * bV ) + ( kPoints[point][2] * cV );

        total += kWeights[point] * area * TMap::GetJacobian( u, v );
    }

    return total;
}

// Integral of a map's Jacobian over the convex polygon a face owns.
//
// Fan-triangulated from the polygon's first vertex.
// The Jacobian is a signed area element; both maps wind their faces outward and the fan uses the unsigned area, so a negative contribution means a face is wound inward and the check fails rather than cancelling itself out.
//-------------------------------------------------------------------------

template< typename TMap >
static double IntegrateJacobianOverRegion( double const* pPolygonU, double const* pPolygonV, uint32_t count )
{
    double total = 0.0;

    for ( uint32_t index = 1; ( index + 1 ) < count; ++index )
    {
        total += IntegrateJacobianOverTriangle< TMap >
            (
                pPolygonU[0], pPolygonV[0],
                pPolygonU[index], pPolygonV[index],
                pPolygonU[index + 1], pPolygonV[index + 1], 1
            );
    }

    return total;
}

//-------------------------------------------------------------------------

static double PolygonArea( double const* pPolygonU, double const* pPolygonV, uint32_t count )
{
    if ( count < 3 )
    {
        return 0.0;
    }

    double doubleArea = 0.0;

    for ( uint32_t index = 0; index < count; ++index )
    {
        uint32_t const next = ( index + 1 ) % count;

        doubleArea += ( pPolygonU[index] * pPolygonV[next] ) - ( pPolygonU[next] * pPolygonV[index] );
    }

    return 0.5 * std::fabs( doubleArea );
}

//-------------------------------------------------------------------------

static void PolygonCentroid
(
    double const* pPolygonU, double const* pPolygonV, uint32_t count,
    double& centroidU, double& centroidV
)
{
    centroidU = 0.0;
    centroidV = 0.0;

    if ( count == 0 )
    {
        return;
    }

    for ( uint32_t index = 0; index < count; ++index )
    {
        centroidU += pPolygonU[index];
        centroidV += pPolygonV[index];
    }

    double const inverse = 1.0 / static_cast<double>( count );

    centroidU *= inverse;
    centroidV *= inverse;
}

//  The tap layout against the table's own axis count
//-------------------------------------------------------------------------

template< typename TMap >
static bool RunTapLayoutCheck( CoefficientTable const& table, char const* pLabel )
{
    bool passed = true;

    uint32_t const resolution = 8;
    uint32_t const numSlices = TMap::NumSlices;

    uint32_t largestTapIndex = 0;
    uint32_t totalTaps = 0;
    uint32_t tapsPerAxis[TMap::NumFrames] = {};
    uint32_t numFramesSeen = 0;

    std::vector<TableTap> taps;

    for ( uint32_t slice = 0; slice < numSlices; ++slice )
    {
        for ( uint32_t texelY = 0; texelY < resolution; ++texelY )
        {
            for ( uint32_t texelX = 0; texelX < resolution; ++texelX )
            {
                BuildTableTaps< TMap >( table, 0, slice, texelX, texelY, resolution, taps );

                for ( TableTap const& tap : taps )
                {
                    if ( tap.m_tapIndex > largestTapIndex )
                    {
                        largestTapIndex = tap.m_tapIndex;
                    }

                    // supertaps are contiguous per axis, so the axis is the index
                    // divided by a quarter of the tap count, per axis
                    uint32_t const axis = ( tap.m_tapIndex / 4 ) / ( table.GetNumTaps() / 4 );

                    if ( axis < TMap::NumFrames )
                    {
                        ++tapsPerAxis[axis];
                    }

                    ++totalTaps;
                }
            }
        }
    }

    for ( uint32_t axis = 0; axis < TMap::NumFrames; ++axis )
    {
        if ( tapsPerAxis[axis] > 0 )
        {
            ++numFramesSeen;
        }
    }

    bool const layoutOk = ( largestTapIndex == ( table.GetTapCount() - 1 ) )
        && ( numFramesSeen == TMap::NumFrames )
        && ( totalTaps > 0 );

    passed = passed && layoutOk;

    std::printf
    (
        "  %-12s %2u axes x %2u taps, indices 0..%u of %u, %u of %u axes addressed : %s\n",
        pLabel, TMap::NumFrames, table.GetNumTaps(), largestTapIndex, table.GetTapCount() - 1,
        numFramesSeen, TMap::NumFrames, layoutOk ? "PASS" : "FAIL"
    );

    return passed;
}

//-------------------------------------------------------------------------

template< typename TMap >
static bool RunTapLayoutSelfCheck()
{
    bool passed = true;

    std::printf( "\n" );
    std::printf( "tap layout against the table's axis count\n" );
    std::printf( "  a table of the frame's own axis count, and the published one where it fits\n" );

    // The table over the frame's own axis count: the same taps per axis and the same coefficient count as a published one, differing only in how many axes it addresses. 
    // The values are synthetic rather than radiance, because what is being checked is the INDEX RANGE, and a zero-filled table would produce no taps at all - every sample direction would degenerate and be skipped, and the check would pass on nothing.
    CoefficientTable allAxes;
    allAxes.Create( TableShape::ForMap( ProbeMapOf< TMap >::Value, 8, 1, "const_8_axes" ) );

    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        for ( uint32_t tap = 0; tap < allAxes.GetTapCount(); ++tap )
        {
            allAxes.SetTapCoefficient( level, tap, CoefficientTable::ParameterDir0, 0, 0.1 );
            allAxes.SetTapCoefficient( level, tap, CoefficientTable::ParameterDir1, 0, 0.1 );
            allAxes.SetTapCoefficient( level, tap, CoefficientTable::ParameterDir2, 0, 1.0 );
            allAxes.SetTapCoefficient( level, tap, CoefficientTable::ParameterLevel, 0, 0.0 );
            allAxes.SetTapCoefficient( level, tap, CoefficientTable::ParameterWeight, 0, 1.0 );
        }
    }

    passed = RunTapLayoutCheck< TMap >( allAxes, "own axes" ) && passed;

    // The published tables are cubemap data, so they are three-axis tables.
    // Pairing one with a four-axis frame is not a peer shape, it is a mismatch, and the accessor's own assertion is what catches it rather than a silent read past the end: index 31 of a 24-tap table aborts.
    // Checked only where the counts agree, and said out loud where they do not.
    if constexpr ( TMap::NumFrames == CoefficientTable::NumCubeAxes )
    {
        CoefficientTable published;
        published.Load( ReferenceTable::Const8 );

        passed = RunTapLayoutCheck< TMap >( published, "published" ) && passed;
    }
    else
    {
        std::printf( "  %-12s the published tables are 3-axis cube data and are not paired with this frame\n", "published" );
    }

    std::fflush( stdout );

    return passed;
}

//-------------------------------------------------------------------------

template< typename TMap >
static bool RunMapProjectionCheck()
{
    bool passed = true;

    std::printf( "\n" );
    std::printf( "map projection\n" );
    std::printf( "  the contract every base map answers, checked generically\n" );

    constexpr uint32_t resolution = 32;

    // The texel's extent, derived from the spacing between texel centres rather than assumed.
    // A cube's face spans 2 in its own coordinates and a tetrahedron's spans 1, so a half-width of 1/resolution is right for one and twice the texel for the other.
    double centreU0 = 0.0;
    double centreV0 = 0.0;
    double centreU1 = 0.0;
    double centreV1 = 0.0;

    TMap::GetTexelCentreUV( &centreU0, &centreV0, 0, 0, resolution );
    TMap::GetTexelCentreUV( &centreU1, &centreV1, 1, 0, resolution );
    double const halfExtentU = 0.5 * std::fabs( centreU1 - centreU0 );

    TMap::GetTexelCentreUV( &centreU1, &centreV1, 0, 1, resolution );
    double const halfExtentV = 0.5 * std::fabs( centreV1 - centreV0 );

    double const expectedSolidAngle = 4.0 * std::numbers::pi_v<double>;

    // 1. The faces tile the sphere: no gap, no overlap
    double totalSolidAngle = 0.0;

    for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
    {
        for ( uint32_t texelY = 0; texelY < resolution; ++texelY )
        {
            for ( uint32_t texelX = 0; texelX < resolution; ++texelX )
            {
                double u = 0.0;
                double v = 0.0;
                TMap::GetTexelCentreUV( &u, &v, texelX, texelY, resolution );

                totalSolidAngle += TMap::GetTexelSolidAngle
                (
                    u - halfExtentU, v - halfExtentV,
                    u + halfExtentU, v + halfExtentV
                );
            }
        }
    }

    double const coverageError = std::fabs( totalSolidAngle - expectedSolidAngle ) / expectedSolidAngle;
    bool const coverageOk = coverageError < 1.0e-4;
    passed = passed && coverageOk;

    std::printf( "  %-42s : %s (relative error %.3e)\n", "solid angle over every texel sums to 4*pi", coverageOk ? "PASS" : "FAIL", coverageError );

    // 2. Texel centre to direction and back lands on the same texel
    uint32_t numRoundTripFailures = 0;
    double worstRoundTrip = 0.0;

    for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
    {
        for ( uint32_t texelY = 0; texelY < resolution; ++texelY )
        {
            for ( uint32_t texelX = 0; texelX < resolution; ++texelX )
            {
                double u = 0.0;
                double v = 0.0;
                TMap::GetTexelCentreUV( &u, &v, texelX, texelY, resolution );

                double direction[3];
                TMap::GetTexelDirection( direction, slice, u, v );

                uint32_t resolvedFace = 0;
                double resolvedU = 0.0;
                double resolvedV = 0.0;
                TMap::GetFaceAndUVFromDirection( direction, resolvedFace, resolvedU, resolvedV );

                double rebuilt[3];
                TMap::GetDirectionFromUV( rebuilt, resolvedU, resolvedV, resolvedFace );

                double const length = std::sqrt( ( direction[0] * direction[0] ) + ( direction[1] * direction[1] ) + ( direction[2] * direction[2] ) );
                double const rebuiltLength = std::sqrt( ( rebuilt[0] * rebuilt[0] ) + ( rebuilt[1] * rebuilt[1] ) + ( rebuilt[2] * rebuilt[2] ) );

                double const dot = ( ( direction[0] * rebuilt[0] ) + ( direction[1] * rebuilt[1] ) + ( direction[2] * rebuilt[2] ) ) / ( length * rebuiltLength );

                double const angularError = std::fabs( 1.0 - dot );

                // Directions only.
                // The face is not compared, because for a single-slice map the face is a consequence of the tile position rather than an argument, so comparing it asks the wrong question.
                if ( angularError > 1.0e-12 )
                {
                    ++numRoundTripFailures;
                }

                if ( angularError > worstRoundTrip )
                {
                    worstRoundTrip = angularError;
                }
            }
        }
    }

    bool const roundTripOk = ( numRoundTripFailures == 0 );
    passed = passed && roundTripOk;

    std::printf
    (
        "  %-42s : %s (%u of %u failed, worst %.3e)\n", "texel centre -> direction -> texel centre",
        roundTripOk ? "PASS" : "FAIL", numRoundTripFailures,
        TMap::NumSlices * resolution * resolution, worstRoundTrip
    );

    // 3. At least one frame stays active at every direction, or the runtime divides by a zero weight sum. 
    // Fibonacci sphere, because a grid would sample the same symmetries the rules are built from.
    uint32_t numCulled = 0;
    uint32_t minActiveFrames = TMap::NumFrames + 1;
    uint32_t maxActiveFrames = 0;

    constexpr uint32_t numDirections = 100000;
    double const goldenAngle = std::numbers::pi_v<double> *( 3.0 - std::sqrt( 5.0 ) );

    for ( uint32_t index = 0; index < numDirections; ++index )
    {
        double const y = 1.0 - ( 2.0 * static_cast<double>( index ) / static_cast<double>( numDirections - 1 ) );
        double const radius = std::sqrt( std::max( 0.0, 1.0 - ( y * y ) ) );
        double const theta = goldenAngle * static_cast<double>( index );

        double const direction[3] = { std::cos( theta ) * radius, y, std::sin( theta ) * radius };

        // The map's own coordinate vector, which is what the weights are defined against and not the direction itself
        uint32_t face = 0;
        double u = 0.0;
        double v = 0.0;
        TMap::GetFaceAndUVFromDirection( direction, face, u, v );

        double coordinate[3];
        TMap::GetDirectionFromUV( coordinate, u, v, face );

        uint32_t numActive = 0;
        for ( uint32_t frame = 0; frame < TMap::NumFrames; ++frame )
        {
            if ( TMap::GetFrameWeight( coordinate, frame ) > 0.0 )
            {
                ++numActive;
            }
        }

        if ( numActive == 0 )
        {
            ++numCulled;
        }

        if ( numActive < minActiveFrames )
        {
            minActiveFrames = numActive;
        }

        if ( numActive > maxActiveFrames )
        {
            maxActiveFrames = numActive;
        }
    }

    bool const coverageOfFramesOk = ( numCulled == 0 );
    passed = passed && coverageOfFramesOk;

    std::printf
    (
        "  %-42s : %s (active frames %u..%u of %u, %u directions culled nothing)\n",
        "at least one frame active at every direction",
        coverageOfFramesOk ? "PASS" : "FAIL",
        minActiveFrames, maxActiveFrames, TMap::NumFrames, numCulled
    );

    // 4. The solid angle function agrees with the parameterisation it came from.
    //
    // For a surface parameterised by (u,v), the solid angle element is
    //
    //        dOmega/dudv = det[ p, dp/du, dp/dv ] / |p|^3
    //
    // which needs no face normal and is therefore checkable for any map that can produce a direction from a coordinate. 
    // Integrating that element over a coordinate region must give the region's exact solid angle, and the map supplies the two sides independently: the Jacobian as a closed form built from its own parameterisation, the exact area as a sum of spherical triangles over the same region mapped to the sphere.
    //
    // This is the law the level correction is derived from, so confirming it is what lets a map's correction be read off analytically instead of fitted.
    //
    // Nothing here is finite-differenced.
    // A central difference straddles a region boundary and reads a derivative belonging to neither side, and a tetrahedral tile has a boundary through the middle of itself where two texel columns do exactly that.
    // The map is instead asked which part of the box each of its faces owns, and each part is integrated in its own chart, which is exact at the boundary rather than merely survivable.
    uint32_t const lawResolution = 8;

    double lawCentreU0 = 0.0;
    double lawCentreV0 = 0.0;
    double lawCentreU1 = 0.0;
    double lawCentreV1 = 0.0;

    TMap::GetTexelCentreUV( &lawCentreU0, &lawCentreV0, 0, 0, lawResolution );
    TMap::GetTexelCentreUV( &lawCentreU1, &lawCentreV1, 1, 0, lawResolution );
    double const lawHalfExtentU = 0.5 * std::fabs( lawCentreU1 - lawCentreU0 );

    TMap::GetTexelCentreUV( &lawCentreU1, &lawCentreV1, 0, 1, lawResolution );
    double const lawHalfExtentV = 0.5 * std::fabs( lawCentreV1 - lawCentreV0 );

    double worstLawError = 0.0;
    uint32_t numLawFailures = 0;
    uint32_t worstLawSlice = 0;
    double worstLawU = 0.0;
    double worstLawV = 0.0;
    double worstLawExact = 0.0;
    double worstLawIntegrated = 0.0;

    for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
    {
        uint32_t sliceFaces[8] = {};
        uint32_t const facesInSlice = TMap::GetSliceFaces( slice, sliceFaces );
        FF_ASSERT( facesInSlice <= 8 );

        for ( uint32_t texelY = 0; texelY < lawResolution; ++texelY )
        {
            for ( uint32_t texelX = 0; texelX < lawResolution; ++texelX )
            {
                double centreU = 0.0;
                double centreV = 0.0;
                TMap::GetTexelCentreUV( &centreU, &centreV, texelX, texelY, lawResolution );

                double const minU = centreU - lawHalfExtentU;
                double const minV = centreV - lawHalfExtentV;
                double const maxU = centreU + lawHalfExtentU;
                double const maxV = centreV + lawHalfExtentV;

                double integrated = 0.0;

                for ( uint32_t faceIndex = 0; faceIndex < facesInSlice; ++faceIndex )
                {
                    double regionU[16] = {};
                    double regionV[16] = {};

                    uint32_t const regionCount = TMap::GetFaceRegion( sliceFaces[faceIndex], minU, minV, maxU, maxV,
                                                                     regionU, regionV );

                    integrated += IntegrateJacobianOverRegion< TMap >( regionU, regionV, regionCount );
                }

                double const exact = TMap::GetTexelSolidAngle( minU, minV, maxU, maxV );
                double const magnitude = ( exact > 1.0e-12 ) ? exact : 1.0e-12;
                double const relativeError = std::fabs( integrated - exact ) / magnitude;

                if ( relativeError > 1.0e-3 )
                {
                    ++numLawFailures;
                }

                if ( relativeError > worstLawError )
                {
                    worstLawError = relativeError;
                    worstLawSlice = slice;
                    worstLawU = centreU;
                    worstLawV = centreV;
                    worstLawExact = exact;
                    worstLawIntegrated = integrated;
                }
            }
        }
    }

    bool const lawOk = ( numLawFailures == 0 );
    passed = passed && lawOk;

    std::printf
    (
        "  %-42s : %s (%u of %u failed, worst %.3e)\n",
        "solid angle agrees with the parameterisation",
        lawOk ? "PASS" : "FAIL", numLawFailures,
        TMap::NumSlices * lawResolution * lawResolution, worstLawError
    );

    if ( !lawOk )
    {
        std::printf( "      worst at slice %u, uv (%.5f, %.5f): exact %.6e, integrated %.6e\n", worstLawSlice, worstLawU, worstLawV, worstLawExact, worstLawIntegrated );
    }

    // 5. The faces of a slice tile its coordinate box, and each face's region really is the preimage of that face.
    // The second half is what couples the two ways a face can be named: the region the map reports, and the face the direction-to-face rule resolves.
    // A map whose region and whose rule disagree would put a tap on one face and read it from another.
    uint32_t numSliceRegionFailures = 0;
    double worstTileError = 0.0;

    for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
    {
        uint32_t sliceFaces[8] = {};
        uint32_t const facesInSlice = TMap::GetSliceFaces( slice, sliceFaces );

        double firstU = 0.0;
        double firstV = 0.0;
        double lastU = 0.0;
        double lastV = 0.0;

        TMap::GetTexelCentreUV( &firstU, &firstV, 0, 0, lawResolution );
        TMap::GetTexelCentreUV( &lastU, &lastV, lawResolution - 1, lawResolution - 1, lawResolution );

        double const sliceMinU = firstU - lawHalfExtentU;
        double const sliceMinV = firstV - lawHalfExtentV;
        double const sliceMaxU = lastU + lawHalfExtentU;
        double const sliceMaxV = lastV + lawHalfExtentV;

        double coveredArea = 0.0;

        for ( uint32_t faceIndex = 0; faceIndex < facesInSlice; ++faceIndex )
        {
            double regionU[16] = {};
            double regionV[16] = {};

            uint32_t const regionCount = TMap::GetFaceRegion( sliceFaces[faceIndex], sliceMinU, sliceMinV, sliceMaxU, sliceMaxV, regionU, regionV );

            double const area = PolygonArea( regionU, regionV, regionCount );
            coveredArea += area;

            if ( area <= 0.0 )
            {
                ++numSliceRegionFailures;
                continue;
            }

            double centroidU = 0.0;
            double centroidV = 0.0;
            PolygonCentroid( regionU, regionV, regionCount, centroidU, centroidV );

            double direction[3];
            TMap::GetDirectionFromUV( direction, centroidU, centroidV, sliceFaces[faceIndex] );

            uint32_t resolvedFace = 0;
            double resolvedU = 0.0;
            double resolvedV = 0.0;
            TMap::GetFaceAndUVFromDirection( direction, resolvedFace, resolvedU, resolvedV );

            if ( resolvedFace != sliceFaces[faceIndex] )
            {
                ++numSliceRegionFailures;
            }
        }

        double const boxArea = std::fabs( ( sliceMaxU - sliceMinU ) * ( sliceMaxV - sliceMinV ) );
        double const tileError = ( boxArea > 0.0 ) ? std::fabs( coveredArea - boxArea ) / boxArea : 0.0;

        if ( tileError > worstTileError )
        {
            worstTileError = tileError;
        }

        if ( tileError > 1.0e-9 )
        {
            ++numSliceRegionFailures;
        }
    }

    bool const regionsOk = ( numSliceRegionFailures == 0 );
    passed = passed && regionsOk;

    std::printf
    (
        "  %-42s : %s (%u failures, worst area error %.3e)\n",
        "a slice's faces tile it and each maps to itself",
        regionsOk ? "PASS" : "FAIL", numSliceRegionFailures, worstTileError
    );

    std::fflush( stdout );

    return passed;
}

//-------------------------------------------------------------------------

template< typename TProfile >
static bool RunRecoveryCheck( CubeReferenceFrame const& baseFrame, TProfile const& profile, EvaluationSettings const& settings )
{
    bool passed = true;

    std::printf( "\n" );
    std::printf( "recovery check\n" );
    std::printf( "  every unknown of a published const_8 level perturbed, then optimized\n" );

    CoefficientTable published;
    published.Load( ReferenceTable::Const8 );

    // The fit is scored on a sparse sample of output texels: min(gridSize, resolution) along each face axis, on all six faces.
    // At level 6 that is 24 samples against 120 unknowns - fewer samples than parameters, so the optimizer can drive the training number down by fitting the sample rather than the filter.
    // Every result is therefore also scored on a denser sample it never saw. 
    // Without that, "recovered below the published value" measures overfitting and nothing else.
    //
    // Doubling gridSize only yields an independent sample where the face resolution exceeds it.
    // At level 3 the resolution is 16, so training takes a 4x4 grid and held-out an interleaved 8x8 grid - 96 and 384 texels, disjoint.
    // At level 6 the resolution is 2, so both are capped to the same 2x2 grid and the held-out column repeats the training value.
    // The level with the fewest samples per unknown is the one level where this check cannot see overfitting.
    EvaluationSettings heldOutSettings = settings;
    heldOutSettings.m_gridSize = 2 * settings.m_gridSize;

    std::printf( "\n  %-7s %-8s %-12s %-12s %-12s %-12s %-8s\n", "level", "unknowns", "published", "perturbed", "recovered", "held-out", "result" );
    std::printf( "  %-7s %-8s %-12s %-12s %-12s %-12s %-8s\n", "", "", "train", "train", "train", "train", "" );

    // A deterministic perturbation, so a failure reproduces.
    // Alternating sign by index avoids moving every unknown the same way, which would be a scale change the fit could absorb trivially.
    constexpr double perturbationMagnitude = 0.05;

    for ( uint32_t level : { 6u, 3u } )
    {
        LevelObjective< CubeReferenceFrame, TProfile > objective;
        objective.Initialize( baseFrame, ReferenceTable::Const8, profile, settings, level );

        LevelObjective< CubeReferenceFrame, TProfile > heldOutObjective;
        heldOutObjective.Initialize( baseFrame, ReferenceTable::Const8, profile, heldOutSettings, level );

        std::vector<double> publishedParameters;
        published.GetLevelParameters( level, publishedParameters );

        double const publishedL1 = objective.Evaluate( publishedParameters );
        double const publishedHeldOutL1 = heldOutObjective.Evaluate( publishedParameters );

        std::vector<double> perturbed = publishedParameters;
        for ( size_t parameterIndex = 0; parameterIndex < perturbed.size(); ++parameterIndex )
        {
            double const magnitude = ( perturbed[parameterIndex] < 0.0 ) ? -perturbed[parameterIndex] : perturbed[parameterIndex];
            double const scale = ( magnitude > 1.0e-3 ) ? magnitude : 1.0e-3;
            double const sign = ( ( parameterIndex % 2 ) == 0 ) ? 1.0 : -1.0;

            perturbed[parameterIndex] += sign * perturbationMagnitude * scale;
        }

        double const perturbedL1 = objective.Evaluate( perturbed );

        OptimizerSettings optimizerSettings;
        optimizerSettings.m_maxIterations = ( level == 6 ) ? 1200u : 400u;
        optimizerSettings.m_memorySize = 20;
        optimizerSettings.m_verbose = false;

        OptimizerResult const result = MinimizeLBFGS( objective, perturbed, optimizerSettings );

        double const recoveredHeldOutL1 = heldOutObjective.Evaluate( result.m_parameters );

        bool const improvedTraining = ( result.m_value <= publishedL1 * ( 1.0 + 1.0e-9 ) );
        bool const improvedHeldOut = ( recoveredHeldOutL1 <= publishedHeldOutL1 * ( 1.0 + 1.0e-9 ) );
        bool const recovered = improvedTraining && improvedHeldOut;

        passed = passed && recovered;

        std::printf
        (
            "  %-7u %-8u %-12.6f %-12.6f %-12.6f %-12.6f %-8s\n",
            level, objective.GetParameterCount(), publishedL1, perturbedL1, result.m_value,
            recoveredHeldOutL1, recovered ? "ok" : "FAIL"
        );
        std::printf
        (
            "          %u texels, %u iterations, %s; held-out %u texels: %.6f -> %.6f\n",
            objective.GetNumOutputTexels(),
            result.m_iterations, GetStopReasonName( result.m_stopReason ),
            heldOutObjective.GetNumOutputTexels(), publishedHeldOutL1, recoveredHeldOutL1
        );
        std::fflush( stdout );
    }

    return passed;
}

//  Fit a whole table
//-------------------------------------------------------------------------

template< typename TFrame, typename TProfile >
static bool RunTableFitCheck
(
    TFrame const& baseFrame, TProfile const& profile, EvaluationSettings const& settings,
    TableShape const& shape, char const* pCheckpointPath, bool seedFromPublished
)
{
    bool passed = true;

    // The refinement path seeds from the published table of the same shape, and only the four shapes the paper published have one - all of them cubemap tables. 
    // Any other map is fitted from the analytic seed, and the reported comparison against a published table has no meaning for it.
    bool const hasPublishedShape = ( shape.GetPublishedShape() != TableShape::PublishedShapeNone );

    CoefficientTable seed;
    if ( hasPublishedShape )
    {
        seed.Load( static_cast<ReferenceTable>( shape.GetPublishedShape() ) );
    }

    std::printf( "\n" );
    std::printf( "table fit\n" );
    std::printf
    (
        "  %s seeded from %s, checkpoint at %s\n",
        shape.m_name,
        ( seedFromPublished && hasPublishedShape ) ? "the published table (refinement)" : "the analytic seed (generation)",
        pCheckpointPath
    );

    // The check works from whatever state the checkpoint is in, so the assertions below are stated against the invariants: every level ends up finished, a level that was refitted improved on its seed, and a level that was skipped was left alone.
    // --reset discards the checkpoint first.
    FitSettings fitSettings;
    fitSettings.m_evaluation = settings;
    fitSettings.m_shape = shape;
    fitSettings.m_pSeed = ( seedFromPublished && hasPublishedShape ) ? &seed : nullptr;
    fitSettings.m_pCheckpointPath = pCheckpointPath;
    fitSettings.m_levelLimit = CoefficientTable::NumLevels;
    fitSettings.m_optimizer.m_maxIterations = 60;   // enough to move, not to converge
    fitSettings.m_optimizer.m_memorySize = 20;
    fitSettings.m_verbose = true;

    // Baseline: the published table scored by the same objective the fit uses, so "improved" is measured on our measure rather than on theirs.
    // Built one level at a time because each objective holds a reference preimage per output texel.
    double publishedBaseline[CoefficientTable::NumLevels] = {};

    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        LevelObjective< TFrame, TProfile > objective;
        objective.Initialize( baseFrame, shape, profile, settings, level );

        std::vector<double> publishedParameters;
        seed.GetLevelParameters( level, publishedParameters );

        publishedBaseline[level] = hasPublishedShape ? objective.Evaluate( publishedParameters ) : 0.0;
    }

    // First run: finishes every level that was not already finished, checkpointing after each
    FitResult const firstRun = RunTableFit( baseFrame, profile, fitSettings );

    bool const firstComplete = firstRun.m_ok && firstRun.m_complete && ( ( firstRun.m_levelsFitted + firstRun.m_levelsSkipped ) == CoefficientTable::NumLevels );

    std::printf
    (
        "\n  first run:  %u fitted, %u skipped, complete %s : %s\n",
        firstRun.m_levelsFitted, firstRun.m_levelsSkipped,
        firstRun.m_complete ? "yes" : "no", firstComplete ? "ok" : "FAIL"
    );

    if ( firstRun.m_ok && ( firstRun.m_levelsSkipped > 0 ) )
    {
        std::printf( "  resumed an existing checkpoint; pass --reset to fit every level\n" );
    }

    passed = passed && firstComplete;

    // A refused checkpoint fits nothing, so the per-level table would be seven rows of zeros labelled "resumed" - which reads like a successful no-op rather than a failure.
    // The refusal message is printed by the driver, above.
    if ( !firstRun.m_ok )
    {
        std::printf( "\n  nothing was fitted: the checkpoint was refused. Move it aside,\n" );
        std::printf( "  or pass --reset to fit over it.\n" );
        std::fflush( stdout );

        return false;
    }

    // What this check asserts is the DRIVER, not the fit quality: every level that was fitted must improve on its own starting point, which is what says the optimizer is doing something.
    // How the result compares to the published table is reported but not asserted - with an analytic seed that comparison is a statement about the seed, and conflicting the two makes a working driver look broken.
    std::printf( "\n  %-6s %-11s %-11s %-10s %-11s %-11s %-6s\n", "level", "seed", "fitted", "improved", "published", "vs pub", "" );

    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        // A level resumed from a complete checkpoint has no seed from this run and was not asked to improve on anything, so it is reported and not asserted.
        // Whether it was left untouched is what the resume comparison below checks.
        bool const wasFitted = firstRun.m_levelFitted[level];
        double const seedValue = wasFitted ? firstRun.m_seedObjective[level] : 0.0;
        double const fitted = firstRun.m_checkpoint.m_levels[level].m_objective;
        double const baseline = publishedBaseline[level];

        bool const improved = !wasFitted || ( fitted <= ( seedValue * ( 1.0 + 1.0e-9 ) ) );
        passed = passed && improved;

        // A mirror level is built from its closed form, which is already the answer, so its starting value is exactly zero and there is no relative change to report
        char improvementText[16] = {};
        if ( !wasFitted )
        {
            std::snprintf( improvementText, sizeof( improvementText ), "resumed" );
        }
        else if ( seedValue > 0.0 )
        {
            std::snprintf( improvementText, sizeof( improvementText ), "%+.2f", 100.0 * ( fitted - seedValue ) / seedValue );
        }
        else
        {
            std::snprintf( improvementText, sizeof( improvementText ), "n/a" );
        }

        // Formatted into strings rather than printed as numbers so that a map with no published table of its shape has something honest to say in the two columns that are about one.
        // The widths agree, so a cubemap run's table reads exactly as it did when these were printed as numbers.
        char baselineText[16] = {};
        char versusPublishedText[16] = {};

        if ( hasPublishedShape )
        {
            std::snprintf( baselineText, sizeof( baselineText ), "%.6f", baseline );
            std::snprintf( versusPublishedText, sizeof( versusPublishedText ), "%+.2f", 100.0 * ( fitted - baseline ) / baseline );
        }
        else
        {
            std::snprintf( baselineText, sizeof( baselineText ), "n/a" );
            std::snprintf( versusPublishedText, sizeof( versusPublishedText ), "n/a" );
        }

        std::printf
        (
            "  %-6u %-11.6f %-11.6f %-10s %-11s %-11s %-6s\n",
            level, seedValue, fitted, improvementText,
            baselineText, versusPublishedText,
            improved ? "ok" : "FAIL"
        );
    }
    std::fflush( stdout );

    // Second run: every level is already finished, so nothing is refitted - except a mirror level, which is derived rather than fitted and is rebuilt every time.
    // A checkpoint written by an older construction must not survive into a new one.
    uint32_t numMirrors = 0;
    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        if ( profile.GetWidth( level ) <= 0.0 )
        {
            ++numMirrors;
        }
    }

    FitResult const secondRun = RunTableFit( baseFrame, profile, fitSettings );

    bool const secondSkippedAll = secondRun.m_ok
        && ( secondRun.m_levelsFitted == numMirrors )
        && ( secondRun.m_levelsSkipped == ( CoefficientTable::NumLevels - numMirrors ) );

    std::printf( "  second run: %u fitted, %u skipped (%u mirror levels are always rebuilt) : %s\n", secondRun.m_levelsFitted, secondRun.m_levelsSkipped, numMirrors, secondSkippedAll ? "ok" : "FAIL" );
    passed = passed && secondSkippedAll;

    // The resumed state has to be the state that was saved, bit for bit
    bool identical = true;
    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        if ( firstRun.m_checkpoint.m_levels[level].m_parameters != secondRun.m_checkpoint.m_levels[level].m_parameters )
        {
            identical = false;
        }
    }

    std::printf( "  resumed state identical to the saved state : %s\n", identical ? "ok" : "FAIL" );
    passed = passed && identical;

    // A checkpoint from a different fit must be refused, not mixed in
    FitSettings mismatchSettings = fitSettings;
    mismatchSettings.m_evaluation.m_gridSize = settings.m_gridSize + 4;
    mismatchSettings.m_verbose = false;

    FitResult const mismatchRun = RunTableFit( baseFrame, profile, mismatchSettings );

    bool const refused = ( !mismatchRun.m_ok ) && ( mismatchRun.m_levelsFitted == 0 );
    std::printf( "  a checkpoint from a different fit is refused : %s\n", refused ? "ok" : "FAIL" );
    passed = passed && refused;

    std::printf( "\n  the refusal message was: %s\n", mismatchRun.m_checkpoint.m_message );

    return passed;
}

// Analytic seed diagnostic
//-------------------------------------------------------------------------
// Scores the analytic initialisation against the published table at every level, without running a fit.
// The point is iteration speed: a full fit is minutes, so the seed's level offset and ring spacing cannot be tuned by fitting.
// A seed within a small factor of the published table at every level is in the right basin and worth fitting from; one that is orders of magnitude worse is not.
//
// Also prints the mips the seed samples, because a tap pattern that samples the wrong mips is wrong regardless of its directions, and that is not visible from the objective alone.

template< typename TProfile >
static bool RunSeedDiagnostic( CubeReferenceFrame const& baseFrame, TProfile const& profile, EvaluationSettings const& settings )
{
    bool passed = true;

    std::printf( "\n" );
    std::printf( "analytic seed against the published table\n" );
    std::printf( "  const_8, no fit, so the seed can be tuned without waiting for one\n" );

    CoefficientTable published;
    published.Load( ReferenceTable::Const8 );

    // Two free constants, swept together because they interact: the offset decides which mips the taps read, the outer ring weight decides whether the footprint is a ring or a difference of Gaussians.
    // The second is the one the published tables' negative weights point at.
    double const sweepOffsets[] = { -1.0, 0.0, 1.0, 2.0, 3.0, 4.0 };
    constexpr uint32_t numSweepOffsets = 6;

    double const sweepOuterWeights[] = { -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0 };
    constexpr uint32_t numSweepOuterWeights = 7;

    std::printf( "\n  %-6s %-11s %-11s %-9s %-8s %-8s %-11s\n", "level", "published", "best seed", "ratio", "offset", "outerW", "seed mips" );

    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        LevelObjective< CubeReferenceFrame, TProfile > objective;
        objective.Initialize( baseFrame, ReferenceTable::Const8, profile, settings, level );

        std::vector<double> publishedParameters;
        published.GetLevelParameters( level, publishedParameters );

        double const publishedL1 = objective.Evaluate( publishedParameters );

        double bestL1 = 1.0e30;
        double bestOffset = 0.0;
        double bestOuterWeight = 0.0;
        double bestLowMip = 0.0;
        double bestHighMip = 0.0;

        std::vector<double> seedParameters;

        for ( uint32_t offsetIndex = 0; offsetIndex < numSweepOffsets; ++offsetIndex )
        {
            for ( uint32_t weightIndex = 0; weightIndex < numSweepOuterWeights; ++weightIndex )
            {
                AnalyticSeedSettings seedSettings;
                seedSettings.m_levelOffset = sweepOffsets[offsetIndex];
                seedSettings.m_outerRingWeightScale = sweepOuterWeights[weightIndex];

                BuildAnalyticSeed( ReferenceTable::Const8, profile, level, settings.m_baseResolution, settings.m_sampleLevelCount, seedSettings, seedParameters );

                double const seedL1 = objective.Evaluate( seedParameters );

                if ( seedL1 < bestL1 )
                {
                    bestL1 = seedL1;
                    bestOffset = seedSettings.m_levelOffset;
                    bestOuterWeight = seedSettings.m_outerRingWeightScale;

                    bestLowMip = 1.0e30;
                    bestHighMip = -1.0e30;

                    uint32_t const activeCoefficients = published.GetActiveCoefficientCount();

                    for ( uint32_t tap = 0; tap < published.GetTapCount(); ++tap )
                    {
                        size_t const levelOffset = ( ( static_cast<size_t>( tap ) * CoefficientTable::NumParameters ) + CoefficientTable::ParameterLevel ) * activeCoefficients;

                        double const tapLevel = seedParameters[levelOffset];

                        bestLowMip = ( tapLevel < bestLowMip ) ? tapLevel : bestLowMip;
                        bestHighMip = ( tapLevel > bestHighMip ) ? tapLevel : bestHighMip;
                    }
                }
            }
        }

        double const ratio = ( publishedL1 > 0.0 ) ? ( bestL1 / publishedL1 ) : 0.0;

        // A seed an order of magnitude worse than the published table is not in a basin anything will climb out of in a few hundred iterations.
        bool const usable = ( ratio < 10.0 );
        passed = passed && usable;

        std::printf
        (
            "  %-6u %-11.6f %-11.6f %-9.2f %-8.1f %-8.2f %-.1f..%.1f %s\n",
            level, publishedL1, bestL1, ratio, bestOffset, bestOuterWeight,
            bestLowMip, bestHighMip, usable ? "" : " - FAIL"
        );
        std::fflush( stdout );
    }

    // The radius floor failing says the seed's problem is not where the taps sit but how they are weighted.
    // Dump the published table's own tap structure so the next attempt is designed from evidence: radius as a multiple of alpha, the mip it reads, and the weight relative to the largest tap.
    std::printf( "\n  published tap structure, per axis, ordered by radius from the axis\n" );

    for ( uint32_t level : { 0u, 3u, 6u } )
    {
        double const alpha = profile.GetWidth( level );

        std::printf( "\n    const_8 level %u, alpha %.6f\n", level, alpha );
        std::printf( "    %-8s %-10s %-10s %-10s %-10s\n", "tap", "radius/a", "radius", "mip", "weight" );

        uint32_t const numSuperTaps = published.GetNumTaps() / 4;

        // Axis 0 only: every axis has the same structure, just rotated
        double largestWeight = 0.0;

        for ( uint32_t tap = 0; tap < published.GetNumTaps(); ++tap )
        {
            uint32_t const index = ( numSuperTaps * 0 ) + ( tap / 4 );
            uint32_t const subTap = tap % 4;

            double const weight = published.GetCoefficient( level, CoefficientTable::ParameterWeight, 0, index, subTap );
            largestWeight = ( weight > largestWeight ) ? weight : largestWeight;
        }

        for ( uint32_t tap = 0; tap < published.GetNumTaps(); ++tap )
        {
            uint32_t const index = ( numSuperTaps * 0 ) + ( tap / 4 );
            uint32_t const subTap = tap % 4;

            double const dir0 = published.GetCoefficient( level, CoefficientTable::ParameterDir0, 0, index, subTap );
            double const dir1 = published.GetCoefficient( level, CoefficientTable::ParameterDir1, 0, index, subTap );
            double const dir2 = published.GetCoefficient( level, CoefficientTable::ParameterDir2, 0, index, subTap );

            double const radius = std::atan2( std::sqrt( ( dir0 * dir0 ) + ( dir1 * dir1 ) ), dir2 );
            double const mip = published.GetCoefficient( level, CoefficientTable::ParameterLevel, 0, index, subTap );
            double const weight = published.GetCoefficient( level, CoefficientTable::ParameterWeight, 0, index, subTap );

            std::printf
            (
                "    %-8u %-10.3f %-10.5f %-10.3f %-10.4f\n",
                tap, ( alpha > 0.0 ) ? ( radius / alpha ) : 0.0, radius, mip,
                ( largestWeight > 0.0 ) ? ( weight / largestWeight ) : 0.0
            );
        }
    }

    std::fflush( stdout );

    return passed;
}

// Weight-subspace headroom probe
//-------------------------------------------------------------------------
// The split rests on one narrow claim: with placements and levels fixed, the objective is convex in the weights.
// That is a statement about a subspace at a point, so it can be measured without any alternation and without a solver.
//
// Freeze every parameter except the weight coefficients and search.
// The drop that becomes available is exactly the total gain in the weight subspace at that point, with no trajectory confound: comparing joint against alternating trajectories cannot distinguish a better method from a luckier path.
//
// If the drop is negligible the stall is in the placement direction, the split cannot address it, and building the solver would be answering a question the measurements do not ask.

template< typename TProfile >
static bool RunWeightSubspaceProbe( CubeReferenceFrame const& baseFrame, TProfile const& profile, EvaluationSettings const& settings )
{
    std::printf( "\n" );
    std::printf( "weight-subspace headroom\n" );
    std::printf( "  published const_8, all parameters frozen except the weight coefficients\n" );

    CoefficientTable published;
    published.Load( ReferenceTable::Const8 );

    uint32_t const activeCoefficients = published.GetActiveCoefficientCount();

    std::printf( "\n  %-6s %-11s %-11s %-10s %-10s %-9s\n", "level", "start", "weights only", "drop", "iterations", "free" );

    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        LevelObjective< CubeReferenceFrame, TProfile > objective;
        objective.Initialize( baseFrame, ReferenceTable::Const8, profile, settings, level );

        std::vector<double> parameters;
        published.GetLevelParameters( level, parameters );

        double const startL1 = objective.Evaluate( parameters );

        // Free only parameter 4 of each tap, which is the weight
        ParameterMask mask;
        mask.SetAllFree( objective.GetParameterCount() );

        uint32_t numFree = 0;

        for ( uint32_t tap = 0; tap < published.GetTapCount(); ++tap )
        {
            for ( uint32_t parameter = 0; parameter < CoefficientTable::NumParameters; ++parameter )
            {
                for ( uint32_t coefficient = 0; coefficient < activeCoefficients; ++coefficient )
                {
                    size_t const index = ( ( static_cast<size_t>( tap ) * CoefficientTable::NumParameters ) + parameter ) * activeCoefficients + coefficient;

                    bool const isWeight = ( parameter == CoefficientTable::ParameterWeight );

                    mask.m_free[index] = isWeight ? 1 : 0;

                    if ( isWeight )
                    {
                        ++numFree;
                    }
                }
            }
        }

        OptimizerSettings optimizerSettings;
        optimizerSettings.m_maxIterations = 300;
        optimizerSettings.m_memorySize = 20;
        optimizerSettings.m_pMask = &mask;

        OptimizerResult const result = MinimizeLBFGS( objective, parameters, optimizerSettings );

        double const drop = 100.0 * ( result.m_value - startL1 ) / startL1;

        std::printf( "  %-6u %-11.6f %-11.6f %+-10.3f %-10u %-9u\n",
                     level, startL1, result.m_value, drop, result.m_iterations, numFree );
        std::fflush( stdout );
    }

    std::printf( "\n  A drop near zero means the weight direction is already optimal at the\n" );
    std::printf( "  published table and the split has nothing to recover there.\n" );

    return true;
}

// Weight-gradient check
//-------------------------------------------------------------------------
// THE FAILURE MODE THIS GUARDS
//
// A parameter-layout mismatch produces a gradient of entirely plausible magnitude pointing the wrong way.
// It does not crash, it does not stall visibly, and it surfaces days later as "the optimizer plateaus and nobody knows why".
// 
// Two such mismatches are possible here:
//
//    * taps addressed by position in the tap list rather than by parameter index, which is wrong because inactive axial frames are skipped - 16 taps for const_8, not 24;
//    * a frame weight carried on the normaliser term but not on the field term, which is invisible while every active frame has a weight of exactly 1 - true at the coarse levels, where the grid clamps and max(other) is 1.
//
//  WHY PER-COEFFICIENT, AND WHY ONE-SIDED
//
// Checking the gradient as a single descent direction is cheaper but nearly blind: it computes g.(g+e) = |g|^2 + g.e, so any error orthogonal to g vanishes.
// Measured, levels 5 and 6 scored a perfect 1.0000 on that test while level 0's every component was wrong.
//
// A one-sided difference along one coordinate is exact here.
// For a piecewise-linear F the one-sided derivative along e_k IS the subgradient component, so a correct gradient gives a ratio of exactly 1.
// F is an aggregate of 9.4M O(1) terms cancelling to 1e-3, and a difference of F never assembles that sum, so it does not suffer the cancellation that makes the gradient components individually ill-conditioned to compute.
//
// THE STEP is a compromise, and both ends fail. 
// 
// Too large and the difference crosses some of the 9.4M kinks, which at 1e-6 showed up as a worst error of 6e-3; too small and the difference falls into the objective's own rounding.
//  At 1e-8 every level lands in a uniform 1e-5..1e-4 band, which is the rounding floor and leaves an order of magnitude of margin against the tolerance.

template< typename TProfile >
static bool RunWeightGradientCheck( CubeReferenceFrame const& baseFrame, TProfile const& profile, EvaluationSettings const& settings )
{
    bool passed = true;

    std::printf( "\n" );
    std::printf( "weight-gradient check\n" );
    std::printf( "  analytic against a one-sided difference, per weight coefficient\n" );

    CoefficientTable published;
    published.Load( ReferenceTable::Const8 );

    uint32_t const activeCoefficients = published.GetActiveCoefficientCount();
    uint32_t const numSuperTaps = published.GetNumTaps() / 4;

    constexpr double step = 1.0e-8;

    std::printf( "\n  %-6s %-14s %-8s %-8s %-6s\n", "level", "worst error", "tap", "axis", "" );

    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        LevelObjective< CubeReferenceFrame, TProfile > objective;
        objective.Initialize( baseFrame, ReferenceTable::Const8, profile, settings, level );

        std::vector<double> parameters;
        published.GetLevelParameters( level, parameters );

        std::vector<double> gradient;
        double const value = objective.EvaluateWithWeightGradient( parameters, gradient );

        double   worstError = 0.0;
        uint32_t worstTap = 0;
        uint32_t worstAxis = 0;

        for ( uint32_t tap = 0; tap < published.GetTapCount(); ++tap )
        {
            size_t const index = ( ( static_cast<size_t>( tap ) * CoefficientTable::NumParameters ) + CoefficientTable::ParameterWeight ) * activeCoefficients;

            std::vector<double> probe = parameters;
            probe[index] -= step;

            double const moved = objective.Evaluate( probe );
            double const oneSided = -( moved - value ) / step;
            double const analytic = gradient[index];

            // Relative, with a floor so a near-zero component cannot manufacture a large ratio out of nothing
            double const scale = ( std::fabs( analytic ) > 1.0e-6 ) ? std::fabs( analytic ) : 1.0e-6;
            double const error = std::fabs( oneSided - analytic ) / scale;

            if ( error > worstError )
            {
                worstError = error;
                worstTap = tap;
                worstAxis = ( tap / 4 ) / ( ( numSuperTaps > 0 ) ? numSuperTaps : 1 );
            }
        }

        // A correct gradient gives exactly 1 per coefficient; the tolerance allows only for the finite step and for the objective's own rounding.
        bool const agrees = ( worstError < 1.0e-3 );
        passed = passed && agrees;

        std::printf( "  %-6u %-14.3e %-8u %-8u %-6s\n", level, worstError, worstTap, worstAxis, agrees ? "ok" : "FAIL" );
        std::fflush( stdout );
    }

    return passed;
}

// Split optimizer against joint
//-------------------------------------------------------------------------
// The split, tested rather than asserted: solve the weight subproblem with the normaliser frozen, which makes it convex, then refresh the normaliser and let placement and level move, and alternate.
//
// Both arms start from the same seed and get the same iteration budget.
// The joint arm is the current optimizer; the split arm alternates a frozen-normaliser weight stage, solved with the analytic gradient, against a placement stage on everything else.
//
// Comparing final objectives alone cannot separate "better method" from "luckier path", which is why the headroom probe did the attribution first.
// What this answers is whether the alternating scheme actually REALISES that headroom.

namespace
{
    using namespace FilterFitter;
}

template< typename TProfile >
static bool RunSplitOptimizerComparison( CubeReferenceFrame const& baseFrame, TProfile const& profile, EvaluationSettings const& settings )
{
    bool passed = true;

    std::printf( "\n" );
    std::printf( "split optimizer against joint\n" );
    std::printf( "  frozen-normaliser weight stage, alternating with placement and level\n" );

    CoefficientTable published;
    published.Load( ReferenceTable::Const8 );

    uint32_t const activeCoefficients = published.GetActiveCoefficientCount();

    ParameterMask weightMask;
    ParameterMask placementMask;

    BuildParameterMask( published.GetTapCount(), activeCoefficients, true, weightMask );
    BuildParameterMask( published.GetTapCount(), activeCoefficients, false, placementMask );

    constexpr uint32_t jointIterations = 60;
    constexpr uint32_t outerRounds = 4;
    constexpr uint32_t weightIterations = 40;
    constexpr uint32_t placementIterations = 15;

    std::printf( "\n  %-6s %-13s %-13s %-13s %-11s %-11s\n",
                 "level", "joint", "alternating", "joint+polish", "vs joint", "vs joint" );

    for ( uint32_t level : { 3u, 4u, 5u, 6u } )
    {
        LevelObjective< CubeReferenceFrame, TProfile > objective;
        objective.Initialize( baseFrame, ReferenceTable::Const8, profile, settings, level );

        std::vector<double> seed;
        published.GetLevelParameters( level, seed );

        // Arm A: joint, the current optimizer
        OptimizerSettings jointSettings;
        jointSettings.m_maxIterations = jointIterations;
        jointSettings.m_memorySize = 20;

        OptimizerResult const joint = MinimizeLBFGS( objective, seed, jointSettings );

        // Arm B: alternate a convex weight stage against placement and level
        std::vector<double> splitParameters = seed;

        // Arm C: joint first, then the same weight stage as a polish.
        // The weight stage needs an analytic gradient rather than a finite-difference one, so polishing is fast and does not compete with placement for iterations.
        std::vector<double> polishedParameters = joint.m_parameters;

        std::vector<double> normalisers;

        OptimizerSettings weightSettings;
        weightSettings.m_maxIterations = weightIterations;
        weightSettings.m_memorySize = 20;
        weightSettings.m_pMask = &weightMask;

        OptimizerSettings placementSettings;
        placementSettings.m_maxIterations = placementIterations;
        placementSettings.m_memorySize = 20;
        placementSettings.m_pMask = &placementMask;

        for ( uint32_t round = 0; round < outerRounds; ++round )
        {
            // Freeze the normaliser from the current parameters, making the weight subproblem affine inside the absolute values and therefore convex
            objective.CaptureNormalisers( splitParameters, normalisers );

            FrozenNormaliserObjective< CubeReferenceFrame, TProfile > frozenSplit;
            frozenSplit.m_pObjective = &objective;
            frozenSplit.m_pNormalisers = &normalisers;

            splitParameters = MinimizeLBFGS( frozenSplit, splitParameters, weightSettings ).m_parameters;
            splitParameters = MinimizeLBFGS( objective, splitParameters, placementSettings ).m_parameters;

            // The polish arm runs only the weight stage, from the joint result
            objective.CaptureNormalisers( polishedParameters, normalisers );

            FrozenNormaliserObjective< CubeReferenceFrame, TProfile > frozenPolish;
            frozenPolish.m_pObjective = &objective;
            frozenPolish.m_pNormalisers = &normalisers;

            polishedParameters = MinimizeLBFGS( frozenPolish, polishedParameters, weightSettings ).m_parameters;
        }

        double const jointValue = objective.Evaluate( joint.m_parameters );
        double const splitValue = objective.Evaluate( splitParameters );
        double const polishedValue = objective.Evaluate( polishedParameters );

        double const splitChange = 100.0 * ( splitValue - jointValue ) / jointValue;
        double const polishedChange = 100.0 * ( polishedValue - jointValue ) / jointValue;

        bool const polishHelps = ( polishedValue <= jointValue * ( 1.0 + 1.0e-9 ) );
        passed = passed && polishHelps;

        std::printf
        (
            "  %-6u %-13.6f %-13.6f %-13.6f %+-11.3f %+-11.3f %s\n",
            level, jointValue, splitValue, polishedValue,
            splitChange, polishedChange, polishHelps ? "ok" : "FAIL"
        );
        std::fflush( stdout );
    }

    std::printf( "\n  All arms start from the published table. Alternating and polishing run\n" );
    std::printf( "  the same convex weight stage; they differ in whether it competes with\n" );
    std::printf( "  placement for iterations or follows it.\n" );

    return passed;
}

// Best-scoring analytic seed for a level
//-------------------------------------------------------------------------
// Shared by the convergence probe and the sample-size cross-check, so that a run of either starts from the same point and the two are comparable.
// The seed's own score does not predict where the fit lands, so this is a way of holding the start fixed, not of picking a good one.

template< typename TFrame, typename TProfile >
static void FindBestAnalyticSeed
(
    LevelObjective< TFrame, TProfile >& objective,
    ReferenceTable shape,
    TProfile const& profile,
    uint32_t level,
    EvaluationSettings const& settings,
    std::vector<double>& seed,
    double& seedValue
)
{
    seedValue = 1.0e30;
    seed.clear();

    std::vector<double> candidate;

    for ( double const offset : { -1.0, 0.0, 1.0, 2.0, 3.0, 4.0 } )
    {
        for ( double const weight : { -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0 } )
        {
            AnalyticSeedSettings seedSettings;
            seedSettings.m_levelOffset = offset;
            seedSettings.m_outerRingWeightScale = weight;

            BuildAnalyticSeed( shape, profile, level, settings.m_baseResolution, settings.m_sampleLevelCount, seedSettings, candidate );

            double const candidateValue = objective.Evaluate( candidate );

            if ( candidateValue < seedValue )
            {
                seedValue = candidateValue;
                seed = candidate;
            }
        }
    }
}

// Convergence probe: truncated, or actually stuck
//-------------------------------------------------------------------------
// Every fit so far has been capped at 60 joint iterations and hit the cap at every level, so "stuck in a local minimum" and "we stopped early" have never been distinguished.
// They want opposite responses - a budget, or a better search - and a final value alone cannot tell them apart.
//
// This runs a level to a large budget from its best analytic seed and reports how the largest gradient component behaves across the run.
// A gradient that keeps falling means the cap was the problem.
// One that flattens means the point is stationary, up to the smoothing that a finite-difference gradient applies to a piecewise-linear objective.

template< typename TProfile >
static bool RunConvergenceProbe( CubeReferenceFrame const& baseFrame, TProfile const& profile, EvaluationSettings const& settings )
{
    bool passed = true;

    std::printf( "\n" );
    std::printf( "convergence probe\n" );
    std::printf( "  joint search, large budget, from the best analytic seed per level\n" );

    CoefficientTable published;
    published.Load( ReferenceTable::Const8 );

    // Sized to finish inside a tool call cap rather than to be generous.
    // Measured, a level-0 run costs about 0.5 s per iteration, so 400 iterations is a few minutes per level and the whole probe is well under half an hour.
    // The trace STREAMS, so a run cut short still leaves every iteration it reached.
    constexpr uint32_t budget = 400;

    std::printf( "\n  %-6s %-12s %-12s %-8s %-12s\n", "level", "iterations", "|g| at end", "value", "verdict" );

    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        LevelObjective< CubeReferenceFrame, TProfile > objective;
        objective.Initialize( baseFrame, ReferenceTable::Const8, profile, settings, level );

        // Best-scoring analytic seed, which is the start the driver would use
        double seedValue = 0.0;
        std::vector<double> seed;

        FindBestAnalyticSeed( objective, ReferenceTable::Const8, profile, level, settings, seed, seedValue );

        OptimizerSettings optimizerSettings;
        optimizerSettings.m_maxIterations = budget;
        optimizerSettings.m_memorySize = 20;

        // Streams every iteration to stdout, flushed.
        // A run that is cut short then still leaves the whole trajectory it reached, which is the only way to make a probe that can outlive its budget useful.
        optimizerSettings.m_verbose = true;

        OptimizerResult const result = MinimizeLBFGS( objective, seed, optimizerSettings );

        // The optimizer reports which rule stopped it, so the verdict is read rather than inferred.
        // Inferring it from the gradient norm could not separate an exhausted line search from an exhausted iteration budget, and that difference decides whether a level wants more budget or a different start.
        std::printf
        (
            "  %-6u %-12u %-12.3e %-8.6f %-12s\n",
            level, result.m_iterations, result.m_gradientNorm, result.m_value,
            GetStopReasonName( result.m_stopReason )
        );

        std::fflush( stdout );
    }

    std::printf( "\n  Each level reports its stop reason directly. The case worth reading is a\n" );
    std::printf( "  value plateau beside a gradient norm that has not shrunk: the objective\n" );
    std::printf( "  stopped improving while the subgradient did not, which is what a sum of\n" );
    std::printf( "  absolute values does at a kink.\n" );
    std::printf( "\n" );
    std::printf( "  A gradient norm of order 1e2 or larger is the central difference straddling\n" );
    std::printf( "  a kink, where it divides by the difference step, so it measures the noise in\n" );
    std::printf( "  the estimate rather than any distance from the optimum. Read the levels whose\n" );
    std::printf( "  norm stays small.\n" );

    return passed;
}

// Sample-size cross-check: a gain in the filter, or in the sample
//-------------------------------------------------------------------------
// The fit scores a level on min(gridSize, resolution) texels per face axis, which at grid 4 is 96 for levels 0 to 5, against 120 unknowns for const_8.
// The optimizer can therefore drive the training number down by fitting the sample rather than the filter, and nothing so far separates the two.
//
// This trains at the configured grid and re-scores the result at twice the grid, with the published coefficients alongside as a reference that was never trained on any of our samples.
// Both tables are scored on the same sample, so the fitted-to-published ratio is the measurement: if it holds as the sample grows, the gain is in the filter, and if it collapses toward one, the gain was in the sample.
//
// Doubling the grid is NOT a disjoint test.
// At level 4 the face is 8 texels, so grid 4 samples texels 1, 3, 5, 7 along each axis and grid 8 samples all eight - three quarters of the denser sample is new, and the repeated quarter is why this is a weaker test than the recovery check's interleaved held-out set.

template< typename TProfile >
static bool RunSampleSizeCheck( CubeReferenceFrame const& baseFrame, TProfile const& profile, EvaluationSettings const& settings )
{
    bool passed = true;

    uint32_t const denseGridSize = 2 * settings.m_gridSize;

    std::printf( "\n" );
    std::printf( "sample-size cross-check\n" );
    std::printf( "  trained at grid %u and at grid %u, both tables scored on both samples\n",
                 settings.m_gridSize, denseGridSize );

    CoefficientTable published;
    published.Load( ReferenceTable::Const8 );

    // The driver's budget is 60, which is enough to move and not to converge.
    // That would confound overfitting with an unfinished search, so this runs the same budget as the convergence probe.
    constexpr uint32_t budget = 400;

    std::printf( "\n  %-6s %-6s %-12s %-12s %-12s %-10s\n", "level", "grid", "published", "train@g4", "train@g8", "g8 vs g4" );

    for ( uint32_t level : { 3u, 4u } )
    {
        LevelObjective< CubeReferenceFrame, TProfile > fineObjective;
        fineObjective.Initialize( baseFrame, ReferenceTable::Const8, profile, settings, level );

        EvaluationSettings denseSettings = settings;
        denseSettings.m_gridSize = denseGridSize;

        LevelObjective< CubeReferenceFrame, TProfile > denseObjective;
        denseObjective.Initialize( baseFrame, ReferenceTable::Const8, profile, denseSettings, level );

        OptimizerSettings optimizerSettings;
        optimizerSettings.m_maxIterations = budget;
        optimizerSettings.m_memorySize = 20;

        // Each grid trains from its own best-scoring analytic seed, because the sweep scores candidates against the objective they will be used on, so the two starts are not the same point.
        double fineSeedValue = 0.0;
        std::vector<double> fineSeed;
        FindBestAnalyticSeed( fineObjective, ReferenceTable::Const8, profile, level, settings, fineSeed, fineSeedValue );

        double denseSeedValue = 0.0;
        std::vector<double> denseSeed;
        FindBestAnalyticSeed( denseObjective, ReferenceTable::Const8, profile, level, denseSettings, denseSeed, denseSeedValue );

        OptimizerResult const fineResult = MinimizeLBFGS( fineObjective, fineSeed, optimizerSettings );
        OptimizerResult const denseResult = MinimizeLBFGS( denseObjective, denseSeed, optimizerSettings );

        std::vector<double> const& fittedFine = fineResult.m_parameters;
        std::vector<double> const& fittedDense = denseResult.m_parameters;

        std::vector<double> publishedParameters;
        published.GetLevelParameters( level, publishedParameters );

        double const finePublished = fineObjective.Evaluate( publishedParameters );
        double const fineFromFine = fineObjective.Evaluate( fittedFine );
        double const fineFromDense = fineObjective.Evaluate( fittedDense );

        double const densePublished = denseObjective.Evaluate( publishedParameters );
        double const denseFromFine = denseObjective.Evaluate( fittedFine );
        double const denseFromDense = denseObjective.Evaluate( fittedDense );

        bool const sane = ( finePublished > 0.0 ) && ( densePublished > 0.0 );

        passed = passed && sane;

        std::printf
        (
            "  %-6u %-6u %-12.6f %-12.6f %-12.6f %+.1f%%\n",
            level, settings.m_gridSize, finePublished, fineFromFine, fineFromDense,
            100.0 * ( ( fineFromDense - fineFromFine ) / fineFromFine )
        );
        std::printf
        (
            "  %-6s %-6u %-12.6f %-12.6f %-12.6f %+.1f%%\n",
            "", denseGridSize, densePublished, denseFromFine, denseFromDense,
            100.0 * ( ( denseFromDense - denseFromFine ) / denseFromFine )
        );
        std::printf
        (
            "         fitted/published %.4f at grid %u, %.4f at grid %u; sample %u -> %u texels\n",
            fineFromFine / finePublished, settings.m_gridSize,
            denseFromFine / densePublished, denseGridSize,
            fineObjective.GetNumOutputTexels(), denseObjective.GetNumOutputTexels()
        );

        // How each arm terminated.
        // Without this a table that came out worse than the published one cannot be told apart from a run that failed to converge, and the two want opposite responses.
        std::printf
        (
            "         grid %u arm: seed %.6f, %u iterations, %s\n",
            settings.m_gridSize, fineSeedValue, fineResult.m_iterations,
            GetStopReasonName( fineResult.m_stopReason )
        );
        std::printf
        (
            "         grid %u arm: seed %.6f, %u iterations, %s\n",
            denseGridSize, denseSeedValue, denseResult.m_iterations,
            GetStopReasonName( denseResult.m_stopReason )
        );

        std::fflush( stdout );
    }

    std::printf( "\n  train@g4 and train@g8 are the same fit procedure at the two sample sizes,\n" );
    std::printf( "  each scored on the row's sample. g8 vs g4 is train@g8 relative to train@g4 on\n" );
    std::printf( "  that sample, so negative means the denser training sample produced the better\n" );
    std::printf( "  table and the training grid is set too low.\n" );
    std::printf( "\n" );
    std::printf( "  The grid-4 row's fitted-to-published ratio overstates the fit, because there\n" );
    std::printf( "  the fit and the score use the same sample. The grid-8 row is the honest\n" );
    std::printf( "  comparison for a table trained at grid 4, since most of that sample is unseen.\n" );

    return passed;
}

//  Writing a finished fit out
//-------------------------------------------------------------------------

static bool WriteFittedTable
(
    char const* pCheckpointPath,
    char const* pHeaderPath,
    char const* pBinaryPath,
    FitFingerprint const& expected
)
{
    bool passed = true;

    std::printf( "\n" );
    std::printf( "table output\n" );
    std::printf( "  %s\n", pCheckpointPath );

    FitCheckpoint checkpoint;

    FitCheckpoint::LoadResult const loadResult = checkpoint.Load( pCheckpointPath, expected, FitCheckpoint::FingerprintCheck::Configuration );

    if ( loadResult != FitCheckpoint::LoadResult::Loaded )
    {
        std::printf( "  FAIL: %s\n", checkpoint.m_message );
        return false;
    }

    if ( !checkpoint.IsComplete() )
    {
        std::printf( "  FAIL: %u of %u levels fitted, and an output table has to be complete\n", checkpoint.CountDone(), CoefficientTable::NumLevels );
        return false;
    }

    // The checkpoint names its own shape rather than naming a published table, so this rebuilds whatever it recorded - a cubemap table or a tetrahedral one - and the layout check is the table's own, which is stricter: it also catches a tap count no table has.
    TableShape const shape = checkpoint.m_fingerprint.GetShape();

    CoefficientTable table;
    table.Create( shape );

    if ( !table.ValidateShape() )
    {
        std::printf( "  FAIL: %s records a shape this build cannot use\n", pCheckpointPath );
        return false;
    }

    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        table.SetLevelParameters( level, checkpoint.m_levels[level].m_parameters );
    }

    std::printf
    (
        "  %s, %u levels, %u taps per axis, %u unknowns per level\n",
        checkpoint.m_fingerprint.m_shapeName, CoefficientTable::NumLevels,
        table.GetNumTaps(), table.GetLevelParameterCount()
    );
    std::printf
    (
        "  profile %s, seed %s\n",
        checkpoint.m_fingerprint.m_profileName,
        ( checkpoint.m_fingerprint.m_seedName[0] != '\0' ) ? checkpoint.m_fingerprint.m_seedName : "analytic"
    );

    std::printf( "  level objectives:" );

    for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
    {
        std::printf( " %.6f", checkpoint.m_levels[level].m_objective );
    }

    std::printf( "\n" );

    if ( pHeaderPath != nullptr )
    {
        bool const written = WriteTableHeader( table, pHeaderPath );

        passed = passed && written;

        std::printf( "  %-4s %s\n", written ? "ok" : "FAIL", pHeaderPath );
    }

    if ( pBinaryPath != nullptr )
    {
        // The identity comes from the checkpoint's fingerprint, which is where a fit records what it was fitted for.
        // The fingerprint carries the widths as the seven alphas, so no name has to be reconstructed or guessed.
        std::vector<double> widths( checkpoint.m_fingerprint.m_alpha, checkpoint.m_fingerprint.m_alpha + CoefficientTable::NumLevels );

        LevelWidthCurve const fittedCurve = LevelWidthCurve::MakeExplicit( widths );

        bool const written = WriteTableBinary( table, checkpoint.m_fingerprint.m_profileName, fittedCurve, pBinaryPath );

        passed = passed && written;

        std::printf( "  %-4s %s\n", written ? "ok" : "FAIL", pBinaryPath );

        if ( written )
        {
            std::printf( "         profile '%s'", checkpoint.m_fingerprint.m_profileName );
            std::printf( ", widths:" );

            for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
            {
                std::printf( " %.6f", widths[level] );
            }

            std::printf( "\n" );
            std::printf( "\n  embed it with:\n" );
            std::printf( "    External\\DataEmbed\\DataEmbed.exe %s <Symbol> <OutputName>\n", pBinaryPath );
        }
    }

    return passed;
}

//  Cached transpose operator
//-------------------------------------------------------------------------

static bool RunUpsampleOperatorSelfCheck()
{
    bool passed = true;

    // The chain the taps read, which is what the operator has to cover
    constexpr uint32_t sampleLevelCount = 8;

    std::printf( "\n" );
    std::printf( "cached transpose operator\n" );
    std::printf( "  base %u, %u levels; compared against UpsampleLevel on random fields\n", g_paperBaseResolution, sampleLevelCount );

    JacobianWeighting const weightings[] = { JacobianWeighting::None, JacobianWeighting::InverseJacobian, JacobianWeighting::Jacobian };
    char const* const       names[] = { "none", "1/J (vendored code)", "J (paper)" };

    std::printf( "\n  %-22s %-14s %-14s %-14s %-22s\n", "weighting", "entries", "MB", "worst rel", "row sum" );

    for ( uint32_t weightingIndex = 0; weightingIndex < 3; ++weightingIndex )
    {
        UpsampleOperator< CubeProjection > upsampleOperator;
        upsampleOperator.Build( g_paperBaseResolution, sampleLevelCount, weightings[weightingIndex] );

        double worstRelative = 0.0;
        double minRowSum = 1.0e30;
        double maxRowSum = 0.0;

        for ( uint32_t level = 1; level < sampleLevelCount; ++level )
        {
            uint32_t const coarseResolution = g_paperBaseResolution >> level;
            uint32_t const numCoarseTexels = CubeReferenceFrame::NumFaces * coarseResolution * coarseResolution;

            std::vector<double> coarser( numCoarseTexels );
            for ( double& value : coarser )
            {
                value = NextRandom();
            }

            std::vector<double> expected;
            UpsampleLevel< CubeProjection >( coarser, coarseResolution, weightings[weightingIndex], expected );

            std::vector<double> actual;
            upsampleOperator.Apply( level, coarser, actual );

            if ( expected.size() != actual.size() )
            {
                std::printf( "       level %u: size mismatch - FAIL\n", level );
                passed = false;
                continue;
            }

            double fieldScale = 0.0;
            double worstAbs = 0.0;

            for ( size_t texelIndex = 0; texelIndex < expected.size(); ++texelIndex )
            {
                fieldScale = std::fmax( fieldScale, std::fabs( expected[texelIndex] ) );
                worstAbs = std::fmax( worstAbs, std::fabs( expected[texelIndex] - actual[texelIndex] ) );
            }

            // The two accumulate in different orders - scatter by coarser texel against gather by finer texel - so this is a rounding comparison, not a bitwise one.
            double const relative = ( fieldScale > 0.0 ) ? ( worstAbs / fieldScale ) : worstAbs;
            worstRelative = std::fmax( worstRelative, relative );

            // Row sum of the transpose, over a coarser field of all ones.
            // The transpose is not normalised, so this is below 1 and varies with the Jacobian weighting; it is reported rather than asserted.
            std::vector<double> ones( numCoarseTexels, 1.0 );
            std::vector<double> rowSums;
            upsampleOperator.Apply( level, ones, rowSums );

            for ( double const rowSum : rowSums )
            {
                minRowSum = std::fmin( minRowSum, rowSum );
                maxRowSum = std::fmax( maxRowSum, rowSum );
            }
        }

        bool const matches = worstRelative < 1.0e-13;
        passed = passed && matches;

        std::printf
        (
            "  %-22s %-14u %-14.2f %-14.3e %-.6f..%.6f %s\n",
            names[weightingIndex], upsampleOperator.GetNumEntries(),
            static_cast<double>( upsampleOperator.GetNumBytes() ) / ( 1024.0 * 1024.0 ),
            worstRelative, minRowSum, maxRowSum, matches ? "" : " - FAIL"
        );
        std::fflush( stdout );
    }

    std::printf( "\n  the operator maps 4 coarse texels to 1 finer texel's worth of rows,\n" );
    std::printf( "  so a row sum is the transpose's, not 1. Constant reproduction is a\n" );
    std::printf( "  property of DownsampleLevel and is checked in the section above.\n" );

    return passed;
}

// Optimizer cost benchmark
//-------------------------------------------------------------------------
// Splits one objective evaluation into the part that depends only on the profile and the output direction, and the part that depends on the unknowns.
//
// The first part, B(x), is fixed for the whole fit and is computed once per output texel.
// The second part - build the taps, splat them, resolve to base, compare - is computed on every evaluation, so it is what sets the fit's wall clock.
//
// Also reports how much of the base cube the approximation leaves at zero, which bounds the saving available from scoring |B - A| only where A is non-zero and treating the rest of |B| as a constant.

static void RunOptimizerBenchmark()
{
    std::printf( "\n" );
    std::printf( "=========================================================================\n" );
    std::printf( " OPTIMIZER COST BENCHMARK\n" );
    std::printf( "=========================================================================\n" );

    ProfileGGX const profile( LobeConvention::NDFCosineHemisphere );

    EvaluationSettings settings;
    settings.m_baseResolution = g_paperBaseResolution;
    settings.m_supersampleRate = g_conformanceSupersampleRate;
    settings.m_gridSize = g_conformanceGridSize;

    CubeReferenceFrame baseFrame;
    baseFrame.Initialize( settings.m_baseResolution );

    ReferenceTable const tables[] = { ReferenceTable::Const8, ReferenceTable::Quad32 };

    std::printf
    (
        "\n  base resolution %u, reference rate %u, %u x %u output texels per face\n",
        settings.m_baseResolution, settings.m_supersampleRate, settings.m_gridSize, settings.m_gridSize
    );
    std::printf
    (
        "\n  %-10s %-6s %-8s %-14s %-14s %-14s %-10s\n",
        "table", "level", "taps", "B (us/texel)", "A (us/texel)", "total", "A zero %"
    );

    // Per-level unknowns: 5 parameters per tap, and a constant table carries one basis coefficient per parameter, a quadratic carries three
    struct Projection
    {
        char const* m_pTable;
        uint32_t    m_level;
        uint32_t    m_unknowns;
        uint32_t    m_numOutputTexels;
        double      m_serialSeconds;    // whole level, all output texels, one thread
    };

    std::vector<Projection> projections;

    for ( ReferenceTable tableId : tables )
    {
        CoefficientTable table;
        table.Load( tableId );

        uint32_t const numTaps = table.GetTapCount();
        uint32_t const numCoefficientTerms = ( tableId == ReferenceTable::Quad32 ) ? CoefficientTable::NumCoefficients : 1;

        std::vector<TableTap> taps;
        PreimageAccumulator< CubeProjection > accumulator;
        accumulator.Initialize( settings.m_baseResolution, settings.m_sampleLevelCount, settings.m_weighting );

        ReferencePreimage<CubeReferenceFrame> reference;

        for ( uint32_t level : { 0u, 3u, 6u } )
        {
            uint32_t const resolution = settings.m_baseResolution >> level;
            uint32_t const faceGridSize = ( settings.m_gridSize < resolution ) ? settings.m_gridSize : resolution;
            uint32_t const numOutputTexels = CubeReferenceFrame::NumFaces * faceGridSize * faceGridSize;

            double const inverseResolution = 1.0 / static_cast<double>( resolution );

            // Collect the sample positions once; both timings replay the same set
            std::vector<double> directions;
            std::vector<uint32_t> texelCoords;
            directions.reserve( static_cast<size_t>( numOutputTexels ) * 3 );
            texelCoords.reserve( static_cast<size_t>( numOutputTexels ) * 3 );

            for ( uint32_t face = 0; face < CubeReferenceFrame::NumFaces; ++face )
            {
                for ( uint32_t gridY = 0; gridY < faceGridSize; ++gridY )
                {
                    for ( uint32_t gridX = 0; gridX < faceGridSize; ++gridX )
                    {
                        uint32_t const texelX = ( ( ( 2 * gridX ) + 1 ) * resolution ) / ( 2 * faceGridSize );
                        uint32_t const texelY = ( ( ( 2 * gridY ) + 1 ) * resolution ) / ( 2 * faceGridSize );

                        double const u = ( ( 2.0 * static_cast<double>( texelX ) ) + 1.0 ) * inverseResolution - 1.0;
                        double const v = -( ( 2.0 * static_cast<double>( texelY ) ) + 1.0 ) * inverseResolution + 1.0;

                        double dir[3];
                        GetCubeFaceDirection( dir, u, v, face );

                        directions.push_back( dir[0] );
                        directions.push_back( dir[1] );
                        directions.push_back( dir[2] );
                        texelCoords.push_back( face );
                        texelCoords.push_back( texelX );
                        texelCoords.push_back( texelY );
                    }
                }
            }

            reference.Initialize( &baseFrame, level, settings.m_supersampleRate );

            std::chrono::steady_clock::time_point const referenceStart = std::chrono::steady_clock::now();

            for ( size_t sampleIndex = 0; sampleIndex < ( directions.size() / 3 ); ++sampleIndex )
            {
                reference.Evaluate( profile, &directions[sampleIndex * 3] );
                reference.Normalize( settings.m_measure );
            }

            double const referenceSeconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - referenceStart ).count();

            // One reference evaluation, kept for the objective timing
            reference.Evaluate( profile, &directions[0] );
            reference.Normalize( settings.m_measure );

            // Warm the allocators so the first iteration does not carry their setup cost
            accumulator.Reset();
            BuildTableTaps< CubeProjection >( table, level, 0, 0, 0, settings.m_baseResolution, taps );
            for ( TableTap const& tap : taps )
            {
                accumulator.AddSample( tap.m_direction, tap.m_level, tap.m_weight );
            }
            accumulator.ResolveToBase();

            uint32_t zeroTexels = 0;
            uint32_t totalTexels = 0;

            double tapsSeconds = 0.0;
            double splitSeconds = 0.0;
            double resolveSeconds = 0.0;
            double compareSeconds = 0.0;

            for ( size_t sampleIndex = 0; sampleIndex < ( texelCoords.size() / 3 ); ++sampleIndex )
            {
                uint32_t const face = texelCoords[( sampleIndex * 3 ) + 0];
                uint32_t const texelX = texelCoords[( sampleIndex * 3 ) + 1];
                uint32_t const texelY = texelCoords[( sampleIndex * 3 ) + 2];

                std::chrono::steady_clock::time_point const tapsStart = std::chrono::steady_clock::now();
                BuildTableTaps< CubeProjection >( table, level, face, texelX, texelY, settings.m_baseResolution, taps );
                tapsSeconds += std::chrono::duration<double>( std::chrono::steady_clock::now() - tapsStart ).count();

                std::chrono::steady_clock::time_point const splitStart = std::chrono::steady_clock::now();
                accumulator.Reset();

                double totalTapWeight = 0.0;
                for ( TableTap const& tap : taps )
                {
                    accumulator.AddSample( tap.m_direction, tap.m_level, tap.m_weight );
                    totalTapWeight += tap.m_weight;
                }
                splitSeconds += std::chrono::duration<double>( std::chrono::steady_clock::now() - splitStart ).count();

                std::chrono::steady_clock::time_point const resolveStart = std::chrono::steady_clock::now();
                std::vector<double> const& approximation = accumulator.ResolveToBase();
                resolveSeconds += std::chrono::duration<double>( std::chrono::steady_clock::now() - resolveStart ).count();

                std::chrono::steady_clock::time_point const compareStart = std::chrono::steady_clock::now();
                ComparePreimages( baseFrame, reference.GetValues(), approximation, totalTapWeight, settings.m_measure );
                compareSeconds += std::chrono::duration<double>( std::chrono::steady_clock::now() - compareStart ).count();

                if ( sampleIndex == 0 )
                {
                    for ( double const coefficient : approximation )
                    {
                        if ( coefficient == 0.0 )
                        {
                            ++zeroTexels;
                        }
                        ++totalTexels;
                    }
                }
            }

            double const objectiveSeconds = tapsSeconds + splitSeconds + resolveSeconds + compareSeconds;

            double const totalSeconds = referenceSeconds + objectiveSeconds;

            std::printf
            (
                "  %-10s %-6u %-8u %-14.1f %-14.1f %-14.1f %-10.1f\n",
                table.GetName(), level, numTaps,
                ( referenceSeconds * 1.0e6 ) / static_cast<double>( numOutputTexels ),
                ( objectiveSeconds * 1.0e6 ) / static_cast<double>( numOutputTexels ),
                ( totalSeconds * 1.0e6 ) / static_cast<double>( numOutputTexels ),
                ( totalTexels > 0 ) ? ( 100.0 * static_cast<double>( zeroTexels ) / static_cast<double>( totalTexels ) ) : 0.0
            );

            double const objectiveMicros = objectiveSeconds * 1.0e6;
            std::printf
            (
                "             A breakdown: taps %.1f%%  splat %.1f%%  resolve %.1f%%  compare %.1f%%\n",
                ( objectiveMicros > 0.0 ) ? ( 100.0 * tapsSeconds * 1.0e6 / objectiveMicros ) : 0.0,
                ( objectiveMicros > 0.0 ) ? ( 100.0 * splitSeconds * 1.0e6 / objectiveMicros ) : 0.0,
                ( objectiveMicros > 0.0 ) ? ( 100.0 * resolveSeconds * 1.0e6 / objectiveMicros ) : 0.0,
                ( objectiveMicros > 0.0 ) ? ( 100.0 * compareSeconds * 1.0e6 / objectiveMicros ) : 0.0
            );
            std::fflush( stdout );

            Projection projection;
            projection.m_pTable = table.GetName();
            projection.m_level = level;
            projection.m_unknowns = numTaps * CoefficientTable::NumParameters * numCoefficientTerms;
            projection.m_numOutputTexels = numOutputTexels;
            projection.m_serialSeconds = objectiveSeconds;
            projections.push_back( projection );
        }
    }

    // Wall clock projection.
    // The reference is cached, so only the objective part repeats.
    // A finite-difference gradient costs one evaluation per unknown plus one, which is the only gradient available until the objective's analytic derivative is derived - and it is what dominates the choice of method.
    //
    // The per-texel columns above are single-threaded.
    // The projection divides by the workers the output-texel loop actually uses, because the fit parallelises over output texels and each worker runs its whole per-texel chain, resolve included, serially.
    uint32_t const maxWorkers = GetHardwareWorkerCount();

    std::printf( "\n  projection, objective only, reference cached, %u workers\n", maxWorkers );
    std::printf( "\n  %-10s %-6s %-10s %-10s %-12s %-16s %-14s\n", "table", "level", "unknowns", "texels", "eval (s)", "FD gradient (s)", "gradients/hour" );

    for ( Projection const& projection : projections )
    {
        uint32_t const workers = ( projection.m_numOutputTexels < maxWorkers ) ? projection.m_numOutputTexels : maxWorkers;

        double const evaluationSeconds = projection.m_serialSeconds / static_cast<double>( ( workers > 0 ) ? workers : 1 );
        double const gradientSeconds = evaluationSeconds * static_cast<double>( projection.m_unknowns + 1 );

        std::printf
        (
            "  %-10s %-6u %-10u %-10u %-12.5f %-16.3f %-14.1f\n",
            projection.m_pTable, projection.m_level, projection.m_unknowns,
            projection.m_numOutputTexels, evaluationSeconds, gradientSeconds,
            ( gradientSeconds > 0.0 ) ? ( 3600.0 / gradientSeconds ) : 0.0
        );
    }

    std::printf( "\n  The reference column is paid once per output texel for the whole fit;\n" );
    std::printf( "  the objective column is paid on every evaluation. Multiplying the whole\n" );
    std::printf( "  table by a larger output-texel count scales only the objective column.\n" );
    std::printf( "  The objective cost per output texel is flat across levels, so a level's\n" );
    std::printf( "  cost is its output texel count divided by the workers.\n" );
    std::printf( "\n" );
    std::fflush( stdout );
}

//-------------------------------------------------------------------------

static void PrintUsage()
{
    std::printf( "\nusage: EsotericaFilterFitter [--diagnostics] [--benchmark]\n" );
    std::printf( "\n" );
    std::printf( "  Runs the per-stage self-checks and the published-table conformance test.\n" );
    std::printf( "\n" );
    std::printf( "  --diagnostics   Also run the gap diagnostics. These attribute the conformance\n" );
    std::printf( "                  difference to the mip chain length, the reference preimage\n" );
    std::printf( "                  definition, the supersampling rate, the texel measure, the\n" );
    std::printf( "                  output texel position and the lobe convention. About ten\n" );
    std::printf( "                  minutes, and only useful when investigating that difference.\n" );
    std::printf( "\n" );
    std::printf( "  --benchmark     Also measure the per-output-texel cost of one objective\n" );
    std::printf( "                  evaluation, split into the profile-only part and the part\n" );
    std::printf( "                  that depends on the unknowns, and project fit wall clock\n" );
    std::printf( "                  from it. Seconds.\n" );
    std::printf( "\n" );
    std::printf( "  --optimizer     Also run the recovery check: perturb every unknown of\n" );
    std::printf( "                  a published const_8 level and require the optimizer to\n" );
    std::printf( "                  recover to at most the published value, on the training\n" );
    std::printf( "                  sample and on a denser held-out sample. Minutes.\n" );
    std::printf( "\n" );
    std::printf( "  --fit           Fit a whole table, one level at a time, then check that a\n" );
    std::printf( "                  second run skips every finished level and that a checkpoint\n" );
    std::printf( "                  from a different fit is refused. Resumes: levels already in\n" );
    std::printf( "                  the checkpoint are kept, so an interrupted fit is continued\n" );
    std::printf( "                  rather than repeated. Minutes on a first run, seconds after.\n" );
    std::printf( "\n" );
    std::printf( "  --reset         Discard the checkpoint before fitting, so --fit starts from\n" );
    std::printf( "                  level 0 again. Without this a fit always resumes.\n" );
    std::printf( "\n" );
    std::printf( "  HDRI RADIANCE VALIDATION\n" );
    std::printf( "  A corpus mode of its own: it measures the tables against a reference\n" );
    std::printf( "  convolution of real environments rather than against the preimage, and it\n" );
    std::printf( "  runs none of the checks above. Results are EXR, one file per slice.\n" );
    std::printf( "\n" );
    std::printf( "  --hdri-scan <root>      List the HDRIs in a dataset and what was rejected.\n" );
    std::printf( "                          Seconds.\n" );
    std::printf( "  --hdri-ingest <root>    Decode each panorama, area downsample it, project it\n" );
    std::printf( "                          to a 128 base map and cache it. Minutes for one, and\n" );
    std::printf( "                          cached afterwards, so re-running is free.\n" );
    std::printf( "  --hdri-validate <root>  Reference and table-driven convolution of every\n" );
    std::printf( "                          ingested HDRI, compared per level. The reference is\n" );
    std::printf( "                          the expensive half and is cached per sample count.\n" );
    std::printf( "  --hdri-dir <path>       Where the projected maps live. Default\n" );
    std::printf( "                          External/FilterFitter/hdri.\n" );
    std::printf( "  --hdri-limit <n>        Stop after n assets, for a first look. Stable order,\n" );
    std::printf( "                          so the same n is the same assets every run.\n" );
    std::printf( "  --hdri-samples <n>      Reference samples per output texel. Default 8192,\n" );
    std::printf( "                          which is converged; 1024 reproduces the engine's own\n" );
    std::printf( "                          realtime estimator instead.\n" );
    std::printf( "  --hdri-equirect <w>     Equirect width before projection. Default 2048.\n" );
    std::printf( "  --hdri-force            Recompute cached stages instead of reusing them.\n" );
    std::printf( "  --hdri-showcase [dir]   Write the showcase EXRs as well: one image per\n" );
    std::printf( "                          interesting environment, holding the source beside the\n" );
    std::printf( "                          brute-force result beside the table's, at every level,\n" );
    std::printf( "                          each level point upsampled so the texel grid shows.\n" );
    std::printf( "                          A cubemap gets esoterica_best, fireflies_paper and\n" );
    std::printf( "                          fireflies_esoterica; a tetrahedral map gets\n" );
    std::printf( "                          tetrahedral_esoterica. `dir` is optional: the images\n" );
    std::printf( "                          go beside the CSV, or into --hdri-dir without one.\n" );
    std::printf( "  --hdri-csv <path>       Dump every per-asset per-table per-level number, so an\n" );
    std::printf( "                          outlier claim can be checked against its value.\n" );
    std::printf( "\n" );
    std::printf( "  --fit-seeded    The same fit seeded from the published const_8 instead of\n" );
    std::printf( "                  the analytic seed: the refinement path rather than the\n" );
    std::printf( "                  generation path. Minutes.\n" );
    std::printf( "\n" );
    std::printf( "  --seed          Score the analytic initialisation against the published\n" );
    std::printf( "                  table at every level, with no fit. Seconds, and the only\n" );
    std::printf( "                  practical way to tune it.\n" );
    std::printf( "\n" );
    std::printf( "  --weight-probe  Freeze everything but the weight coefficients at the\n" );
    std::printf( "                  published table and search, to measure how much the\n" );
    std::printf( "                  weight subspace alone has left to give. Minutes.\n" );
    std::printf( "\n" );
    std::printf( "  --gradient-check  Compare the analytic weight gradient against a one-sided\n" );
    std::printf( "                  difference, per coefficient, at every level. Seconds. Also\n" );
    std::printf( "                  runs whenever the optimizer does, because a parameter-layout\n" );
    std::printf( "                  mismatch is silent and produces a descent direction of\n" );
    std::printf( "                  entirely plausible magnitude.\n" );
    std::printf( "\n" );
    std::printf( "  --irls          Compare the split optimizer against the joint one: a\n" );
    std::printf( "                  frozen-normaliser weight stage, which is convex, alternating\n" );
    std::printf( "                  with placement and level. Both arms from the same seed and\n" );
    std::printf( "                  budget. Minutes.\n" );
    std::printf( "\n" );
    std::printf( "  --converge      Run each level to a larger budget and report how the largest\n" );
    std::printf( "                  gradient component behaves, to tell a truncation apart from\n" );
    std::printf( "                  a stationary point. Streams every iteration. About 20 min.\n" );
    std::printf( "\n" );
    std::printf( "  --sample-size   Train levels 3 and 4 at the configured grid and at twice it, then\n" );
    std::printf( "                  score both tables on both samples beside the published table.\n" );
    std::printf( "                  Separates a gain in the filter from one in the training sample,\n" );
    std::printf( "                  and tests whether training denser gives a better table. About 40 min.\n" );
    std::printf( "\n" );
    std::printf( "  --write-header [path]\n" );
    std::printf( "                  Write the fitted table as a C header, shaped like the vendored\n" );
    std::printf( "                  reference tables. Needs a complete checkpoint, so run --fit\n" );
    std::printf( "                  first. With no path the name is derived from the profile and\n" );
    std::printf( "                  curve. Seconds.\n" );
    std::printf( "\n" );
    std::printf( "  --write-binary [path]\n" );
    std::printf( "                  Write the same table as a raw binary for the engine's DataEmbed\n" );
    std::printf( "                  tool: a 144-byte header then the float4 payload. Can be given\n" );
    std::printf( "                  with --write-header, and the checkpoint is read once for both.\n" );
    std::printf( "                  With no path the name is derived, as above. Seconds.\n" );
    std::printf( "\n" );
    std::printf( "  Checkpoints and derived output names carry the profile, the curve and the\n" );
    std::printf( "  map they are for:\n" );
    std::printf( "\n" );
    std::printf( "      FilterFitter_<profile>_<curve>.fit                  checkpoint, cubemap\n" );
    std::printf( "      FilterFitter_<profile>_<curve>_<projection>.fit     checkpoint, every other map\n" );
    std::printf( "      ReflectionProbeTable_<profile>_<curve>_<projection>.h/.bin   outputs\n" );
    std::printf( "\n" );
    std::printf( "  so two configurations can be fitted and kept side by side, and a table is\n" );
    std::printf( "  never picked up for the wrong map. The cubemap checkpoint is the one name\n" );
    std::printf( "  without a projection: it was written before the flag existed, and a fit is\n" );
    std::printf( "  minutes to hours. A checkpoint renamed to another configuration is refused\n" );
    std::printf( "  on load rather than written out under a name it does not have.\n" );
    std::printf( "\n" );
    std::printf( "  --profile <ggx|beckmann>\n" );
    std::printf( "                  Which NDF to fit or measure. Default ggx, which is what the\n" );
    std::printf( "                  engine's DistributionGGX implements.\n" );
    std::printf( "\n" );
    std::printf( "  --curve <paper|esoterica>\n" );
    std::printf( "                  The width at each level. paper is the reference's gloss curve at\n" );
    std::printf( "                  spec power 18, which the published tables were fitted against and\n" );
    std::printf( "                  which conformance is measured through. esoterica is roughness\n" );
    std::printf( "                  linear across the levels, which is what this engine's radiance\n" );
    std::printf( "                  chain does. Default paper, because esoterica has not been fitted\n" );
    std::printf( "                  yet; a table has to be fitted with the curve the runtime selects\n" );
    std::printf( "                  with. Overridden by --widths.\n" );
    std::printf( "\n" );
    std::printf( "  --spec-power <f>\n" );
    std::printf( "                  The paper curve's spec power. 18 reproduces the published tables.\n" );
    std::printf( "\n" );
    std::printf( "  --widths <w0,...,w6>\n" );
    std::printf( "                  One width per level, for a fork whose roughness remap is neither.\n" );
    std::printf( "\n" );
    std::printf( "  --projection <cube|tetrahedron>\n" );
    std::printf( "                  Which base map this run is for. Default cube. It selects the\n" );
    std::printf( "                  frame the fit and the checks build, the shape the fit produces,\n" );
    std::printf( "                  the name of every artifact and checkpoint, and the map the HDRI\n" );
    std::printf( "                  stages are cached and convolved under. A table belongs to one\n" );
    std::printf( "                  map, and the four published tables are cubemap data, so a\n" );
    std::printf( "                  tetrahedral run validates the table it fitted and no other.\n" );
    std::printf( "\n" );
}

//-------------------------------------------------------------------------

struct RunSelection
{
    bool m_optimizer = false;
    bool m_fit = false;
    bool m_fitSeeded = false;
    bool m_seed = false;
    bool m_weightProbe = false;
    bool m_gradientCheck = false;
    bool m_split = false;
    bool m_converge = false;
    bool m_sampleSize = false;
    bool m_reset = false;

    // One checkpoint per configuration, named for the profile and curve.
    // The fingerprint already refuses to load one written by a different fit, but a single shared filename means the second fit replaces the first, so only one configuration can exist on disk at a time.
    char const* m_pCheckpointPath = nullptr;
    char const* m_pSeededCheckpointPath = nullptr;

    char const* m_pWriteHeaderPath = nullptr;
    char const* m_pWriteBinaryPath = nullptr;
};

//-------------------------------------------------------------------------

// Comma-separated widths. Returns false on anything that is not a number, so a typo fails rather than silently producing a shorter curve.
static bool ParseWidths( char const* pText, std::vector<double>& widths )
{
    widths.clear();

    char const* pCursor = pText;

    while ( *pCursor != '\0' )
    {
        char* pEnd = nullptr;
        double const value = std::strtod( pCursor, &pEnd );

        if ( pEnd == pCursor )
        {
            return false;
        }

        widths.push_back( value );
        pCursor = pEnd;

        if ( *pCursor == ',' )
        {
            ++pCursor;
        }
        else if ( *pCursor != '\0' )
        {
            return false;
        }
    }

    return !widths.empty();
}

//-------------------------------------------------------------------------

template< typename TFrame, typename TProfile >
static bool RunSelectedChecks
(
    TFrame const& frame,
    TProfile const& profile,
    EvaluationSettings const& settings,
    RunSelection const& selection,
    TableShape const& shape
)
{
    bool passed = true;

    // The per-stage checks are built on the cube chart: they name a face, a face-local kernel or the published cube-Jacobian correction, and their expected values come from the paper's cubemap construction.
    // They are checks of that construction, so they run for a cubemap.
    // The invariants they stand for are checked for either map by the self-checks that are templated on the map and run before this (frame, recurrence, tap layout, projection law) - what is cube-only here is the reference numbers, not the generality.
    if constexpr ( std::is_same_v< TFrame, CubeReferenceFrame > )
    {
        passed = RunObjectiveCrossCheck( frame, profile, settings ) && passed;
        passed = RunMirrorLevelCheck( frame, profile, settings ) && passed;

        if ( selection.m_optimizer )
        {
            passed = RunRecoveryCheck( frame, profile, settings ) && passed;
        }
    }

    // A fit resumes by default, which is the point of checkpointing one that runs for hours.
    // --reset is the deliberate way to ask for level 0 again.
    if ( selection.m_reset )
    {
        if ( selection.m_fit && ( selection.m_pCheckpointPath != nullptr ) )
        {
            std::remove( selection.m_pCheckpointPath );
        }

        if ( selection.m_fitSeeded && ( selection.m_pSeededCheckpointPath != nullptr ) )
        {
            std::remove( selection.m_pSeededCheckpointPath );
        }
    }

    if ( selection.m_fit )
    {
        passed = RunTableFitCheck( frame, profile, settings, shape, selection.m_pCheckpointPath, false ) && passed;
    }

    if ( selection.m_fitSeeded )
    {
        passed = RunTableFitCheck( frame, profile, settings, shape, selection.m_pSeededCheckpointPath, true ) && passed;
    }

    if constexpr ( std::is_same_v< TFrame, CubeReferenceFrame > )
    {
        if ( selection.m_seed )
        {
            passed = RunSeedDiagnostic( frame, profile, settings ) && passed;
        }

        if ( selection.m_weightProbe )
        {
            passed = RunWeightSubspaceProbe( frame, profile, settings ) && passed;
        }

        // The gradient check rides with the checks that fit, because it is those runs whose gradient it validates.
        if ( selection.m_gradientCheck || selection.m_optimizer || selection.m_fit )
        {
            passed = RunWeightGradientCheck( frame, profile, settings ) && passed;
        }

        if ( selection.m_split )
        {
            passed = RunSplitOptimizerComparison( frame, profile, settings ) && passed;
        }

        if ( selection.m_converge )
        {
            passed = RunConvergenceProbe( frame, profile, settings ) && passed;
        }

        if ( selection.m_sampleSize )
        {
            passed = RunSampleSizeCheck( frame, profile, settings ) && passed;
        }
    }

    if ( ( selection.m_pWriteHeaderPath != nullptr ) || ( selection.m_pWriteBinaryPath != nullptr ) )
    {
        // The seed is deliberately left out, so a seeded fit can still be written out by a run that does not repeat the seeding flags.
        FitFingerprint const expected = FitFingerprint::Make( settings, shape, profile, "" );

        passed = WriteFittedTable( selection.m_pCheckpointPath, selection.m_pWriteHeaderPath, selection.m_pWriteBinaryPath, expected ) && passed;
    }

    return passed;
}

//-------------------------------------------------------------------------

int main( int argc, char** argv )
{
    bool runDiagnostics = false;
    bool runBenchmark = false;
    bool runOptimizer = false;
    bool runFit = false;
    bool runFitSeeded = false;
    bool runSeed = false;
    bool runWeightProbe = false;
    bool runGradientCheck = false;
    bool runSplit = false;
    bool runConverge = false;
    bool runSampleSize = false;
    bool runReset = false;

    // HDRI radiance validation. 
    // These are a mode of their own rather than another check, because they run against a dataset rather than against the tables' own reference, and because a corpus run is hours long.
    //
    // The defaults are read from HDRIRunSettings rather than repeated here, so the heap text, the struct and the behaviour cannot drift apart. 
    // The object is named rather than a temporary because the output root is taken from it as a string, which a temporary would not keep alive.
    HDRIRunSettings const hdriDefaults;

    char const* pDatasetRoot = nullptr;
    char const* pCubemapRoot = hdriDefaults.m_outputRoot.c_str();
    uint32_t    hdriMode = 0;      // 1 scan, 2 ingest, 3 validate
    uint32_t    hdriLimit = 0;
    uint32_t    hdriSamples = hdriDefaults.m_samples;
    uint32_t    hdriEquirect = hdriDefaults.m_ingest.m_equirectWidth;
    bool        hdriForce = false;
    char const* pCsvPath = nullptr;

    // Showcase EXRs: gated, because they are for looking at rather than for measuring.
    // The directory is optional and derived when it is not given.
    bool        hdriShowcase = false;
    char const* pShowcaseDirectory = nullptr;

    // Requesting an output and naming it are separate, so an omitted path can be filled in from the profile and curve once those are known.
    // Parsing cannot derive it, because the flags that determine it may come later on the command line.
    bool        writeHeaderRequested = false;
    bool        writeBinaryRequested = false;
    char const* pWriteHeaderPath = nullptr;
    char const* pWriteBinaryPath = nullptr;

    // Which NDF and which width curve. Both default to what reproduces the reference, so an unqualified run behaves as it always has.
    char const* pProfileName = "ggx";
    char const* pCurveName = "paper";
    char const* pWidthsText = nullptr;
    double      specPower = ProfileGGX::MaxSpecPower;

    // Which base map everything after this is for: the frame the checks build, the shape the fit produces, the name of every artifact and checkpoint, and the map the HDRI stages are keyed by.
    // One flag rather than four, because a run that fitted a tetrahedral table and then validated a cubemap one would be measuring the wrong file.
    ProbeMap    projection = ProbeMap::Cube;

    for ( int argumentIndex = 1; argumentIndex < argc; ++argumentIndex )
    {
        char const* pArgument = argv[argumentIndex];

        if ( ( std::strcmp( pArgument, "--diagnostics" ) == 0 ) || ( std::strcmp( pArgument, "-d" ) == 0 ) )
        {
            runDiagnostics = true;
        }
        else if ( std::strcmp( pArgument, "--benchmark" ) == 0 )
        {
            runBenchmark = true;
        }
        else if ( std::strcmp( pArgument, "--optimizer" ) == 0 )
        {
            runOptimizer = true;
        }
        else if ( std::strcmp( pArgument, "--fit" ) == 0 )
        {
            runFit = true;
        }
        else if ( std::strcmp( pArgument, "--fit-seeded" ) == 0 )
        {
            runFitSeeded = true;
        }
        else if ( std::strcmp( pArgument, "--seed" ) == 0 )
        {
            runSeed = true;
        }
        else if ( std::strcmp( pArgument, "--weight-probe" ) == 0 )
        {
            runWeightProbe = true;
        }
        else if ( std::strcmp( pArgument, "--gradient-check" ) == 0 )
        {
            runGradientCheck = true;
        }
        else if ( std::strcmp( pArgument, "--irls" ) == 0 )
        {
            runSplit = true;
        }
        else if ( std::strcmp( pArgument, "--converge" ) == 0 )
        {
            runConverge = true;
        }
        else if ( std::strcmp( pArgument, "--sample-size" ) == 0 )
        {
            runSampleSize = true;
        }
        else if ( std::strcmp( pArgument, "--reset" ) == 0 )
        {
            runReset = true;
        }
        else if ( ( std::strcmp( pArgument, "--hdri-scan" ) == 0 )
              || ( std::strcmp( pArgument, "--hdri-ingest" ) == 0 )
              || ( std::strcmp( pArgument, "--hdri-validate" ) == 0 ) )
        {
            // The dataset root is required and is the only positional thing these modes take, so a missing one is an error rather than a default
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "%s needs a dataset root\n", pArgument );
                return 1;
            }

            pDatasetRoot = argv[++argumentIndex];

            if ( std::strcmp( pArgument, "--hdri-scan" ) == 0 )
            {
                hdriMode = 1;
            }
            else if ( std::strcmp( pArgument, "--hdri-ingest" ) == 0 )
            {
                hdriMode = 2;
            }
            else
            {
                hdriMode = 3;
            }
        }
        else if ( std::strcmp( pArgument, "--hdri-dir" ) == 0 )
        {
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "--hdri-dir needs a path\n" );
                return 1;
            }

            pCubemapRoot = argv[++argumentIndex];
        }
        else if ( std::strcmp( pArgument, "--hdri-limit" ) == 0 )
        {
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "--hdri-limit needs a count\n" );
                return 1;
            }

            hdriLimit = static_cast<uint32_t>( std::strtoul( argv[++argumentIndex], nullptr, 10 ) );
        }
        else if ( std::strcmp( pArgument, "--hdri-samples" ) == 0 )
        {
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "--hdri-samples needs a count\n" );
                return 1;
            }

            hdriSamples = static_cast<uint32_t>( std::strtoul( argv[++argumentIndex], nullptr, 10 ) );
        }
        else if ( std::strcmp( pArgument, "--hdri-equirect" ) == 0 )
        {
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "--hdri-equirect needs a width\n" );
                return 1;
            }

            hdriEquirect = static_cast<uint32_t>( std::strtoul( argv[++argumentIndex], nullptr, 10 ) );
        }
        else if ( std::strcmp( pArgument, "--hdri-force" ) == 0 )
        {
            hdriForce = true;
        }
        else if ( std::strcmp( pArgument, "--hdri-csv" ) == 0 )
        {
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "--hdri-csv needs a path\n" );
                return 1;
            }

            pCsvPath = argv[++argumentIndex];
        }
        else if ( std::strcmp( pArgument, "--hdri-showcase" ) == 0 )
        {
            // The directory is optional: omitted, the images go beside the CSV, or into the output root when there is no CSV.
            // A path that starts with a dash is never taken as one, so the flag can precede another option.
            hdriShowcase = true;

            if ( ( ( argumentIndex + 1 ) < argc ) && ( argv[argumentIndex + 1][0] != '-' ) )
            {
                pShowcaseDirectory = argv[++argumentIndex];
            }
        }
        else if ( std::strcmp( pArgument, "--write-header" ) == 0 )
        {
            // The path is optional.
            // Omitted, the name is derived from the profile and curve, so a header cannot be written under a name that describes a different configuration.
            writeHeaderRequested = true;

            if ( ( ( argumentIndex + 1 ) < argc ) && ( argv[argumentIndex + 1][0] != '-' ) )
            {
                pWriteHeaderPath = argv[++argumentIndex];
            }
        }
        else if ( std::strcmp( pArgument, "--write-binary" ) == 0 )
        {
            writeBinaryRequested = true;

            if ( ( ( argumentIndex + 1 ) < argc ) && ( argv[argumentIndex + 1][0] != '-' ) )
            {
                pWriteBinaryPath = argv[++argumentIndex];
            }
        }
        else if ( std::strcmp( pArgument, "--profile" ) == 0 )
        {
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "--profile needs a name\n" );
                return 1;
            }

            pProfileName = argv[++argumentIndex];
        }
        else if ( std::strcmp( pArgument, "--curve" ) == 0 )
        {
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "--curve needs a name\n" );
                return 1;
            }

            pCurveName = argv[++argumentIndex];
        }
        else if ( std::strcmp( pArgument, "--projection" ) == 0 )
        {
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "--projection needs a map name\n" );
                return 1;
            }

            char const* pProjectionName = argv[++argumentIndex];

            if ( !TryGetProbeMapFromName( pProjectionName, projection ) )
            {
                std::printf( "unknown projection '%s'; expected cube or tetrahedron\n", pProjectionName );
                return 1;
            }
        }
        else if ( std::strcmp( pArgument, "--spec-power" ) == 0 )
        {
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "--spec-power needs a value\n" );
                return 1;
            }

            specPower = std::strtod( argv[++argumentIndex], nullptr );
        }
        else if ( std::strcmp( pArgument, "--widths" ) == 0 )
        {
            if ( ( argumentIndex + 1 ) >= argc )
            {
                std::printf( "--widths needs a comma-separated list\n" );
                return 1;
            }

            pWidthsText = argv[++argumentIndex];
        }
        else if ( ( std::strcmp( pArgument, "--help" ) == 0 ) || ( std::strcmp( pArgument, "-h" ) == 0 ) )
        {
            PrintUsage();
            return 0;
        }
        else
        {
            std::printf( "unknown argument: %s\n", pArgument );
            PrintUsage();
            return 1;
        }
    }

    // Arguments are validated before any work runs.
    // The self-checks below take a few seconds and the fits take hours, so refusing a bad --profile at the end would mean doing all of it before saying no.
    bool const isBeckmann = ( std::strcmp( pProfileName, "beckmann" ) == 0 );
    bool const isGGX = ( std::strcmp( pProfileName, "ggx" ) == 0 );

    if ( !isBeckmann && !isGGX )
    {
        std::printf( "unknown profile '%s'; expected ggx or beckmann\n", pProfileName );
        return 1;
    }

    // --reset only has meaning next to a fit, and silently ignoring it would look exactly like a checkpoint that had already been discarded
    if ( runReset && !runFit && !runFitSeeded )
    {
        std::printf( "--reset needs --fit or --fit-seeded; there is no checkpoint to discard\n" );
        return 1;
    }

    LevelWidthCurve curve;

    // A short name for this curve, for filenames.
    // The paper curve is fully determined by its spec power and esoterica by the level count, so both are readable.
    // An explicit curve has no name beyond the numbers themselves, so it gets a hash of them - opaque, but it is the difference between two explicit curves sharing a checkpoint name and not.
    char curveSlug[32] = {};

    if ( pWidthsText != nullptr )
    {
        std::vector<double> widths;

        if ( !ParseWidths( pWidthsText, widths ) || ( widths.size() != ProfileGGX::NumLevels ) )
        {
            std::printf( "--widths needs %u comma-separated values\n", ProfileGGX::NumLevels );
            return 1;
        }

        curve = LevelWidthCurve::MakeExplicit( widths );

        uint32_t hash = 2166136261u;

        for ( double const width : widths )
        {
            uint64_t bits = 0;
            std::memcpy( &bits, &width, sizeof( bits ) );

            for ( uint32_t byteIndex = 0; byteIndex < 8; ++byteIndex )
            {
                hash ^= static_cast<uint32_t>( ( bits >> ( byteIndex * 8 ) ) & 0xFFu );
                hash *= 16777619u;
            }
        }

        std::snprintf( curveSlug, sizeof( curveSlug ), "explicit_%08x", hash );
    }
    else if ( std::strcmp( pCurveName, "esoterica" ) == 0 )
    {
        curve = LevelWidthCurve::MakeEsotericaLinear( ProfileGGX::NumLevels );
        std::snprintf( curveSlug, sizeof( curveSlug ), "esoterica" );
    }
    else if ( std::strcmp( pCurveName, "paper" ) == 0 )
    {
        curve = LevelWidthCurve::MakePaperGloss( ProfileGGX::NumLevels, specPower );
        std::snprintf( curveSlug, sizeof( curveSlug ), "paper%.4g", specPower );
    }
    else
    {
        std::printf( "unknown curve '%s'; expected paper, esoterica or --widths\n", pCurveName );
        return 1;
    }

    // The fit's shape follows from the map: the published const_8 layout is a cubemap table from Reference/, and any other map gets the same tap and coefficient counts under its own axis count, which is the closest analogue and what the per-map self-checks are written against.
    // The checkpoint records the layout rather than a published enum, so this is the only place the choice is made.
    char fitShapeName[32] = {};
    std::snprintf( fitShapeName, sizeof( fitShapeName ), "%s_const_8", GetProbeMapName( projection ) );

    TableShape const fitShape = ( projection == ProbeMap::Cube )
        ? TableShape( ReferenceTable::Const8 )
        : TableShape::ForMap( projection, 8, 1, fitShapeName );

    // One checkpoint per configuration.
    // Without this a second fit with a different profile or curve replaces the first, and the fingerprint - which correctly refuses to load the wrong one - turns that into a mystery rather than a merge.
    //
    // The cubemap name carries no projection because a completed cubemap fit predates the flag, and a fit is minutes to hours: renaming it would orphan the one checkpoint on disk that this build can still read.
    // Every other map appends its projection, which is also what keeps a tetrahedral fit from being written over the cubemap one.
    char checkpointPath[160] = {};
    char seededCheckpointPath[160] = {};

    if ( projection == ProbeMap::Cube )
    {
        std::snprintf( checkpointPath, sizeof( checkpointPath ), "FilterFitter_%s_%s.fit", pProfileName, curveSlug );
        std::snprintf( seededCheckpointPath, sizeof( seededCheckpointPath ), "FilterFitter_%s_%s_seeded.fit", pProfileName, curveSlug );
    }
    else
    {
        std::snprintf( checkpointPath, sizeof( checkpointPath ), "FilterFitter_%s_%s_%s.fit", pProfileName, curveSlug, GetProbeMapName( projection ) );
        std::snprintf( seededCheckpointPath, sizeof( seededCheckpointPath ), "FilterFitter_%s_%s_%s_seeded.fit", pProfileName, curveSlug, GetProbeMapName( projection ) );
    }

    // Output names follow the same scheme, so a generated header or binary says which profile, curve and PROJECTION it is for without anyone having to remember.
    // The projection is in the name because a table is for exactly one map: a build script that picked up a cubemap table for a tetrahedral probe would otherwise have to be told, by a file that reads as if it were the right one.
    // An explicit path still wins, because a caller embedding this in a build system needs to choose.
    char derivedHeaderPath[192] = {};
    char derivedBinaryPath[192] = {};

    std::snprintf( derivedHeaderPath, sizeof( derivedHeaderPath ), "ReflectionProbeTable_%s_%s_%s.h", pProfileName, curveSlug, GetProbeMapName( projection ) );
    std::snprintf( derivedBinaryPath, sizeof( derivedBinaryPath ), "ReflectionProbeTable_%s_%s_%s.bin", pProfileName, curveSlug, GetProbeMapName( projection ) );

    char const* const headerOutputPath = ( pWriteHeaderPath != nullptr ) ? pWriteHeaderPath : derivedHeaderPath;
    char const* const binaryOutputPath = ( pWriteBinaryPath != nullptr ) ? pWriteBinaryPath : derivedBinaryPath;

    std::printf( "Esoterica FilterFitter\n" );
    std::printf( "offline prefiltered-radiance table generator\n" );

    bool passed = true;
    passed = RunMapFrameCheck< CubeProjection >( g_paperBaseResolution ) && passed;
    passed = RunMapFrameCheck< TetrahedralProjection >( g_paperBaseResolution ) && passed;
    passed = RunProfileSelfCheck() && passed;
    passed = RunReferencePreimageSelfCheck() && passed;
    passed = RunBsplineRecurrenceSelfCheck< CubeProjection >() && passed;
    passed = RunBsplineRecurrenceSelfCheck< TetrahedralProjection >() && passed;
    passed = RunUpsampleOperatorSelfCheck() && passed;
    passed = RunPreimageAccumulatorSelfCheck< CubeReferenceFrame >() && passed;
    passed = RunPreimageAccumulatorSelfCheck< MapReferenceFrame< TetrahedralProjection > >() && passed;
    passed = RunPreimageErrorSelfCheck() && passed;

    RunPublishedTableConformanceTest();

    passed = RunTableRoundTripCheck() && passed;

    passed = RunMapProjectionCheck< CubeProjection >() && passed;

    passed = RunMapProjectionCheck< TetrahedralProjection >() && passed;

    passed = RunTapLayoutSelfCheck< CubeProjection >() && passed;

    passed = RunTapLayoutSelfCheck< TetrahedralProjection >() && passed;

    passed = RunTableWriterSelfCheck() && passed;

    passed = RunShowcaseSelfCheck() && passed;

    {
        EvaluationSettings optimizerSettings;
        optimizerSettings.m_baseResolution = g_paperBaseResolution;
        optimizerSettings.m_gridSize = g_conformanceGridSize;
        optimizerSettings.m_supersampleRate = g_conformanceSupersampleRate;

        // The frame every map-generic path below builds against.
        // The cubemap one is named because the cube-only checks take it by type; the tetrahedral one is the same MapReferenceFrame over the other projection, and the two are never both used by one run.
        CubeReferenceFrame optimizerFrame;
        optimizerFrame.Initialize( optimizerSettings.m_baseResolution );

        MapReferenceFrame< TetrahedralProjection > tetrahedralFrame;
        tetrahedralFrame.Initialize( optimizerSettings.m_baseResolution );

        // Printed before anything runs, because it decides what every table produced here means. 
        // A table fitted with one curve and selected with another is a quality loss with no error attached to it.
        std::printf( "\nprofile %s, curve %s\n", pProfileName, curve.GetName() );
        std::printf( "  projection %s, %s\n", GetProbeMapName( projection ), fitShape.m_name );
        std::printf( "  widths:" );

        for ( uint32_t level = 0; level < curve.GetNumLevels(); ++level )
        {
            std::printf( " %.6f", curve.GetWidth( level ) );
        }

        std::printf( "\n" );
        std::printf( "  checkpoint %s\n", checkpointPath );
        std::printf( "  outputs    %s  %s\n", headerOutputPath, binaryOutputPath );

        // HDRI radiance validation is a mode of its own: it runs against a dataset rather than against the tables' own reference, and it reads none of the per-profile checks.
        // It dispatches here rather than earlier because it needs the curve the run selected - that is what decides which widths the reference is convolved at - and the checkpoint path derived from it.
        if ( hdriMode != 0 )
        {
            HDRIRunSettings hdriSettings;
            hdriSettings.m_datasetRoot = pDatasetRoot;
            hdriSettings.m_outputRoot = pCubemapRoot;
            hdriSettings.m_limit = hdriLimit;
            hdriSettings.m_force = hdriForce;
            hdriSettings.m_samples = hdriSamples;
            hdriSettings.m_ingest.m_equirectWidth = hdriEquirect;
            hdriSettings.m_ingest.m_baseWidth = optimizerSettings.m_baseResolution;
            hdriSettings.m_curve = curve;
            hdriSettings.m_evaluation = optimizerSettings;
            hdriSettings.m_map = projection;
            hdriSettings.m_shape = fitShape;
            hdriSettings.pCheckpointPath = checkpointPath;
            hdriSettings.pCsvPath = pCsvPath;
            hdriSettings.m_showcase = hdriShowcase;
            hdriSettings.m_showcaseDirectory = ( pShowcaseDirectory != nullptr ) ? pShowcaseDirectory : "";

            if ( hdriMode == 1 )
            {
                return RunHDRIScan( hdriSettings );
            }

            if ( hdriMode == 2 )
            {
                return RunHDRIIngest( hdriSettings );
            }

            return RunHDRIValidate( hdriSettings );
        }

        RunSelection selection;
        selection.m_optimizer = runOptimizer;
        selection.m_fit = runFit;
        selection.m_fitSeeded = runFitSeeded;
        selection.m_seed = runSeed;
        selection.m_weightProbe = runWeightProbe;
        selection.m_gradientCheck = runGradientCheck;
        selection.m_split = runSplit;
        selection.m_converge = runConverge;
        selection.m_sampleSize = runSampleSize;
        selection.m_reset = runReset;
        selection.m_pCheckpointPath = checkpointPath;
        selection.m_pSeededCheckpointPath = seededCheckpointPath;
        selection.m_pWriteHeaderPath = writeHeaderRequested ? headerOutputPath : nullptr;
        selection.m_pWriteBinaryPath = writeBinaryRequested ? binaryOutputPath : nullptr;

        // Each profile is a distinct template instantiation, which is the point of the concept-not-base-class design, and so is each map: the two together are the four instantiations below. 
        // A third profile or map adds a branch here and an explicit instantiation at the bottom of the translation units that hold the templated definitions.
        if ( projection == ProbeMap::Tetrahedron )
        {
            if ( isBeckmann )
            {
                passed = RunSelectedChecks( tetrahedralFrame, ProfileBeckmann( curve, LobeConvention::NDFCosineHemisphere ), optimizerSettings, selection, fitShape ) && passed;
            }
            else
            {
                passed = RunSelectedChecks( tetrahedralFrame, ProfileGGX( curve, LobeConvention::NDFCosineHemisphere ), optimizerSettings, selection, fitShape ) && passed;
            }
        }
        else if ( isBeckmann )
        {
            passed = RunSelectedChecks( optimizerFrame, ProfileBeckmann( curve, LobeConvention::NDFCosineHemisphere ), optimizerSettings, selection, fitShape ) && passed;
        }
        else
        {
            passed = RunSelectedChecks( optimizerFrame, ProfileGGX( curve, LobeConvention::NDFCosineHemisphere ), optimizerSettings, selection, fitShape ) && passed;
        }
    }

    if ( runDiagnostics )
    {
        RunGapDiagnostics();
    }

    if ( runBenchmark )
    {
        RunOptimizerBenchmark();
    }

    std::printf( "\n" );
    std::printf( "=========================================================\n" );
    std::printf( " OVERALL : %s\n", passed ? "PASS" : "FAIL" );
    std::printf( "=========================================================\n" );
    std::printf( "\n" );

    return passed ? 0 : 1;
}
