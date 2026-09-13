#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

#include "CoefficientTable.h"

// analytic initialisation
//-------------------------------------------------------------------------
// Places taps without reference to any published table, so the tool can fit a table shape or a profile the paper never published.
// This is what turns the driver from a refiner into a generator.
//
// The seed follows the structure the table already has.
// Tap parameters are stored in the OUTPUT TEXEL'S TANGENT FRAME - dir0/1/2 are components along frameX, frameY, frameZ, and frameZ is the output direction - so a single set of parameters is frame-relative and correct for every output texel of the level at once.
// Only the polynomial arguments (theta, phi) vary per texel, and a constant seed leaves those at zero.
//
// Taps are laid out on concentric rings about the axis:
//
//      radius   theta = alpha * ( innerScale + ringSpacing * ring )
//      azimuth  spread evenly, offset by half a step per ring so successive
//               rings do not line up radially
//
// and the sampled mip is the one whose texel angular size matches the tap's offset from the axis.
// A tap at theta represents detail at scale theta, and the mip l has texels of about 2^(l+1) / R radians, so
//
//      theta = 2^(l+1) / R     =>     l = log2( theta * R / 2 )
//
// with an additive offset to tune, clamped into the readable chain. 
// This is the only part of the seed with a free constant, and it is the part worth sweeping: a seed whose taps sample the wrong mips is in the wrong basin regardless of how good its directions are.
//
// Measured against the published const_8, the rule gives taps at mips 0 for level 0 (where the lobe is sub-texel, so a near-delta filter is right) up to 4.6-5.8 for level 6 (where the lobe is two texels wide).
// That is the shape the paper's own tables have.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    struct AnalyticSeedSettings
    {
        // Ring radii, in units of the level's lobe half-width:
        //     theta = alpha * ( innerRadiusScale + ringSpacingScale * ring )
        //
        // Measured against the published table, the seed's quality is sensitive to the ring geometry and these values are the best found.
        // Two attempts to improve it were both refuted by the sweep:
        //
        //   * A floor on the ring radius, in base texels, over { 0, 0.5, 1, 2, 3 }. Picked 0 at every level: widening a uniformly-weighted ring past the lobe makes the reconstruction broader than the target, not sharper.
        //   * A single outer-radius parameter with the rings distributed inside it, swept over { 0.75 .. 5 }. Worse at five of seven levels, because it moved the inner ring from 0.75 alpha to 0.25 of the outer radius.
        //
        // What the published tables actually do, measured over one axis:
        //
        //     level 0   radii up to 4.82 alpha, mips -0.3..0.9, weights 0.37..1.00
        //     level 3   radii up to 4.39 alpha, mips  2.3..4.3, weights 0.34..1.00
        //     level 6   radii up to 0.83 alpha, mips  5.1..6.4, weights -0.31..1.00
        //
        // So the span is not a constant multiple of alpha, the mip rule below is close (level 3 predicts 2.6..3.8 against a published 2.3..4.3), and weights are allowed to go negative - level 6 uses one.
        // The next thing to try is the radial WEIGHT profile, which is uniform here and clearly is not in the published tables.
        double m_innerRadiusScale = 0.75;
        double m_ringSpacingScale = 1.0;

        // Added to log2( theta * R / 2 ). Negative samples finer mips than the footprint match suggests.
        double m_levelOffset = -1.0;

        // Weight of the OUTERMOST ring, relative to the innermost, which is 1.
        // Rings in between interpolate.
        //
        // This exists because uniform weights cannot express the kernel. 
        // A ring of equally weighted taps reconstructs a ring, and at the fine levels the target is a spike - which is why the seed scored 0.82 against the published 0.094 at level 0 and the optimizer could not close any of it.
        //
        // Measured, the published tables do not weight uniformly either: over one axis the weights span 0.34..1.00 at level 3 and -0.31..1.00 at level 6, with a NEGATIVE weight in the outermost ring.
        // That is the signature of a difference of Gaussians - a positive core with a negative shoulder that cancels the ring's skirt - and it is the one shape a positive-only uniform seed cannot reach.
        //
        // 1.0 reproduces the old uniform behaviour.
        double m_outerRingWeightScale = 1.0;
    };

    //-------------------------------------------------------------------------

    template< typename TProfile >
    void BuildAnalyticSeed
    (
        TableShape const& shape,
        TProfile const& profile,
        uint32_t level,
        uint32_t baseResolution,
        uint32_t sampleLevelCount,
        AnalyticSeedSettings const& settings,
        std::vector<double>& parameters
    )
    {
        CoefficientTable seed;
        seed.Create( shape );

        double const levelAlpha = profile.GetWidth( level );
        uint32_t const numTapsPerAxis = seed.GetNumTaps();
        uint32_t const numSuperTaps = numTapsPerAxis / 4;

        // Four azimuths per ring
        constexpr uint32_t tapsPerRing = 4;

        double const pi = std::numbers::pi_v<double>;

        uint32_t const numRings = ( numSuperTaps > 0 ) ? numSuperTaps : 1;

        // One ring set per AXIS, and the axis count is the map's: three for a cubemap, four for a regular tetrahedron.
        // A seed built over three of a tetrahedral table's four quarters would leave the fourth at zero, which is not a starting point - it is a table with no taps on one symmetry axis.
        for ( uint32_t axis = 0; axis < seed.GetNumAxes(); ++axis )
        {
            for ( uint32_t superTap = 0; superTap < numSuperTaps; ++superTap )
            {
                for ( uint32_t subTap = 0; subTap < 4; ++subTap )
                {
                    uint32_t const tapWithinAxis = ( superTap * 4 ) + subTap;

                    uint32_t const ring = tapWithinAxis / tapsPerRing;
                    uint32_t const slot = tapWithinAxis % tapsPerRing;

                    double const radiusScale = settings.m_innerRadiusScale + ( settings.m_ringSpacingScale * static_cast<double>( ring ) );
                    double const theta = levelAlpha * radiusScale;

                    // Half a step of rotation per ring, so rings interleave rather
                    // than stack radially
                    double const phi = ( 2.0 * pi * ( static_cast<double>( slot ) + ( 0.5 * static_cast<double>( ring ) ) ) ) / static_cast<double>( tapsPerRing );

                    // The mip whose texel angular size matches the tap's offset
                    double levelValue = std::log2( ( theta * static_cast<double>( baseResolution ) ) / 2.0 ) + settings.m_levelOffset;

                    double const highestLevel = static_cast<double>( ( sampleLevelCount > 0 ) ? ( sampleLevelCount - 1 ) : 0 );

                    levelValue = ( levelValue < 0.0 ) ? 0.0 : levelValue;
                    levelValue = ( levelValue > highestLevel ) ? highestLevel : levelValue;

                    uint32_t const tapIndex = ( ( numSuperTaps * axis ) + superTap ) * 4 + subTap;

                    // Innermost ring 1, outermost m_outerRingWeightScale, linear in between.
                    // A negative outer weight gives the difference of Gaussians shape the published tables use.
                    double const ringFraction = ( numRings > 1 )
                        ? ( static_cast<double>( ring ) / static_cast<double>( numRings - 1 ) )
                        : 0.0;

                    double const ringWeight = 1.0 + ( ( settings.m_outerRingWeightScale - 1.0 ) * ringFraction );

                    seed.SetTapCoefficient( level, tapIndex, CoefficientTable::ParameterDir0, 0, std::sin( theta ) * std::cos( phi ) );
                    seed.SetTapCoefficient( level, tapIndex, CoefficientTable::ParameterDir1, 0, std::sin( theta ) * std::sin( phi ) );
                    seed.SetTapCoefficient( level, tapIndex, CoefficientTable::ParameterDir2, 0, std::cos( theta ) );
                    seed.SetTapCoefficient( level, tapIndex, CoefficientTable::ParameterLevel, 0, levelValue );
                    seed.SetTapCoefficient( level, tapIndex, CoefficientTable::ParameterWeight, 0, ringWeight );

                    // A quadratic table's theta^2 and phi^2 terms stay zero, which makes the seed's polynomial a constant.
                    // The optimiser is free to introduce curvature.
                }
            }
        }

        seed.GetLevelParameters( level, parameters );
    }

    // The mirror level
    //-------------------------------------------------------------------------
    // A width of zero is not a narrow lobe, it is the identity: the filter returns its own source texel unchanged.
    // That happens whenever a width curve starts at zero roughness, which the esoterica curve does.
    //
    // It has a closed form in the tap parameterisation and needs no search - and no search can find it.
    // The objective is piecewise constant in the tap direction: 0 when the tap lands in the output texel and 2.0 when it lands in any other, so a finite-difference gradient is zero almost everywhere and there is no direction to descend. 
    // Measured on the esoterica curve, three starts from three seeds each returned 0.899579 unchanged, bit for bit.
    //
    // frameZ is by construction the output direction, so dir0 = dir1 = 0 with dir2 = 1 puts the tap exactly on it, and every tap carries unit weight so the weight sum the runtime divides by stays positive whichever axial frame the direction selects.
    // Which frame that is varies per output texel - it is the one whose pole is roughly perpendicular to the direction, and a direction needs two of the three - so a weight on one nominated tap leaves most texels with a non-positive normaliser.
    //
    // The sampled level cannot be a plain zero, because the tap builder adds the MAP's level correction to whatever the table says, and the two maps differ:
    //
    //      cube         level = tableLevel + 0.75 * log2( 1 / major^2 )
    //
    // which is 0 at a face centre and 0.75 * log2( 3 ) at a corner.
    // Cancelling that needs log2( major ), and major is sqrt( (1 - phi^2) / (1 + theta^2) ), so no (1, theta^2, phi^2) polynomial reaches it and no fit can place this level.
    // A table level of -0.75 * log2( 3 ) instead makes the sum non-positive at every direction, so the clamp in the sampler holds the tap on source mip 0 - which is the unfiltered source, and is what a runtime does at roughness zero.
    // A negative level is not extrapolation: SampleLevel clamps it in the same place.
    //
    // A tetrahedral map's correction is -1.5 * log2( n . L ), which is 0 at a face centre and 1.5 * log2( 3 ) at a face's corner - exactly twice the cube's, because its Jacobian varies over 27:1 against the cube's 5.2:1.
    // So its mirror level is -1.5 * log2( 3 ), and the cube's constant would leave the tap a full span short of mip 0. Which value to use is the map's own answer: TMap::GetMirrorLevel.
    //
    // Every tap keeps a valid direction and level, not just the active one.
    // A tap of zero weight still has its direction normalised by the runtime and by the accumulator, and a zero vector there divides by zero.
    //-------------------------------------------------------------------------

    template< typename TMap >
    inline void BuildMirrorLevelParameters( TableShape const& shape, std::vector<double>& parameters )
    {
        CoefficientTable table;
        table.Create( shape );

        uint32_t const numTaps = table.GetTapCount();
        uint32_t const numCoefficients = table.GetActiveCoefficientCount();

        parameters.assign( static_cast<size_t>( numTaps ) * CoefficientTable::NumParameters * numCoefficients, 0.0 );

        double const mirrorLevel = TMap::GetMirrorLevel();

        for ( uint32_t tap = 0; tap < numTaps; ++tap )
        {
            size_t const dir2Index = ( ( static_cast<size_t>( tap ) * CoefficientTable::NumParameters ) + CoefficientTable::ParameterDir2 ) * numCoefficients;
            size_t const levelIndex = ( ( static_cast<size_t>( tap ) * CoefficientTable::NumParameters ) + CoefficientTable::ParameterLevel ) * numCoefficients;
            size_t const weightIndex = ( ( static_cast<size_t>( tap ) * CoefficientTable::NumParameters ) + CoefficientTable::ParameterWeight ) * numCoefficients;

            // The constant terms only; the theta^2 and phi^2 terms stay zero, so the tap sits on the axis for every output texel rather than for one
            parameters[dir2Index] = 1.0;
            parameters[levelIndex] = mirrorLevel;
            parameters[weightIndex] = 1.0;
        }
    }
}
