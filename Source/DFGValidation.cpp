#include "DFGValidation.h"

#include "Assert.h"
#include "DFGOutput.h"
#include "ParallelFor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    static void BuildDFGReferenceTable( uint32_t resolution, uint32_t sampleCount, DFGTable& table )
    {
        table.m_resolution = resolution;
        table.m_sampleCount = sampleCount;
        table.m_texels.assign( static_cast<size_t>( resolution ) * resolution, DFGTexel() );

        ParallelForRanges( resolution, [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            (void) workerIndex;

            for ( uint32_t row = begin; row < end; ++row )
            {
                double const roughness = table.GetRoughness( row );

                for ( uint32_t column = 0; column < resolution; ++column )
                {
                    table.SetTexel
                    (
                        column, row,
                        EvaluateDFGTexel( table.GetNdotV( column ), roughness, sampleCount, DFGSampleSequence::Stratified )
                    );
                }
            }
        } );
    }

    //-------------------------------------------------------------------------

    void CompareDFGTables( DFGTable const& table, DFGTable const& reference, DFGComparison& comparison )
    {
        FF_ASSERT( table.GetResolution() == reference.GetResolution() );

        uint32_t const resolution = table.GetResolution();

        comparison = DFGComparison();
        comparison.m_resolution = resolution;
        comparison.m_numTexels = resolution * resolution;

        // A reference to the array member, not a copy of it, and not const: this accumulates into each region as it walks the grid.
        DFGRegion( &regions )[DFGComparison::NumBands][DFGComparison::NumBands] = comparison.m_regions;

        double sumAbsoluteScale = 0.0;
        double sumAbsoluteBias = 0.0;
        double sumSignedScale = 0.0;
        double sumSignedBias = 0.0;

        for ( uint32_t row = 0; row < resolution; ++row )
        {
            // Both axes span zero to one, so the bands are equal counts of rows and columns rather than equal widths rounded.
            uint32_t const rowBand = ( ( row * DFGComparison::NumBands ) / resolution );

            for ( uint32_t column = 0; column < resolution; ++column )
            {
                uint32_t const columnBand = ( ( column * DFGComparison::NumBands ) / resolution );

                DFGTexel const& value = table.GetTexel( column, row );
                DFGTexel const& expected = reference.GetTexel( column, row );

                double const scaleDelta = value.m_scale - expected.m_scale;
                double const biasDelta = value.m_bias - expected.m_bias;

                double const absoluteScale = std::fabs( scaleDelta );
                double const absoluteBias = std::fabs( biasDelta );

                DFGRegion& region = regions[rowBand][columnBand];

                if ( region.m_numTexels == 0 )
                {
                    region.m_columnBegin = column;
                    region.m_rowBegin = row;
                }

                region.m_columnEnd = column + 1;
                region.m_rowEnd = row + 1;
                ++region.m_numTexels;

                region.m_meanAbsoluteScale += absoluteScale;
                region.m_meanAbsoluteBias += absoluteBias;
                region.m_signedMeanScale += scaleDelta;
                region.m_signedMeanBias += biasDelta;
                region.m_worstAbsoluteScale = ( absoluteScale > region.m_worstAbsoluteScale ) ? absoluteScale : region.m_worstAbsoluteScale;
                region.m_worstAbsoluteBias = ( absoluteBias > region.m_worstAbsoluteBias ) ? absoluteBias : region.m_worstAbsoluteBias;

                sumAbsoluteScale += absoluteScale;
                sumAbsoluteBias += absoluteBias;
                sumSignedScale += scaleDelta;
                sumSignedBias += biasDelta;

                if ( absoluteScale > comparison.m_worstAbsoluteScale )
                {
                    comparison.m_worstAbsoluteScale = absoluteScale;
                    comparison.m_worstScale.m_absolute = absoluteScale;
                    comparison.m_worstScale.m_column = column;
                    comparison.m_worstScale.m_row = row;
                    comparison.m_worstScale.m_tableScale = value.m_scale;
                    comparison.m_worstScale.m_referenceScale = expected.m_scale;
                }

                if ( absoluteBias > comparison.m_worstAbsoluteBias )
                {
                    comparison.m_worstAbsoluteBias = absoluteBias;
                    comparison.m_worstBias.m_absolute = absoluteBias;
                    comparison.m_worstBias.m_column = column;
                    comparison.m_worstBias.m_row = row;
                    comparison.m_worstBias.m_tableBias = value.m_bias;
                    comparison.m_worstBias.m_referenceBias = expected.m_bias;
                }
            }
        }

        double const inverseCount = 1.0 / static_cast<double>( comparison.m_numTexels );

        comparison.m_meanAbsoluteScale = sumAbsoluteScale * inverseCount;
        comparison.m_meanAbsoluteBias = sumAbsoluteBias * inverseCount;
        comparison.m_signedMeanScale = sumSignedScale * inverseCount;
        comparison.m_signedMeanBias = sumSignedBias * inverseCount;

        for ( uint32_t rowBand = 0; rowBand < DFGComparison::NumBands; ++rowBand )
        {
            for ( uint32_t columnBand = 0; columnBand < DFGComparison::NumBands; ++columnBand )
            {
                DFGRegion& region = regions[rowBand][columnBand];

                if ( region.m_numTexels == 0 )
                {
                    continue;
                }

                double const inverseRegionCount = 1.0 / static_cast<double>( region.m_numTexels );

                region.m_meanAbsoluteScale *= inverseRegionCount;
                region.m_meanAbsoluteBias *= inverseRegionCount;
                region.m_signedMeanScale *= inverseRegionCount;
                region.m_signedMeanBias *= inverseRegionCount;
            }
        }
    }

    //  The deterministic quadrature
    //-------------------------------------------------------------------------

    DFGTexel EvaluateDFGTexelQuadrature( double ndotV, double roughness, uint32_t numThetaCells, uint32_t numPhiCells )
    {
        DFGTexel result;

        if ( ( numThetaCells == 0 ) || ( numPhiCells == 0 ) )
        {
            return result;
        }

        double const cosV = ( ndotV < DFGIntegrand::MinCosine ) ? DFGIntegrand::MinCosine : ( ( ndotV > 1.0 ) ? 1.0 : ndotV );
        double const sinV = std::sqrt( ( 1.0 - cosV ) * ( 1.0 + cosV ) );

        // The BRDF's own parameters, longhand rather than through the estimator's expression: the width the NDF takes, its square, and the shadowing term.
        double const alpha = DFGIntegrand::EngineAlpha( roughness );
        double const alphaSquared = alpha * alpha;
        double const k = DFGIntegrand::EngineShadowingK( roughness );

        double const g1V = DFGIntegrand::GeometrySchlick( cosV, k );

        double const thetaStep = ( 0.5 * std::numbers::pi_v<double> ) / static_cast<double>( numThetaCells );
        double const phiStep = ( 2.0 * std::numbers::pi_v<double> ) / static_cast<double>( numPhiCells );

        double scale = 0.0;
        double bias = 0.0;

        for ( uint32_t thetaCell = 0; thetaCell < numThetaCells; ++thetaCell )
        {
            double const theta = ( static_cast<double>( thetaCell ) + 0.5 ) * thetaStep;
            double const cosTheta = std::cos( theta );
            double const sinTheta = std::sin( theta );

            // The solid angle of the cell, which is what is being integrated against.
            double const solidAngle = sinTheta * thetaStep * phiStep;

            for ( uint32_t phiCell = 0; phiCell < numPhiCells; ++phiCell )
            {
                double const phi = ( static_cast<double>( phiCell ) + 0.5 ) * phiStep;

                double const lightX = sinTheta * std::cos( phi );
                double const dotNL = cosTheta;

                // The half-vector of this pair. Its length is what both cosines below are divided by, and the view against it is half that length.
                double const vdotL = ( sinV * lightX ) + ( cosV * dotNL );
                double const halfLength = std::sqrt( 2.0 + ( 2.0 * vdotL ) );

                if ( halfLength <= 0.0 )
                {
                    continue;
                }

                double const vdotH = 0.5 * halfLength;
                double const dotNH = ( cosV + dotNL ) / halfLength;

                if ( ( vdotH <= 0.0 ) || ( dotNH <= 0.0 ) )
                {
                    continue;
                }

                // The NDF at the half-vector, in the form the engine's own distribution is written in, with the roughness squared into the alpha the NDF takes.
                double const denominator = ( ( dotNH * dotNH ) * ( alphaSquared - 1.0 ) ) + 1.0;
                double const distribution = alphaSquared / ( DFGIntegrand::Pi * denominator * denominator );

                double const g1L = DFGIntegrand::GeometrySchlick( dotNL, k );

                // The BRDF against the cosine, with the split's Fresnel term kept apart: D G / ( 4 ( N.V ) ), which is what is left of D G F / ( 4 ( N.V ) ( N.L ) ) once the N.L the pixel shader folds in has cancelled the one in the denominator.
                double const value = ( distribution * g1V * g1L ) / ( 4.0 * cosV );
                double const fresnel = DFGIntegrand::FresnelBase( vdotH );

                scale += ( 1.0 - fresnel ) * value * solidAngle;
                bias += fresnel * value * solidAngle;
            }
        }

        result.m_scale = scale;
        result.m_bias = bias;

        return result;
    }

    //  Reporting
    //-------------------------------------------------------------------------

    static void PrintDFGBandTable( char const* pTitle, char const* pDetail, DFGTable const& table, DFGComparison const& comparison, uint32_t referenceSampleCount )
    {
        DFGRegion const ( &regions )[DFGComparison::NumBands][DFGComparison::NumBands] = comparison.m_regions;

        std::printf( "\n" );
        std::printf( "%s\n", pTitle );
        std::printf( "  %s\n", pDetail );
        std::printf( "  %u x %u grid, %u samples per texel in the table, %u in the reference\n", comparison.m_resolution, comparison.m_resolution, table.GetSampleCount(), referenceSampleCount );
        std::printf( "\n" );
        std::printf( "  %-15s %-15s %-13s %-13s %-13s %-13s %-13s\n", "roughness", "N.V", "mean |d s|", "max |d s|", "mean |d b|", "max |d b|", "signed d b" );

        for ( uint32_t rowBand = 0; rowBand < DFGComparison::NumBands; ++rowBand )
        {
            for ( uint32_t columnBand = 0; columnBand < DFGComparison::NumBands; ++columnBand )
            {
                DFGRegion const& region = regions[rowBand][columnBand];

                char roughnessText[32] = {};
                char ndotVText[32] = {};

                std::snprintf( roughnessText, sizeof( roughnessText ), "%.3f-%.3f", table.GetRoughness( region.m_rowBegin ), table.GetRoughness( region.m_rowEnd - 1 ) );
                std::snprintf( ndotVText, sizeof( ndotVText ), "%.3f-%.3f", table.GetNdotV( region.m_columnBegin ), table.GetNdotV( region.m_columnEnd - 1 ) );

                std::printf
                (
                    "  %-15s %-15s %-13.4e %-13.4e %-13.4e %-13.4e %+-13.4e\n",
                    roughnessText, ndotVText,
                    region.m_meanAbsoluteScale, region.m_worstAbsoluteScale,
                    region.m_meanAbsoluteBias, region.m_worstAbsoluteBias,
                    region.m_signedMeanBias
                );
            }
        }

        std::printf( "\n" );
        std::printf( "  whole grid\n" );
        std::printf( "    mean |d scale|                 : %.6e\n", comparison.m_meanAbsoluteScale );
        std::printf( "    mean |d bias|                  : %.6e\n", comparison.m_meanAbsoluteBias );
        std::printf( "    mean signed d scale            : %+.6e\n", comparison.m_signedMeanScale );
        std::printf( "    mean signed d bias             : %+.6e\n", comparison.m_signedMeanBias );
        std::printf( "    worst |d scale|                : %.6e at N.V %.6f, roughness %.6f\n", comparison.m_worstAbsoluteScale, table.GetNdotV( comparison.m_worstScale.m_column ), table.GetRoughness( comparison.m_worstScale.m_row ) );
        std::printf( "      table %.9f / reference %.9f\n", comparison.m_worstScale.m_tableScale, comparison.m_worstScale.m_referenceScale );
        std::printf( "    worst |d bias|                 : %.6e at N.V %.6f, roughness %.6f\n", comparison.m_worstAbsoluteBias, table.GetNdotV( comparison.m_worstBias.m_column ), table.GetRoughness( comparison.m_worstBias.m_row ) );
        std::printf( "      table %.9f / reference %.9f\n", comparison.m_worstBias.m_tableBias, comparison.m_worstBias.m_referenceBias );
    }

    //  The quadrature check
    //-------------------------------------------------------------------------
    // Two estimators that share a weight cannot check that weight, and a deterministic quadrature that resolves the lobe is the only thing here that can.
    // It cannot resolve a grazing sample at a narrow width, though, so each texel it runs at is also evaluated by the table at four times its sample count: a texel whose own estimate moves by more than the tolerance between those two is sampling limited, and the quadrature says nothing about it.
    // Those are counted and reported rather than silently passed, and a row that has none is reported as saying nothing.

    static constexpr double DFGQuadratureTolerance = 5.0e-3;

    // How much the table's own estimate is allowed to move when the sample count is quadrupled before the texel counts as converged.
    static constexpr double DFGQuadratureConvergenceTolerance = 1.0e-3;

    static bool CheckDFGQuadratureRows( DFGTable const& table, uint32_t sampleCount, bool& anyPassed )
    {
        uint32_t const resolution = table.GetResolution();

        bool passed = true;
        anyPassed = false;

        std::printf( "\n" );
        std::printf( "deterministic quadrature\n" );
        std::printf( "  the BRDF written longhand over the hemisphere, at %u x %u cells, against two\n", DFGQuadratureThetaCells, DFGQuadraturePhiCells );
        std::printf( "  rows of the table. A texel counts only where quadrupling the table's own\n" );
        std::printf( "  sample count moves its estimate by less than %.0e, which is where a sample\n", DFGQuadratureConvergenceTolerance );
        std::printf( "  count can still say what the integral is.\n" );
        std::printf( "\n" );
        std::printf( "  %-10s %-11s %-13s %-13s %-13s %-13s\n", "roughness", "converged", "max |d s|", "max |d b|", "table s", "quad s" );

        for ( uint32_t const row : DFGQuadratureRows )
        {
            double const roughness = table.GetRoughness( row );

            uint32_t converged = 0;
            double worstScale = 0.0;
            double worstBias = 0.0;
            double tableScale = 0.0;
            double quadratureScale = 0.0;

            for ( uint32_t column = 0; column < resolution; ++column )
            {
                double const ndotV = table.GetNdotV( column );

                DFGTexel const fine = EvaluateDFGTexel( ndotV, roughness, sampleCount, DFGSampleSequence::Hammersley );
                DFGTexel const coarser = EvaluateDFGTexel( ndotV, roughness, sampleCount * 4u, DFGSampleSequence::Hammersley );

                if ( ( std::fabs( fine.m_scale - coarser.m_scale ) > DFGQuadratureConvergenceTolerance )
                  || ( std::fabs( fine.m_bias - coarser.m_bias ) > DFGQuadratureConvergenceTolerance ) )
                {
                    continue;
                }

                ++converged;

                DFGTexel const quadrature = EvaluateDFGTexelQuadrature( ndotV, roughness, DFGQuadratureThetaCells, DFGQuadraturePhiCells );

                double const scaleDelta = std::fabs( fine.m_scale - quadrature.m_scale );
                double const biasDelta = std::fabs( fine.m_bias - quadrature.m_bias );

                if ( scaleDelta > worstScale )
                {
                    worstScale = scaleDelta;
                    tableScale = fine.m_scale;
                    quadratureScale = quadrature.m_scale;
                }

                worstBias = ( biasDelta > worstBias ) ? biasDelta : worstBias;
            }

            if ( converged == 0 )
            {
                std::printf( "  %-10.4f %-11s %-13s %-13s %-13s %-13s\n", roughness, "none", "-", "-", "-", "-" );

                continue;
            }

            anyPassed = true;

            bool const rowPassed = ( worstScale <= DFGQuadratureTolerance ) && ( worstBias <= DFGQuadratureTolerance );

            passed = passed && rowPassed;

            std::printf
            (
                "  %-10.4f %-11u %-13.4e %-13.4e %-13.6f %-13.6f\n",
                roughness, converged, worstScale, worstBias, tableScale, quadratureScale
            );

            if ( !rowPassed )
            {
                std::printf( "    above %.0e against an estimator-independent quadrature\n", DFGQuadratureTolerance );
            }
        }

        if ( !anyPassed )
        {
            std::printf( "\n  no texel in any of these rows is converged, so this check measured nothing.\n" );
        }

        return passed;
    }

    //  Output encoding
    //-------------------------------------------------------------------------

    static bool CheckDFGEncoding( DFGTable const& table )
    {
        uint32_t const resolution = table.GetResolution();

        // Half has ten mantissa bits, so a normal value is stored to within one part in 1024.
        // Below the smallest normal half the spacing stops shrinking, so the allowance there becomes an absolute one denormal step: a relative tolerance alone would fail on a value the format simply cannot resolve more finely, which is not the same thing as being stored wrongly.
        double const relativeTolerance = 1.05e-3;
        double const denormalStep = 5.9604645e-8;

        // What each stored value is allowed to be off by, as a fraction of that. One is the format's own rounding and nothing beyond it.
        double worstScaleAllowance = 0.0;
        double worstBiasAllowance = 0.0;
        double largestScale = 0.0;
        double largestBias = 0.0;
        uint32_t largestScaleColumn = 0;
        uint32_t largestScaleRow = 0;
        uint32_t largestBiasColumn = 0;
        uint32_t largestBiasRow = 0;

        // The texel each is worst at, with both numbers, because a ratio on its own says nothing about which of the two moved.
        double worstScaleValue = 0.0;
        double worstScaleStored = 0.0;
        double worstBiasValue = 0.0;
        double worstBiasStored = 0.0;
        uint32_t worstScaleColumn = 0;
        uint32_t worstScaleRow = 0;
        uint32_t worstBiasColumn = 0;
        uint32_t worstBiasRow = 0;

        for ( uint32_t row = 0; row < resolution; ++row )
        {
            for ( uint32_t column = 0; column < resolution; ++column )
            {
                DFGHalfTexel half;
                GetHalfTexel( table, column, row, half );

                DFGTexel const& value = table.GetTexel( column, row );

                double const scale = static_cast<double>( HalfToFloat( half.m_scale ) );
                double const bias = static_cast<double>( HalfToFloat( half.m_bias ) );

                if ( std::fabs( value.m_scale ) > largestScale )
                {
                    largestScale = std::fabs( value.m_scale );
                    largestScaleColumn = column;
                    largestScaleRow = row;
                }

                if ( std::fabs( value.m_bias ) > largestBias )
                {
                    largestBias = std::fabs( value.m_bias );
                    largestBiasColumn = column;
                    largestBiasRow = row;
                }

                double const scaleAllowance = ( std::fabs( scale - value.m_scale ) ) / std::max( relativeTolerance * std::fabs( value.m_scale ), denormalStep );
                double const biasAllowance = ( std::fabs( bias - value.m_bias ) ) / std::max( relativeTolerance * std::fabs( value.m_bias ), denormalStep );

                if ( scaleAllowance > worstScaleAllowance )
                {
                    worstScaleAllowance = scaleAllowance;
                    worstScaleValue = value.m_scale;
                    worstScaleStored = scale;
                    worstScaleColumn = column;
                    worstScaleRow = row;
                }

                if ( biasAllowance > worstBiasAllowance )
                {
                    worstBiasAllowance = biasAllowance;
                    worstBiasValue = value.m_bias;
                    worstBiasStored = bias;
                    worstBiasColumn = column;
                    worstBiasRow = row;
                }
            }
        }

        std::printf( "\n" );
        std::printf( "  payload encoding\n" );
        std::printf( "    channels                       : scale, bias\n" );
        std::printf( "    format                         : two IEEE 754 binary16 per texel\n" );
        std::printf( "    payload bytes                  : %u\n", resolution * resolution * 4u );
        std::printf( "    largest scale                  : %.6e at N.V %.6f, roughness %.6f\n", largestScale, table.GetNdotV( largestScaleColumn ), table.GetRoughness( largestScaleRow ) );
        std::printf( "    largest bias                   : %.6e at N.V %.6f, roughness %.6f\n", largestBias, table.GetNdotV( largestBiasColumn ), table.GetRoughness( largestBiasRow ) );
        std::printf( "    worst scale, of its allowance  : %.6f\n", worstScaleAllowance );
        std::printf( "      %.9g evaluated, %.9g stored, at N.V %.6f, roughness %.6f\n", worstScaleValue, worstScaleStored, table.GetNdotV( worstScaleColumn ), table.GetRoughness( worstScaleRow ) );
        std::printf( "    worst bias, of its allowance   : %.6f\n", worstBiasAllowance );
        std::printf( "      %.9g evaluated, %.9g stored, at N.V %.6f, roughness %.6f\n", worstBiasValue, worstBiasStored, table.GetNdotV( worstBiasColumn ), table.GetRoughness( worstBiasRow ) );

        std::printf( "\n" );
        std::printf( "  corners of the grid\n" );
        std::printf( "    %-15s %-15s %-15s %-15s\n", "N.V", "roughness", "scale", "bias" );

        uint32_t const last = resolution - 1u;
        uint32_t const cornerColumn[4] = { 0u, last, 0u, last };
        uint32_t const cornerRow[4] = { 0u, 0u, last, last };

        for ( uint32_t corner = 0; corner < 4; ++corner )
        {
            DFGTexel const& value = table.GetTexel( cornerColumn[corner], cornerRow[corner] );

            std::printf
            (
                "    %-15.6f %-15.6f %-15.6e %-15.6e\n",
                table.GetNdotV( cornerColumn[corner] ), table.GetRoughness( cornerRow[corner] ),
                value.m_scale, value.m_bias
            );
        }

        return ( worstScaleAllowance <= 1.0 ) && ( worstBiasAllowance <= 1.0 );
    }

    //-------------------------------------------------------------------------

    int RunDFGValidation
    (
        char const* pBinaryPath,
        char const* pHeaderPath,
        uint32_t resolution,
        uint32_t sampleCount,
        uint32_t referenceSampleCount
    )
    {
        uint32_t const effectiveReferenceSamples = ( referenceSampleCount > sampleCount ) ? referenceSampleCount : ( sampleCount * 2u );
        uint32_t const coarseReferenceSamples = ( ( effectiveReferenceSamples / DFGConvergenceDivisor ) > 0 ) ? ( effectiveReferenceSamples / DFGConvergenceDivisor ) : 1u;

        std::printf( "\n" );
        std::printf( "preintegrated split-sum DFG term\n" );
        std::printf( "  the integral the engine's own pass evaluates: alpha = roughness squared,\n" );
        std::printf( "  k = roughness / 2, and the half-vector sampled from the NDF\n" );
        std::printf( "\n" );
        std::printf( "  grid                  %u x %u texels\n", resolution, resolution );
        std::printf( "  texel coordinates     N.V = ( x + 0.5 ) / %u, roughness = ( y + 0.5 ) / %u\n", resolution, resolution );
        std::printf( "  table sampling        %u samples per texel, Hammersley\n", sampleCount );
        std::printf( "  reference sampling    %u samples per texel, shifted Cartesian stratification\n", effectiveReferenceSamples );
        std::printf( "  outputs               %s\n", pBinaryPath );
        std::printf( "                        %s\n", ( pHeaderPath != nullptr ) ? pHeaderPath : "no C header requested" );
        std::printf( "\n" );
        std::printf( "  evaluating the table...\n" );
        std::fflush( stdout );

        DFGOptions options;
        options.m_resolution = resolution;
        options.m_sampleCount = sampleCount;

        DFGTable table;
        BuildDFGTable( options, table );

        std::printf( "  evaluating the reference...\n" );
        std::fflush( stdout );

        DFGTable reference;
        BuildDFGReferenceTable( resolution, effectiveReferenceSamples, reference );

        DFGComparison comparison;
        CompareDFGTables( table, reference, comparison );
        PrintDFGBandTable( "table against the reference", "two sampling sequences over one integrand, so what is below is the table's quadrature error", table, comparison, effectiveReferenceSamples );

        bool passed = true;

        bool anyQuadratureRow = false;

        passed = CheckDFGQuadratureRows( table, sampleCount, anyQuadratureRow ) && passed;

        // The reference's own error, measured rather than assumed: the same grid at a quarter of the samples.
        // No deviation above can be smaller than this, and where this is the larger of the two the comparison measured the reference.
        std::printf( "\n" );
        std::printf( "  evaluating the reference's own convergence...\n" );
        std::fflush( stdout );

        DFGTable coarseReference;
        BuildDFGReferenceTable( resolution, coarseReferenceSamples, coarseReference );

        DFGComparison convergence;
        CompareDFGTables( reference, coarseReference, convergence );
        PrintDFGBandTable( "the reference against itself", "the same estimator at a quarter of the samples, which bounds every number above", reference, convergence, coarseReferenceSamples );

        passed = CheckDFGEncoding( table ) && passed;

        // Output
        //---------------------------------------------------------------------

        std::printf( "\n" );
        std::printf( "output\n" );

        bool const binaryWritten = WriteDFGTableBinary( table, pBinaryPath );

        std::printf( "  %-40s : %s\n", pBinaryPath, binaryWritten ? "written" : "FAILED" );

        passed = passed && binaryWritten;

        if ( pHeaderPath != nullptr )
        {
            bool const headerWritten = WriteDFGTableHeader( table, pHeaderPath );

            std::printf( "  %-40s : %s\n", pHeaderPath, headerWritten ? "written" : "FAILED" );

            passed = passed && headerWritten;
        }

        std::printf( "\n" );
        std::printf( "  RESULT                             : %s\n", passed ? "PASS" : "FAIL" );

        return passed ? 0 : 1;
    }
}
