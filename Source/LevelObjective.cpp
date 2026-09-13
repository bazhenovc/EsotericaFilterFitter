#include "Assert.h"
#include "LevelObjective.h"

#include "CubeReferenceFrame.h"
#include "MapProjection.h"
#include "ParallelFor.h"
#include "PreimageError.h"
#include "ProfileBeckmann.h"
#include "ProfileGGX.h"
#include "TetrahedralProjection.h"

#include <cmath>

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    void LevelObjective< TFrame, TProfile >::Initialize
    (
        TFrame const& baseFrame,
        TableShape const& shape,
        TProfile const& profile,
        EvaluationSettings const& settings,
        uint32_t level
    )
    {
        FF_ASSERT( baseFrame.IsInitialized() );
        FF_ASSERT( level < CoefficientTable::NumLevels );

        m_pBaseFrame = &baseFrame;
        m_settings = settings;
        m_level = level;
        m_shape = shape;

        m_parameters.Create( shape );

        uint32_t const resolution = settings.m_baseResolution >> level;

        BuildOutputTexelList< typename TFrame::Map >( resolution, settings.m_gridSize, m_texels );

        uint32_t const numItems = static_cast<uint32_t>( m_texels.size() / 3 );

        // Enough lanes for whichever of the two parallel regions uses more: the per-texel evaluation, or the 2n finite-difference probes. 
        // At a coarse level the probes win by a wide margin - level 6 has 24 output texels and 240 probes - and sizing from the texel count alone indexes past the end of the worker array.
        uint32_t const numProbes = 2 * m_parameters.GetLevelParameterCount();
        uint32_t const numLanes = GetWorkerCount( ( numItems > numProbes ) ? numItems : numProbes );

        m_workers.resize( numLanes );

        for ( Worker& worker : m_workers )
        {
            worker.m_accumulator.Initialize( settings.m_baseResolution, settings.m_sampleLevelCount, settings.m_weighting );
            worker.m_reference.Initialize( &baseFrame, level, settings.m_supersampleRate );
            worker.m_parameters.Create( shape );
        }

        // B(x) is independent of the table, so it is paid once here rather than once per evaluation. This is the whole reason the class exists.
        m_referenceValues.resize( numItems );

        auto evaluateReferenceRange = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            ReferencePreimage< TFrame >& reference = m_workers[workerIndex].m_reference;

            for ( uint32_t item = begin; item < end; ++item )
            {
                uint32_t const face = m_texels[( item * 3 ) + 0];
                uint32_t const texelX = m_texels[( item * 3 ) + 1];
                uint32_t const texelY = m_texels[( item * 3 ) + 2];

                // The map's own texel -> direction, not a cubemap expression: this is the output direction the reference is built for, and on a single-slice tetrahedral map a "face" is one of four triangles in one texture rather than one of six textures.
                double outputDirection[3];
                TFrame::Map::GetDirection( outputDirection, face, texelX, texelY, resolution );

                reference.Evaluate( profile, outputDirection );

                for ( uint32_t smoothingPass = 0; smoothingPass < settings.m_referenceSmoothing; ++smoothingPass )
                {
                    reference.Smooth();
                }

                reference.Normalize( settings.m_measure );

                m_referenceValues[item] = reference.GetValues();
            }
        };

        ParallelForRanges( numItems, evaluateReferenceRange );

        m_perTexelL1.assign( numItems, 0.0 );
        m_probeValues.assign( 2 * m_parameters.GetLevelParameterCount(), 0.0 );
        m_numEvaluations = 0;
    }

    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    double LevelObjective< TFrame, TProfile >::EvaluateItem( Worker& worker, uint32_t item )
    {
        uint32_t const face = m_texels[( item * 3 ) + 0];
        uint32_t const texelX = m_texels[( item * 3 ) + 1];
        uint32_t const texelY = m_texels[( item * 3 ) + 2];

        BuildTableTaps< typename TFrame::Map >( worker.m_parameters, m_level, face, texelX, texelY, m_settings.m_baseResolution, worker.m_taps );

        worker.m_accumulator.Reset();

        double totalTapWeight = 0.0;
        for ( TableTap const& tap : worker.m_taps )
        {
            worker.m_accumulator.AddSample( tap.m_direction, tap.m_level, tap.m_weight );
            totalTapWeight += tap.m_weight;
        }

        std::vector<double> const& approximation = worker.m_accumulator.ResolveToBase();

        // Frozen while solving the convex weight subproblem, recomputed otherwise
        double const normaliser = ( m_pFrozenNormalisers != nullptr )
            ? ( *m_pFrozenNormalisers )[item]
            : totalTapWeight;

        // A non-positive normaliser is a degenerate table: the runtime divides by the weight sum, so zero is a division by zero and negative flips the sign of the result.
        // ComparePreimages only requires a non-zero sum, so negative passes there silently and the fit will walk into it.
        //
        // Penalised rather than asserted, because the optimizer has to be pushed out of the region rather than crashed in it. 
        // The magnitude is large enough that any descent direction away from it is preferred.
        if ( normaliser <= 0.0 )
        {
            return 1.0e30;
        }

        PreimageComparison const comparison = ComparePreimages( *m_pBaseFrame, m_referenceValues[item], approximation, normaliser, m_settings.m_measure );

        return comparison.m_l1;
    }

    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    double LevelObjective< TFrame, TProfile >::Evaluate( std::vector<double> const& parameters )
    {
        FF_ASSERT( IsInitialized() );
        FF_ASSERT( parameters.size() == GetParameterCount() );

        // Every lane's table gets the same parameters, because the parallel region below splits by output texel and any lane may take any texel.
        for ( Worker& worker : m_workers )
        {
            worker.m_parameters.SetLevelParameters( m_level, parameters );
        }

        uint32_t const numItems = static_cast<uint32_t>( m_referenceValues.size() );

        auto evaluateRange = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            Worker& worker = m_workers[workerIndex];

            for ( uint32_t item = begin; item < end; ++item )
            {
                m_perTexelL1[item] = EvaluateItem( worker, item );
            }
        };

        ParallelForRanges( numItems, evaluateRange );

        ++m_numEvaluations;

        double total = 0.0;
        for ( uint32_t item = 0; item < numItems; ++item )
        {
            total += m_perTexelL1[item];
        }

        return total / static_cast<double>( numItems );
    }

    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    double LevelObjective< TFrame, TProfile >::EvaluateOnWorker( uint32_t workerIndex, std::vector<double> const& parameters )
    {
        Worker& worker = m_workers[workerIndex];

        worker.m_parameters.SetLevelParameters( m_level, parameters );

        uint32_t const numItems = static_cast<uint32_t>( m_referenceValues.size() );

        double total = 0.0;
        for ( uint32_t item = 0; item < numItems; ++item )
        {
            total += EvaluateItem( worker, item );
        }

        return total / static_cast<double>( numItems );
    }

    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    double LevelObjective< TFrame, TProfile >::EvaluateWithGradient( std::vector<double> const& parameters, std::vector<double>& gradient )
    {
        FF_ASSERT( IsInitialized() );

        uint32_t const numParameters = GetParameterCount();

        FF_ASSERT( parameters.size() == numParameters );

        double const value = Evaluate( parameters );

        uint32_t const numProbes = 2 * numParameters;

        gradient.resize( numParameters );
        m_probeValues.resize( numProbes );

        std::vector<double> steps( numParameters );
        for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
        {
            steps[parameterIndex] = GetFiniteDifferenceStep( parameters[parameterIndex] );
        }

        // Both probes of every coordinate, in one parallel region.
        // A lane owns its scratch, so it can run whole evaluations serially; the shared result array is written at disjoint indices.
        auto probeRange = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            std::vector<double> probe = parameters;

            for ( uint32_t probeIndex = begin; probeIndex < end; ++probeIndex )
            {
                uint32_t const parameterIndex = probeIndex >> 1;
                bool const     isForward = ( ( probeIndex & 1 ) == 0 );

                probe[parameterIndex] = parameters[parameterIndex] + ( isForward ? steps[parameterIndex] : -steps[parameterIndex] );

                m_probeValues[probeIndex] = EvaluateOnWorker( workerIndex, probe );

                probe[parameterIndex] = parameters[parameterIndex];
            }
        };

        ParallelForRanges( numProbes, probeRange );

        m_numEvaluations += numProbes;

        for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
        {
            double const forward = m_probeValues[2 * parameterIndex];
            double const backward = m_probeValues[( 2 * parameterIndex ) + 1];

            gradient[parameterIndex] = ( forward - backward ) / ( 2.0 * steps[parameterIndex] );
        }

        return value;
    }

    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    double LevelObjective< TFrame, TProfile >::EvaluateWithWeightGradient( std::vector<double> const& parameters, std::vector<double>& gradient )
    {
        FF_ASSERT( IsInitialized() );

        uint32_t const numParameters = GetParameterCount();
        uint32_t const activeCoefficients = m_parameters.GetActiveCoefficientCount();

        FF_ASSERT( parameters.size() == numParameters );

        double const value = Evaluate( parameters );

        gradient.assign( numParameters, 0.0 );

        for ( Worker& worker : m_workers )
        {
            worker.m_parameters.SetLevelParameters( m_level, parameters );
            worker.m_sign.assign( m_referenceValues.empty() ? 0 : m_referenceValues[0].size(), 0.0 );
            worker.m_singleSplat.assign( worker.m_sign.size(), 0.0 );
            worker.m_weightGradient.assign( numParameters, 0.0 );
        }

        uint32_t const numItems = static_cast<uint32_t>( m_referenceValues.size() );

        auto gradientRange = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            Worker& worker = m_workers[workerIndex];

            for ( uint32_t item = begin; item < end; ++item )
            {
                uint32_t const face = m_texels[( item * 3 ) + 0];
                uint32_t const texelX = m_texels[( item * 3 ) + 1];
                uint32_t const texelY = m_texels[( item * 3 ) + 2];

                BuildTableTaps< typename TFrame::Map >( worker.m_parameters, m_level, face, texelX, texelY, m_settings.m_baseResolution, worker.m_taps );

                worker.m_accumulator.Reset();

                double totalTapWeight = 0.0;
                for ( TableTap const& tap : worker.m_taps )
                {
                    worker.m_accumulator.AddSample( tap.m_direction, tap.m_level, tap.m_weight );
                    totalTapWeight += tap.m_weight;
                }

                std::vector<double> const& approximation = worker.m_accumulator.ResolveToBase();

                double const normaliser = ( m_pFrozenNormalisers != nullptr )
                    ? ( *m_pFrozenNormalisers )[item]
                    : totalTapWeight;

                // Degenerate, already penalised by EvaluateItem; contributes no gradient rather than a division by a non-positive normaliser
                if ( normaliser <= 0.0 )
                {
                    continue;
                }

                std::vector<double> const& referenceValues = m_referenceValues[item];

                uint32_t const numBaseTexels = static_cast<uint32_t>( approximation.size() );

                // Residual sign field, and v = sum s * A
                double v = 0.0;

                for ( uint32_t baseTexel = 0; baseTexel < numBaseTexels; ++baseTexel )
                {
                    double const measure = m_pBaseFrame->GetTexelMeasure( m_pBaseFrame->GetTexel( baseTexel ), m_settings.m_measure );

                    double const density = approximation[baseTexel] / ( normaliser * measure );
                    double const residual = referenceValues[baseTexel] - density;

                    double const sign = ( residual > 0.0 ) ? 1.0 : ( ( residual < 0.0 ) ? -1.0 : 0.0 );

                    worker.m_sign[baseTexel] = sign;

                    v += sign * approximation[baseTexel];
                }

                // Per-tap footprints, one resolve each.
                // This is the K-resolve cost the adjoint would remove; at K = 24 against a 200us resolve it is about 5 ms per output texel, which is affordable.
                for ( uint32_t tapIndex = 0; tapIndex < worker.m_taps.size(); ++tapIndex )
                {
                    TableTap const& tap = worker.m_taps[tapIndex];

                    worker.m_accumulator.Reset();
                    worker.m_accumulator.AddSample( tap.m_direction, tap.m_level, 1.0 );

                    std::vector<double> const& singleSplat = worker.m_accumulator.ResolveToBase();

                    double u = 0.0;
                    for ( uint32_t baseTexel = 0; baseTexel < numBaseTexels; ++baseTexel )
                    {
                        u += worker.m_sign[baseTexel] * singleSplat[baseTexel];
                    }

                    double const inverseD = 1.0 / normaliser;

                    // BOTH terms carry the frame weight, and that is easy to get wrong.
                    // A tap's stored weight is parameter[4] * frameWeight, so dA/dc carries frameWeight as well as dD/dc.
                    // Putting it on the normaliser term only is a silent error: it is invisible while every active frame has a weight of exactly 1, which is the case at the coarse levels where the grid clamps and max(other) is 1, and it shows up as soon as a third frame activates partially.
                    //
                    // The normaliser term vanishes while frozen, because a held-constant D has no derivative - that is what makes the subproblem convex.
                    double const normaliserTerm = ( m_pFrozenNormalisers != nullptr )
                        ? 0.0
                        : ( v * inverseD * inverseD );

                    double const frameTerm = tap.m_frameWeight * ( ( -u * inverseD ) + normaliserTerm );

                    double const basis[3] = { 1.0, tap.m_theta2, tap.m_phi2 };

                    for ( uint32_t parameter = 0; parameter < CoefficientTable::NumParameters; ++parameter )
                    {
                        for ( uint32_t coefficient = 0; coefficient < activeCoefficients; ++coefficient )
                        {
                            // Address the tap's coefficients by its PARAMETER index, not by its position in the tap list: inactive frames are skipped, so the two differ.
                            size_t const index = ( ( static_cast<size_t>( tap.m_tapIndex ) * CoefficientTable::NumParameters ) + parameter ) * activeCoefficients + coefficient;

                            // Only the weight parameter carries a derivative; the placement and level parameters are held fixed here
                            if ( parameter == CoefficientTable::ParameterWeight )
                            {
                                worker.m_weightGradient[index] += basis[coefficient] * frameTerm;
                            }
                        }
                    }
                }
            }
        };

        ParallelForRanges( numItems, gradientRange );

        // Reduce in lane order, so the result does not depend on the scheduler
        for ( Worker const& worker : m_workers )
        {
            for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
            {
                gradient[parameterIndex] += worker.m_weightGradient[parameterIndex];
            }
        }

        double const inverseCount = 1.0 / static_cast<double>( ( numItems > 0 ) ? numItems : 1 );

        for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
        {
            gradient[parameterIndex] *= inverseCount;
        }

        return value;
    }

    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    void LevelObjective< TFrame, TProfile >::CaptureNormalisers( std::vector<double> const& parameters, std::vector<double>& normalisers )
    {
        FF_ASSERT( IsInitialized() );

        for ( Worker& worker : m_workers )
        {
            worker.m_parameters.SetLevelParameters( m_level, parameters );
        }

        uint32_t const numItems = static_cast<uint32_t>( m_referenceValues.size() );

        normalisers.assign( numItems, 0.0 );

        auto captureRange = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            Worker& worker = m_workers[workerIndex];

            for ( uint32_t item = begin; item < end; ++item )
            {
                uint32_t const face = m_texels[( item * 3 ) + 0];
                uint32_t const texelX = m_texels[( item * 3 ) + 1];
                uint32_t const texelY = m_texels[( item * 3 ) + 2];

                BuildTableTaps< typename TFrame::Map >( worker.m_parameters, m_level, face, texelX, texelY, m_settings.m_baseResolution, worker.m_taps );

                double totalTapWeight = 0.0;
                for ( TableTap const& tap : worker.m_taps )
                {
                    totalTapWeight += tap.m_weight;
                }

                normalisers[item] = totalTapWeight;
            }
        };

        ParallelForRanges( numItems, captureRange );
    }

    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    double LevelObjective< TFrame, TProfile >::EvaluateFrozen
    (
        std::vector<double> const& parameters,
        std::vector<double> const& normalisers,
        std::vector<double>* pGradient
    )
    {
        FF_ASSERT( IsInitialized() );
        FF_ASSERT( normalisers.size() == m_referenceValues.size() );

        m_pFrozenNormalisers = &normalisers;

        double value = 0.0;

        if ( pGradient != nullptr )
        {
            value = EvaluateWithWeightGradient( parameters, *pGradient );
        }
        else
        {
            value = Evaluate( parameters );
        }

        m_pFrozenNormalisers = nullptr;

        return value;
    }

    //-------------------------------------------------------------------------

    template class LevelObjective< CubeReferenceFrame, ProfileGGX >;
    template class LevelObjective< CubeReferenceFrame, ProfileBeckmann >;

    template class LevelObjective< MapReferenceFrame< TetrahedralProjection >, ProfileGGX >;
    template class LevelObjective< MapReferenceFrame< TetrahedralProjection >, ProfileBeckmann >;
}
