#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "Assert.h"
#include "CoefficientTable.h"
#include "FitCheckpoint.h"
#include "LevelObjective.h"
#include "Optimizer.h"
#include "TableHarness.h"
#include "TableSeed.h"

// multi-level fit driver
//-------------------------------------------------------------------------
// Fits one table, one level at a time. Levels are independent - the table is indexed by level and a level's coefficients never enter another level's objective - so the loop is a sequence, not a joint optimisation, and a level  that finishes never has to be revisited.
//
// That is what makes the checkpoint useful: a finished level is final, so a run that dies partway keeps every level it completed, and a resumed run skips them entirely.
//
//  WHAT THIS DELIBERATELY DOES NOT DO YET
//
// The starting point is supplied by the caller. The only seed wired up so far is a published table of the same shape, which is enough to exercise the driver and the checkpoint but is not a generator: it cannot fit a shape the paper did not publish, and it cannot fit a different profile.
// An analytic initialisation - taps placed on rings at radii scaled by the level's lobe width - slots in at exactly this point.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    // The convex weight stage, and how it is attached to the fit
    //-------------------------------------------------------------------------
    // The objective is a ratio of two functions linear in the weights, so it is not convex and a weight-only descent can stall.
    // Freezing the normaliser makes it affine inside the absolute values and therefore convex, and the normaliser is then refreshed from the solved weights.
    //
    // Measured on const_8, running this stage AFTER a joint search is a strict improvement at every level tested (-0.02% to -14.8%), while alternating it with placement loses 5% at two levels because interleaving costs placement iterations. 
    // 
    // Level 5 is the one where the joint search stalls, and this is what unsticks it: 0.186967, where 60 joint iterations moved it 0.01%, to 0.159362.
    //-------------------------------------------------------------------------

    // Presents a frozen-normaliser subproblem through the optimizer's interface
    template< typename TFrame, typename TProfile >
    struct FrozenNormaliserObjective
    {
        LevelObjective< TFrame, TProfile >* m_pObjective;
        std::vector<double> const*          m_pNormalisers;

        uint32_t GetParameterCount() const { return m_pObjective->GetParameterCount(); }

        double Evaluate( std::vector<double> const& parameters )
        {
            return m_pObjective->EvaluateFrozen( parameters, *m_pNormalisers, nullptr );
        }

        double EvaluateWithGradient( std::vector<double> const& parameters, std::vector<double>& gradient )
        {
            return m_pObjective->EvaluateFrozen( parameters, *m_pNormalisers, &gradient );
        }
    };

    // Free either the weight coefficients or everything else
    inline void BuildParameterMask( uint32_t tapCount, uint32_t activeCoefficients, bool weightsFree, ParameterMask& mask )
    {
        mask.SetAllFree( tapCount * CoefficientTable::NumParameters * activeCoefficients );

        for ( uint32_t tap = 0; tap < tapCount; ++tap )
        {
            for ( uint32_t parameter = 0; parameter < CoefficientTable::NumParameters; ++parameter )
            {
                for ( uint32_t coefficient = 0; coefficient < activeCoefficients; ++coefficient )
                {
                    size_t const index = ( ( static_cast<size_t>( tap ) * CoefficientTable::NumParameters ) + parameter ) * activeCoefficients + coefficient;

                    bool const isWeight = ( parameter == CoefficientTable::ParameterWeight );

                    mask.m_free[index] = ( isWeight == weightsFree ) ? 1 : 0;
                }
            }
        }
    }

    //-------------------------------------------------------------------------

    struct FitSettings
    {
        EvaluationSettings  m_evaluation;
        OptimizerSettings   m_optimizer;

        // Which table to fit.
        // A published shape, or any shape a map needs: the shape carries the map, the axis count and the tap count as one value, so a fit for a tetrahedral probe is a different shape rather than a flag.
        TableShape          m_shape = ReferenceTable::Const8;

        // Seed for every level.
        // Null means build one analytically, which is the generator path; a published table of the same shape is a refinement path, used to check the optimizer against a known answer.
        CoefficientTable const* m_pSeed = nullptr;

        // Level offsets and outer ring weights swept when building an analyticseed. 
        // Both are free constants and the sweep costs seconds against a fit's minutes, so they are measured rather than fixed.
        //
        // Measured on const_8, the outer weight is what the seed was missing: the published tables use negative weights, which a uniformly weighted ring cannot express, and sweeping it took the seed from 1.3..8.7x the  published table down to 1.0..4.5x.
        // Coarse levels pick 0.0 or -0.5, a centre-weighted kernel that cancels the ring's skirt; fine levels pick 1.0, which is the old uniform behaviour.
        double              m_seedLevelOffsets[6] = { -1.0, 0.0, 1.0, 2.0, 3.0, 4.0 };
        uint32_t            m_numSeedLevelOffsets = 6;

        double              m_seedOuterRingWeights[7] = { -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0 };
        uint32_t            m_numSeedOuterRingWeights = 7;

        char const*         m_pCheckpointPath = "FilterFitter.fit";

        // Levels [0, m_levelLimit) are fitted. Less than the table's level count is how a partial run is produced, and therefore how resume is tested.
        uint32_t            m_levelLimit = CoefficientTable::NumLevels;

        // Convex weight stages run after the joint search on each level. The measured result is a strict improvement at every level, so the default is on.
        uint32_t            m_polishRounds = 4;
        uint32_t            m_polishIterations = 40;

        // Starts per level, taken from the best-scoring seed of each distinct outer-ring weight and kept by lowest fitted objective.
        // More than one because seed score does not predict fit quality: see the note in the level loop. Set to 1 to reproduce single-start behaviour.
        uint32_t            m_numSeedStarts = 3;

        bool                m_verbose = true;
    };

    //-------------------------------------------------------------------------

    struct FitResult
    {
        FitCheckpoint   m_checkpoint;
        uint32_t        m_levelsFitted = 0;
        uint32_t        m_levelsSkipped = 0;
        double          m_seconds = 0.0;
        bool            m_complete = false;
        bool            m_ok = false;   // false on a refused checkpoint

        // Objective at the starting point of each fitted level, so a caller can tell "the optimizer did nothing" from "the seed was already good".
        double          m_seedObjective[CoefficientTable::NumLevels] = {};

        // Which levels this run actually worked on. 
        // A level resumed from a checkpoint keeps its stored objective and has no seed from this run, so a caller cannot separate the two from m_seedObjective alone - a skipped level would otherwise look like a fit that scored zero against a zero seed.
        bool            m_levelFitted[CoefficientTable::NumLevels] = {};
    };

    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    FitResult RunTableFit( TFrame const& baseFrame, TProfile const& profile, FitSettings const& settings )
    {
        FF_ASSERT( settings.m_pCheckpointPath != nullptr );
        FF_ASSERT( settings.m_levelLimit <= CoefficientTable::NumLevels );
        FF_ASSERT( ( settings.m_pSeed != nullptr ) || ( settings.m_numSeedLevelOffsets > 0 ) );

        std::chrono::steady_clock::time_point const startTime = std::chrono::steady_clock::now();

        FitResult result;

        result.m_checkpoint.m_fingerprint = FitFingerprint::Make( settings.m_evaluation, settings.m_shape, profile, ( settings.m_pSeed != nullptr ) ? settings.m_pSeed->GetName() : "analytic" );

        FitCheckpoint::LoadResult const loadResult = result.m_checkpoint.Load( settings.m_pCheckpointPath, result.m_checkpoint.m_fingerprint, FitCheckpoint::FingerprintCheck::Exact );

        if ( settings.m_verbose )
        {
            if ( loadResult == FitCheckpoint::LoadResult::Loaded )
            {
                std::printf( "\nresuming\n" );
                result.m_checkpoint.PrintSummary( "  " );
            }
            else if ( loadResult == FitCheckpoint::LoadResult::Absent )
            {
                std::printf( "\nno checkpoint, starting a new fit\n" );
            }
            else
            {
                std::printf( "\n%s\n", result.m_checkpoint.m_message );
                if ( loadResult == FitCheckpoint::LoadResult::Incompatible )
                {
                    std::printf( "  refusing to resume rather than mix two fits; move or delete the file\n" );
                }
                std::fflush( stdout );
                return result;
            }

            std::fflush( stdout );
        }
        else if ( ( loadResult == FitCheckpoint::LoadResult::Incompatible ) || ( loadResult == FitCheckpoint::LoadResult::Corrupt ) )
        {
            return result;
        }

        result.m_ok = true;

        for ( uint32_t level = 0; level < settings.m_levelLimit; ++level )
        {
            FitCheckpoint::LevelState& state = result.m_checkpoint.m_levels[level];

            // A zero width is the identity, which comes from the closed form rather than from a search. 
            // Rebuilding it is free and resuming it would pin a row that the construction, not the fit, is responsible for - so a checkpoint written by an older construction cannot survive into a new one.
            bool const isMirror = ( profile.GetWidth( level ) <= 0.0 );

            if ( state.m_done && !isMirror )
            {
                ++result.m_levelsSkipped;
                continue;
            }

            if ( settings.m_verbose )
            {
                std::printf( "  level %u: fitting\n", level );
                std::fflush( stdout );
            }

            LevelObjective< TFrame, TProfile > objective;
            objective.Initialize( baseFrame, settings.m_shape, profile, settings.m_evaluation, level );

            std::vector<double> probe;

            // Candidate starts.
            // One per outer-ring weight, keeping each one's best mip offset, so the candidates differ in the SHAPE of the kernel rather than merely in where it samples.
            //
            // Not just the single best-scoring seed, because seed score does not predict fit quality: measured on const_8, replacing the uniform seed with a better-scoring centre-weighted one took level 5's seed from 0.318 to 0.186 - nearly the published table - and the fit then landed at 0.140 instead of 0.117.
            // The far, worse seed found the better basin.
            // So the fit runs from several and keeps the best.
            uint32_t const numCandidates = settings.m_numSeedOuterRingWeights;

            std::vector<double>              candidateValues( numCandidates, 1.0e30 );
            std::vector<std::vector<double>> candidateSeeds( numCandidates );

            if ( isMirror )
            {
                // A mirror level is built, not searched. 
                // See BuildMirrorLevelParameters: the objective is piecewise constant in the tap direction, so descent has nothing to follow. 
                // One candidate only, which the ordering below turns into one start - the search still runs, finds nothing, and its result is the closed form it started from.
                BuildMirrorLevelParameters< typename TFrame::Map >( settings.m_shape, candidateSeeds[0] );
                candidateValues[0] = objective.Evaluate( candidateSeeds[0] );

                if ( settings.m_verbose )
                {
                    std::printf( "  level %u: mirror (zero width), built analytically -> %.6f\n",
                                 level, candidateValues[0] );
                    std::fflush( stdout );
                }
            }
            else if ( settings.m_pSeed != nullptr )
            {
                settings.m_pSeed->GetLevelParameters( level, probe );

                candidateSeeds[0] = probe;
                candidateValues[0] = objective.Evaluate( probe );
            }
            else
            {
                for ( uint32_t offsetIndex = 0; offsetIndex < settings.m_numSeedLevelOffsets; ++offsetIndex )
                {
                    for ( uint32_t weightIndex = 0; weightIndex < numCandidates; ++weightIndex )
                    {
                        AnalyticSeedSettings seedSettings;
                        seedSettings.m_levelOffset = settings.m_seedLevelOffsets[offsetIndex];
                        seedSettings.m_outerRingWeightScale = settings.m_seedOuterRingWeights[weightIndex];

                        BuildAnalyticSeed
                        (
                            settings.m_shape, profile, level,
                            settings.m_evaluation.m_baseResolution,
                            settings.m_evaluation.m_sampleLevelCount,
                            seedSettings, probe
                        );

                        double const candidateValue = objective.Evaluate( probe );

                        if ( candidateValue < candidateValues[weightIndex] )
                        {
                            candidateValues[weightIndex] = candidateValue;
                            candidateSeeds[weightIndex] = probe;
                        }
                    }
                }
            }

            // Best-scoring candidates first, so a budget of one start reproduces the old behaviour
            std::vector<uint32_t> order;

            for ( uint32_t candidateIndex = 0; candidateIndex < numCandidates; ++candidateIndex )
            {
                if ( !candidateSeeds[candidateIndex].empty() )
                {
                    order.push_back( candidateIndex );
                }
            }

            std::sort( order.begin(), order.end(), [&] ( uint32_t left, uint32_t right )
            {
                return candidateValues[left] < candidateValues[right];
            } );

            uint32_t const numStarts = ( settings.m_numSeedStarts < order.size() ) ? settings.m_numSeedStarts : static_cast<uint32_t>( order.size() );

            double              bestFittedValue = 1.0e30;
            std::vector<double> bestFitted;

            for ( uint32_t startIndex = 0; startIndex < numStarts; ++startIndex )
            {
                std::vector<double> seed = candidateSeeds[order[startIndex]];

                OptimizerResult const optimizerResult = MinimizeLBFGS( objective, seed, settings.m_optimizer );

                std::vector<double> fitted = optimizerResult.m_parameters;
                double fittedValue = optimizerResult.m_value;

                // Convex weight stages, after the joint search rather than competing with it. 
                // Each stage freezes the normaliser from the current parameters, which makes the subproblem convex, then frees it again.
                if ( settings.m_polishRounds > 0 )
                {
                    ParameterMask weightMask;
                    BuildParameterMask( objective.GetTapCount(), objective.GetActiveCoefficientCount(), true, weightMask );

                    OptimizerSettings polishSettings = settings.m_optimizer;
                    polishSettings.m_maxIterations = settings.m_polishIterations;
                    polishSettings.m_memorySize = 20;
                    polishSettings.m_pMask = &weightMask;
                    polishSettings.m_verbose = false;

                    std::vector<double> normalisers;

                    for ( uint32_t round = 0; round < settings.m_polishRounds; ++round )
                    {
                        objective.CaptureNormalisers( fitted, normalisers );

                        FrozenNormaliserObjective< TFrame, TProfile > frozen;
                        frozen.m_pObjective = &objective;
                        frozen.m_pNormalisers = &normalisers;

                        std::vector<double> const polished = MinimizeLBFGS( frozen, fitted, polishSettings ).m_parameters;
                        double const polishedValue = objective.Evaluate( polished );

                        // The stage minimises the FROZEN objective, which is not the true one, so it is not a descent method for what we care about - measured, unguarded it improved some levels by 26% and made others 3% worse. 
                        // Accept only a genuine improvement.
                        if ( polishedValue < fittedValue )
                        {
                            fitted = polished;
                            fittedValue = polishedValue;
                        }
                        else
                        {
                            break;
                        }
                    }
                }

                if ( settings.m_verbose )
                {
                    std::printf( "  level %u: start %u of %u seed %.6f -> fitted %.6f\n",
                                 level, startIndex + 1, numStarts, candidateValues[order[startIndex]], fittedValue );
                    std::fflush( stdout );
                }

                if ( fittedValue < bestFittedValue )
                {
                    bestFittedValue = fittedValue;
                    bestFitted = fitted;
                }

                // One start when a seed table is supplied: the refinement path has no candidate diversity to exploit
                if ( settings.m_pSeed != nullptr )
                {
                    break;
                }
            }

            std::vector<double> const& fitted = bestFitted;
            double const               fittedValue = bestFittedValue;

            state.m_done = true;
            state.m_objective = fittedValue;
            state.m_iterations = 0;
            state.m_parameters = fitted;

            result.m_seedObjective[level] = candidateValues[order[0]];
            result.m_levelFitted[level] = true;

            ++result.m_levelsFitted;

            // After every level, and only after a level: a finished level is final, so the checkpoint never holds a half-fitted one and a resume never repeats work.
            if ( !result.m_checkpoint.Save( settings.m_pCheckpointPath ) )
            {
                std::printf( "  level %u: checkpoint write to %s failed\n", level, settings.m_pCheckpointPath );
                std::fflush( stdout );
                result.m_ok = false;
                return result;
            }

            if ( settings.m_verbose )
            {
                std::printf( "  level %u: objective %.6f from %u starts, checkpointed\n",
                             level, state.m_objective, numStarts );
                std::fflush( stdout );
            }
        }

        result.m_complete = result.m_checkpoint.IsComplete();
        result.m_seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - startTime ).count();

        if ( settings.m_verbose )
        {
            std::printf( "\n  %u levels fitted, %u skipped, %.1f s\n",
                         result.m_levelsFitted, result.m_levelsSkipped, result.m_seconds );
            std::fflush( stdout );
        }

        return result;
    }
}
