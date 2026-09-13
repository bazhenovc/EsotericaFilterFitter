#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "Assert.h"

//  L-BFGS
//-------------------------------------------------------------------------
// Limited-memory BFGS with an Armijo backtracking line search.
// L-BFGS rather than the paper's plain BFGS because the level's unknown count is in the hundreds and a dense inverse Hessian is not worth the memory or the factorisation.
//
// The objective is taken as a template parameter and is expected to provide
//
//      double Evaluate( std::vector<double> const& )
//      double EvaluateWithGradient( std::vector<double> const&, std::vector<double>& )
//      uint32_t GetParameterCount()
//
// Gradients are central differences, which is what LevelObjective supplies.
// That is the starting point: it costs 2n objective evaluations per iteration, and the cost model says a const_8 gradient is about 0.15 s at grid 4, so a few hundred iterations is minutes rather than days.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    // Optional per-parameter freeze
    //-------------------------------------------------------------------------
    // A zero entry freezes the parameter: its gradient component is zeroed, so the search never leaves the free subspace and the reported gradient norm covers only the free coordinates.
    //
    // This exists to measure a subspace rather than to optimise one.
    // Freezing everything except the weights turns the optimizer into a probe for how much the weight direction alone has left to give at a point where the joint search has stalled.
    //-------------------------------------------------------------------------

    struct ParameterMask
    {
        std::vector<uint8_t> m_free;

        void SetAllFree( uint32_t numParameters )
        {
            m_free.assign( numParameters, 1 );
        }

        inline bool IsFree( uint32_t parameterIndex ) const
        {
            return ( parameterIndex < m_free.size() ) && ( m_free[parameterIndex] != 0 );
        }
    };

    //-------------------------------------------------------------------------

    struct OptimizerSettings
    {
        uint32_t    m_maxIterations = 300;
        uint32_t    m_memorySize = 10;
        double      m_gradientTolerance = 1.0e-10;

        // Improvements smaller than this, measured over m_valueWindow iterations rather than one, count as the objective no longer moving.
        // This is the only rule that catches a creep: a sum of absolute values can keep decreasing by less than any useful amount while Armijo still accepts every step, so neither the gradient nor an exhausted line search fires.
        //
        // The window is what separates a creep from a single flat step mid-descent. 
        // One one step against the next cannot distinguish them at any tolerance: loosening it far enough to catch level 4's creep also terminates level 1, which is still descending at 2.5e-6 per iteration when its worst single step stops improving. 
        // Measured over ten iterations, level 4 moves the objective by at most 1.1e-8 relative while level 1 moves it by 1.4e-4, so 1e-6 separates them with two orders of margin either side.
        double      m_valueTolerance = 1.0e-6;
        uint32_t    m_valueWindow = 10;
        double      m_initialStep = 1.0;
        double      m_armijoConstant = 1.0e-4;
        double      m_stepShrink = 0.5;
        uint32_t    m_maxLineSearchSteps = 40;
        bool        m_verbose = false;

        // Null leaves every parameter free
        ParameterMask const* m_pMask = nullptr;

        // Sample the value and gradient norm every N iterations into the result.
        uint32_t    m_traceInterval = 0;
    };

    //-------------------------------------------------------------------------

    struct OptimizerTraceSample
    {
        uint32_t    m_iteration;
        double      m_value;
        double      m_gradientNorm;
    };

    // Why the search stopped
    //-------------------------------------------------------------------------
    // A gradient tolerance and a value tolerance both read as "converged" and mean opposite things. 
    // The first says the point is stationary. The second says the objective stopped improving by more than it can resolve, which a sum of absolute values does readily while its subgradient is still large.
    //
    // An exhausted line search is a third outcome again: Armijo could not be satisfied at any step the search tried, so the run is stuck at a point that is neither stationary nor out of budget. 
    // Reporting it as an iteration limit hides the one failure mode worth investigating.
    //-------------------------------------------------------------------------

    enum class OptimizerStopReason
    {
        IterationLimit,
        GradientTolerance,
        ValueTolerance,
        LineSearchFailure,
    };

    inline char const* GetStopReasonName( OptimizerStopReason reason )
    {
        switch ( reason )
        {
            case OptimizerStopReason::GradientTolerance: return "gradient tolerance";
            case OptimizerStopReason::ValueTolerance:    return "value plateau";
            case OptimizerStopReason::LineSearchFailure: return "line search failed";
            default:                                     return "iteration limit";
        }
    }

    //-------------------------------------------------------------------------

    inline double LargestMagnitude( std::vector<double> const& values )
    {
        double largest = 0.0;

        for ( double const value : values )
        {
            double const magnitude = ( value < 0.0 ) ? -value : value;
            largest = ( magnitude > largest ) ? magnitude : largest;
        }

        return largest;
    }

    //-------------------------------------------------------------------------

    struct OptimizerResult
    {
        std::vector<double> m_parameters;
        double              m_initialValue = 0.0;
        double              m_value = 0.0;
        uint32_t            m_iterations = 0;

        // IterationLimit is the default because falling out of the loop is what reaching the limit means.
        OptimizerStopReason m_stopReason = OptimizerStopReason::IterationLimit;

        // Largest absolute gradient component, measured at m_parameters so that the norm and m_value describe the same point
        double              m_gradientNorm = 0.0;

        // Populated only when m_traceInterval is non-zero
        std::vector<OptimizerTraceSample> m_trace;
    };

    //-------------------------------------------------------------------------

    template< typename TObjective >
    OptimizerResult MinimizeLBFGS
    (
        TObjective& objective,
        std::vector<double> const& initialParameters,
        OptimizerSettings const& settings
    )
    {
        uint32_t const numParameters = static_cast<uint32_t>( initialParameters.size() );

        FF_ASSERT( numParameters > 0 );

        OptimizerResult result;
        result.m_parameters = initialParameters;

        std::vector<double> gradient( numParameters );
        std::vector<double> candidate( numParameters );
        std::vector<double> candidateGradient( numParameters );

        // Zeroing the frozen components keeps the direction inside the free subspace, and keeps the curvature pairs consistent with it.
        auto applyMask = [&] ( std::vector<double>& values )
        {
            if ( settings.m_pMask == nullptr )
            {
                return;
            }

            for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
            {
                if ( !settings.m_pMask->IsFree( parameterIndex ) )
                {
                    values[parameterIndex] = 0.0;
                }
            }
        };

        // L-BFGS correction history, oldest first
        std::vector< std::vector<double> > steps;
        std::vector< std::vector<double> > gradientDeltas;
        std::vector<double> stepDotGradientDelta;

        result.m_initialValue = objective.EvaluateWithGradient( result.m_parameters, gradient );
        applyMask( gradient );
        result.m_value = result.m_initialValue;

        double const initialValue = result.m_initialValue;
        double const valueFloor = ( initialValue > 0.0 ) ? initialValue : 1.0;

        // Value history for the plateau test, oldest first.
        // Until the window fills, the oldest entry is the initial value, so the early iterations compare against the starting point rather than against a value that does not exist yet, and a run that stalls immediately is still caught inside the first window.
        uint32_t const valueWindow = ( settings.m_valueWindow > 0 ) ? settings.m_valueWindow : 1;

        std::vector<double> valueHistory( valueWindow, initialValue );
        uint32_t            valueHistoryIndex = 0;

        for ( uint32_t iteration = 0; iteration < settings.m_maxIterations; ++iteration )
        {
            // Two-loop recursion: build the search direction from the history
            std::vector<double> direction = gradient;

            uint32_t const historySize = static_cast<uint32_t>( steps.size() );
            std::vector<double> alpha( historySize );

            for ( uint32_t reverseIndex = historySize; reverseIndex > 0; --reverseIndex )
            {
                uint32_t const historyIndex = reverseIndex - 1;

                double dot = 0.0;
                for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
                {
                    dot += steps[historyIndex][parameterIndex] * direction[parameterIndex];
                }

                alpha[historyIndex] = stepDotGradientDelta[historyIndex] > 0.0
                    ? ( dot / stepDotGradientDelta[historyIndex] )
                    : 0.0;

                for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
                {
                    direction[parameterIndex] -= alpha[historyIndex] * gradientDeltas[historyIndex][parameterIndex];
                }
            }

            // Initial Hessian scaling from the most recent pair
            double scale = 1.0;

            if ( historySize > 0 )
            {
                uint32_t const last = historySize - 1;

                double stepDotStep = 0.0;
                double deltaDotDelta = 0.0;

                for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
                {
                    stepDotStep += steps[last][parameterIndex] * steps[last][parameterIndex];
                    deltaDotDelta += gradientDeltas[last][parameterIndex] * gradientDeltas[last][parameterIndex];
                }

                if ( deltaDotDelta > 0.0 )
                {
                    scale = stepDotStep / deltaDotDelta;
                }
            }

            for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
            {
                direction[parameterIndex] *= scale;
            }

            for ( uint32_t forwardIndex = 0; forwardIndex < historySize; ++forwardIndex )
            {
                double dot = 0.0;
                for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
                {
                    dot += gradientDeltas[forwardIndex][parameterIndex] * direction[parameterIndex];
                }

                double const denominator = stepDotGradientDelta[forwardIndex];
                double const beta = ( denominator > 0.0 )
                    ? ( dot / denominator )
                    : 0.0;

                double const coefficient = alpha[forwardIndex] - beta;

                for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
                {
                    direction[parameterIndex] += coefficient * steps[forwardIndex][parameterIndex];
                }
            }

            // Descend
            for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
            {
                direction[parameterIndex] = -direction[parameterIndex];
            }

            double directionalDerivative = 0.0;
            for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
            {
                directionalDerivative += direction[parameterIndex] * gradient[parameterIndex];
            }

            double const largestGradient = LargestMagnitude( gradient );

            result.m_gradientNorm = largestGradient;

            if ( ( settings.m_traceInterval > 0 ) && ( ( iteration % settings.m_traceInterval ) == 0 ) )
            {
                OptimizerTraceSample sample;
                sample.m_iteration = iteration;
                sample.m_value = result.m_value;
                sample.m_gradientNorm = largestGradient;

                result.m_trace.push_back( sample );
            }

            if ( largestGradient < settings.m_gradientTolerance )
            {
                result.m_stopReason = OptimizerStopReason::GradientTolerance;
                break;
            }

            // A non-descent direction means the curvature pairs have gone bad; restart from steepest descent rather than line-searching uphill.
            if ( directionalDerivative >= 0.0 )
            {
                steps.clear();
                gradientDeltas.clear();
                stepDotGradientDelta.clear();

                for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
                {
                    direction[parameterIndex] = -gradient[parameterIndex];
                }

                directionalDerivative = 0.0;
                for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
                {
                    directionalDerivative -= gradient[parameterIndex] * gradient[parameterIndex];
                }
            }

            // Armijo backtracking
            double stepLength = settings.m_initialStep;
            bool accepted = false;
            double candidateValue = result.m_value;

            for ( uint32_t lineSearchStep = 0; lineSearchStep < settings.m_maxLineSearchSteps; ++lineSearchStep )
            {
                for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
                {
                    candidate[parameterIndex] = result.m_parameters[parameterIndex] + ( stepLength * direction[parameterIndex] );
                }

                candidateValue = objective.Evaluate( candidate );

                if ( candidateValue <= ( result.m_value + ( settings.m_armijoConstant * stepLength * directionalDerivative ) ) )
                {
                    accepted = true;
                    break;
                }

                stepLength *= settings.m_stepShrink;
            }

            if ( !accepted )
            {
                // Armijo could not be satisfied at any step the line search tried, so the smallest step it reached is already below the difference the objective can resolve.
                // That is a stall, not an exhausted budget, and folding it into the iteration limit hides it.
                result.m_stopReason = OptimizerStopReason::LineSearchFailure;
                break;
            }

            std::vector<double> const previousGradient = gradient;

            objective.EvaluateWithGradient( candidate, candidateGradient );
            applyMask( candidateGradient );

            // Curvature pair
            std::vector<double> step( numParameters );
            std::vector<double> gradientDelta( numParameters );

            double stepDotDelta = 0.0;
            double stepDotStep = 0.0;
            double deltaDotDelta = 0.0;

            for ( uint32_t parameterIndex = 0; parameterIndex < numParameters; ++parameterIndex )
            {
                step[parameterIndex] = candidate[parameterIndex] - result.m_parameters[parameterIndex];
                gradientDelta[parameterIndex] = candidateGradient[parameterIndex] - previousGradient[parameterIndex];
                stepDotDelta += step[parameterIndex] * gradientDelta[parameterIndex];
                stepDotStep += step[parameterIndex] * step[parameterIndex];
                deltaDotDelta += gradientDelta[parameterIndex] * gradientDelta[parameterIndex];
            }

            result.m_parameters = candidate;
            result.m_value = candidateValue;
            gradient = candidateGradient;
            result.m_iterations = iteration + 1;

            // The value and the gradient norm have to describe the same point.
            // The norm from the top of the iteration belongs to the point before the step, so reusing it here pairs a value with a gradient taken from either side of the step - which is what the convergence verdict reads.
            double const candidateGradientNorm = LargestMagnitude( candidateGradient );
            result.m_gradientNorm = candidateGradientNorm;

            // Against the oldest entry, which is the value from valueWindow iterations ago once the history has filled.
            bool const valueConverged = ( valueHistory[valueHistoryIndex] - candidateValue ) <= ( settings.m_valueTolerance * valueFloor );

            valueHistory[valueHistoryIndex] = candidateValue;
            valueHistoryIndex = ( valueHistoryIndex + 1 ) % valueWindow;

            // Require the curvature to be a non-negligible fraction of |s||y|.
            // An absolute floor admits pairs that are numerically orthogonal, and one degenerate pair corrupts the two-loop recursion for as long as it stays in the history.
            double const curvatureScale = std::sqrt( stepDotStep * deltaDotDelta );

            if ( stepDotDelta > ( 1.0e-10 * curvatureScale ) )
            {
                if ( steps.size() >= settings.m_memorySize )
                {
                    steps.erase( steps.begin() );
                    gradientDeltas.erase( gradientDeltas.begin() );
                    stepDotGradientDelta.erase( stepDotGradientDelta.begin() );
                }

                steps.push_back( step );
                gradientDeltas.push_back( gradientDelta );
                stepDotGradientDelta.push_back( stepDotDelta );
            }

            if ( settings.m_verbose )
            {
                std::printf( "    iter %4u  value %.8f  |g|inf %.3e  step %.3e\n",
                             iteration + 1, candidateValue, candidateGradientNorm, stepLength );
                std::fflush( stdout );
            }

            if ( valueConverged )
            {
                result.m_stopReason = OptimizerStopReason::ValueTolerance;
                break;
            }
        }

        return result;
    }
}
