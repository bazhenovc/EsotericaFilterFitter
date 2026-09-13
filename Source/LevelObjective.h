#pragma once

#include <cstdint>
#include <vector>

#include "CoefficientTable.h"
#include "PreimageAccumulator.h"
#include "ReferencePreimage.h"
#include "TableHarness.h"

// The optimizer's objective, for one level
//-------------------------------------------------------------------------
// A level's unknown vector is CoefficientTable's [tap][parameter][coefficient] layout: 120 doubles for const_8, 1440 for quad_32.
// Evaluate writes that vector into a table, builds the taps it prescribes, splats them, and returns the mean L1 over the sampled output texels.
//
// That is the same quantity, through the same building blocks, that EvaluatePublishedTableLevel reports - so a fit optimizes the function the conformance test measures rather than a second implementation of it. 
//
// The reason this class exists rather than a loop around the harness is that B(x) depends only on the frame, the profile and the level - never on the table - and costs about 320 us per output texel against roughly 430 us for the splat-and-compare it feeds. 
// Recomputing it per evaluation would put the reference back in the inner loop, which is the one thing the cost model says not to do. So Initialize pays it once and keeps it.
//
//  WORKER SCRATCH
//
// Every lane owns a parameter table, a tap list and an accumulator, so that a finite-difference gradient can evaluate all 2n probes inside ONE parallel region with each lane running whole evaluations serially.
// Evaluating probes as separate parallel regions instead costs 2n barriers per gradient, which measured as the largest remaining overhead once thread creation was removed.
// The consequence is that a lane must only ever touch its own worker state, and that the shared probe-result array is written at disjoint indices.
//
// TFrame is the frame contract in ReferenceFrame.h, TProfile the profile contract in Profile.h. Definitions are in the translation unit; the instantiations the tool uses are at the bottom of it.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    template< typename TFrame, typename TProfile >
    class LevelObjective
    {
    public:

        // Precomputes B for every sampled output texel of level.
        // shape selects the table geometry - tap count and active coefficient count - and is not read for its values.
        void Initialize
        (
            TFrame const& baseFrame,
            TableShape const& shape,
            TProfile const& profile,
            EvaluationSettings const& settings,
            uint32_t level
        );

        inline bool IsInitialized() const { return m_pBaseFrame != nullptr; }

        // Mean L1 over the sampled output texels
        double Evaluate( std::vector<double> const& parameters );

        // The same value, plus a central-difference gradient.
        // Central rather than forward because the objective is a sum of absolute values and therefore has kinks; a one-sided difference steps across them.
        double EvaluateWithGradient( std::vector<double> const& parameters, std::vector<double>& gradient );

        // The same value, plus the ANALYTIC gradient with respect to the weight coefficients. 
        // Every other component is left at zero.
        //
        // This is the piece that makes the weight split affordable.
        // Solving the weight subproblem by forming its K x K normal equations costs K^2 per residual, which is 576 x 9.4M per pass for const_8 - minutes per solve.
        // Differentiating the objective directly costs K per gradient instead, and K resolves to obtain the per-tap footprints.
        //
        // Derivative, with A the resolved field, D the frame-weighted tap sum, rho = B - A/(D*m) the residual and s = sign( rho ):
        //
        //     dF/dc[i,p] = (1/N) * sum_t basis_p(t) * fw[i] *
        //                    [ -u[i] / D  +  v / D^2 ]
        //
        //     u[i] = sum_x s(x) * R( e[i] )(x)    R( e[i] ) = resolve of tap i
        //     v    = sum_x s(x) * A(x)
        //
        // The minus term is the residual's dependence on A, the plus term its dependence on the normaliser D, and BOTH carry the frame weight: a tap's stored weight is parameter[4] * frameWeight, so dA/dc carries frameWeight as well as dD/dc.
        // Getting that wrong on one term only is silent, because it is invisible while every active frame has a weight of exactly 1.
        double EvaluateWithWeightGradient( std::vector<double> const& parameters, std::vector<double>& gradient );

        // The convex weight subproblem
        //---------------------------------------------------------------------
        // The objective is a ratio of two functions linear in the weights, so it is not convex and a weight-only descent can stall.
        // Freezing the normaliser makes it affine inside the absolute values, and therefore convex, and the normaliser is then refreshed from the solved weights.
        //
        // Reaching it by iteratively reweighted least squares would need the K x K normal equations; minimising the frozen L1 directly with the analytic gradient reaches the same subproblem without them, so that is what is implemented.
        // The normaliser term disappears from the gradient while frozen, because a held-constant D has no derivative.
        //---------------------------------------------------------------------

        // The normaliser each output texel currently has, which the caller freezes for the next subproblem
        void CaptureNormalisers( std::vector<double> const& parameters, std::vector<double>& normalisers );

        // Objective, and optionally gradient, with the normaliser held at normalisers rather than recomputed from the parameters
        double EvaluateFrozen
        (
            std::vector<double> const& parameters,
            std::vector<double> const& normalisers,
            std::vector<double>* pGradient
        );

        inline uint32_t GetParameterCount() const { return m_parameters.GetLevelParameterCount(); }
        inline uint32_t GetTapCount() const { return m_parameters.GetTapCount(); }
        inline uint32_t GetActiveCoefficientCount() const { return m_parameters.GetActiveCoefficientCount(); }
        inline uint32_t GetNumOutputTexels() const { return static_cast<uint32_t>( m_referenceValues.size() ); }
        inline uint32_t GetLevel() const { return m_level; }
        inline uint32_t GetNumEvaluations() const { return m_numEvaluations; }
        inline uint32_t GetNumGradientProbes() const { return 2 * GetParameterCount(); }

        // Per-output-texel L1 from the last Evaluate, for localising a residual
        std::vector<double> const& GetLastPerTexelL1() const { return m_perTexelL1; }

        // Relative step for a central difference
        static double GetFiniteDifferenceStep( double value )
        {
            double const magnitude = ( value < 0.0 ) ? -value : value;
            return 1.0e-6 * ( ( magnitude > 1.0e-3 ) ? magnitude : 1.0e-3 );
        }

    private:

        // One lane's scratch. Kept on the object so an evaluation allocates nothing: the optimizer runs hundreds of thousands of them.
        struct Worker
        {
            PreimageAccumulator< typename TFrame::Map > m_accumulator;
            std::vector<TableTap>       m_taps;
            ReferencePreimage< TFrame > m_reference;
            CoefficientTable            m_parameters;

            // Weight-gradient scratch, per lane
            std::vector<double>         m_sign;
            std::vector<double>         m_weightGradient;
            std::vector<double>         m_singleSplat;
        };

        // Splat one output texel's taps and score it.
        // Caller-supplied worker, so this is safe to call concurrently from different worker indices and never touches shared state.
        double EvaluateItem( Worker& worker, uint32_t item );

        // A whole evaluation on one lane's scratch, serially over output texels. Used by the gradient probes.
        double EvaluateOnWorker( uint32_t workerIndex, std::vector<double> const& parameters );

    private:

        TFrame const*                           m_pBaseFrame = nullptr;
        EvaluationSettings                      m_settings;
        uint32_t                                m_level = 0;
        TableShape                              m_shape;

        CoefficientTable                        m_parameters;
        std::vector<uint32_t>                   m_texels;
        std::vector<std::vector<double>>        m_referenceValues;
        std::vector<double>                     m_perTexelL1;
        std::vector<double>                     m_probeValues;

        std::vector<Worker>                     m_workers;

        // Non-null while a frozen-normaliser subproblem is being solved.
        // Null means the normaliser is recomputed from the parameters, which is the true objective.
        std::vector<double> const*              m_pFrozenNormalisers = nullptr;

        uint32_t                                m_numEvaluations = 0;
    };
}
