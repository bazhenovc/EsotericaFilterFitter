#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "BsplineRecurrence.h"
#include "CoefficientTable.h"
#include "HDRIDataset.h"
#include "LevelWidthCurve.h"
#include "ProbeMap.h"
#include "TableHarness.h"

//  HDRI radiance validation
//-------------------------------------------------------------------------
// The fit is scored on a preimage, which measures how well the taps reproduce the lobe and says nothing about what that does to an environment.
// This is the other half. For each HDRI:
//
//      ingest      equirect EXR -> area downsample -> a 128 probe level
//      reference   the engine's brute-force convolution, on the CPU
//      fast        the same convolution from a table
//      compare     relative error of fast against reference, per level
//
// Reference and fast read the same source chain by the same sampler, so the only difference between them is where the samples come from and how they are weighted, which is what is being validated.
// Both are instantiated for either base map; m_map selects which.
//
// Every stage is cached under the output root and keyed by the configuration that produced it - the map included - so a re-run with another table reuses the ingest and the reference. 
// Nothing is recomputed because a file exists; a file is reused because it is the answer to the same question.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    struct HDRIRunSettings
    {
        std::string         m_datasetRoot;
        std::string         m_outputRoot = "External/FilterFitter/hdri";

        HDRIIngestSettings  m_ingest;

        // Which base map the chains, the convolutions and the cache are for. 
        // The whole pipeline is instantiated per map, so this selects an instantiationrather than a branch inside one.
        ProbeMap            m_map = ProbeMap::Cube;

        // 0 means every asset the scan accepted
        uint32_t            m_limit = 0;

        // Recompute cached stages instead of reusing them
        bool                m_force = false;

        // Reference convolution sample count. 
        // The estimator's own noise falls as this rises, so it is part of the answer and part of the cache key.
        //
        // 8192 rather than the engine's 1024: the reference's job is to be the converged answer, not to reproduce a realtime budget. 
        // Measured on DayEnvironmentHDRI001, going from 1024 to 8192 moves every table's level-6 error by 6-20%, and going from 8192 to 131072 - sixteen times the samples - moves nothing beyond the fourth decimal.
        // 1024 stays reachable and reproduces RadianceFiltering.esf's own number.
        uint32_t            m_samples = 8192;

        // Pass-1 weighting used to read the source chain, and by the taps.
        // The tables are fitted against the paper's J, so that is the default.
        JacobianWeighting   m_weighting = JacobianWeighting::Jacobian;

        // Which profile and curve the run selected.
        // The same flags that fit a table decide what its validation means, so they are read here rather than duplicated as their own options.
        LevelWidthCurve     m_curve;

        // Settings the fitted checkpoint was written with, so its fingerprint can be reconstructed and its table loaded.
        // Only Configuration is checked: the seed is not part of the table.
        EvaluationSettings  m_evaluation;

        // The shape the fitted table has.
        // The fingerprint names a layout rather than a published table, so the reader has to be told which one it is looking for - the map alone does not say how many taps the fit used.
        TableShape          m_shape = ReferenceTable::Const8;

        // Derived path of the fitted checkpoint for the selected curve, or null tos validate the published tables only.
        char const*         pCheckpointPath = nullptr;

        // Spec power the published tables are validated at, matching conformance.
        uint32_t            m_specPower = 18;

        // Optional per-asset, per-table, per-level dump.
        // A corpus run is thousands of numbers and the console can only show a summary of them; this is the raw data an outlier claim has to be checked against.
        char const*         pCsvPath = nullptr;

        // Write the showcase EXRs: one image per interesting environment, holding the source beside the brute-force result beside the table's result at every level. 
        // Off by default - it is for looking at, not for measuring, and picking what to look at costs a pass over the results.
        bool                m_showcase = false;

        // Where they go. Empty means beside the CSV, or the output root when there is no CSV.
        std::string         m_showcaseDirectory;
    };

    //-------------------------------------------------------------------------

    // Lists what the scan found and what it rejected, and writes nothing.
    int RunHDRIScan( HDRIRunSettings const& settings );

    // Ingests every accepted HDRI into the output root as one cubemap each.
    int RunHDRIIngest( HDRIRunSettings const& settings );

    // Reference and table-driven convolution, compared level by level.
    int RunHDRIValidate( HDRIRunSettings const& settings );
}
