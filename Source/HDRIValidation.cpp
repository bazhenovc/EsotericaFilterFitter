#include "Assert.h"
#include "HDRIValidation.h"

#include "FitCheckpoint.h"
#include "HDRIConvolve.h"
#include "HDRIExr.h"
#include "HDRIImage.h"
#include "HDRIShowcase.h"
#include "ParallelFor.h"
#include "ProfileGGX.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace FilterFitter
{
    namespace
    {
        //---------------------------------------------------------------------

        HDRIStageConfig MakeSourceConfig( HDRIRunSettings const& settings, std::string const& assetId )
        {
            HDRIStageConfig config;

            config.m_stage = "source";
            config.m_sourceId = assetId;
            config.m_projectionName = GetProbeMapName( settings.m_map );
            config.m_baseWidth = settings.m_ingest.m_baseWidth;
            config.m_numLevels = 1;
            config.m_numSlices = GetProbeMapSliceCount( settings.m_map );

            // The requested width, not the effective one.
            // The effective width depends on the decoded panorama's size, and asking for that would mean decoding the EXR before deciding whether to decode the EXR - which is the whole cost the cache exists to avoid. 
            // The asset's size never changes, so the key is still unambiguous.
            config.m_equirectWidth = settings.m_ingest.m_equirectWidth;

            return config;
        }

        //---------------------------------------------------------------------

        uint32_t GetEffectiveEquirectWidth( HDRIRunSettings const& settings, uint32_t sourceWidth )
        {
            uint32_t const requested = ( settings.m_ingest.m_equirectWidth > 0 ) ? settings.m_ingest.m_equirectWidth : sourceWidth;

            return ( requested < sourceWidth ) ? requested : sourceWidth;
        }
    }

    //-------------------------------------------------------------------------

    int RunHDRIScan( HDRIRunSettings const& settings )
    {
        std::vector<HDRIAsset> assets;
        ScanHDRIDataset( settings.m_datasetRoot, assets );

        std::printf( "\n" );
        std::printf( "HDRI dataset scan\n" );
        std::printf( "  root %s\n", settings.m_datasetRoot.c_str() );

        if ( assets.empty() )
        {
            std::printf( "  no asset.json with type 'hdri' found\n" );
            return 1;
        }

        uint32_t numUsable = 0;

        std::printf( "\n  %-34s %-12s %-22s %-8s\n", "asset", "resolution", "technique", "" );

        for ( HDRIAsset const& asset : assets )
        {
            if ( asset.m_skipReason.empty() )
            {
                ++numUsable;
            }

            std::printf
            (
                "  %-34s %-12s %-22s %-8s\n",
                asset.m_id.c_str(),
                asset.m_declaredResolution.empty() ? "-" : asset.m_declaredResolution.c_str(),
                asset.m_technique.empty() ? "-" : asset.m_technique.c_str(),
                asset.m_skipReason.empty() ? "ok" : "SKIP"
            );

            if ( !asset.m_skipReason.empty() )
            {
                std::printf( "  %-34s %s\n", "", asset.m_skipReason.c_str() );
            }
        }

        std::printf
        (
            "\n  %u HDRI assets, %u usable, %u rejected before decode\n",
            static_cast<uint32_t>( assets.size() ), numUsable,
            static_cast<uint32_t>( assets.size() ) - numUsable
        );

        std::fflush( stdout );

        return 0;
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    static int RunHDRIIngestFor( HDRIRunSettings const& settings )
    {
        std::vector<HDRIAsset> assets;
        ScanHDRIDataset( settings.m_datasetRoot, assets );

        std::printf( "\n" );
        std::printf( "HDRI ingest\n" );
        std::printf( "  root        %s\n", settings.m_datasetRoot.c_str() );
        std::printf( "  output      %s\n", settings.m_outputRoot.c_str() );
        std::printf( "  projection  %s, %u slice%s per level\n", GetProbeMapName( settings.m_map ), TMap::NumSlices, ( TMap::NumSlices == 1 ) ? "" : "s" );
        std::printf( "  base width  %u, equirect down to %u wide before projection\n", settings.m_ingest.m_baseWidth, settings.m_ingest.m_equirectWidth );

        std::vector<HDRIAsset const*> candidates;
        uint32_t numRejected = 0;

        for ( HDRIAsset const& asset : assets )
        {
            if ( asset.m_skipReason.empty() )
            {
                candidates.push_back( &asset );
            }
            else
            {
                ++numRejected;
            }
        }

        uint32_t const numToIngest = ( ( settings.m_limit != 0 ) && ( settings.m_limit < candidates.size() ) )
            ? settings.m_limit
            : static_cast<uint32_t>( candidates.size() );

        std::printf( "  candidates  %u of %u HDRI assets (%u rejected before decode)\n", numToIngest, static_cast<uint32_t>( assets.size() ), numRejected );

        if ( numToIngest == 0 )
        {
            std::printf( "\n  nothing to ingest\n" );
            return 1;
        }

        std::printf( "\n  %-30s %-12s %-12s %-11s %-9s %-8s %-6s\n", "asset", "source", "equirect", "luminance", "seconds", "state", "" );

        uint32_t numIngested = 0;
        uint32_t numCached = 0;
        uint32_t numFailed = 0;

        double totalSeconds = 0.0;

        for ( uint32_t index = 0; index < numToIngest; ++index )
        {
            HDRIAsset const& asset = *candidates[index];

            HDRIStageConfig const config = MakeSourceConfig( settings, asset.m_id );

            std::vector<HDRIImage> levels;
            std::string            cacheMessage;

            bool loaded = false;
            std::string state;
            double seconds = 0.0;
            uint32_t sourceWidth = 0;
            uint32_t sourceHeight = 0;

            std::chrono::steady_clock::time_point const startTime = std::chrono::steady_clock::now();

            if ( !settings.m_force && TryLoadCachedStage( settings.m_outputRoot, config, levels, cacheMessage ) )
            {
                loaded = true;
                state = "cached";
                ++numCached;
            }
            else
            {
                std::vector<float> equirect;
                uint32_t           width = 0;
                uint32_t           height = 0;
                std::string        loadMessage;

                if ( !LoadEquirectEXR( asset, settings.m_ingest, equirect, width, height, loadMessage ) )
                {
                    std::printf( "  %-30s %-12s %-12s %-11s %-9s %-8s %-6s\n", asset.m_id.c_str(), "-", "-", "-", "-", "SKIP", "skip" );
                    std::printf( "  %-30s %s\n", "", loadMessage.c_str() );
                    std::fflush( stdout );

                    ++numFailed;
                    continue;
                }

                sourceWidth = width;
                sourceHeight = height;

                uint32_t const targetWidth = GetEffectiveEquirectWidth( settings, width );
                uint32_t const targetHeight = ( targetWidth > 1 ) ? ( targetWidth / 2 ) : 1;

                std::vector<float> reduced;

                if ( !DownsampleEquirect( equirect.data(), width, height, targetWidth, targetHeight, reduced ) )
                {
                    std::printf( "  %-30s %-12s %-12s %-11s %-9s %-8s %-6s\n", asset.m_id.c_str(), "-", "-", "-", "-", "FAIL", "resize" );
                    std::fflush( stdout );

                    ++numFailed;
                    continue;
                }

                levels.resize( 1 );
                ResampleEquirectToMap< TMap >( reduced.data(), targetWidth, targetHeight, 3,
                                        settings.m_ingest.m_baseWidth, levels[0] );

                if ( !WriteStageEXR( GetStageDirectory( settings.m_outputRoot, config ), levels, cacheMessage )
                 || !WriteStageManifest( GetStageDirectory( settings.m_outputRoot, config ), config, cacheMessage ) )
                {
                    std::printf( "  %-30s %-12s %-12s %-11s %-9s %-8s %-6s\n", asset.m_id.c_str(), "-", "-", "-", "-", "FAIL", "write" );
                    std::printf( "  %-30s %s\n", "", cacheMessage.c_str() );
                    std::fflush( stdout );

                    ++numFailed;
                    continue;
                }

                loaded = true;
                state = "ingested";
                ++numIngested;
            }

            seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - startTime ).count();
            totalSeconds += seconds;

            double meanLuminance = 0.0;
            double numNonFinite = 0.0;
            GetImageStatistics< TMap >( levels[0], meanLuminance, numNonFinite );

            char sourceText[32] = {};
            if ( sourceWidth != 0 )
            {
                std::snprintf( sourceText, sizeof( sourceText ), "%ux%u", sourceWidth, sourceHeight );
            }
            else
            {
                std::snprintf( sourceText, sizeof( sourceText ), "cached" );
            }

            // The cached path never decoded, so the source fields are unknown and
            // the line says so rather than reporting a zero
            char equirectText[32] = {};
            std::snprintf( equirectText, sizeof( equirectText ), "%u", config.m_equirectWidth );

            std::printf
            (
                "  %-30s %-12s %-12s %-11.5f %-9.2f %-8s %-6s\n",
                asset.m_id.c_str(), sourceText, equirectText,
                meanLuminance, seconds, state.c_str(),
                ( numNonFinite > 0.0 ) ? "warn" : "ok"
            );

            if ( numNonFinite > 0.0 )
            {
                std::printf( "  %-30s %.4f%% of cube texels are not finite\n", "", numNonFinite * 100.0 );
            }

            std::fflush( stdout );
        }

        std::printf( "\n  %u ingested, %u cached, %u failed, %.1f s total\n", numIngested, numCached, numFailed, totalSeconds );

        std::fflush( stdout );

        return ( numFailed == 0 ) ? 0 : 1;
    }

    //  Validation
    //-------------------------------------------------------------------------

    namespace
    {
        struct ValidationCase
        {
            std::string         m_name;
            CoefficientTable    m_table;
            LevelWidthCurve     m_curve;

            // The paper's own tables are scored under the paper's curve whatever the run selected, so they are the ground truth a firefly claim is about.
            // Everything else here was fitted for this run.
            bool                m_isPublished = false;
        };

        //---------------------------------------------------------------------

        HDRIStageConfig MakeConvolutionConfig
        (
            HDRIRunSettings const& settings,
            std::string const& assetId,
            char const* pStage,
            char const* pTableName,
            LevelWidthCurve const& curve
        )
        {
            HDRIStageConfig config;

            config.m_stage = pStage;
            config.m_sourceId = assetId;
            config.m_profileName = "ggx";
            config.m_curveName = curve.GetName();
            config.m_tableName = pTableName;
            config.m_projectionName = GetProbeMapName( settings.m_map );
            config.m_baseWidth = settings.m_ingest.m_baseWidth;
            config.m_numLevels = CoefficientTable::NumLevels;
            config.m_numSlices = GetProbeMapSliceCount( settings.m_map );

            // The reference is an estimator, so its sample count is part of the answer and part of the key. 
            // A table's convolution has no samples.
            config.m_samples = ( std::strcmp( pStage, "reference" ) == 0 ) ? settings.m_samples : 0;

            config.m_levelAlpha.reserve( curve.GetNumLevels() );
            for ( uint32_t level = 0; level < curve.GetNumLevels(); ++level )
            {
                config.m_levelAlpha.push_back( static_cast<float>( curve.GetWidth( level ) ) );
            }

            return config;
        }

        //---------------------------------------------------------------------

        bool LoadFittedTable
        (
            char const* pCheckpointPath,
            EvaluationSettings const& evaluation,
            TableShape const& shape,
            LevelWidthCurve const& curve,
            CoefficientTable& table,
            std::string& message
        )
        {
            message.clear();

            ProfileGGX const profile( curve, LobeConvention::NDFCosineHemisphere );

            FitFingerprint const expected = FitFingerprint::Make( evaluation, shape, profile, "" );

            FitCheckpoint checkpoint;

            // Configuration rather than Exact: the seed decides how a table was reached, not what it means, and a validation run does not repeat the seeding flags.
            if ( checkpoint.Load( pCheckpointPath, expected, FitCheckpoint::FingerprintCheck::Configuration ) != FitCheckpoint::LoadResult::Loaded )
            {
                message = checkpoint.m_message;
                return false;
            }

            if ( !checkpoint.IsComplete() )
            {
                message = "checkpoint is not complete";
                return false;
            }

            // The shape comes out of the checkpoint rather than being repeated here, so a tetrahedral fit's table is rebuilt with its own axes
            table.Create( checkpoint.m_fingerprint.GetShape() );

            for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
            {
                table.SetLevelParameters( level, checkpoint.m_levels[level].m_parameters );
            }

            return true;
        }

        //---------------------------------------------------------------------

        template< typename TMap >
        bool BuildReferenceChain
        (
            HDRIRunSettings const& settings,
            std::string const& assetId,
            LevelWidthCurve const& curve,
            std::vector<HDRIImage> const& sourceChain,
            std::vector<HDRIImage>& reference,
            bool& wasCached,
            std::string& message
        )
        {
            message.clear();

            HDRIStageConfig const config = MakeConvolutionConfig( settings, assetId, "reference", "", curve );

            if ( !settings.m_force && TryLoadCachedStage( settings.m_outputRoot, config, reference, message ) )
            {
                wasCached = true;
                return true;
            }

            wasCached = false;

            ProfileGGX const profile( curve, LobeConvention::NDFCosineHemisphere );

            reference.resize( CoefficientTable::NumLevels );

            for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
            {
                ConvolveReference< TMap, ProfileGGX >( sourceChain, profile, level, settings.m_samples, reference[level] );
            }

            std::string const directory = GetStageDirectory( settings.m_outputRoot, config );

            if ( !WriteStageEXR( directory, reference, message ) || !WriteStageManifest( directory, config, message ) )
            {
                return false;
            }

            return true;
        }

        //---------------------------------------------------------------------

        template< typename TMap >
        bool BuildFastChain
        (
            HDRIRunSettings const& settings,
            std::string const& assetId,
            ValidationCase const& validationCase,
            std::vector<HDRIImage> const& sourceChain,
            std::vector<HDRIImage>& fast,
            bool& wasCached,
            std::string& message
        )
        {
            message.clear();

            HDRIStageConfig const config = MakeConvolutionConfig( settings, assetId, "fast", validationCase.m_name.c_str(), validationCase.m_curve );

            if ( !settings.m_force && TryLoadCachedStage( settings.m_outputRoot, config, fast, message ) )
            {
                wasCached = true;
                return true;
            }

            wasCached = false;

            fast.resize( CoefficientTable::NumLevels );

            for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
            {
                ConvolveTable< TMap >( sourceChain, validationCase.m_table, level, settings.m_ingest.m_baseWidth, fast[level] );
            }

            std::string const directory = GetStageDirectory( settings.m_outputRoot, config );

            if ( !WriteStageEXR( directory, fast, message ) || !WriteStageManifest( directory, config, message ) )
            {
                return false;
            }

            return true;
        }

        // Showcase images
        //---------------------------------------------------------------------
        // For looking at rather than for measuring, and chosen from the numbers the run measured, so a picture cannot disagree with the table it came from. 
        // Each one reloads the stages it needs, which are the ones the run cached, so it costs a composition and not a convolution.
        //---------------------------------------------------------------------

        // One ( asset, table, level ) line of the run's results, which is also the input to the selection below
        struct ErrorRecord
        {
            std::string m_asset;
            std::string m_table;
            uint32_t    m_level = 0;
            double      m_l1 = 0.0;
            double      m_rms = 0.0;
            double      m_maxAbsolute = 0.0;
            double      m_referencePeak = 0.0;
            double      m_approximatePeak = 0.0;
        };

        bool HasRecords( std::vector<ErrorRecord> const& records, std::string const& assetId )
        {
            for ( ErrorRecord const& record : records )
            {
                if ( record.m_asset == assetId )
                {
                    return true;
                }
            }

            return false;
        }

        // Mean relative L1 over the levels the published tables are scored on, which is the number every table in a run is ranked by
        double GetRecordedMeanL1( std::vector<ErrorRecord> const& records, std::string const& assetId, std::string const& tableName )
        {
            double   sum = 0.0;
            uint32_t count = 0;

            for ( ErrorRecord const& record : records )
            {
                if ( ( record.m_asset == assetId ) && ( record.m_table == tableName ) && ( record.m_level > 0 ) )
                {
                    sum += record.m_l1;
                    ++count;
                }
            }

            return ( count > 0 ) ? ( sum / static_cast<double>( count ) ) : 0.0;
        }

        // The brightest texel of the approximation over the brightest texel of the reference, at one level. 
        // Above 1 is a firefly: the table has put more light in one texel than the reference ever had there.
        double GetRecordedPeakRatio( std::vector<ErrorRecord> const& records, std::string const& assetId, std::string const& tableName, uint32_t level )
        {
            for ( ErrorRecord const& record : records )
            {
                if ( ( record.m_asset == assetId ) && ( record.m_table == tableName ) && ( record.m_level == level ) )
                {
                    return ( record.m_referencePeak > 0.0 ) ? ( record.m_approximatePeak / record.m_referencePeak ) : 0.0;
                }
            }

            return 0.0;
        }

        //---------------------------------------------------------------------

        template< typename TMap >
        bool ComposeShowcaseFor
        (
            HDRIRunSettings const& settings,
            std::string const& assetId,
            ValidationCase const& validationCase,
            ShowcaseImage& image,
            std::string& message
        )
        {
            message.clear();

            HDRIStageConfig const sourceConfig = MakeSourceConfig( settings, assetId );

            std::vector<HDRIImage> sourceLevels;

            if ( !TryLoadCachedStage( settings.m_outputRoot, sourceConfig, sourceLevels, message ) )
            {
                return false;
            }

            std::vector<HDRIImage> sourceChain;
            BuildSourceChain< TMap >( sourceLevels[0], HDRIImage::MaxLevels, settings.m_weighting, sourceChain );

            std::vector<HDRIImage> reference;
            std::vector<HDRIImage> fast;
            bool referenceCached = false;
            bool fastCached = false;

            if ( !BuildReferenceChain< TMap >( settings, assetId, validationCase.m_curve, sourceChain, reference, referenceCached, message ) )
            {
                return false;
            }

            if ( !BuildFastChain< TMap >( settings, assetId, validationCase, sourceChain, fast, fastCached, message ) )
            {
                return false;
            }

            ShowcaseSource showcase;
            showcase.m_pSource = &sourceLevels[0];
            showcase.m_pReference = &reference;
            showcase.m_pFast = &fast;
            showcase.m_pNaive = &sourceChain;

            ComposeShowcaseImage< TMap >( showcase, image );

            return true;
        }

        //---------------------------------------------------------------------

        template< typename TMap >
        bool WriteShowcase
        (
            HDRIRunSettings const& settings,
            std::string const& directory,
            char const* pFileName,
            HDRIAsset const& asset,
            ValidationCase const& validationCase,
            char const* pReason
        )
        {
            ShowcaseImage image;
            std::string   message;

            if ( !ComposeShowcaseFor< TMap >( settings, asset.m_id, validationCase, image, message ) )
            {
                std::printf( "    %-26s FAIL %s\n", pFileName, message.c_str() );
                return false;
            }

            std::string const path = directory + "/" + pFileName;

            if ( !image.Save( path, message ) )
            {
                std::printf( "    %-26s FAIL %s\n", pFileName, message.c_str() );
                return false;
            }

            // Read back what was written, so a showcase that a viewer would show as a plausible picture of the wrong thing is a failure here instead
            if ( !image.Verify( path, message ) )
            {
                std::printf( "    %-26s FAIL %s\n", pFileName, message.c_str() );
                return false;
            }

            std::printf
            (
                "    %-26s %ux%u  %s  %s  %s\n",
                pFileName, image.GetWidth(), image.GetHeight(),
                asset.m_id.c_str(), validationCase.m_name.c_str(), pReason
            );

            return true;
        }

        // Which environments to show, and why
        //---------------------------------------------------------------------
        // The cube layout answers three questions: which environment this curve filters best, which one makes the paper's own tables firefly most, and what this curve does to that same environment.
        // The single-slice layout answers one: that the map works, on the first environment that has a result.
        //---------------------------------------------------------------------

        template< typename TMap >
        void EmitShowcases
        (
            HDRIRunSettings const& settings,
            std::vector<HDRIAsset const*> const& candidates,
            std::vector<ValidationCase> const& cases,
            std::vector<ErrorRecord> const& records,
            uint32_t numToValidate
        )
        {
            if ( records.empty() || cases.empty() )
            {
                return;
            }

            std::string directory = settings.m_showcaseDirectory;

            if ( directory.empty() )
            {
                // Beside the CSV when there is one, because that is the file the numbers in the picture come from
                if ( settings.pCsvPath != nullptr )
                {
                    std::filesystem::path const csv( settings.pCsvPath );
                    directory = csv.has_parent_path() ? csv.parent_path().string() : std::string( "." );
                }
                else
                {
                    directory = settings.m_outputRoot;
                }
            }

            std::error_code errorCode;
            std::filesystem::create_directories( directory, errorCode );

            int32_t fittedCase = -1;

            for ( uint32_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex )
            {
                if ( !cases[caseIndex].m_isPublished )
                {
                    fittedCase = static_cast<int32_t>( caseIndex );
                }
            }

            // The first candidate the run actually produced a result for
            HDRIAsset const* pFirstValidated = nullptr;

            for ( uint32_t assetIndex = 0; assetIndex < numToValidate; ++assetIndex )
            {
                if ( HasRecords( records, candidates[assetIndex]->m_id ) )
                {
                    pFirstValidated = candidates[assetIndex];
                    break;
                }
            }

            std::printf( "\n  showcase\n" );
            std::printf( "    directory   %s\n", directory.c_str() );

            if ( TMap::NumSlices == 1 )
            {
                std::printf( "    layout      row 0 the source, row 1 brute force, row 2 the table's, row 3 the\n" );
                std::printf( "                naive mip chain, one level per column, every level point upsampled\n" );

                if ( ( fittedCase >= 0 ) && ( pFirstValidated != nullptr ) )
                {
                    WriteShowcase< TMap >( settings, directory, "tetrahedral_esoterica.exr", *pFirstValidated, cases[fittedCase], "first environment with a result" );
                }
                else
                {
                    std::printf( "    tetrahedral_esoterica.exr  skipped: no fitted table for this curve\n" );
                }

                std::fflush( stdout );

                return;
            }

            std::printf( "    layout      row 0 the source, rows 1..%u brute force then the table's then the\n", CoefficientTable::NumLevels );
            std::printf( "                naive mip chain, the map's slices across each block, every level point\n" );
            std::printf( "                upsampled to the source\n" );

            if ( fittedCase >= 0 )
            {
                // Best: the smallest mean relative L1 on the table this run fitted
                HDRIAsset const* pBest = nullptr;
                double           bestMeanL1 = 0.0;

                for ( uint32_t assetIndex = 0; assetIndex < numToValidate; ++assetIndex )
                {
                    HDRIAsset const* pAsset = candidates[assetIndex];

                    if ( !HasRecords( records, pAsset->m_id ) )
                    {
                        continue;
                    }

                    double const meanL1 = GetRecordedMeanL1( records, pAsset->m_id, cases[fittedCase].m_name );

                    if ( ( pBest == nullptr ) || ( meanL1 < bestMeanL1 ) )
                    {
                        pBest = pAsset;
                        bestMeanL1 = meanL1;
                    }
                }

                if ( pBest != nullptr )
                {
                    char reason[96] = {};
                    std::snprintf( reason, sizeof( reason ), "lowest mean L1 1-6, %.4f", bestMeanL1 );

                    WriteShowcase< TMap >( settings, directory, "esoterica_best.exr", *pBest, cases[fittedCase], reason );
                }
            }
            else
            {
                std::printf( "    esoterica_best.exr         skipped: no fitted table for this curve\n" );
            }

            // The firefly: the largest level-0 peak ratio over the paper's own tables, which are scored under the paper's curve whatever this run selected. 
            // The same environment is then shown under this run's curve.
            HDRIAsset const* pFireflyAsset = nullptr;
            int32_t          fireflyCase = -1;
            double           worstPeakRatio = 0.0;

            for ( uint32_t assetIndex = 0; assetIndex < numToValidate; ++assetIndex )
            {
                HDRIAsset const* pAsset = candidates[assetIndex];

                if ( !HasRecords( records, pAsset->m_id ) )
                {
                    continue;
                }

                for ( uint32_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex )
                {
                    if ( !cases[caseIndex].m_isPublished )
                    {
                        continue;
                    }

                    double const peakRatio = GetRecordedPeakRatio( records, pAsset->m_id, cases[caseIndex].m_name, 0 );

                    if ( peakRatio > worstPeakRatio )
                    {
                        worstPeakRatio = peakRatio;
                        pFireflyAsset = pAsset;
                        fireflyCase = static_cast<int32_t>( caseIndex );
                    }
                }
            }

            if ( ( pFireflyAsset != nullptr ) && ( fireflyCase >= 0 ) )
            {
                char reason[96] = {};
                std::snprintf( reason, sizeof( reason ), "level 0 peak ratio %.3f", worstPeakRatio );

                WriteShowcase< TMap >( settings, directory, "fireflies_paper.exr", *pFireflyAsset, cases[fireflyCase], reason );

                if ( fittedCase >= 0 )
                {
                    double const fittedPeakRatio = GetRecordedPeakRatio( records, pFireflyAsset->m_id, cases[fittedCase].m_name, 0 );

                    char fittedReason[96] = {};
                    std::snprintf( fittedReason, sizeof( fittedReason ), "same environment, peak ratio %.3f", fittedPeakRatio );

                    WriteShowcase< TMap >( settings, directory, "fireflies_esoterica.exr", *pFireflyAsset, cases[fittedCase], fittedReason );
                }
                else
                {
                    std::printf( "    fireflies_esoterica.exr    skipped: no fitted table for this curve\n" );
                }
            }
            else
            {
                std::printf( "    fireflies_paper.exr        skipped: no published table in this run\n" );
            }

            std::fflush( stdout );
        }
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    static int RunHDRIValidateFor( HDRIRunSettings const& settings )
    {
        std::vector<HDRIAsset> assets;
        ScanHDRIDataset( settings.m_datasetRoot, assets );

        std::printf( "\n" );
        std::printf( "HDRI radiance validation\n" );
        std::printf( "  root        %s\n", settings.m_datasetRoot.c_str() );
        std::printf( "  cubemaps    %s\n", settings.m_outputRoot.c_str() );
        std::printf( "  projection  %s, %u slice%s per level\n", GetProbeMapName( settings.m_map ),
                     GetProbeMapSliceCount( settings.m_map ), ( GetProbeMapSliceCount( settings.m_map ) == 1 ) ? "" : "s" );
        std::printf( "  reference   %u samples per output texel\n", settings.m_samples );

        // The cases.
        // The published tables are the paper's own and are only meaningful under the paper's curve, so they are validated at the spec power the conformance test uses regardless of what the run selected, at a fixed reference sample count so their peak ratios do not move with it.
        // The fitted table is validated under the curve it was fitted for, which is the only curve its widths mean anything at.
        //
        // The four published shapes are cubemap data from Reference/, so they are cases for a cubemap run and for no other map: sampling one on a tetrahedral probe would be reading a table whose axes are not the map's.
        // A tetrahedral run has the fitted table and nothing else.
        std::vector<ValidationCase> cases;

        if ( settings.m_map == ProbeMap::Cube )
        {
            LevelWidthCurve const paperCurve = LevelWidthCurve::MakePaperGloss( CoefficientTable::NumLevels, static_cast<double>( settings.m_specPower ) );

            ReferenceTable const published[] = { ReferenceTable::Const8, ReferenceTable::Const16, ReferenceTable::Const32, ReferenceTable::Quad32 };

            for ( ReferenceTable shape : published )
            {
                ValidationCase validationCase;
                validationCase.m_table.Load( shape );
                validationCase.m_name = validationCase.m_table.GetName();
                validationCase.m_curve = paperCurve;
                validationCase.m_isPublished = true;

                cases.push_back( validationCase );
            }
        }

        bool haveFitted = false;

        if ( settings.pCheckpointPath != nullptr )
        {
            ValidationCase fitted;

            std::string message;
            if ( LoadFittedTable( settings.pCheckpointPath, settings.m_evaluation, settings.m_shape, settings.m_curve, fitted.m_table, message ) )
            {
                fitted.m_name = std::string( "fitted_" ) + settings.m_curve.GetName();
                fitted.m_curve = settings.m_curve;

                cases.push_back( fitted );
                haveFitted = true;
            }
            else
            {
                std::printf( "  fitted table: not validated - %s\n", message.c_str() );
            }
        }

        std::printf( "  tables      %u", static_cast<uint32_t>( cases.size() ) );
        for ( ValidationCase const& validationCase : cases )
        {
            std::printf( " %s", validationCase.m_name.c_str() );
        }
        std::printf( "%s\n", haveFitted ? "" : " (no fitted table for this curve)" );

        // Which assets to run.
        // Only ingested ones can be validated, and the ingest is a separate and much slower pass, so a missing cubemap is reported rather than silently skipped.
        std::vector<HDRIAsset const*> candidates;

        for ( HDRIAsset const& asset : assets )
        {
            if ( asset.m_skipReason.empty() )
            {
                candidates.push_back( &asset );
            }
        }

        uint32_t const numToValidate = ( ( settings.m_limit != 0 ) && ( settings.m_limit < candidates.size() ) )
            ? settings.m_limit
            : static_cast<uint32_t>( candidates.size() );

        std::printf( "\n  %-30s %-16s %-9s %-9s %-10s %-9s %-8s\n", "asset", "table", "L1 1-6", "RMS 1-6", "max abs", "peak x", "state" );

        // Per case and level, summed over assets, for the closing table
        std::vector<std::vector<double>> levelL1( cases.size(), std::vector<double>( CoefficientTable::NumLevels, 0.0 ) );
        std::vector<std::vector<double>> levelRms( cases.size(), std::vector<double>( CoefficientTable::NumLevels, 0.0 ) );
        std::vector<std::vector<double>> levelMeanRatio( cases.size(), std::vector<double>( CoefficientTable::NumLevels, 0.0 ) );

        // Every number, kept rather than only summarised.
        // A mean over 425 assets cannot show a firefly, and a claim that one environment is bad has to be checkable against the value it came from.
        std::vector<ErrorRecord> records;

        uint32_t numValidated = 0;
        uint32_t numMissing = 0;
        uint32_t numFailed = 0;

        std::chrono::steady_clock::time_point const startTime = std::chrono::steady_clock::now();

        for ( uint32_t assetIndex = 0; assetIndex < numToValidate; ++assetIndex )
        {
            HDRIAsset const& asset = *candidates[assetIndex];

            HDRIStageConfig const sourceConfig = MakeSourceConfig( settings, asset.m_id );

            std::vector<HDRIImage> sourceLevels;
            std::string            sourceMessage;

            if ( !TryLoadCachedStage( settings.m_outputRoot, sourceConfig, sourceLevels, sourceMessage ) )
            {
                std::printf( "  %-30s %-16s %-10s %-10s %-10s %-8s\n", asset.m_id.c_str(), "-", "-", "-", "-", "not ingested" );
                std::fflush( stdout );

                ++numMissing;
                continue;
            }

            // Pass 1 once. Both convolutions read this chain, so they cannot disagree about what the source's own levels contain.
            std::vector<HDRIImage> sourceChain;
            BuildSourceChain< TMap >( sourceLevels[0], HDRIImage::MaxLevels, settings.m_weighting, sourceChain );

            for ( uint32_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex )
            {
                ValidationCase const& validationCase = cases[caseIndex];

                std::vector<HDRIImage> reference;
                std::vector<HDRIImage> fast;
                bool referenceCached = false;
                bool fastCached = false;
                std::string message;

                if ( !BuildReferenceChain< TMap >( settings, asset.m_id, validationCase.m_curve, sourceChain, reference, referenceCached, message ) )
                {
                    std::printf( "  %-30s %-16s reference failed: %s\n", asset.m_id.c_str(), validationCase.m_name.c_str(), message.c_str() );
                    std::fflush( stdout );

                    ++numFailed;
                    continue;
                }

                if ( !BuildFastChain< TMap >( settings, asset.m_id, validationCase, sourceChain, fast, fastCached, message ) )
                {
                    std::printf( "  %-30s %-16s table failed: %s\n", asset.m_id.c_str(), validationCase.m_name.c_str(), message.c_str() );
                    std::fflush( stdout );

                    ++numFailed;
                    continue;
                }

                double sumL1 = 0.0;
                double sumRms = 0.0;
                double assetMaxAbsolute = 0.0;
                double assetPeakRatio = 0.0;

                for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
                {
                    ConvolutionError error;
                    CompareConvolutions< TMap >( reference[level], fast[level], error );

                    levelL1[caseIndex][level] += error.m_relativeL1;
                    levelRms[caseIndex][level] += error.m_relativeRms;

                    if ( error.m_referenceMean > 0.0 )
                    {
                        levelMeanRatio[caseIndex][level] += error.m_approximateMean / error.m_referenceMean;
                    }
                    else
                    {
                        levelMeanRatio[caseIndex][level] += 1.0;
                    }

                    if ( error.m_maxAbsolute > assetMaxAbsolute )
                    {
                        assetMaxAbsolute = error.m_maxAbsolute;
                    }

                    // The peak ratio is the firefly signal.
                    // Above 1 the table has put more energy in one texel than the reference ever had there; far below 1 the reference has a spike the table smoothed away.
                    if ( error.m_referencePeak > 0.0 )
                    {
                        double const peakRatio = error.m_approximatePeak / error.m_referencePeak;
                        if ( peakRatio > assetPeakRatio )
                        {
                            assetPeakRatio = peakRatio;
                        }
                    }

                    records.push_back
                    (
                        {
                            asset.m_id, validationCase.m_name, level,
                            error.m_relativeL1, error.m_relativeRms, error.m_maxAbsolute,
                            error.m_referencePeak, error.m_approximatePeak
                        }
                    );

                    if ( level > 0 )
                    {
                        sumL1 += error.m_relativeL1;
                        sumRms += error.m_relativeRms;
                    }
                }

                double const inverseSix = 1.0 / static_cast<double>( CoefficientTable::NumLevels - 1 );

                char stateText[32] = {};
                std::snprintf
                (
                    stateText, sizeof( stateText ), "%s/%s",
                    referenceCached ? "ref-cache" : "ref-new",
                    fastCached ? "tab-cache" : "tab-new"
                );

                std::printf
                (
                    "  %-30s %-16s %-9.4f %-9.4f %-10.4g %-9.4f %-8s\n",
                    asset.m_id.c_str(), validationCase.m_name.c_str(),
                    sumL1 * inverseSix, sumRms * inverseSix,
                    assetMaxAbsolute, assetPeakRatio,
                    stateText
                );

                std::fflush( stdout );
            }

            ++numValidated;
        }

        double const seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - startTime ).count();

        // Outliers
        //
        // The mean is reported, but the reason this ran is the tail.
        // A firefly is a failure of the mean: one texel carrying an error of order the peak  moves a 98304-texel L1 by nothing, so the ranking here is by the largest single-texel error relative to the reference's own peak, not by L1.
        //
        // Two other columns matter for reading a bad environment. A high peak ratio means the table put energy where the reference had none, which is the technique's own artifact.
        // A high reference peak against a high approximate peak with a low ratio means both agree on a very bright source, i.e. the environment is extreme and the filter handled it.
        //-------------------------------------------------------------------------

        if ( !records.empty() )
        {
            struct AssetOutlier
            {
                std::string m_asset;
                std::string m_table;
                uint32_t    m_level = 0;
                double      m_maxAbsolute = 0.0;
                double      m_referencePeak = 0.0;
                double      m_peakRatio = 0.0;
                double      m_l1 = 0.0;
            };

            std::vector<AssetOutlier> worstPeak;
            std::vector<AssetOutlier> worstAbsolute;

            for ( ErrorRecord const& record : records )
            {
                AssetOutlier outlier;
                outlier.m_asset = record.m_asset;
                outlier.m_table = record.m_table;
                outlier.m_level = record.m_level;
                outlier.m_maxAbsolute = record.m_maxAbsolute;
                outlier.m_referencePeak = record.m_referencePeak;
                outlier.m_l1 = record.m_l1;

                // Guarded: a level whose reference is black everywhere has no peak to divide by, and an infinite ratio would top every table.
                outlier.m_peakRatio = ( record.m_referencePeak > 0.0 )
                    ? ( record.m_approximatePeak / record.m_referencePeak )
                    : 0.0;

                worstPeak.push_back( outlier );
                worstAbsolute.push_back( outlier );
            }

            std::sort( worstPeak.begin(), worstPeak.end(), [] ( AssetOutlier const& left, AssetOutlier const& right ) { return left.m_peakRatio > right.m_peakRatio; } );

            std::sort( worstAbsolute.begin(), worstAbsolute.end(), [] ( AssetOutlier const& left, AssetOutlier const& right ) { return left.m_maxAbsolute > right.m_maxAbsolute; } );

            std::printf( "\n  worst peak ratio - the table putting energy the reference did not\n" );
            std::printf( "  %-30s %-16s %-6s %-11s %-13s %-11s\n", "asset", "table", "level", "peak ratio", "ref peak", "L1" );

            for ( uint32_t index = 0; ( index < 15 ) && ( index < worstPeak.size() ); ++index )
            {
                AssetOutlier const& outlier = worstPeak[index];

                std::printf
                (
                    "  %-30s %-16s %-6u %-11.3f %-13.4g %-11.4f\n",
                    outlier.m_asset.c_str(), outlier.m_table.c_str(), outlier.m_level,
                    outlier.m_peakRatio, outlier.m_referencePeak, outlier.m_l1
                );
            }

            std::printf( "\n  worst single-texel absolute error, in linear radiance\n" );
            std::printf( "  %-30s %-16s %-6s %-11s %-13s %-11s\n", "asset", "table", "level", "max abs", "ref peak", "L1" );

            for ( uint32_t index = 0; ( index < 15 ) && ( index < worstAbsolute.size() ); ++index )
            {
                AssetOutlier const& outlier = worstAbsolute[index];

                std::printf
                (
                    "  %-30s %-16s %-6u %-11.4g %-13.4g %-11.4f\n",
                    outlier.m_asset.c_str(), outlier.m_table.c_str(), outlier.m_level,
                    outlier.m_maxAbsolute, outlier.m_referencePeak, outlier.m_l1
                );
            }

            // Per-asset worst, because one bad level in one table is a different claim from an environment that is bad for all of them
            std::printf( "\n  worst per-asset peak ratio across every table and level\n" );
            std::printf( "  %-30s %-11s %-13s\n", "asset", "peak ratio", "max abs" );

            std::vector<AssetOutlier> perAsset;

            for ( ErrorRecord const& record : records )
            {
                double const peakRatio = ( record.m_referencePeak > 0.0 )
                    ? ( record.m_approximatePeak / record.m_referencePeak )
                    : 0.0;

                bool found = false;
                for ( AssetOutlier& outlier : perAsset )
                {
                    if ( outlier.m_asset == record.m_asset )
                    {
                        if ( peakRatio > outlier.m_peakRatio )
                        {
                            outlier.m_peakRatio = peakRatio;
                        }

                        if ( record.m_maxAbsolute > outlier.m_maxAbsolute )
                        {
                            outlier.m_maxAbsolute = record.m_maxAbsolute;
                        }

                        found = true;
                        break;
                    }
                }

                if ( !found )
                {
                    AssetOutlier outlier;
                    outlier.m_asset = record.m_asset;
                    outlier.m_peakRatio = peakRatio;
                    outlier.m_maxAbsolute = record.m_maxAbsolute;

                    perAsset.push_back( outlier );
                }
            }

            std::sort( perAsset.begin(), perAsset.end(), [] ( AssetOutlier const& left, AssetOutlier const& right ) { return left.m_peakRatio > right.m_peakRatio; } );

            for ( uint32_t index = 0; ( index < 15 ) && ( index < perAsset.size() ); ++index )
            {
                std::printf( "  %-30s %-11.3f %-13.4g\n", perAsset[index].m_asset.c_str(), perAsset[index].m_peakRatio, perAsset[index].m_maxAbsolute );
            }
        }

        //-------------------------------------------------------------------------

        if ( ( settings.pCsvPath != nullptr ) && !records.empty() )
        {
            std::FILE* pFile = std::fopen( settings.pCsvPath, "wb" );

            if ( pFile == nullptr )
            {
                std::printf( "\n  cannot write %s\n", settings.pCsvPath );
            }
            else
            {
                std::fprintf( pFile, "asset,table,level,l1,rms,max_abs,ref_peak,approx_peak,peak_ratio\n" );

                for ( ErrorRecord const& record : records )
                {
                    double const peakRatio = ( record.m_referencePeak > 0.0 )
                        ? ( record.m_approximatePeak / record.m_referencePeak )
                        : 0.0;

                    std::fprintf
                    (
                        pFile, "%s,%s,%u,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        record.m_asset.c_str(), record.m_table.c_str(), record.m_level,
                        record.m_l1, record.m_rms, record.m_maxAbsolute,
                        record.m_referencePeak, record.m_approximatePeak, peakRatio
                    );
                }

                std::fclose( pFile );

                std::printf( "\n  %u rows written to %s\n", static_cast<uint32_t>( records.size() ), settings.pCsvPath );
            }
        }

        //-------------------------------------------------------------------------

        if ( numValidated > 0 )
        {
            double const inverseCount = 1.0 / static_cast<double>( numValidated );

            std::printf( "\n  relative L1 in radiance against the reference, by level\n" );
            std::printf( "  (levels 1-6 are the ones the published tables are scored on; level 0 is\n" );
            std::printf( "   the mirror under the esoterica curve and a very narrow lobe under paper)\n\n" );

            std::printf( "  %-16s", "table" );
            for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
            {
                std::printf( " %-9u", level );
            }
            std::printf( " %-9s\n", "1-6" );

            for ( uint32_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex )
            {
                std::printf( "  %-16s", cases[caseIndex].m_name.c_str() );

                double sum = 0.0;
                for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
                {
                    double const value = levelL1[caseIndex][level] * inverseCount;
                    std::printf( " %-9.4f", value );

                    if ( level > 0 )
                    {
                        sum += value;
                    }
                }

                std::printf( " %-9.4f\n", sum / static_cast<double>( CoefficientTable::NumLevels - 1 ) );
            }

            // A single normalized filter preserves the mean exactly, so this column should be flat at 1.0000 for every level.
            // A table whose weights do not sum to its own normaliser drifts here and nowhere else, and level 0 being exactly 1 is the mirror reproducing the source.
            std::printf( "\n  mean of the table's convolution over the mean of the reference\n" );
            std::printf( "  (1.0000 everywhere is a correctly normalised table)\n\n" );

            std::printf( "  %-16s", "table" );
            for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
            {
                std::printf( " %-9u", level );
            }
            std::printf( "\n" );

            for ( uint32_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex )
            {
                std::printf( "  %-16s", cases[caseIndex].m_name.c_str() );

                for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
                {
                    std::printf( " %-9.4f", levelMeanRatio[caseIndex][level] * inverseCount );
                }

                std::printf( "\n" );
            }
        }

        std::printf( "\n  %u validated, %u not ingested, %u failed, %.1f s\n", numValidated, numMissing, numFailed, seconds );

        std::fflush( stdout );

        if ( settings.m_showcase )
        {
            EmitShowcases< TMap >( settings, candidates, cases, records, numToValidate );
        }

        return ( ( numFailed == 0 ) && ( numValidated > 0 ) ) ? 0 : 1;
    }

    // One entry point, both maps behind it
    //-------------------------------------------------------------------------
    // The pipeline is instantiated per base map rather than branched inside it, so "the same sampler" is the same code and not two implementations that agree.
    // The map is chosen once, here, from the run's settings; everything below is a template parameter.
    // An invalid value cannot reach this - the CLI validates it - so the default reports rather than silently picking one.
    //-------------------------------------------------------------------------

    int RunHDRIIngest( HDRIRunSettings const& settings )
    {
        switch ( settings.m_map )
        {
            case ProbeMap::Cube:        return RunHDRIIngestFor< CubeProjection >( settings );
            case ProbeMap::Tetrahedron: return RunHDRIIngestFor< TetrahedralProjection >( settings );
        }

        std::printf( "ingest: unknown projection value\n" );

        return 1;
    }

    //-------------------------------------------------------------------------

    int RunHDRIValidate( HDRIRunSettings const& settings )
    {
        switch ( settings.m_map )
        {
            case ProbeMap::Cube:        return RunHDRIValidateFor< CubeProjection >( settings );
            case ProbeMap::Tetrahedron: return RunHDRIValidateFor< TetrahedralProjection >( settings );
        }

        std::printf( "validate: unknown projection value\n" );

        return 1;
    }
}