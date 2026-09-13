#include "Assert.h"
#include "TableHarness.h"

#include "BsplineRecurrence.h"
#include "CubeReferenceFrame.h"
#include "MapProjection.h"
#include "ParallelFor.h"
#include "PreimageAccumulator.h"
#include "PreimageError.h"
#include "ProfileBeckmann.h"
#include "ProfileGGX.h"
#include "ReferencePreimage.h"

#include <cmath>
#include <cstdio>
#include <numbers>

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    static double Dot3( double const* pA, double const* pB )
    {
        return ( pA[0] * pB[0] ) + ( pA[1] * pB[1] ) + ( pA[2] * pB[2] );
    }

    static void Cross3( double* pOut, double const* pA, double const* pB )
    {
        pOut[0] = ( pA[1] * pB[2] ) - ( pA[2] * pB[1] );
        pOut[1] = ( pA[2] * pB[0] ) - ( pA[0] * pB[2] );
        pOut[2] = ( pA[0] * pB[1] ) - ( pA[1] * pB[0] );
    }

    // Tap reconstruction
    //-------------------------------------------------------------------------
    // Mirrors Reference/filter_using_table_128.txt. Three details are easy to misread:
    //
    //    * the frame weight is ( max - 0.75 ) / 0.25 with no explicit saturate. The shader skips the frame when the value is <= 0, which is equivalent to saturate( 4 * max - 3 ) followed by a cull;
    //    * Nx and Ny use the signed direction components and Nz the absolute one, so the reference's "Nz <= -0.999" branch is unreachable;
    //    * each of the five per-tap parameters is a polynomial in theta^2 and phi^2 with no linear term, per the paper's continuity constraint.s
    //-------------------------------------------------------------------------

    template< typename TMap >
    void BuildTableTaps
    (
        CoefficientTable const& table,
        uint32_t level,
        uint32_t slice,
        uint32_t texelX,
        uint32_t texelY,
        uint32_t baseResolution,
        std::vector<TableTap>& taps
    )
    {
        taps.clear();

        uint32_t const resolution = baseResolution >> level;

        // Everything below is the same for every base map.
        // The map answers the six questions in MapProjection.h and the construction does not know which map it is answering them.
        double dir[3];
        TMap::GetDirection( dir, slice, texelX, texelY, resolution );

        double const dirLength = std::sqrt( Dot3( dir, dir ) );
        double const frameZ[3] = { dir[0] / dirLength, dir[1] / dirLength, dir[2] / dirLength };

        uint32_t const numSuperTaps = table.GetNumTaps() / 4;

        for ( uint32_t axis = 0; axis < TMap::NumFrames; ++axis )
        {
            double const frameWeight = TMap::GetFrameWeight( dir, axis );
            if ( frameWeight <= 0.0 )
            {
                continue;
            }

            double up[3];
            TMap::GetFrameUp( up, axis );

            double frameX[3];
            Cross3( frameX, up, frameZ );

            // The tangent frame degenerates where the direction is parallel to the pole, which is exactly where the frame weight culls the frame
            double const frameXLength = std::sqrt( Dot3( frameX, frameX ) );
            if ( frameXLength <= 0.0 )
            {
                continue;
            }

            frameX[0] /= frameXLength;
            frameX[1] /= frameXLength;
            frameX[2] /= frameXLength;

            double frameY[3];
            Cross3( frameY, frameZ, frameX );

            double theta = 0.0;
            double phi = 0.0;
            TMap::GetPolynomialArguments( dir, axis, theta, phi );

            double const theta2 = theta * theta;
            double const phi2 = phi * phi;

            for ( uint32_t superTap = 0; superTap < numSuperTaps; ++superTap )
            {
                uint32_t const index = ( numSuperTaps * axis ) + superTap;

                for ( uint32_t subTap = 0; subTap < 4; ++subTap )
                {
                    double parameter[CoefficientTable::NumParameters];
                    for ( uint32_t parameterIndex = 0; parameterIndex < CoefficientTable::NumParameters; ++parameterIndex )
                    {
                        double const constantTerm = table.GetCoefficient( level, parameterIndex, 0, index, subTap );
                        double const thetaTerm = table.GetCoefficient( level, parameterIndex, 1, index, subTap );
                        double const phiTerm = table.GetCoefficient( level, parameterIndex, 2, index, subTap );

                        parameter[parameterIndex] = constantTerm + ( thetaTerm * theta2 ) + ( phiTerm * phi2 );
                    }

                    double sampleDir[3];
                    sampleDir[0] = ( frameX[0] * parameter[0] ) + ( frameY[0] * parameter[1] ) + ( frameZ[0] * parameter[2] );
                    sampleDir[1] = ( frameX[1] * parameter[0] ) + ( frameY[1] * parameter[1] ) + ( frameZ[1] * parameter[2] );
                    sampleDir[2] = ( frameX[2] * parameter[0] ) + ( frameY[2] * parameter[1] ) + ( frameZ[2] * parameter[2] );

                    double sampleLevel = parameter[3];
                    double sampleWeight = parameter[4] * frameWeight;

                    // Jacobian adjustment, exactly as the reference does it
                    double const maxComponent = std::fmax( std::fabs( sampleDir[0] ), std::fmax( std::fabs( sampleDir[1] ), std::fabs( sampleDir[2] ) ) );
                    if ( maxComponent <= 0.0 )
                    {
                        continue;
                    }

                    sampleDir[0] /= maxComponent;
                    sampleDir[1] /= maxComponent;
                    sampleDir[2] /= maxComponent;

                    sampleLevel += TMap::GetLevelCorrection( sampleDir );

                    TableTap tap;
                    tap.m_direction[0] = sampleDir[0];
                    tap.m_direction[1] = sampleDir[1];
                    tap.m_direction[2] = sampleDir[2];
                    tap.m_level = sampleLevel;
                    tap.m_weight = sampleWeight;
                    tap.m_tapIndex = ( ( numSuperTaps * axis ) + superTap ) * 4 + subTap;
                    tap.m_frameWeight = frameWeight;
                    tap.m_theta2 = theta2;
                    tap.m_phi2 = phi2;

                    taps.push_back( tap );
                }
            }
        }
    }

    //-------------------------------------------------------------------------

    template void BuildTableTaps< CubeProjection >( CoefficientTable const&, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, std::vector<TableTap>& );

    template void BuildTableTaps< TetrahedralProjection >( CoefficientTable const&, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, std::vector<TableTap>& );

    //-------------------------------------------------------------------------

    namespace
    {
        using namespace FilterFitter;

        // One worker's scratch and partial sums for a single level.
        //
        // The reference preimage is the big allocation here: base texels * 8 bytes, per worker.
        // It is per worker rather than shared because it is overwritten for every output direction.
        template< typename TFrame >
        struct LevelWorker
        {
            ReferencePreimage< TFrame >                 m_reference;
            PreimageAccumulator< typename TFrame::Map > m_accumulator;
            std::vector<TableTap>                       m_taps;

            double                                      m_averageL1 = 0.0;
            double                                      m_averageTapWeight = 0.0;
            double                                      m_minL1 = 1.0e30;
            double                                      m_maxL1 = 0.0;
            uint32_t                                    m_numTexels = 0;
            uint32_t                                    m_numTaps = 0;

            double                                      m_bandAverageL1[TableEvaluationLevel::NumPositionBands] = {};
            uint32_t                                    m_bandNumTexels[TableEvaluationLevel::NumPositionBands] = {};
        };
    }

    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    TableEvaluationLevel EvaluatePublishedTableLevel
    (
        CoefficientTable const& table,
        TFrame const& baseFrame,
        TProfile const& profile,
        EvaluationSettings const& settings,
        uint32_t level
    )
    {
        uint32_t const resolution = settings.m_baseResolution >> level;

        // Only the face-coordinate band metric below uses this
        double const inverseResolution = 1.0 / static_cast<double>( resolution );

        // Flattened so the parallel split is over a plain index.
        // Built by the shared helper so the fit's cached reference preimages are guaranteed to be in the same order.
        std::vector<uint32_t> texels;
        BuildOutputTexelList< typename TFrame::Map >( resolution, settings.m_gridSize, texels );

        uint32_t const numItems = static_cast<uint32_t>( texels.size() / 3 );

        uint32_t const numWorkers = GetWorkerCount( numItems );

        // One accumulator, one reference preimage and one tap list per worker.
        // The reference lives on the base cube and is the largest of these, so this is the fit's dominant memory: workers * ( base texels * 8 bytes ).
        std::vector<LevelWorker< TFrame >> workers( numWorkers );

        for ( LevelWorker< TFrame >& worker : workers )
        {
            worker.m_reference.Initialize( &baseFrame, level, settings.m_supersampleRate );
            worker.m_accumulator.Initialize( settings.m_baseResolution, settings.m_sampleLevelCount, settings.m_weighting );
        }

        // Captured by reference and inlined into the worker function.
        // The body is the whole per-output-texel chain, so nothing here goes through a function pointer or a type-erased context.
        auto evaluateRange = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            LevelWorker< TFrame >& worker = workers[workerIndex];

            for ( uint32_t item = begin; item < end; ++item )
            {
                uint32_t const slice = texels[( item * 3 ) + 0];
                uint32_t const texelX = texels[( item * 3 ) + 1];
                uint32_t const texelY = texels[( item * 3 ) + 2];

                // The map's own texel -> direction, so this is not a cubemap expression wearing a template parameter.
                double outputDirection[3];
                TFrame::Map::GetDirection( outputDirection, slice, texelX, texelY, resolution );

                worker.m_reference.Evaluate( profile, outputDirection );

                for ( uint32_t smoothingPass = 0; smoothingPass < settings.m_referenceSmoothing; ++smoothingPass )
                {
                    worker.m_reference.Smooth();
                }

                worker.m_reference.Normalize( settings.m_measure );

                BuildTableTaps< typename TFrame::Map >( table, level, slice, texelX, texelY, settings.m_baseResolution, worker.m_taps );

                worker.m_accumulator.Reset();

                double totalTapWeight = 0.0;
                for ( TableTap const& tap : worker.m_taps )
                {
                    worker.m_accumulator.AddSample( tap.m_direction, tap.m_level, tap.m_weight );
                    totalTapWeight += tap.m_weight;
                }

                std::vector<double> const& approximation = worker.m_accumulator.ResolveToBase();

                PreimageComparison const comparison = ComparePreimages( baseFrame, worker.m_reference.GetValues(), approximation, totalTapWeight, settings.m_measure );

                worker.m_averageL1 += comparison.m_l1;
                worker.m_averageTapWeight += totalTapWeight;
                worker.m_minL1 = ( comparison.m_l1 < worker.m_minL1 ) ? comparison.m_l1 : worker.m_minL1;
                worker.m_maxL1 = ( comparison.m_l1 > worker.m_maxL1 ) ? comparison.m_l1 : worker.m_maxL1;
                ++worker.m_numTexels;
                worker.m_numTaps += static_cast<uint32_t>( worker.m_taps.size() );

                // The band metric is the paper's cube-face statistic: its Table 1 splits a face into centre and edge tiles by max( |u|, |v| ) of the face coordinate, which is centred on the face and runs to +-1 at its edges.
                // Only a map with one face per slice has such a coordinate, so the bands are measured there and left empty elsewhere rather than measured against something that is not the quantity the paper's table is about.
                // Nothing prints them for a map that has no published table to compare against.
                if ( TFrame::Map::NumFaces == TFrame::Map::NumSlices )
                {
                    double const u = ( ( 2.0 * static_cast<double>( texelX ) ) + 1.0 ) * inverseResolution - 1.0;
                    double const v = -( ( 2.0 * static_cast<double>( texelY ) ) + 1.0 ) * inverseResolution + 1.0;

                    uint32_t const band = GetPositionBand( u, v );
                    worker.m_bandAverageL1[band] += comparison.m_l1;
                    ++worker.m_bandNumTexels[band];
                }
            }
        };

        ParallelForRanges( numItems, evaluateRange );

        // Reduce in worker order, so the result does not depend on how the scheduler interleaved. 
        // A worker that got no items has to be skipped rather than folded in: its minimum is still the sentinel.
        TableEvaluationLevel levelResult;
        levelResult.m_minL1 = 1.0e30;

        for ( LevelWorker< TFrame > const& worker : workers )
        {
            if ( worker.m_numTexels == 0 )
            {
                continue;
            }

            levelResult.m_averageL1 += worker.m_averageL1;
            levelResult.m_averageTapWeight += worker.m_averageTapWeight;
            levelResult.m_minL1 = ( worker.m_minL1 < levelResult.m_minL1 ) ? worker.m_minL1 : levelResult.m_minL1;
            levelResult.m_maxL1 = ( worker.m_maxL1 > levelResult.m_maxL1 ) ? worker.m_maxL1 : levelResult.m_maxL1;
            levelResult.m_numTexels += worker.m_numTexels;
            levelResult.m_numTaps += worker.m_numTaps;

            for ( uint32_t band = 0; band < TableEvaluationLevel::NumPositionBands; ++band )
            {
                levelResult.m_bandAverageL1[band] += worker.m_bandAverageL1[band];
                levelResult.m_bandNumTexels[band] += worker.m_bandNumTexels[band];
            }
        }

        if ( levelResult.m_numTexels > 0 )
        {
            levelResult.m_averageL1 /= levelResult.m_numTexels;
            levelResult.m_averageTapWeight /= levelResult.m_numTexels;

            for ( uint32_t band = 0; band < TableEvaluationLevel::NumPositionBands; ++band )
            {
                if ( levelResult.m_bandNumTexels[band] > 0 )
                {
                    levelResult.m_bandAverageL1[band] /= levelResult.m_bandNumTexels[band];
                }
            }
        }

        return levelResult;
    }

    //-------------------------------------------------------------------------

    template< typename TFrame, typename TProfile >
    TableEvaluation EvaluatePublishedTable
    (
        ReferenceTable table,
        TFrame const& baseFrame,
        TProfile const& profile,
        EvaluationSettings const& settings
    )
    {
        CoefficientTable coefficientTable;
        coefficientTable.Load( table );

        uint32_t const numLevels = CoefficientTable::NumLevels;

        TableEvaluation result;

        double totalL1 = 0.0;
        uint32_t totalLevels = 0;

        for ( uint32_t level = 0; level < numLevels; ++level )
        {
            TableEvaluationLevel const levelResult = EvaluatePublishedTableLevel( coefficientTable, baseFrame, profile, settings, level );

            result.m_levels[level] = levelResult;
            result.m_numTexels += levelResult.m_numTexels;

            if ( settings.m_verbose )
            {
                std::printf
                (
                    "    level %u  res %3u  texels %3u  taps/texel %.1f  L1 %.4f (min %.4f max %.4f)\n",
                    level, settings.m_baseResolution >> level, levelResult.m_numTexels,
                    ( levelResult.m_numTexels > 0 ) ? ( static_cast<double>( levelResult.m_numTaps ) / levelResult.m_numTexels ) : 0.0,
                    levelResult.m_averageL1, levelResult.m_minL1, levelResult.m_maxL1
                );
                std::fflush( stdout );
            }

            // The paper excludes level 0 from its average
            if ( level > 0 )
            {
                totalL1 += levelResult.m_averageL1;
                ++totalLevels;
            }
        }

        if ( totalLevels > 0 )
        {
            result.m_averageL1 = totalL1 / totalLevels;
        }

        return result;
    }

    //-------------------------------------------------------------------------

    template< typename TProfile >
    double GetLobeToTexelRatio( TProfile const& profile, uint32_t level, uint32_t baseResolution )
    {
        // The profile's width parameter is alpha, the GGX NDF's half-angle width, so the lobe spans about 2*alpha in the environment angle.
        //
        // The texel size is taken at the cube face centre, where it is smallest.
        // Face space maps to direction as normalize( u, v, 1 ), so at the centre d(angle)/du is 1 and a texel of face-space size 2/R subtends 2/R radians.
        // Over the whole face the mean texel subtends pi/(2R), which is 4/pi smaller, so quoting the mean would understate the ratio by 1.27x; the face centre is the worst case for resolving a lobe, which is what this number is used to judge.
        double const resolution = static_cast<double>( baseResolution >> level );
        double const lobeWidth = 2.0 * profile.GetWidth( level );
        double const texelWidth = 2.0 / resolution;

        return lobeWidth / texelWidth;
    }

    //-------------------------------------------------------------------------

    uint32_t GetPositionBand( double u, double v )
    {
        double const maxCoordinate = std::fmax( std::fabs( u ), std::fabs( v ) );

        if ( maxCoordinate < 0.5 )
        {
            return 0;
        }

        return ( maxCoordinate < 0.75 ) ? 1 : 2;
    }

    //-------------------------------------------------------------------------

    uint32_t GetFaceGridSize( uint32_t resolution, uint32_t gridSize )
    {
        return ( gridSize < resolution ) ? gridSize : resolution;
    }

    //-------------------------------------------------------------------------

    template< typename TMap >
    void BuildOutputTexelList( uint32_t resolution, uint32_t gridSize, std::vector<uint32_t>& texels )
    {
        uint32_t const faceGridSize = GetFaceGridSize( resolution, gridSize );

        texels.clear();
        texels.reserve( static_cast<size_t>( TMap::NumSlices ) * faceGridSize * faceGridSize * 3 );

        // Centre of each cell of a faceGridSize x faceGridSize grid over the slice.
        // A linear stride is not usable: a stride that is a multiple of the resolution puts every sample in the same column.
        //
        // A slice, not a face: a map whose slice holds several faces samples each of its faces through the same coordinate box, so walking the faces would visit the same texels once per face and weight them that many times.
        for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
        {
            for ( uint32_t gridY = 0; gridY < faceGridSize; ++gridY )
            {
                for ( uint32_t gridX = 0; gridX < faceGridSize; ++gridX )
                {
                    texels.push_back( slice );
                    texels.push_back( ( ( ( 2 * gridX ) + 1 ) * resolution ) / ( 2 * faceGridSize ) );
                    texels.push_back( ( ( ( 2 * gridY ) + 1 ) * resolution ) / ( 2 * faceGridSize ) );
                }
            }
        }
    }

    template void BuildOutputTexelList< CubeProjection >( uint32_t, uint32_t, std::vector<uint32_t>& );
    template void BuildOutputTexelList< TetrahedralProjection >( uint32_t, uint32_t, std::vector<uint32_t>& );

    //-------------------------------------------------------------------------

    template TableEvaluation EvaluatePublishedTable< CubeReferenceFrame, ProfileGGX >( ReferenceTable table, CubeReferenceFrame const& baseFrame, ProfileGGX const& profile, EvaluationSettings const& settings );
    template TableEvaluationLevel EvaluatePublishedTableLevel< CubeReferenceFrame, ProfileGGX >( CoefficientTable const& table, CubeReferenceFrame const& baseFrame, ProfileGGX const& profile, EvaluationSettings const& settings, uint32_t level );
    template double GetLobeToTexelRatio< ProfileGGX >( ProfileGGX const& profile, uint32_t level, uint32_t baseResolution );

    template TableEvaluation EvaluatePublishedTable< CubeReferenceFrame, ProfileBeckmann >( ReferenceTable table, CubeReferenceFrame const& baseFrame, ProfileBeckmann const& profile, EvaluationSettings const& settings );
    template TableEvaluationLevel EvaluatePublishedTableLevel< CubeReferenceFrame, ProfileBeckmann >( CoefficientTable const& table, CubeReferenceFrame const& baseFrame, ProfileBeckmann const& profile, EvaluationSettings const& settings, uint32_t level );
    template double GetLobeToTexelRatio< ProfileBeckmann >( ProfileBeckmann const& profile, uint32_t level, uint32_t baseResolution );
}
