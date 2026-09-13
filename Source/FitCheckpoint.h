#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "CoefficientTable.h"
#include "TableHarness.h"

// Resumable fit state
//-------------------------------------------------------------------------
// A fit is minutes to hours, so a run has to be resumable rather than merely restartable.
//
// The checkpoint carries a FINGERPRINT of everything that changes an answer - the sampling settings, the measure, the b-spline weighting, the table shape, and the profile's whole roughness curve - and refuses to load against a different one rather than silently mixing two fits.
// Two runs with different settings produce different tables, and averaging their levels is worse than either.
//
// The format is binary with a magic and a version. Doubles round-trip exactly, which matters because a resumed level's parameters are the input to a further descent; a text format would need enough digits to be exact anyway, and would then be more code than this.
//
// Writes are atomic: the state goes to a temporary file which is then renamed over the target, so a crash mid-write cannot destroy the last good checkpoint.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    struct FitFingerprint
    {
        uint32_t    m_baseResolution = 0;
        uint32_t    m_sampleLevelCount = 0;
        uint32_t    m_gridSize = 0;
        uint32_t    m_supersampleRate = 0;
        uint32_t    m_referenceSmoothing = 0;
        uint32_t    m_measure = 0;
        uint32_t    m_weighting = 0;

        // The table's LAYOUT, as values rather than as a published shape's enum.
        // A published shape is one of these three plus its name; a shape outside the published set - a tetrahedral table - has no enum. 
        // The axis count is not stored: it follows from the projection, so storing it would only create a second place for the two to disagree.
        uint32_t    m_numTapsPerAxis = 0;
        uint32_t    m_numActiveCoefficients = 0;
        uint32_t    m_projection = 0;

        double      m_alpha[CoefficientTable::NumLevels] = {};

        char        m_profileName[64] = {};
        char        m_shapeName[32] = {};

        // Which initialisation the fit used.
        // Two runs that differ only in the seed produce different tables, so a checkpoint must not be resumable across them - and without this the fingerprint would accept exactly that, and a seeded run would silently skip every level.
        char        m_seedName[32] = {};

        // TProfile is the profile contract in Profile.h. The whole roughness curve goes in, not the profile's name: two runs that differ only in an alpha curve are different fits.
        template< typename TProfile >
        static FitFingerprint Make( EvaluationSettings const& settings, TableShape const& shape, TProfile const& profile, char const* pSeedName )
        {
            FitFingerprint fingerprint;

            fingerprint.m_baseResolution = settings.m_baseResolution;
            fingerprint.m_sampleLevelCount = settings.m_sampleLevelCount;
            fingerprint.m_gridSize = settings.m_gridSize;
            fingerprint.m_supersampleRate = settings.m_supersampleRate;
            fingerprint.m_referenceSmoothing = settings.m_referenceSmoothing;
            fingerprint.m_measure = static_cast<uint32_t>( settings.m_measure );
            fingerprint.m_weighting = static_cast<uint32_t>( settings.m_weighting );
            fingerprint.m_numTapsPerAxis = shape.m_numTapsPerAxis;
            fingerprint.m_numActiveCoefficients = shape.m_numActiveCoefficients;
            fingerprint.m_projection = static_cast<uint32_t>( shape.m_projection );

            for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
            {
                fingerprint.m_alpha[level] = profile.GetWidth( level );
            }

            std::snprintf( fingerprint.m_profileName, sizeof( fingerprint.m_profileName ), "%s", profile.GetName() );
            std::snprintf( fingerprint.m_shapeName, sizeof( fingerprint.m_shapeName ), "%s", shape.m_name );
            std::snprintf( fingerprint.m_seedName, sizeof( fingerprint.m_seedName ), "%s", ( pSeedName != nullptr ) ? pSeedName : "" );

            return fingerprint;
        }

        // The shape these fields describe, so anything holding a checkpoint can rebuild the table it names without going through a published enum
        TableShape GetShape() const;

        // True when the three layout fields and the name could have been written by this build. 
        // A checkpoint is read from a file that may be truncated or edited, and the table it names is allocated from these fields, so they are checked where they are read rather than where they are used.
        bool IsShapeUsable() const;

        // Names the first field that differs, or returns nullptr when there is none. 
        // Compared field by field rather than as a struct: the struct has padding, and two value-initialised fingerprints need not agree on it.
        //
        // includeSeed false excludes the seed, which is what reading a finished fit to report or write it out wants.
        // The seed separates two fits for resume purposes; it does not change what a table is valid for, since two fits from different seeds are both correct tables for the same profile and curve.
        char const* DescribeDifference( FitFingerprint const& other, bool includeSeed, char* pBuffer, size_t bufferSize ) const;
    };

    //-------------------------------------------------------------------------

    class FitCheckpoint
    {
    public:

        enum class LoadResult : uint8_t
        {
            Absent,         // no file, so this is a new fit
            Loaded,
            Incompatible,   // present, valid, but a different fingerprint
            Corrupt,        // unreadable or truncated; message says why
        };

        // How much of the fingerprint has to match.
        //
        // Exact is for resuming, where the seed has to agree because continuing one fit from another's state is wrong.
        //
        // Configuration is for reading a finished fit to report or write it out.
        // The seed does not change what a table is valid for, but everything else does, and checking it is what makes a renamed or copied checkpoint fail loudly instead of being written out under an identity it does not have.
        enum class FingerprintCheck : uint8_t
        {
            Configuration,
            Exact,
        };

        struct LevelState
        {
            bool                m_done = false;
            double              m_objective = 0.0;
            uint32_t            m_iterations = 0;
            std::vector<double> m_parameters;
        };

    public:

        FitFingerprint  m_fingerprint;
        LevelState      m_levels[CoefficientTable::NumLevels];
        char            m_message[256] = {};

        // Reads pPath and checks the stored fingerprint against expected. On Incompatible or Corrupt nothing is loaded and m_message explains why.
        LoadResult Load( char const* pPath, FitFingerprint const& expected, FingerprintCheck check );

        // Atomic: temporary file, then rename over the target
        bool Save( char const* pPath ) const;

        uint32_t CountDone() const;
        bool IsComplete() const;

        void PrintSummary( char const* pIndent ) const;

        // Total parameters across the finished levels, for the summary
        size_t CountStoredParameters() const;
    };
}
