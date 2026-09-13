#include "Assert.h"
#include "FitCheckpoint.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    static constexpr char     g_checkpointMagic[4] = { 'F', 'F', 'I', 'T' };

    // 1 predates the seed name. 2 added it. 3 replaced the published shape's enum with the layout it stands for, so a shape the paper never published - a tetrahedral table - can be recorded at all.
    // Older files still load: their enum names a cubemap shape, which is what those fits produced.
    static constexpr uint32_t g_checkpointVersion = 3;
    static constexpr uint32_t g_oldestCheckpointVersion = 2;

    //-------------------------------------------------------------------------

    template< typename TValue >
    static bool WriteValue( std::FILE* pFile, TValue const& value )
    {
        return std::fwrite( &value, sizeof( TValue ), 1, pFile ) == 1;
    }

    template< typename TValue >
    static bool ReadValue( std::FILE* pFile, TValue& value )
    {
        return std::fread( &value, sizeof( TValue ), 1, pFile ) == 1;
    }

    //-------------------------------------------------------------------------

    char const* FitFingerprint::DescribeDifference( FitFingerprint const& other, bool includeSeed, char* pBuffer, size_t bufferSize ) const
    {
        // this is the fingerprint of the run that is starting, other is the one read back from the file.
        // Both helpers take them in that order so the message cannot end up with the two swapped.
        auto report = [&] ( char const* pField, char const* pThisRun, char const* pCheckpoint ) -> char const*
        {
            std::snprintf( pBuffer, bufferSize, "%s differs: checkpoint has %s, this run has %s", pField, pCheckpoint, pThisRun );
            return pBuffer;
        };

        auto reportNumber = [&] ( char const* pField, uint32_t thisRun, uint32_t checkpoint ) -> char const*
        {
            std::snprintf( pBuffer, bufferSize, "%s differs: checkpoint %u, this run %u", pField, checkpoint, thisRun );
            return pBuffer;
        };

        if ( m_baseResolution != other.m_baseResolution )
        {
            return reportNumber( "base resolution", m_baseResolution, other.m_baseResolution );
        }

        if ( m_sampleLevelCount != other.m_sampleLevelCount )
        {
            return reportNumber( "mip chain length", m_sampleLevelCount, other.m_sampleLevelCount );
        }

        if ( m_gridSize != other.m_gridSize )
        {
            return reportNumber( "sample grid size", m_gridSize, other.m_gridSize );
        }

        if ( m_supersampleRate != other.m_supersampleRate )
        {
            return reportNumber( "supersample rate", m_supersampleRate, other.m_supersampleRate );
        }

        if ( m_referenceSmoothing != other.m_referenceSmoothing )
        {
            return reportNumber( "reference smoothing", m_referenceSmoothing, other.m_referenceSmoothing );
        }

        if ( m_measure != other.m_measure )
        {
            return reportNumber( "error measure", m_measure, other.m_measure );
        }

        if ( m_weighting != other.m_weighting )
        {
            return reportNumber( "b-spline weighting", m_weighting, other.m_weighting );
        }

        if ( m_projection != other.m_projection )
        {
            return report
            (
                "projection",
                GetProbeMapName( static_cast<ProbeMap>( m_projection ) ),
                GetProbeMapName( static_cast<ProbeMap>( other.m_projection ) )
            );
        }

        if ( m_numTapsPerAxis != other.m_numTapsPerAxis )
        {
            return reportNumber( "taps per axis", m_numTapsPerAxis, other.m_numTapsPerAxis );
        }

        if ( m_numActiveCoefficients != other.m_numActiveCoefficients )
        {
            return reportNumber( "active coefficients", m_numActiveCoefficients, other.m_numActiveCoefficients );
        }

        // Last, because it is the weakest of the four: two shapes with the same name and the same layout are the same shape, and a name that differs while the layout agrees is a renamed table rather than a different one.
        // Reporting the name first would hide the numbers that explain it.
        if ( std::strcmp( m_shapeName, other.m_shapeName ) != 0 )
        {
            return report( "table shape", m_shapeName, other.m_shapeName );
        }

        // The seed separates two fits that would otherwise look interchangeable, so resuming has to compare it.
        // It does not change what a table is valid for, though, so reading one out to report or write compares everything but this.
        if ( includeSeed && ( std::strcmp( m_seedName, other.m_seedName ) != 0 ) )
        {
            return report( "seed", m_seedName, other.m_seedName );
        }

        if ( std::strcmp( m_profileName, other.m_profileName ) != 0 )
        {
            return report( "profile", m_profileName, other.m_profileName );
        }

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            // Bit comparison, because the roughness curve is an input and any difference at all means a different fit
            if ( std::memcmp( &m_alpha[level], &other.m_alpha[level], sizeof( double ) ) != 0 )
            {
                std::snprintf( pBuffer, bufferSize, "alpha at level %u differs: checkpoint %.17g, this run %.17g", level, other.m_alpha[level], m_alpha[level] );
                return pBuffer;
            }
        }

        return nullptr;
    }

    //-------------------------------------------------------------------------

    TableShape FitFingerprint::GetShape() const
    {
        return TableShape::ForMap( static_cast<ProbeMap>( m_projection ), m_numTapsPerAxis, m_numActiveCoefficients, m_shapeName );
    }

    //-------------------------------------------------------------------------

    bool FitFingerprint::IsShapeUsable() const
    {
        // The largest published table is 32 taps per axis, and a table's tap count is what the runtime's gather loops are built around.
        // A field beyond this is a corrupt read rather than a plan, and the table it would allocate is not one this build could use anyway.
        return IsProbeMapValue( m_projection )
            && ( m_numTapsPerAxis > 0 )
            && ( ( m_numTapsPerAxis % 4 ) == 0 )
            && ( m_numTapsPerAxis <= 256 )
            && ( ( m_numActiveCoefficients == 1 ) || ( m_numActiveCoefficients == CoefficientTable::NumCoefficients ) );
    }

    //-------------------------------------------------------------------------

    FitCheckpoint::LoadResult FitCheckpoint::Load( char const* pPath, FitFingerprint const& expected, FingerprintCheck check )
    {
        FF_ASSERT( pPath != nullptr );

        m_message[0] = '\0';

        std::FILE* pFile = std::fopen( pPath, "rb" );
        if ( pFile == nullptr )
        {
            std::snprintf( m_message, sizeof( m_message ), "no checkpoint at %s", pPath );
            return LoadResult::Absent;
        }

        char magic[4] = {};
        if ( ( std::fread( magic, 1, sizeof( magic ), pFile ) != sizeof( magic ) ) || ( std::memcmp( magic, g_checkpointMagic, sizeof( magic ) ) != 0 ) )
        {
            std::fclose( pFile );
            std::snprintf( m_message, sizeof( m_message ), "%s is not a FilterFitter checkpoint", pPath );
            return LoadResult::Corrupt;
        }

        uint32_t version = 0;
        if ( !ReadValue( pFile, version ) || ( version > g_checkpointVersion ) || ( version < g_oldestCheckpointVersion ) )
        {
            std::fclose( pFile );
            std::snprintf( m_message, sizeof( m_message ), "%s has version %u, this build reads %u through %u", pPath, version, g_oldestCheckpointVersion, g_checkpointVersion );
            return LoadResult::Corrupt;
        }

        // Read the fingerprint field by field rather than as a struct, so the file does not depend on padding
        FitFingerprint stored;
        bool readable = true;

        readable = readable && ReadValue( pFile, stored.m_baseResolution );
        readable = readable && ReadValue( pFile, stored.m_sampleLevelCount );
        readable = readable && ReadValue( pFile, stored.m_gridSize );
        readable = readable && ReadValue( pFile, stored.m_supersampleRate );
        readable = readable && ReadValue( pFile, stored.m_referenceSmoothing );
        readable = readable && ReadValue( pFile, stored.m_measure );
        readable = readable && ReadValue( pFile, stored.m_weighting );

        if ( version >= 3 )
        {
            readable = readable && ReadValue( pFile, stored.m_numTapsPerAxis );
            readable = readable && ReadValue( pFile, stored.m_numActiveCoefficients );
            readable = readable && ReadValue( pFile, stored.m_projection );
        }
        else
        {
            // Version 2 stored a published shape's enum instead of the layout it stands for.
            // Every shape that version could name is a cubemap table from Reference/, so the layout follows from the enum - and a value that is not one of those is a corrupt field, not a table.
            uint32_t publishedShape = 0;

            readable = readable && ReadValue( pFile, publishedShape );

            if ( readable && ( publishedShape > static_cast<uint32_t>( ReferenceTable::Quad32 ) ) )
            {
                std::fclose( pFile );
                std::snprintf( m_message, sizeof( m_message ), "%s records table shape %u, which is not a known shape", pPath, publishedShape );
                return LoadResult::Corrupt;
            }

            TableShape const legacy( static_cast<ReferenceTable>( publishedShape ) );

            stored.m_numTapsPerAxis = legacy.m_numTapsPerAxis;
            stored.m_numActiveCoefficients = legacy.m_numActiveCoefficients;
            stored.m_projection = static_cast<uint32_t>( legacy.m_projection );
        }

        readable = readable && ( std::fread( stored.m_alpha, sizeof( double ), CoefficientTable::NumLevels, pFile ) == CoefficientTable::NumLevels );

        readable = readable && ( std::fread( stored.m_profileName, 1, sizeof( stored.m_profileName ), pFile ) == sizeof( stored.m_profileName ) );
        readable = readable && ( std::fread( stored.m_shapeName, 1, sizeof( stored.m_shapeName ), pFile ) == sizeof( stored.m_shapeName ) );
        readable = readable && ( std::fread( stored.m_seedName, 1, sizeof( stored.m_seedName ), pFile ) == sizeof( stored.m_seedName ) );

        if ( !readable )
        {
            std::fclose( pFile );
            std::snprintf( m_message, sizeof( m_message ), "%s is truncated in its header", pPath );
            return LoadResult::Corrupt;
        }

        // Checked here rather than where the table is built: the table is allocated from these fields, and a corrupt read has to be a message rather than an abort inside the allocator's own assertion.
        if ( !stored.IsShapeUsable() )
        {
            std::fclose( pFile );
            std::snprintf( m_message, sizeof( m_message ), "%s records a table shape this build cannot use: %u taps per axis, %u coefficients, projection %u", pPath, stored.m_numTapsPerAxis, stored.m_numActiveCoefficients, stored.m_projection );
            return LoadResult::Corrupt;
        }

        // The fingerprint is checked before any level data is touched, so a refusal cannot leave a half-loaded state behind
        char difference[256] = {};
        if ( expected.DescribeDifference( stored, ( check == FingerprintCheck::Exact ), difference, sizeof( difference ) ) != nullptr )
        {
            std::fclose( pFile );
            std::snprintf( m_message, sizeof( m_message ), "%s was written by a different fit - %s", pPath, difference );
            return LoadResult::Incompatible;
        }

        // One level's unknowns are the whole parameter set for a shape, so the count stored with a level is not free to vary.
        // Checking it here also bounds the allocation below, which a corrupt count would otherwise drive to whatever it liked.
        uint32_t const levelParameterCount = GetProbeMapAxisCount( static_cast<ProbeMap>( stored.m_projection ) )
            * stored.m_numTapsPerAxis
            * CoefficientTable::NumParameters
            * stored.m_numActiveCoefficients;

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            uint8_t done = 0;
            if ( !ReadValue( pFile, done ) )
            {
                std::fclose( pFile );
                std::snprintf( m_message, sizeof( m_message ), "%s is truncated at level %u", pPath, level );
                return LoadResult::Corrupt;
            }

            m_levels[level].m_done = ( done != 0 );

            if ( m_levels[level].m_done )
            {
                uint32_t count = 0;

                if ( !ReadValue( pFile, m_levels[level].m_objective ) || !ReadValue( pFile, m_levels[level].m_iterations ) || !ReadValue( pFile, count ) )
                {
                    std::fclose( pFile );
                    std::snprintf( m_message, sizeof( m_message ), "%s is truncated at level %u", pPath, level );
                    return LoadResult::Corrupt;
                }

                if ( count != levelParameterCount )
                {
                    std::fclose( pFile );
                    std::snprintf( m_message, sizeof( m_message ), "%s stores %u parameters for level %u, and its shape has %u",
                                   pPath, count, level, levelParameterCount );
                    return LoadResult::Corrupt;
                }

                m_levels[level].m_parameters.resize( count );

                if ( ( count > 0 ) && ( std::fread( m_levels[level].m_parameters.data(), sizeof( double ), count, pFile ) != count ) )
                {
                    std::fclose( pFile );
                    std::snprintf( m_message, sizeof( m_message ), "%s is truncated in level %u's parameters", pPath, level );
                    return LoadResult::Corrupt;
                }
            }
        }

        std::fclose( pFile );

        m_fingerprint = stored;

        std::snprintf( m_message, sizeof( m_message ), "resumed from %s", pPath );
        return LoadResult::Loaded;
    }

    //-------------------------------------------------------------------------

    bool FitCheckpoint::Save( char const* pPath ) const
    {
        FF_ASSERT( pPath != nullptr );

        std::string temporaryPath = std::string( pPath ) + ".tmp";

        std::FILE* pFile = std::fopen( temporaryPath.c_str(), "wb" );
        if ( pFile == nullptr )
        {
            return false;
        }

        bool written = ( std::fwrite( g_checkpointMagic, 1, sizeof( g_checkpointMagic ), pFile ) == sizeof( g_checkpointMagic ) );

        written = written && WriteValue( pFile, g_checkpointVersion );

        written = written && WriteValue( pFile, m_fingerprint.m_baseResolution );
        written = written && WriteValue( pFile, m_fingerprint.m_sampleLevelCount );
        written = written && WriteValue( pFile, m_fingerprint.m_gridSize );
        written = written && WriteValue( pFile, m_fingerprint.m_supersampleRate );
        written = written && WriteValue( pFile, m_fingerprint.m_referenceSmoothing );
        written = written && WriteValue( pFile, m_fingerprint.m_measure );
        written = written && WriteValue( pFile, m_fingerprint.m_weighting );
        written = written && WriteValue( pFile, m_fingerprint.m_numTapsPerAxis );
        written = written && WriteValue( pFile, m_fingerprint.m_numActiveCoefficients );
        written = written && WriteValue( pFile, m_fingerprint.m_projection );

        written = written && ( std::fwrite( m_fingerprint.m_alpha, sizeof( double ), CoefficientTable::NumLevels, pFile ) == CoefficientTable::NumLevels );
        written = written && ( std::fwrite( m_fingerprint.m_profileName, 1, sizeof( m_fingerprint.m_profileName ), pFile ) == sizeof( m_fingerprint.m_profileName ) );
        written = written && ( std::fwrite( m_fingerprint.m_shapeName, 1, sizeof( m_fingerprint.m_shapeName ), pFile ) == sizeof( m_fingerprint.m_shapeName ) );
        written = written && ( std::fwrite( m_fingerprint.m_seedName, 1, sizeof( m_fingerprint.m_seedName ), pFile ) == sizeof( m_fingerprint.m_seedName ) );

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            uint8_t const done = m_levels[level].m_done ? 1 : 0;
            written = written && WriteValue( pFile, done );

            if ( m_levels[level].m_done )
            {
                uint32_t const count = static_cast<uint32_t>( m_levels[level].m_parameters.size() );

                written = written && WriteValue( pFile, m_levels[level].m_objective );
                written = written && WriteValue( pFile, m_levels[level].m_iterations );
                written = written && WriteValue( pFile, count );

                written = written && ( ( count == 0 )
                                    || ( std::fwrite( m_levels[level].m_parameters.data(), sizeof( double ), count, pFile ) == count ) );
            }
        }

        // Flush before the rename, so the rename never promotes a partial file
        written = written && ( std::fflush( pFile ) == 0 );
        written = written && ( std::fclose( pFile ) == 0 );

        if ( !written )
        {
            std::remove( temporaryPath.c_str() );
            return false;
        }

        std::error_code errorCode;
        std::filesystem::rename( temporaryPath, pPath, errorCode );

        if ( errorCode )
        {
            std::remove( temporaryPath.c_str() );
            return false;
        }

        return true;
    }

    //-------------------------------------------------------------------------

    uint32_t FitCheckpoint::CountDone() const
    {
        uint32_t count = 0;

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            if ( m_levels[level].m_done )
            {
                ++count;
            }
        }

        return count;
    }

    bool FitCheckpoint::IsComplete() const
    {
        return CountDone() == CoefficientTable::NumLevels;
    }

    size_t FitCheckpoint::CountStoredParameters() const
    {
        size_t total = 0;

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            if ( m_levels[level].m_done )
            {
                total += m_levels[level].m_parameters.size();
            }
        }

        return total;
    }

    //-------------------------------------------------------------------------

    void FitCheckpoint::PrintSummary( char const* pIndent ) const
    {
        std::printf( "%s%s, %u of %u levels done\n",
                     pIndent, m_fingerprint.m_shapeName, CountDone(), CoefficientTable::NumLevels );

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            LevelState const& state = m_levels[level];

            if ( state.m_done )
            {
                std::printf( "%s  level %u  objective %.6f  %u iterations  %zu parameters\n", pIndent, level, state.m_objective, state.m_iterations, state.m_parameters.size() );
            }
        }

        std::fflush( stdout );
    }
}
