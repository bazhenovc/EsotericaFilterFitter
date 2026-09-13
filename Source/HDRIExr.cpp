#include "Assert.h"
#include "HDRIExr.h"

#include "ParallelFor.h"
#include "TinyEXR/tinyexr.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace FilterFitter
{
    namespace
    {
        //---------------------------------------------------------------------

        void AppendNumber( std::string& text, uint32_t value, bool leadingDot )
        {
            if ( leadingDot )
            {
                text += '.';
            }

            char buffer[32] = {};
            std::snprintf( buffer, sizeof( buffer ), "%u", value );
            text += buffer;
        }

        void AppendFloat( std::string& text, float value )
        {
            // %.9g round-trips a float exactly, which a cache key needs: two widths that differ in the last bit are different filters
            char buffer[64] = {};
            std::snprintf( buffer, sizeof( buffer ), "%.9g", static_cast<double>( value ) );
            text += buffer;
        }

        //  Stage key
        //
        //  Readable and ordered so the tree sorts sensibly, and complete enough that a directory cannot be mistaken for a different configuration.
        //---------------------------------------------------------------------

        // A name that is safe as one path component. 
        // The curve name is the only one that can carry punctuation - the paper curve is called "paper(18)" - and parentheses in a directory name make the tree awkward to use from a shell for no benefit. 
        // Anything outside [A-Za-z0-9_.-] is dropped.
        std::string SanitizeKeyPart( std::string const& value )
        {
            std::string sanitized;
            sanitized.reserve( value.size() );

            for ( char character : value )
            {
                bool const isLetter = ( ( character >= 'a' ) && ( character <= 'z' ) ) || ( ( character >= 'A' ) && ( character <= 'Z' ) );
                bool const isDigit = ( character >= '0' ) && ( character <= '9' );
                bool const isSafe = ( character == '_' ) || ( character == '.' ) || ( character == '-' );

                if ( isLetter || isDigit || isSafe )
                {
                    sanitized.push_back( character );
                }
            }

            return sanitized;
        }

        //---------------------------------------------------------------------

        std::string GetStageKey( HDRIStageConfig const& config )
        {
            std::string key = SanitizeKeyPart( config.m_stage );

            if ( config.m_stage == "source" )
            {
                key += ".e";
                AppendNumber( key, config.m_equirectWidth, false );
                key += ".b";
                AppendNumber( key, config.m_baseWidth, false );
            }
            else
            {
                if ( !config.m_profileName.empty() )
                {
                    key += '.';
                    key += SanitizeKeyPart( config.m_profileName );
                }

                if ( !config.m_tableName.empty() )
                {
                    key += '.';
                    key += SanitizeKeyPart( config.m_tableName );
                }

                if ( !config.m_curveName.empty() )
                {
                    key += '.';
                    key += SanitizeKeyPart( config.m_curveName );
                }

                if ( config.m_samples != 0 )
                {
                    key += ".s";
                    AppendNumber( key, config.m_samples, false );
                }

                key += ".b";
                AppendNumber( key, config.m_baseWidth, false );
            }

            // The projection, last and always. 
            // A chain of textures belongs to one map, and the two maps have different slices at the same resolution, so a stage computed for one is not a stage for the other.
            // Including it changes every key, which is correct rather than unfortunate: a tree written before this field existed is a cube tree that no longer matches its own key, and it is recomputed rather than reused.
            key += ".p";
            key += SanitizeKeyPart( config.m_projectionName );

            return key;
        }

        //---------------------------------------------------------------------

        std::string GetSlicePath( std::string const& directory, uint32_t level, uint32_t slice )
        {
            char buffer[64] = {};
            std::snprintf( buffer, sizeof( buffer ), "/L%u/S%u.exr", level, slice );

            return directory + buffer;
        }

        std::string GetLevelDirectory( std::string const& directory, uint32_t level )
        {
            char buffer[32] = {};
            std::snprintf( buffer, sizeof( buffer ), "/L%u", level );

            return directory + buffer;
        }

        //---------------------------------------------------------------------

        bool SaveFaceExr( std::string const& path, float const* pRgb, uint32_t resolution, std::string& message )
        {
            char const* pError = nullptr;

            // 32-bit float, matching what the pipeline holds.
            // Half would halve the cache and put a rounding step inside the thing being validated.
            int const result = SaveEXR( pRgb, static_cast<int>( resolution ), static_cast<int>( resolution ), 3, 0, path.c_str(), &pError );

            if ( result == TINYEXR_SUCCESS )
            {
                return true;
            }

            message = "cannot write " + path;
            if ( pError != nullptr )
            {
                message += ": ";
                message += pError;
                FreeEXRErrorMessage( pError );
            }

            return false;
        }

        //---------------------------------------------------------------------

        bool LoadFaceEXR( std::string const& path, uint32_t expectedResolution, float* pRgb, std::string& message )
        {
            float* pRgba = nullptr;
            int    width = 0;
            int    height = 0;

            char const* pError = nullptr;
            int const result = LoadEXR( &pRgba, &width, &height, path.c_str(), &pError );

            if ( result != TINYEXR_SUCCESS )
            {
                message = "cannot read " + path;
                if ( pError != nullptr )
                {
                    message += ": ";
                    message += pError;
                    FreeEXRErrorMessage( pError );
                }

                if ( pRgba != nullptr )
                {
                    std::free( pRgba );
                }

                return false;
            }

            if ( ( pRgba == nullptr ) || ( width != static_cast<int>( expectedResolution ) ) || ( height != static_cast<int>( expectedResolution ) ) )
            {
                char buffer[192] = {};
                std::snprintf( buffer, sizeof( buffer ), "%s is %d x %d, expected %u x %u", path.c_str(), width, height, expectedResolution, expectedResolution );
                message = buffer;

                if ( pRgba != nullptr )
                {
                    std::free( pRgba );
                }

                return false;
            }

            size_t const numTexels = static_cast<size_t>( expectedResolution ) * expectedResolution;

            for ( size_t texelIndex = 0; texelIndex < numTexels; ++texelIndex )
            {
                float const* const pSource = pRgba + ( texelIndex * 4 );
                float* const       pTarget = pRgb + ( texelIndex * 3 );

                pTarget[0] = pSource[0];
                pTarget[1] = pSource[1];
                pTarget[2] = pSource[2];
            }

            std::free( pRgba );

            return true;
        }

        //---------------------------------------------------------------------

        bool ReadWholeFile( std::string const& path, std::string& text )
        {
            std::FILE* pFile = std::fopen( path.c_str(), "rb" );
            if ( pFile == nullptr )
            {
                return false;
            }

            std::fseek( pFile, 0, SEEK_END );
            long const fileSize = std::ftell( pFile );
            std::fseek( pFile, 0, SEEK_SET );

            if ( fileSize < 0 )
            {
                std::fclose( pFile );
                return false;
            }

            text.resize( static_cast<size_t>( fileSize ) );

            size_t const read = ( fileSize > 0 ) ? std::fread( &text[0], 1, text.size(), pFile ) : 0;
            std::fclose( pFile );

            return read == text.size();
        }

        //---------------------------------------------------------------------

        bool ReadManifestLine( std::string const& text, std::string const& key, std::string& value )
        {
            std::string const prefix = key + "=";

            size_t position = 0;
            while ( position < text.size() )
            {
                size_t const lineEnd = text.find( '\n', position );
                size_t const end = ( lineEnd == std::string::npos ) ? text.size() : lineEnd;

                if ( ( end - position ) >= prefix.size() )
                {
                    if ( text.compare( position, prefix.size(), prefix ) == 0 )
                    {
                        size_t valueEnd = end;
                        if ( ( valueEnd > position ) && ( text[valueEnd - 1] == '\r' ) )
                        {
                            --valueEnd;
                        }

                        value.assign( text, position + prefix.size(), valueEnd - position - prefix.size() );

                        return true;
                    }
                }

                if ( lineEnd == std::string::npos )
                {
                    break;
                }

                position = lineEnd + 1;
            }

            return false;
        }

        //---------------------------------------------------------------------

        uint32_t ParseU32( std::string const& text, uint32_t fallback )
        {
            if ( text.empty() )
            {
                return fallback;
            }

            return static_cast<uint32_t>( std::strtoul( text.c_str(), nullptr, 10 ) );
        }

        //---------------------------------------------------------------------

        void ParseAlphaList( std::string const& text, std::vector<float>& alpha )
        {
            alpha.clear();

            char const* pCursor = text.c_str();

            while ( *pCursor != '\0' )
            {
                char* pEnd = nullptr;
                double const value = std::strtod( pCursor, &pEnd );

                if ( pEnd == pCursor )
                {
                    break;
                }

                alpha.push_back( static_cast<float>( value ) );
                pCursor = pEnd;
            }
        }
    }

    //-------------------------------------------------------------------------

    std::string GetStageDirectory( std::string const& root, HDRIStageConfig const& config )
    {
        return root + "/" + config.m_sourceId + "/" + GetStageKey( config );
    }

    //-------------------------------------------------------------------------

    bool WriteStageManifest( std::string const& directory, HDRIStageConfig const& config, std::string& message )
    {
        message.clear();

        std::string text;
        text += "stage=" + config.m_stage + "\n";
        text += "source=" + config.m_sourceId + "\n";
        text += "projection=" + config.m_projectionName + "\n";
        text += "profile=" + config.m_profileName + "\n";
        text += "curve=" + config.m_curveName + "\n";
        text += "table=" + config.m_tableName + "\n";

        char buffer[64] = {};
        std::snprintf( buffer, sizeof( buffer ), "%u", config.m_baseWidth );
        text += "baseWidth=" + std::string( buffer ) + "\n";

        std::snprintf( buffer, sizeof( buffer ), "%u", config.m_numLevels );
        text += "numLevels=" + std::string( buffer ) + "\n";

        std::snprintf( buffer, sizeof( buffer ), "%u", config.m_numSlices );
        text += "numSlices=" + std::string( buffer ) + "\n";

        std::snprintf( buffer, sizeof( buffer ), "%u", config.m_samples );
        text += "samples=" + std::string( buffer ) + "\n";

        std::snprintf( buffer, sizeof( buffer ), "%u", config.m_equirectWidth );
        text += "equirectWidth=" + std::string( buffer ) + "\n";

        text += "levelAlpha=";
        for ( size_t index = 0; index < config.m_levelAlpha.size(); ++index )
        {
            if ( index != 0 )
            {
                text += ' ';
            }

            AppendFloat( text, config.m_levelAlpha[index] );
        }

        text += "\n";

        std::string const path = directory + "/manifest.txt";

        std::FILE* pFile = std::fopen( path.c_str(), "wb" );
        if ( pFile == nullptr )
        {
            message = "cannot write " + path;
            return false;
        }

        size_t const written = std::fwrite( text.data(), 1, text.size(), pFile );
        bool const flushed = ( std::fflush( pFile ) == 0 );
        bool const closed = ( std::fclose( pFile ) == 0 );

        if ( ( written != text.size() ) || !flushed || !closed )
        {
            message = "short write to " + path;
            return false;
        }

        return true;
    }

    //-------------------------------------------------------------------------

    bool ReadStageManifest( std::string const& directory, HDRIStageConfig& config, std::string& message )
    {
        message.clear();
        config = HDRIStageConfig();

        std::string text;
        if ( !ReadWholeFile( directory + "/manifest.txt", text ) )
        {
            return false;
        }

        ReadManifestLine( text, "stage", config.m_stage );
        ReadManifestLine( text, "source", config.m_sourceId );
        ReadManifestLine( text, "projection", config.m_projectionName );
        ReadManifestLine( text, "profile", config.m_profileName );
        ReadManifestLine( text, "curve", config.m_curveName );
        ReadManifestLine( text, "table", config.m_tableName );

        std::string value;

        if ( ReadManifestLine( text, "baseWidth", value ) )
        {
            config.m_baseWidth = ParseU32( value, 0 );
        }

        if ( ReadManifestLine( text, "numLevels", value ) )
        {
            config.m_numLevels = ParseU32( value, 0 );
        }

        if ( ReadManifestLine( text, "numSlices", value ) )
        {
            config.m_numSlices = ParseU32( value, 0 );
        }

        if ( ReadManifestLine( text, "samples", value ) )
        {
            config.m_samples = ParseU32( value, 0 );
        }

        if ( ReadManifestLine( text, "equirectWidth", value ) )
        {
            config.m_equirectWidth = ParseU32( value, 0 );
        }

        if ( ReadManifestLine( text, "levelAlpha", value ) )
        {
            ParseAlphaList( value, config.m_levelAlpha );
        }

        return true;
    }

    //-------------------------------------------------------------------------

    bool WriteStageEXR( std::string const& directory, std::vector<HDRIImage> const& levels, std::string& message )
    {
        message.clear();

        FF_ASSERT( !levels.empty() );

        std::error_code errorCode;

        for ( uint32_t level = 0; level < levels.size(); ++level )
        {
            std::filesystem::create_directories( GetLevelDirectory( directory, level ), errorCode );
        }

        // A slice, not a face: a level is one file per texture, and a single-slice map puts all four of its faces in one.
        // The path is still named by the slice index, which for a cubemap is the face, so the cache a cubemap wrote is read back unchanged.
        struct SliceTask
        {
            uint32_t    m_level;
            uint32_t    m_slice;
        };

        uint32_t const numSlices = levels[0].GetNumSlices();

        std::vector<SliceTask> tasks;
        tasks.reserve( levels.size() * numSlices );

        for ( uint32_t level = 0; level < levels.size(); ++level )
        {
            FF_ASSERT( levels[level].GetNumSlices() == numSlices );

            for ( uint32_t slice = 0; slice < numSlices; ++slice )
            {
                tasks.push_back( { level, slice } );
            }
        }

        std::vector<std::string> failures( tasks.size() );

        auto writeSlices = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            (void) workerIndex;

            for ( uint32_t taskIndex = begin; taskIndex < end; ++taskIndex )
            {
                SliceTask const& task = tasks[taskIndex];

                HDRIImage const& image = levels[task.m_level];
                float const* const pSlice = image.Texel( task.m_slice, 0, 0 );

                SaveFaceExr( GetSlicePath( directory, task.m_level, task.m_slice ), pSlice, image.GetResolution(), failures[taskIndex] );
            }
        };

        ParallelForRanges( static_cast<uint32_t>( tasks.size() ), writeSlices );

        for ( uint32_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex )
        {
            if ( !failures[taskIndex].empty() )
            {
                message = failures[taskIndex];
                return false;
            }
        }

        return true;
    }

    //-------------------------------------------------------------------------

    bool ReadStageEXR
    (
        std::string const& directory,
        std::vector<uint32_t> const& resolutions,
        uint32_t numSlices,
        std::vector<HDRIImage>& levels,
        std::string& message
    )
    {
        message.clear();
        levels.clear();

        levels.resize( resolutions.size() );

        for ( uint32_t level = 0; level < resolutions.size(); ++level )
        {
            levels[level].Create( resolutions[level], numSlices );
        }

        struct SliceTask
        {
            uint32_t    m_level;
            uint32_t    m_slice;
        };

        std::vector<SliceTask> tasks;
        tasks.reserve( resolutions.size() * numSlices );

        for ( uint32_t level = 0; level < resolutions.size(); ++level )
        {
            for ( uint32_t slice = 0; slice < numSlices; ++slice )
            {
                tasks.push_back( { level, slice } );
            }
        }

        std::vector<std::string> failures( tasks.size() );

        auto readSlices = [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            (void) workerIndex;

            for ( uint32_t taskIndex = begin; taskIndex < end; ++taskIndex )
            {
                SliceTask const& task = tasks[taskIndex];

                HDRIImage& image = levels[task.m_level];
                float* const pSlice = image.Texel( task.m_slice, 0, 0 );

                LoadFaceEXR( GetSlicePath( directory, task.m_level, task.m_slice ), image.GetResolution(), pSlice, failures[taskIndex] );
            }
        };

        ParallelForRanges( static_cast<uint32_t>( tasks.size() ), readSlices );

        for ( uint32_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex )
        {
            if ( !failures[taskIndex].empty() )
            {
                message = failures[taskIndex];
                levels.clear();
                return false;
            }
        }

        return true;
    }

    //-------------------------------------------------------------------------

    bool IsStageComplete( std::string const& root, HDRIStageConfig const& config )
    {
        std::string const directory = GetStageDirectory( root, config );

        HDRIStageConfig cached;
        std::string ignored;

        if ( !ReadStageManifest( directory, cached, ignored ) )
        {
            return false;
        }

        if ( cached.m_stage != config.m_stage )
        {
            return false;
        }

        std::error_code errorCode;

        for ( uint32_t level = 0; level < config.m_numLevels; ++level )
        {
            for ( uint32_t slice = 0; slice < config.m_numSlices; ++slice )
            {
                if ( !std::filesystem::is_regular_file( GetSlicePath( directory, level, slice ), errorCode ) )
                {
                    return false;
                }
            }
        }

        return true;
    }

    //-------------------------------------------------------------------------

    bool TryLoadCachedStage
    (
        std::string const& root,
        HDRIStageConfig const& config,
        std::vector<HDRIImage>& levels,
        std::string& message
    )
    {
        message.clear();

        std::string const directory = GetStageDirectory( root, config );

        std::error_code errorCode;
        if ( !std::filesystem::is_directory( directory, errorCode ) )
        {
            return false;
        }

        HDRIStageConfig cached;
        std::string manifestMessage;

        if ( !ReadStageManifest( directory, cached, manifestMessage ) )
        {
            message = "cache has no readable manifest";
            return false;
        }

        // Every field that changes the pixels, compared rather than trusted.
        // A mismatched cache is recomputed, never mixed in.
        if ( cached.m_sourceId != config.m_sourceId )
        {
            message = "cache is for source '" + cached.m_sourceId + "'";
            return false;
        }

        if ( cached.m_stage != config.m_stage )
        {
            message = "cache is for stage '" + cached.m_stage + "'";
            return false;
        }

        if ( cached.m_profileName != config.m_profileName )
        {
            message = "cache profile '" + cached.m_profileName + "' does not match";
            return false;
        }

        if ( cached.m_curveName != config.m_curveName )
        {
            message = "cache curve '" + cached.m_curveName + "' does not match";
            return false;
        }

        if ( cached.m_tableName != config.m_tableName )
        {
            message = "cache table '" + cached.m_tableName + "' does not match";
            return false;
        }

        // The map is part of what the pixels are: the same HDRI at the same width is a different chain of textures for a cubemap than for a tetrahedral tile, so a stage is never reused across them.
        if ( cached.m_projectionName != config.m_projectionName )
        {
            message = "cache projection '" + cached.m_projectionName + "' does not match";
            return false;
        }

        if ( cached.m_numLevels != config.m_numLevels )
        {
            message = "cache level count does not match";
            return false;
        }

        if ( cached.m_numSlices != config.m_numSlices )
        {
            message = "cache slice count does not match";
            return false;
        }

        if ( cached.m_samples != config.m_samples )
        {
            message = "cache sample count does not match";
            return false;
        }

        if ( cached.m_equirectWidth != config.m_equirectWidth )
        {
            message = "cache equirect width does not match";
            return false;
        }

        if ( cached.m_levelAlpha != config.m_levelAlpha )
        {
            message = "cache level widths do not match";
            return false;
        }

        std::vector<uint32_t> resolutions( config.m_numLevels );

        for ( uint32_t level = 0; level < config.m_numLevels; ++level )
        {
            resolutions[level] = GetChainResolution( config.m_baseWidth, level );
        }

        return ReadStageEXR( directory, resolutions, config.m_numSlices, levels, message );
    }
}
