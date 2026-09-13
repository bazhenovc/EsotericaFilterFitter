#include "Assert.h"
#include "HDRIDataset.h"

#include "ParallelFor.h"
#include "TinyEXR/tinyexr.h"
#include "stb/stb_image_resize2.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numbers>

namespace FilterFitter
{
    namespace
    {
        //  Targeted JSON
        //
        //  asset.json is written by the dataset's own downloader: flat, one
        //  object, no nesting beyond textures, and never reformatted. Two keys
        //  are needed, so two lookups are cheaper than a parser and cannot
        //  disagree with one about a file this simple.
        //---------------------------------------------------------------------

        void SkipWhitespace( std::string const& text, size_t& cursor, size_t limit )
        {
            while ( ( cursor < limit ) && ( ( text[cursor] == ' ' ) || ( text[cursor] == '\t' ) || ( text[cursor] == '\r' ) || ( text[cursor] == '\n' ) ) )
            {
                ++cursor;
            }
        }

        //---------------------------------------------------------------------

        bool FindJsonString( std::string const& text, std::string const& key, size_t from, size_t limit, std::string& value )
        {
            std::string const token = "\"" + key + "\"";

            size_t const keyPosition = text.find( token, from );
            if ( ( keyPosition == std::string::npos ) || ( keyPosition >= limit ) )
            {
                return false;
            }

            size_t cursor = keyPosition + token.size();
            SkipWhitespace( text, cursor, limit );

            if ( ( cursor >= limit ) || ( text[cursor] != ':' ) )
            {
                return false;
            }

            ++cursor;
            SkipWhitespace( text, cursor, limit );

            if ( ( cursor >= limit ) || ( text[cursor] != '"' ) )
            {
                return false;
            }

            ++cursor;

            size_t const valueBegin = cursor;
            size_t const valueEnd = text.find( '"', valueBegin );

            if ( ( valueEnd == std::string::npos ) || ( valueEnd > limit ) )
            {
                return false;
            }

            value.assign( text, valueBegin, valueEnd - valueBegin );

            return true;
        }

        //---------------------------------------------------------------------

        bool FindJsonObject( std::string const& text, std::string const& key, size_t& begin, size_t& end )
        {
            std::string const token = "\"" + key + "\"";

            size_t const keyPosition = text.find( token );
            if ( keyPosition == std::string::npos )
            {
                return false;
            }

            size_t cursor = keyPosition + token.size();
            SkipWhitespace( text, cursor, text.size() );

            if ( ( cursor >= text.size() ) || ( text[cursor] != ':' ) )
            {
                return false;
            }

            ++cursor;
            SkipWhitespace( text, cursor, text.size() );

            if ( ( cursor >= text.size() ) || ( text[cursor] != '{' ) )
            {
                return false;
            }

            size_t const objectBegin = cursor;
            uint32_t     depth = 0;

            while ( cursor < text.size() )
            {
                if ( text[cursor] == '{' )
                {
                    ++depth;
                }
                else if ( text[cursor] == '}' )
                {
                    if ( --depth == 0 )
                    {
                        begin = objectBegin + 1;
                        end = cursor;

                        return true;
                    }
                }

                ++cursor;
            }

            return false;
        }

        //---------------------------------------------------------------------

        bool ReadTextFile( std::filesystem::path const& path, std::string& text )
        {
            std::FILE* pFile = std::fopen( path.string().c_str(), "rb" );
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

        bool EndsWithEXR( std::string const& value )
        {
            if ( value.size() < 4 )
            {
                return false;
            }

            std::string tail = value.substr( value.size() - 4 );

            for ( char& character : tail )
            {
                character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );
            }

            return tail == ".exr";
        }

        //---------------------------------------------------------------------

        void AppendAsset
        (
            std::filesystem::path const& assetJsonPath,
            std::string const& text,
            std::vector<HDRIAsset>& assets
        )
        {
            HDRIAsset asset;
            asset.m_assetJsonPath = assetJsonPath.string();
            asset.m_id = assetJsonPath.parent_path().filename().string();

            std::string type;
            if ( !FindJsonString( text, "type", 0, text.size(), type ) )
            {
                return;
            }

            // Not an HDRI, so it is not returned at all rather than returned as a rejection: the dataset is mostly materials and they are not the business of this scan
            if ( type != "hdri" )
            {
                return;
            }

            FindJsonString( text, "resolution", 0, text.size(), asset.m_declaredResolution );
            FindJsonString( text, "technique", 0, text.size(), asset.m_technique );

            size_t texturesBegin = 0;
            size_t texturesEnd = 0;
            std::string hdrEntry;

            if ( !FindJsonObject( text, "textures", texturesBegin, texturesEnd ) )
            {
                asset.m_skipReason = "asset.json has no textures object";
                assets.push_back( asset );
                return;
            }

            if ( !FindJsonString( text, "HDR", texturesBegin, texturesEnd, hdrEntry ) )
            {
                asset.m_skipReason = "asset.json names no HDR map";
                assets.push_back( asset );
                return;
            }

            if ( !EndsWithEXR( hdrEntry ) )
            {
                asset.m_skipReason = "HDR map is not an EXR: " + hdrEntry;
                assets.push_back( asset );
                return;
            }

            std::filesystem::path const exrPath = assetJsonPath.parent_path() / hdrEntry;

            std::error_code errorCode;
            if ( !std::filesystem::is_regular_file( exrPath, errorCode ) )
            {
                asset.m_skipReason = "HDR map is missing on disk: " + hdrEntry;
                assets.push_back( asset );
                return;
            }

            asset.m_exrPath = exrPath.string();
            assets.push_back( asset );
        }
    }

    //-------------------------------------------------------------------------

    void ScanHDRIDataset( std::string const& root, std::vector<HDRIAsset>& assets )
    {
        assets.clear();

        std::error_code errorCode;

        if ( !std::filesystem::is_directory( root, errorCode ) )
        {
            return;
        }

        std::vector<std::filesystem::path> assetJsonPaths;

        std::filesystem::recursive_directory_iterator iterator( root, std::filesystem::directory_options::skip_permission_denied, errorCode );
        std::filesystem::recursive_directory_iterator const endIterator;

        while ( !errorCode && ( iterator != endIterator ) )
        {
            std::filesystem::directory_entry const& entry = *iterator;

            if ( entry.is_regular_file( errorCode ) && ( entry.path().filename() == "asset.json" ) )
            {
                assetJsonPaths.push_back( entry.path() );
            }

            errorCode.clear();
            iterator.increment( errorCode );
        }

        // Sorted, so a limited run takes the same assets on every machine and two runs are comparable
        std::sort( assetJsonPaths.begin(), assetJsonPaths.end() );

        assets.reserve( assetJsonPaths.size() );

        for ( std::filesystem::path const& assetJsonPath : assetJsonPaths )
        {
            std::string text;
            if ( !ReadTextFile( assetJsonPath, text ) )
            {
                continue;
            }

            AppendAsset( assetJsonPath, text, assets );
        }
    }

    //-------------------------------------------------------------------------

    bool LoadEquirectEXR
    (
        HDRIAsset const& asset,
        HDRIIngestSettings const& settings,
        std::vector<float>& pixels,
        uint32_t& width,
        uint32_t& height,
        std::string& message
    )
    {
        message.clear();
        pixels.clear();
        width = 0;
        height = 0;

        float* pRgba = nullptr;
        int    decodedWidth = 0;
        int    decodedHeight = 0;

        char const* pError = nullptr;
        int const result = LoadEXR( &pRgba, &decodedWidth, &decodedHeight, asset.m_exrPath.c_str(), &pError );

        if ( result != TINYEXR_SUCCESS )
        {
            message = "EXR decode failed";
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

        if ( ( decodedWidth <= 0 ) || ( decodedHeight <= 0 ) || ( pRgba == nullptr ) )
        {
            if ( pRgba != nullptr )
            {
                std::free( pRgba );
            }

            message = "EXR decoded to an empty image";
            return false;
        }

        uint32_t const sourceWidth = static_cast<uint32_t>( decodedWidth );
        uint32_t const sourceHeight = static_cast<uint32_t>( decodedHeight );

        // A panorama is 2:1. Anything else in this dataset is a different projection - angular, cross, or a light probe - and this pipeline has no equirect mapping for it, so it is skipped rather than misinterpreted.
        int64_t const aspectError = static_cast<int64_t>( sourceWidth ) - ( 2 * static_cast<int64_t>( sourceHeight ) );
        int64_t const aspectSlack = 1 + ( static_cast<int64_t>( sourceWidth ) / 100 );

        if ( ( aspectError > aspectSlack ) || ( aspectError < -aspectSlack ) )
        {
            char buffer[160] = {};
            std::snprintf( buffer, sizeof( buffer ), "not a 2:1 equirect: %u x %u", sourceWidth, sourceHeight );
            message = buffer;

            std::free( pRgba );

            return false;
        }

        if ( sourceWidth < settings.m_minimumSourceWidth )
        {
            char buffer[160] = {};
            std::snprintf( buffer, sizeof( buffer ), "source is %u wide, below the %u minimum",
                           sourceWidth, settings.m_minimumSourceWidth );
            message = buffer;

            std::free( pRgba );

            return false;
        }

        // Count non-finite texels before using any of it.
        // A NaN that reaches the convolution spreads through the mip chain and poisons a whole level, and finding that out from a validation number is far more expensive than finding it out here.
        uint64_t nonFinite = 0;
        size_t const numSourceTexels = static_cast<size_t>( sourceWidth ) * sourceHeight;

        for ( size_t texelIndex = 0; texelIndex < numSourceTexels; ++texelIndex )
        {
            float const* const pTexel = pRgba + ( texelIndex * 4 );

            if ( !std::isfinite( pTexel[0] ) || !std::isfinite( pTexel[1] ) || !std::isfinite( pTexel[2] ) )
            {
                ++nonFinite;
            }
        }

        double const nonFiniteFraction = static_cast<double>( nonFinite ) / static_cast<double>( numSourceTexels );

        if ( nonFiniteFraction > 0.05 )
        {
            char buffer[160] = {};
            std::snprintf( buffer, sizeof( buffer ), "%.2f%% of texels are not finite", nonFiniteFraction * 100.0 );
            message = buffer;

            std::free( pRgba );

            return false;
        }

        // RGBA to RGB
        pixels.resize( numSourceTexels * 3 );

        for ( size_t texelIndex = 0; texelIndex < numSourceTexels; ++texelIndex )
        {
            float const* const pSource = pRgba + ( texelIndex * 4 );
            float* const       pTarget = &pixels[texelIndex * 3];

            pTarget[0] = pSource[0];
            pTarget[1] = pSource[1];
            pTarget[2] = pSource[2];
        }

        std::free( pRgba );

        width = sourceWidth;
        height = sourceHeight;

        return true;
    }

    //-------------------------------------------------------------------------

    bool DownsampleEquirect
    (
        float const* pSource,
        uint32_t sourceWidth,
        uint32_t sourceHeight,
        uint32_t targetWidth,
        uint32_t targetHeight,
        std::vector<float>& target
    )
    {
        FF_ASSERT( pSource != nullptr );
        FF_ASSERT( sourceWidth > 0 );
        FF_ASSERT( sourceHeight > 0 );
        FF_ASSERT( targetWidth > 0 );
        FF_ASSERT( targetHeight > 0 );

        target.assign( static_cast<size_t>( targetWidth ) * targetHeight * 3, 0.0f );

        if ( ( targetWidth == sourceWidth ) && ( targetHeight == sourceHeight ) )
        {
            std::memcpy( target.data(), pSource, target.size() * sizeof( float ) );
            return true;
        }

        STBIR_RESIZE resize;

        stbir_resize_init
        (
            &resize,
            pSource, static_cast<int>( sourceWidth ), static_cast<int>( sourceHeight ), 0,
            target.data(), static_cast<int>( targetWidth ), static_cast<int>( targetHeight ), 0,
            STBIR_RGB, STBIR_TYPE_FLOAT
        );

        // Explicit, because stb's default for a downsample is Mitchell and this is an area reduction rather than a resample.
        stbir_set_filters( &resize, STBIR_FILTER_BOX, STBIR_FILTER_BOX );
        stbir_set_edgemodes( &resize, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP );

        return stbir_resize_extended( &resize ) != 0;
    }
}
