#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "HDRIImage.h"

// HDRI probe storage on disk
//-------------------------------------------------------------------------
// Every stage of the validation writes its output as EXR and reads the previous stage's instead of recomputing it. 
// EXR rather than a private container for one reason: these are the artefacts someone will want to look at.
// A validation number says the table is wrong; the image says how, and it says it to whatever viewer is already installed.
//
// A level is one EXR per SLICE, because a slice is one texture and every tool reads a square image. 
// A cubemap has one face per slice, so its tree is unchanged; a single-slice tetrahedral map writes one file per level holding all four faces. 
// A chain is one directory per level:
//
//      <root>/<assetId>/<stageKey>/L3/S4.exr
//
//  S rather than F for the reason the slice count is a map property: the file is a texture, and for a cubemap the slice index is the face index.
//
//  The stage key encodes the configuration that produced the stage - including the projection - so the tree can be navigated without opening anything, and a stage computed for a different profile, curve, table or map cannot be mistaken for a hit. 
// A manifest.txt beside it records the same thing in a form that can be read and diffed.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    //  What a cached stage was produced for.
    //
    //  Everything that changes the pixels is here, and nothing that does not.
    //  Two runs whose configurations compare equal must produce identical payloads, or a cache hit is a wrong answer.
    //-------------------------------------------------------------------------

    struct HDRIStageConfig
    {
        // "source", "reference" or "fast"
        std::string             m_stage;
        std::string             m_sourceId;
        std::string             m_profileName;
        std::string             m_curveName;
        std::string             m_tableName;

        // "cube" or "tetrahedron". 
        // A stage is a chain of textures for one map, and the two maps have different slices at the same resolution, so a stage computed for one is not a hit for the other.
        std::string             m_projectionName;

        uint32_t                m_baseWidth = 0;
        uint32_t                m_numLevels = 0;
        uint32_t                m_numSlices = 0;

        // Reference convolution only
        uint32_t                m_samples = 0;

        // Ingest only
        uint32_t                m_equirectWidth = 0;

        // The width the profile was read at, per level. 
        // This is the curve's output, so recording it makes a curve that changed under the same name a cache miss rather than a silent mixture.
        std::vector<float>      m_levelAlpha;
    };

    //-------------------------------------------------------------------------

    // Directory a stage lives in, including the asset. The key is derived from the configuration so it is stable and readable.
    std::string GetStageDirectory( std::string const& root, HDRIStageConfig const& config );

    //-------------------------------------------------------------------------

    bool WriteStageManifest( std::string const& directory, HDRIStageConfig const& config, std::string& message );

    bool ReadStageManifest( std::string const& directory, HDRIStageConfig& config, std::string& message );

    //  Slices
    //-------------------------------------------------------------------------

    // One EXR per slice per level: <directory>/L<level>/S<slice>.exr. 
    // Written as 32-bit float, which is what the pipeline holds in memory, so a cache round trip introduces no error of its own.
    bool WriteStageEXR( std::string const& directory, std::vector<HDRIImage> const& levels, std::string& message );

    bool ReadStageEXR
    (
        std::string const& directory,
        std::vector<uint32_t> const& resolutions,
        uint32_t numSlices,
        std::vector<HDRIImage>& levels,
        std::string& message
    );

    //  Cache
    //-------------------------------------------------------------------------

    // Loads a stage when the directory holds a complete one for exactly this configuration.
    // Returns false with an empty message when there is no usable cache, which is the signal to compute rather than an error.
    bool TryLoadCachedStage
    (
        std::string const& root,
        HDRIStageConfig const& config,
        std::vector<HDRIImage>& levels,
        std::string& message
    );

    bool IsStageComplete( std::string const& root, HDRIStageConfig const& config );
}
