#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "HDRIImage.h"

//  HDRI dataset scan and ingest
//-------------------------------------------------------------------------
// The dataset is a cache directory of unrelated assets: materials, atlases and HDRIs side by side, one folder each, and the only thing that says which is which is the asset.json beside it.
// So the scan is driven by that file and everything it does not positively identify as an HDRI is skipped. So is anything that an HDRI should have and does not, or has and cannot be decoded: a validation corpus is worth more with 400 images that all loaded than with 500 where 100 are black.
//
// The JSON is read with a targeted scan rather than a parser.
// The files are machine-generated, flat, and one screen long, and the two facts needed are "type": "hdri" and the HDR entry of "textures".
// Anything that does not yield both is skipped with a reason.
//
//  INGEST ORDER: EQUIRECT, THEN CUBE
//
// A source HDRI is an equirect panorama and is stored at 4K or 8K.
// It is downsampled in equirect space first and converted to a cubemap second, rather than being projected straight to the cube.
//
// The order matters because the two steps want opposite things. 
// The projection needs a source whose texels are small enough to represent the cube's, and the cube at 128 wants roughly a 512 x 256 equirect; everything a 4K panorama carries beyond that cannot survive the projection and only costs time.
// The equirect downsample is a plain area average, so it is exact, fast, and keeps the panorama's full dynamic range. 
// What the projection then receives is a source it can actually resolve, which is what stops the cube from aliasing.
//
// A source that is not 2:1 is not an equirect - the dataset contains a few - and is skipped.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    struct HDRIAsset
    {
        std::string     m_id;
        std::string     m_assetJsonPath;
        std::string     m_exrPath;
        std::string     m_declaredResolution;
        std::string     m_technique;

        uint32_t        m_sourceWidth = 0;
        uint32_t        m_sourceHeight = 0;

        // Non-empty when the asset was rejected, and the reason it was
        std::string     m_skipReason;
    };

    //  Ingest settings
    //-------------------------------------------------------------------------

    struct HDRIIngestSettings
    {
        // Base cubemap width. 128 throughout, because that is what the engine captures and what every table in this tool is fitted at.
        uint32_t    m_baseWidth = 128;

        // Equirect width the source is reduced to before projection, so the height is half of it.
        // The default is four times the cube's equivalent equirect width of 512, which leaves the projection about 21 source texels per cube texel. 
        // A source narrower than the request is left alone rather than upsampled.
        uint32_t    m_equirectWidth = 2048;

        // A panorama narrower than this cannot fill a 128 cube without aliasing and is skipped.
        uint32_t    m_minimumSourceWidth = 512;
    };

    //-------------------------------------------------------------------------

    // Every asset.json under root whose type is hdri, in a stable order.
    // Assets that are not HDRIs are not returned at all; HDRIs that cannot be used are returned with m_skipReason set, so a scan reports what it rejected and why rather than silently shortening the corpus.
    void ScanHDRIDataset( std::string const& root, std::vector<HDRIAsset>& assets );

    //-------------------------------------------------------------------------

    // Decodes an EXR equirect into float RGB. Rejects anything that is not a usable equirect: a non-2:1 aspect, a resolution below the minimum, a decode failure, or enough non-finite texels that it is not an environment.
    bool LoadEquirectEXR
    (
        HDRIAsset const& asset,
        HDRIIngestSettings const& settings,
        std::vector<float>& pixels,
        uint32_t& width,
        uint32_t& height,
        std::string& message
    );

    // Equirect area downsample
    //-------------------------------------------------------------------------
    // A plain area average, which is what "reduce this panorama" means and what the projection below assumes.
    // A target texel stores the mean radiance over its own (u,v) footprint, so the projection is then free to apply the solid-angle measure once, where it belongs.
    //
    // Weighting here by sin(theta) as well would apply that measure twice - once in the reduction and once in the projection - which over-weights the horizon and darkens the poles.
    //
    // stb's default downsample filter is Mitchell, which has negative lobes.
    // On an HDR panorama whose sun is orders of magnitude above its sky that rings, and the ring survives into the reference as a darkening no convolution performed. 
    // A box is the area average, so the filter is set explicitly.
    //-------------------------------------------------------------------------

    // False when stb could not allocate its samplers.
    bool DownsampleEquirect
    (
        float const* pSource,
        uint32_t sourceWidth,
        uint32_t sourceHeight,
        uint32_t targetWidth,
        uint32_t targetHeight,
        std::vector<float>& target
    );
}
