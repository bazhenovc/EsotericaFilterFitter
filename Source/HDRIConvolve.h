#pragma once

#include <cstdint>
#include <vector>

#include "CoefficientTable.h"
#include "HDRIImage.h"
#include "ProfileGGX.h"

//  Reference and table-driven convolution
//-------------------------------------------------------------------------
// Two convolutions of the same environment, sharing the same source chain and the same sampler, so the only difference between them is where the samples come from and how they are weighted. 
// That is the thing being validated, and anything else shared between the two would hide part of it.
//
// Both are templated on the map, so "the same sampler" means the same implementation rather than two that agree: a cubemap and a tetrahedral tile are convolved by the same code, and a table fitted for one is measured against the chain of the same one.
//
//  REFERENCE
//
// The engine's radiance prefilter, moved to the CPU and run at a high sample count.
// It importance-samples the half-vector from the profile's NDF, reflects, weights by cos(theta_l), and samples the source chain at the mip whose texel solid angle matches the sample's own. 
// That term is not decoration: at a narrow width the lobe covers a fraction of a texel, and without it the estimator aliases the source instead of averaging it.
//
// At a high sample count that term goes to zero and the estimator converges to the exact convolution of the 128 chain against the profile's normalized zonal lobe - which is the same B(x) the fit is scored on.
// So this is the fit's quantity, measured in radiance rather than in preimage error.
//
// The sampler is written for the GGX family and is the one part of this file that is not profile-generic: importance sampling needs the NDF's own half-vector distribution, and Beckmann's is a different formula. 
// A second shipping NDF adds a sampler beside SampleGGXHalfVector, not a branch in it.
//
// A width of exactly zero is a mirror rather than a narrow lobe, and both paths take it as the identity: the reference samples mip 0 along the output direction, and the table's level row for that width is built to clamp there.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    // Error of an approximation against the reference, over one level.
    //
    // Weighted by solid angle, because a cube's texels are not equal in area and an unweighted mean over them over-counts the corners. 
    // Measured on luminance: the filter is identical on all three channels, so a chromatic  difference cannot come from the table.
    //-------------------------------------------------------------------------

    struct ConvolutionError
    {
        double      m_relativeL1 = 0.0;
        double      m_relativeRms = 0.0;
        double      m_maxAbsolute = 0.0;
        double      m_referenceMean = 0.0;
        double      m_approximateMean = 0.0;

        // Peaks, because a mean cannot see a firefly. 
        // A bright source spread over a few texels barely moves an L1 summed over 98304 of them, and a firefly is the failure this technique is reported to have.
        // The two peaks say which side produced one: a table whose peak exceeds the reference's has put energy where the reference did not, and a reference whose peak exceeds the source's is the estimator itself aliasing a small bright source.
        double      m_referencePeak = 0.0;
        double      m_approximatePeak = 0.0;

        //  Diagnostic: the same comparison split by region of the map
        //-------------------------------------------------------------------------
        //  So that a hypothesis about WHERE an approximation is wrong is tested rather than argued.
        //  The regions are the map's OWN decomposition - its faces, from MapTexel::m_face, which the projection already resolves - crossed with a band of the solid-angle Jacobian, so nothing here re-derives the geometry.
        //
        //  The band is the point of it: J = 1 / |p|^3 with |p| the length of the L1-normalized direction at the texel's centre, which runs from 1 at the octahedral map's finest texels (the diamond's vertices and the square's corners) to 5.196 at its coarsest (the eight octant centres).
        //  A quadrature over texels and an analytic integral disagree most where the texels are coarsest, so if that is what the octahedral map's residual error is made of, these bands show it as a trend instead of a flat floor.
        //
        //  Absolute sums rather than ratios, because a region's own relative error is its absolute sum over its own reference sum; both are carried so either can be printed.
        //  Only the octahedral map uses them: for the cube and the tetrahedral map every texel lands in band 0 of its own face, no band trend exists to test, and nothing accumulates differently.
        //-------------------------------------------------------------------------

        static constexpr uint32_t NumRegionBands = 3;
        static constexpr uint32_t NumMaxFaces = 8;
        static constexpr uint32_t NumRegions = NumMaxFaces * NumRegionBands;

        double      m_regionAbsoluteSum[NumRegions] = {};
        double      m_regionReferenceSum[NumRegions] = {};
        uint32_t    m_regionCount[NumRegions] = {};
        double      m_regionMaxAbsolute[NumRegions] = {};

        double      m_faceAbsoluteSum[NumMaxFaces] = {};
        double      m_faceReferenceSum[NumMaxFaces] = {};
        uint32_t    m_faceCount[NumMaxFaces] = {};
    };

    //-------------------------------------------------------------------------

    // One mip level of the reference convolution. level selects the profile's width and the output resolution, chain[0] is the base level of the map.
    template< typename TMap, typename TProfile >
    void ConvolveReference
    (
        std::vector<HDRIImage> const& chain,
        TProfile const& profile,
        uint32_t level,
        uint32_t sampleCount,
        HDRIImage& out
    );

    // The same level from a table, gathered through the same chain.
    template< typename TMap >
    void ConvolveTable
    (
        std::vector<HDRIImage> const& chain,
        CoefficientTable const& table,
        uint32_t level,
        uint32_t baseWidth,
        HDRIImage& out
    );

    //-------------------------------------------------------------------------

    template< typename TMap >
    void CompareConvolutions( HDRIImage const& reference, HDRIImage const& approximate, ConvolutionError& error );

    // Luminance the comparison uses, exported so a caller can print a texel.
    double GetLuminance( float const* pRgb );

    //-------------------------------------------------------------------------

    extern template void ConvolveReference< CubeProjection, ProfileGGX >( std::vector<HDRIImage> const&, ProfileGGX const&, uint32_t, uint32_t, HDRIImage& );
    extern template void ConvolveReference< TetrahedralProjection, ProfileGGX >( std::vector<HDRIImage> const&, ProfileGGX const&, uint32_t, uint32_t, HDRIImage& );

    extern template void ConvolveTable< CubeProjection >( std::vector<HDRIImage> const&, CoefficientTable const&, uint32_t, uint32_t, HDRIImage& );
    extern template void ConvolveTable< TetrahedralProjection >( std::vector<HDRIImage> const&, CoefficientTable const&, uint32_t, uint32_t, HDRIImage& );

    extern template void CompareConvolutions< CubeProjection >( HDRIImage const&, HDRIImage const&, ConvolutionError& );
    extern template void CompareConvolutions< TetrahedralProjection >( HDRIImage const&, HDRIImage const&, ConvolutionError& );
}
