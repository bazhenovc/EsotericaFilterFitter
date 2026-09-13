#pragma once

#include <cstdint>

// reference frame contract
//-------------------------------------------------------------------------
// A reference frame is the discretisation of the sphere that the fit scores against: an ordered set of texels, each with a direction, a solid angle and - for frames that have one - a face and a two-dimensional coordinate within it.
//
// Like the profile, this is a CONCEPT, not a base class.
// A frame is passed as a template parameter and its members are called directly, so the frame is monomorphic in the fit's loops, and a type that fails to provide a member fails to compile rather than failing at run time.
// MapReferenceFrame< TMap > in MapReferenceFrame.h implements it for any map MapProjection.h describes, and is the only implementation; Profile.h documents the same pattern.
//
//  WHAT A FRAME PROVIDES
//
//      using Texel = ...;                      // per-texel record, see below
//
//      void     Initialize( uint32_t resolution );
//      bool     IsInitialized() const;
//      uint32_t GetResolution() const;
//      uint32_t GetNumTexels() const;
//      double   GetTotalSolidAngle() const;
//      Texel const& GetTexel( uint32_t texelIndex ) const;
//      std::vector<Texel> const& GetTexels() const;
//
//      // Measure of one texel under one of the ErrorMeasure conventions
//      double GetTexelMeasure( Texel const& texel, ErrorMeasure measure ) const;
//
//      // Direction at a fractional offset within a texel's own footprint, offsets in [-0.5, 0.5]. Used to average the lobe over a texel.
//      void GetTexelSampleDirection( double* pOutDir, uint32_t texelIndex, double offsetX, double offsetY ) const;
//
//      // One separable (0.25, 0.5, 0.25) pass over the frame's neighbour structure, for the reference-smoothing experiment
//      void SmoothInPlace( std::vector<double>& values ) const;
//
// A Texel provides at least m_dir[3], a unit direction. 
// What else it carries is the frame's business; the fit reads only the direction and asks the frame for everything else.
//
//  WHAT A FRAME DOES NOT COVER
//
// The frame abstracts the BASE DOMAIN - the grid B(x) is sampled on and measured over.
// It does not abstract the mip chain: that is the paper's algorithm rather than a property of the sphere, and the chain is map-generic. BsplineRecurrence, PreimageAccumulator, HDRIImage and the sampler are all templates on the map with instantiations for both, and the whole validation pipeline selects one at run time.
// What is left cube-flavoured is one diagnostic:
//
//      GetLobeToTexelRatio       assumes a cube face's texel size at its centre
//-------------------------------------------------------------------------

namespace FilterFitter
{
    // How the discretised error integral weights a texel.
    // The paper writes the objective as an integral over the sphere without fixing the measure; evaluating the published tables picks solid angle.
    //
    // This lives with the frame rather than with the preimage because the frame is what computes it: each frame defines its own approximation, which has to be fast, and the two approximations here are the cube's.
    //-------------------------------------------------------------------------

    enum class ErrorMeasure : uint8_t
    {
        // dOmega. Natural for a lobe averaged over a texel, and the one that reproduces the published errors.
        SolidAngle,

        // J * (the coordinate area one texel covers): the texel solid angle as the vendored shader approximates it and as the paper defines SAtexel in Eq. 4.
        // For a cubemap that product is 4J/R^2, because a cube face spans 2 in its own coordinates; for a single-slice tetrahedral map it is J/R^2.
        // The paper's expression is a cube expression and applying it to a tetrahedral map overstates the total by four.
        //
        // On a cubemap it differs from the exact solid angle by up to 2.7x at face corners, and evaluated against the published tables that is indistinguishable from the exact value, because the total comes to 12.5664 either way.
        JacobianTexelArea,

        // Uniform in the map's own face-space coordinates, matching the domain the b-spline recurrence and the hardware trilinear filter both work in
        CoordinateArea,
    };
}
