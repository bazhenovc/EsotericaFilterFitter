#pragma once

#include <cmath>
#include <cstdint>

#include "Assert.h"
#include "MapReferenceFrame.h"
#include "ProbeMap.h"

// base map projection contract
//-------------------------------------------------------------------------
// The environment a probe stores is a map: a set of faces, each a grid of texels, covering the sphere.
// A cubemap is one. A tetrahedral map is another. 
// Neither is the default; the tap construction is written against  the questions below and not against either of them.
//
// Six questions, and nothing else. 
// Everything the construction does with the answers - the tangent frames, the tap directions, the weights, the level offsets, the accumulation - is the same for every map.
//
//      NumFrames                                       how many frames
//      GetDirection( slice, texelX, texelY, resolution ) texel -> direction. The first argument is a SLICE, which is what a level is made of: a map with one face per slice reads that face, and a map holding several faces in one slice derives the face from the coordinate
//      GetFrameUp( frame )                             the frame's pole
//      GetFrameWeight( direction, frame )              the frame's blend weight
//      GetPolynomialArguments( direction, frame )      the two polynomial arguments
//      GetLevelCorrection( sampleDirection )           the mip correction
//
// The answers are not independent, and a map has to answer all six consistently or the construction silently produces a plausible wrong answer.
// Two properties in particular decide whether the answers work, and both are properties of a map, not of the sphere:
//
//  FRAME WEIGHTS
//
// The weight culls the frame whose pole is nearly parallel to the direction, which is where the tangent frame degenerates, and it must keep at least one frame active everywhere or the runtime divides by a zero weight sum.
//
// The cube manages both because its face coordinate vector always has one component of magnitude exactly 1: the two frames that are not the face's own always include that 1, so they are never culled, and the face's own frame is culled exactly where it degenerates.
// A rule written against a normalized direction has neither property - at (1,1,0)/sqrt(2) every axis would weight -0.17, all three cull, and the result is 0/0.
//
//  THE LEVEL CORRECTION
//
// A tap is placed in the tangent frame's coordinates and read from a mip chain, and the two do not have the same density.
// The correction converts a distance in frame coordinates into the mip level that has the matching texel density at that point.
//
// It must be fast enough to run per tap: this is the one part of the construction that cannot be computed offline. 
// For the cube it reduces to the largest absolute component of the sample direction, which the shader already has.
//
//  MAPPING
//
// A map also answers where its texels are and what solid angle each covers.
// That is the second group, used by the sampler, the preimage accumulator and pass 1 rather than by the tap construction directly:
//
//      NumFaces                                        how many faces
//      GetDirectionFromUV( u, v, face )                face coordinate -> direction
//      GetFaceAndUVFromDirection( direction )          direction -> face coordinate
//      GetTexelCentreUV( texelX, texelY, resolution )  texel -> face coordinate
//      GetTexelCentreUVOutside( x, y, resolution )     the same, extrapolated past the map's edge
//      GetTexelCoordinateFromUV( u, v, resolution )    coordinate -> continuous texel index
//      GetTexelStep( resolution )                      the signed step between texels
//      GetTexelIndex( face, texelX, texelY, resolution ) texel -> storage index
//      GetTexelSolidAngle( uv box )                    exact solid angle
//      GetJacobian( u, v )                             density, for the measure
//      GetInverseTexelSolidAngle( direction, base )    the reciprocal of the solid angle one texel covers, which is what a sample's own footprint is compared against
//      GetSliceFaces( slice )                          which faces a slice holds
//      GetFaceRegion( face, uv box )                   the part of a box that face owns
//
// GetInverseTexelSolidAngle and GetLevelCorrection are two readings of one quantity and have to agree in sign: the correction is -0.5 * log2( dOmega/dudv ) up to a constant, and the reciprocal is base^2 / ( dOmega/dudv * texel uv area ).
// The constant is absorbed by the  table's own level coefficient, so only the variation matters - which is why a sign error in one of them stays invisible until the table and a reference convolution are compared.
//
// GetSliceFaces and GetFaceRegion exist because the two maps relate a face to a coordinate differently. 
// A cube face owns its own coordinate space, so a box means one thing and one face owns all of it. 
// A tetrahedral slice holds all four faces in one coordinate space, so a box generally straddles two of them and each owns a piece. 
// Both questions have to be askable without the caller knowing which map it has, or every consumer re-derives them.
//
// A face coordinate vector is unnormalized and its scale is the map's own.
// The two texel-to-coordinate conventions above are what the rest of the tool already assumes, so a map that does not reproduce them will disagree with the accumulator about which texel a direction lands in.
//
//  FACES AND SLICES
//
// These are two different numbers and neither follows from the other:
//
//      cubemap        6 faces, 6 slices     one face per slice
//      tetrahedral    4 faces, 1 slice      all four triangles in one texture
//
// A texel's face and the slice holding it are different properties, and neither follows from the other.
// NumFaces is how many frames and how many parameterisations there are. 
// NumSlices is how many textures a level occupies, so a consumer sizes a buffer by slices and walks a level texel by texel.
// Anything that uses the face count as a texel multiplier is correct for a cubemap and four times too large for a tetrahedral map, which is why the distinction is a constant here rather than an assumption at each call site.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    //  Cube face-space helpers. They are declared here because a map's projection is what they implement.
    //-------------------------------------------------------------------------

    // Face-space UV in [-1,1] to a non-normalized cubemap direction. u and v match the vendored get_dir_0 .. get_dir_5.
    void GetCubeFaceDirection( double* pOutDir, double u, double v, uint32_t face );

    // Inverse of the above: the face whose axis dominates the direction, plus the UV that reproduces it.
    // This is the same face-selection rule hardware uses, and it is what resolves a tap that falls outside its own face onto the neighbouring one. 
    // Ties break x, then y, then z; exact ties only occur at a cube corner, where the parameterization is ambiguous.
    void GetCubeFaceAndUVFromDirection( double const* pDir, uint32_t& face, double& u, double& v );

    // Exact solid angle of the face-space rectangle [minU,maxU] x [minV,maxV]
    double GetCubeTexelSolidAngle( double minU, double minV, double maxU, double maxV );

    // Sphere-to-cube Jacobian at a face-space position. u and v match the vendored calcWeight( u, v ).
    double GetCubeJacobian( double u, double v );

    //  Cubemap: six faces, three axial frames.
    //-------------------------------------------------------------------------

    struct CubeProjection
    {
        static constexpr uint32_t NumFrames = 3;

        //---------------------------------------------------------------------

        static void GetDirection
        (
            double* pOutDirection,
            uint32_t slice,
            uint32_t texelX,
            uint32_t texelY,
            uint32_t resolution
        )
        {
            double const inverseResolution = 1.0 / static_cast<double>( resolution );

            double const u = ( ( 2.0 * static_cast<double>( texelX ) ) + 1.0 ) * inverseResolution - 1.0;
            double const v = -( ( 2.0 * static_cast<double>( texelY ) ) + 1.0 ) * inverseResolution + 1.0;

            GetCubeFaceDirection( pOutDirection, u, v, slice );
        }

        //---------------------------------------------------------------------

        static void GetFrameUp( double* pOutUp, uint32_t frame )
        {
            pOutUp[0] = 0.0;
            pOutUp[1] = 0.0;
            pOutUp[2] = 0.0;

            pOutUp[frame] = 1.0;
        }

        //  The weight is computed from the UNNORMALIZED face coordinate vector, whose constant 1 is what makes it work. See the note above.
        //---------------------------------------------------------------------

        static double GetFrameWeight( double const* pDirection, uint32_t frame )
        {
            uint32_t const otherFrame0 = 1 - ( frame & 1u ) - ( frame >> 1 );
            uint32_t const otherFrame1 = 2 - ( frame >> 1 );

            double const absOther0 = std::fabs( pDirection[otherFrame0] );
            double const absOther1 = std::fabs( pDirection[otherFrame1] );

            double const maxOther = ( absOther0 > absOther1 ) ? absOther0 : absOther1;

            return ( maxOther - 0.75 ) / 0.25;
        }

        //  theta and phi are the two map coordinates the tap polynomials are evaluated at. They are not angles and they are bounded by 1.
        //---------------------------------------------------------------------

        static void GetPolynomialArguments( double const* pDirection, uint32_t frame, double& theta, double& phi )
        {
            uint32_t const otherFrame0 = 1 - ( frame & 1u ) - ( frame >> 1 );
            uint32_t const otherFrame1 = 2 - ( frame >> 1 );

            double const nx = pDirection[otherFrame0];
            double const ny = pDirection[otherFrame1];
            double const nz = std::fabs( pDirection[frame] );

            double const maxXY = ( std::fabs( ny ) > std::fabs( nx ) ) ? std::fabs( ny ) : std::fabs( nx );
            FF_ASSERT( maxXY > 0.0 );

            double const scaledNx = nx / maxXY;
            double const scaledNy = ny / maxXY;

            if ( scaledNy < scaledNx )
            {
                theta = ( scaledNy <= -0.999 ) ? scaledNx : scaledNy;
            }
            else
            {
                theta = ( scaledNy >= 0.999 ) ? -scaledNx : -scaledNy;
            }

            // nz is absolute here, so the reference's nz <= -0.999 case cannot occur
            phi = ( nz >= 0.999 ) ? maxXY : nz;
        }

        //  1 / (the solid angle one texel covers), which is what a sample's own footprint is divided by to pick a source mip.
        //
        //  One texel covers 4 * J / R^2, where R is the base resolution and J the cube Jacobian, and J is major^3 with major the largest absolute component of the normalized direction.
        // So the reciprocal is R^2 / ( 4 * major^3 ), and the whole of its variation is the -1.5 * log2( major ) that GetLevelCorrection adds.
        //---------------------------------------------------------------------

        static double GetInverseTexelSolidAngle( double const* pDirection, uint32_t baseResolution )
        {
            double const major = std::fmax( std::fabs( pDirection[0] ), std::fmax( std::fabs( pDirection[1] ), std::fabs( pDirection[2] ) ) );

            double const resolution = static_cast<double>( baseResolution );
            double const sourceTexelScale = ( resolution * resolution ) / 4.0;

            return sourceTexelScale / ( major * major * major );
        }

        // 1.5 is the cube's Jacobian exponent and the halving is the mip level being the log2 of a linear size.
        // The result is log2( |d|^1.5 ) with |d| measured after the direction was divided by its largest component, i.e. the distance from the face centre.
        //---------------------------------------------------------------------

        static double GetLevelCorrection( double const* pSampleDirection )
        {
            double const squaredLength = ( pSampleDirection[0] * pSampleDirection[0] )
                + ( pSampleDirection[1] * pSampleDirection[1] )
                + ( pSampleDirection[2] * pSampleDirection[2] );

            return 0.75 * std::log2( squaredLength );
        }

        //  The table level at which a zero-width (mirror) level samples the  unfiltered source at every direction: the negative of the largest correction GetLevelCorrection can add.
        // A zero-width level is the identity, so what the table says has to leave the sampler's level at or below zero wherever the correction cannot reach it.
        //
        //  Here the correction is 0.75 * log2( |d|^2 ) with |d| at most sqrt(3), the distance from a face centre to a corner, so the largest it adds is  0.75 * log2( 3 ).
        //---------------------------------------------------------------------

        static double GetMirrorLevel()
        {
            return -0.75 * std::log2( 3.0 );
        }

        //  Mapping: where the texels are and what they cover
        //---------------------------------------------------------------------

        static constexpr uint32_t NumFaces = 6;

        // One face per slice: a level occupies six R x R textures
        static constexpr uint32_t NumSlices = 6;

        //---------------------------------------------------------------------

        static uint32_t GetSliceFaces( uint32_t slice, uint32_t* pOutFace )
        {
            FF_ASSERT( slice < NumSlices );

            pOutFace[0] = slice;

            return 1;
        }

        // A slice's coordinate space is its own face's, so the face owns the whole box and the polygon is the box.
        // The face argument is therefore not consulted; a box handed in here is already in that face's space.
        //---------------------------------------------------------------------

        static uint32_t GetFaceRegion( uint32_t face, double minU, double minV, double maxU, double maxV, double* pOutU, double* pOutV )
        {
            (void) face;

            pOutU[0] = minU;
            pOutV[0] = minV;
            pOutU[1] = maxU;
            pOutV[1] = minV;
            pOutU[2] = maxU;
            pOutV[2] = maxV;
            pOutU[3] = minU;
            pOutV[3] = maxV;

            return 4;
        }

        //---------------------------------------------------------------------

        static void GetDirectionFromUV( double* pOutDirection, double u, double v, uint32_t face )
        {
            GetCubeFaceDirection( pOutDirection, u, v, face );
        }

        //---------------------------------------------------------------------

        static void GetFaceAndUVFromDirection( double const* pDirection, uint32_t& face, double& u, double& v )
        {
            GetCubeFaceAndUVFromDirection( pDirection, face, u, v );
        }

        //  What a consumer calls.
        // One slice per face here, so a slice index names its own face; a single-slice map derives the face from the coordinate instead, which is why this exists rather than callers passing a face.
        //---------------------------------------------------------------------

        static void GetTexelDirection( double* pOutDirection, uint32_t slice, double u, double v )
        {
            GetDirectionFromUV( pOutDirection, u, v, slice );
        }

        // Texel centre in face coordinates. 
        // This is the convention the accumulator and the sampler already assume: a texel centre maps to an integer in the continuous coordinate the splat uses.
        //---------------------------------------------------------------------

        // Where a texture sits in the tile, so that a consumer that has to walk it or land on it does not have to know which map it is:
        //
        //      GetTexelCentreUV( x, y )            texel index -> its coordinate
        //      GetTexelCentreUVOutside( x, y )     the same, for an index outside the map, extrapolated
        //      GetTexelCoordinateFromUV( u, v )    coordinate -> continuous index, an integer at a texel centre
        //      GetTexelStep( resolution )          the signed step between texels
        //
        // The last three exist for pass 1 and the accumulator, both of which place phantom texels past a texel's own footprint on purpose: a tap  inside a texel has a bilinear footprint that reaches outside it, and outside the map entirely at the last texel.
        //---------------------------------------------------------------------

        // The coordinate step from one texel to the next along each axis, signed.
        // A consumer turns a fraction of a texel into a coordinate offset with it, which is what keeps the sign of the v axis - down a cube face, up a tetrahedral tile - out of every consumer.
        //---------------------------------------------------------------------

        static void GetTexelStep( double* pOutStepU, double* pOutStepV, uint32_t resolution )
        {
            double const inverseResolution = 1.0 / static_cast<double>( resolution );

            *pOutStepU = 2.0 * inverseResolution;
            *pOutStepV = -2.0 * inverseResolution;
        }

        //---------------------------------------------------------------------

        static void GetTexelCentreUV( double* pOutU, double* pOutV, uint32_t texelX, uint32_t texelY, uint32_t resolution )
        {
            GetTexelCentreUVOutside( pOutU, pOutV, static_cast<int32_t>( texelX ), static_cast<int32_t>( texelY ), resolution );
        }

        //---------------------------------------------------------------------

        static void GetTexelCentreUVOutside( double* pOutU, double* pOutV, int32_t texelX, int32_t texelY, uint32_t resolution )
        {
            double const inverseResolution = 1.0 / static_cast<double>( resolution );

            *pOutU = ( ( 2.0 * static_cast<double>( texelX ) ) + 1.0 ) * inverseResolution - 1.0;
            *pOutV = -( ( 2.0 * static_cast<double>( texelY ) ) + 1.0 ) * inverseResolution + 1.0;
        }

        //---------------------------------------------------------------------

        static void GetTexelCoordinateFromUV( double* pOutX, double* pOutY, double u, double v, uint32_t resolution )
        {
            double const resolutionDouble = static_cast<double>( resolution );

            *pOutX = ( ( ( u + 1.0 ) * resolutionDouble ) - 1.0 ) * 0.5;
            *pOutY = ( ( ( 1.0 - v ) * resolutionDouble ) - 1.0 ) * 0.5;
        }

        //---------------------------------------------------------------------

        static size_t GetTexelIndex( uint32_t slice, uint32_t texelX, uint32_t texelY, uint32_t resolution )
        {
            return ( ( static_cast<size_t>( slice ) * resolution ) + texelY ) * resolution + texelX;
        }

        //---------------------------------------------------------------------

        static double GetTexelSolidAngle( double minU, double minV, double maxU, double maxV )
        {
            return GetCubeTexelSolidAngle( minU, minV, maxU, maxV );
        }

        //---------------------------------------------------------------------

        static double GetJacobian( double u, double v )
        {
            return GetCubeJacobian( u, v );
        }
    };

    //-------------------------------------------------------------------------

    template<>
    struct ProbeMapOf< CubeProjection >
    {
        static constexpr ProbeMap Value = ProbeMap::Cube;
    };
}
