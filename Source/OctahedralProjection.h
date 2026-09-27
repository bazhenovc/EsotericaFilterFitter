#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "Assert.h"
#include "ProbeMap.h"

// octahedral base map, single slice
//-------------------------------------------------------------------------
// The sphere mapped to an octahedron, cut along the two midlines and unwrapped into ONE square texture.
// This is the parameterization the engine already uses for point light shadows, described in Docs/Rendering/Point Light Shadows.md after Praun and Hoppe ( 2003 ) with the form given by Cigolle et al.: normalize, divide by the L1 norm, and reflect the back hemisphere into the square's corners.
//
//      front hemisphere    p.z >= 0        radius <= 1        the square's central diamond, four triangles
//      back hemisphere     p.z <  0        radius >  1        the square's four corners, one triangle each
//
// A level is one R x R texture covering the whole sphere, so NumFaces is 8 and NumSlices is 1: the eight octant faces share one coordinate space and the face a texel belongs to is a CONSEQUENCE of where the texel is, not an argument. See MapProjection.h for why faces and slices are counted separately.
//
//  WHY THIS PARAMETERIZATION
//
// The two midlines are the square's u = 0 and v = 0 axes, and the diamond's edge is |u| + |v| = 1. Each of the eight regions therefore has FIXED AXIS SIGNS, which makes the L1 norm linear in world position across it, so edges stay straight, depth interpolates exactly and the texel footprint stays near-uniform where the naive folding distorts badly.
//
//  THREE FRAMES, NOT EIGHT
//
// The taps are organised by the three axial frames, exactly as they are for a cubemap, because the tap construction works in DIRECTION space and the map only decides where a texel is.
// The octahedron's eight faces are a fact about the texture, not about the filter, so they appear in the mapping half of this contract ( NumFaces, GetSliceFaces, GetFaceRegion ) and not in the frame half.
//
//  MEASURED
//
// One frame per octant face was tried, with eight frames and four taps per axis to stay inside the frame budget, and it is a worse model: 0.1264 mean relative L1 against 0.0946 for these three frames with eight taps, at the same fit configuration, worse at every level and worse on the objective as well.
// The map checks pass either way, so both frame sets are valid answers to the contract; the axial frames simply fit better. Recorded here so the experiment is not repeated.
//
// FRAME WEIGHTS AND POLYNOMIAL ARGUMENTS ARE THE CUBE'S
//
// Both take the direction as a vector whose LARGEST component is 1, which is the convention a cube face coordinate already has and which is what makes the cube's weight rule keep at least two frames active everywhere.
// An octahedral direction is L1-normalized instead, so it is scaled by its own major component before either of the two is applied. That is the whole difference; the rules themselves are the cube's and are copied from CubeProjection deliberately rather than reinvented, because they are known to have the property the construction needs.
//
//  EXTENT
//
// The map's coordinate space is u, v in [-1, 1], an extent of 2 per axis, which the cube's face space also is.
// That is what fixes the constant in GetInverseTexelSolidAngle and in GetLevelCorrection: a texel's uv area is 4 / R^2, not 1 / R^2 as it is on a single-slice tetrahedral map whose tile spans 1.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    //  Diagnostic scale on this map's level correction, 1.0 in every normal run
    //-------------------------------------------------------------------------
    //  It exists so the correction's scale can be swept against measured error without a rebuild.
    //  A level-independent error floor that survives a change of base resolution and a change of fit effort is the signature of a modelling constant being wrong rather than of the fit, and the level correction is the constant this map has that the cube and the tetrahedral map do not.
    //  Scaling it to zero, to one and to two answers whether the floor tracks it; a floor that does not move is not the correction's.
    //
    //  Nothing but GetLevelCorrection reads it, the flag that sets it is documented as diagnostic, and no shipped configuration sets it away from 1.
    //-------------------------------------------------------------------------

    extern double g_octahedralLevelCorrectionScale;

    struct OctahedralProjection
    {
        static constexpr uint32_t NumFaces = 8;
        static constexpr uint32_t NumFrames = 3;
        static constexpr uint32_t NumSlices = 1;

        // The cull threshold on the largest of the two components the frame does not own, after major normalization. 0.75 mirrors the cube; 1 or above would let all three frames cull at once and the runtime would divide by a zero weight sum.
        static constexpr double kBlendThreshold = 0.75;

        // The map's coordinate extent per axis. Every constant below that looks arbitrary is this number.
        static constexpr double kExtent = 2.0;

        //---------------------------------------------------------------------
        //  Faces
        //
        //      face = ( back ? 4 : 0 ) + ( u >= 0 ? 1 : 0 ) + ( v >= 0 ? 2 : 0 )
        //
        // so 0..3 are the central diamond's four triangles in order (-,-), (+,-), (-,+), (+,+) and 4..7 are the corner triangles with the same sign order.
        // The sign of zero is taken as positive, which is only ambiguous for a direction exactly on a midline; those directions lie on a region boundary, where both regions agree on the direction.
        //---------------------------------------------------------------------

        static uint32_t GetFaceFromUV( double u, double v )
        {
            bool const back = ( std::fabs( u ) + std::fabs( v ) ) > 1.0;

            return GetFaceFromSigns( ( u >= 0.0 ), ( v >= 0.0 ), back );
        }

        //---------------------------------------------------------------------

        static uint32_t GetFaceFromSigns( bool uPositive, bool vPositive, bool back )
        {
            return ( back ? 4u : 0u ) + ( uPositive ? 1u : 0u ) + ( vPositive ? 2u : 0u );
        }

        //---------------------------------------------------------------------

        static double GetFaceSignU( uint32_t face )
        {
            FF_ASSERT( face < NumFaces );

            return ( ( face & 1u ) != 0u ) ? 1.0 : -1.0;
        }

        //---------------------------------------------------------------------

        static double GetFaceSignV( uint32_t face )
        {
            FF_ASSERT( face < NumFaces );

            return ( ( face & 2u ) != 0u ) ? 1.0 : -1.0;
        }

        //---------------------------------------------------------------------

        static bool IsBackFace( uint32_t face )
        {
            FF_ASSERT( face < NumFaces );

            return ( face & 4u ) != 0u;
        }

        //  Coordinate to direction
        //
        //  Normalized by the L1 norm and not by the Euclidean one, so the returned vector satisfies | p |_1 = 1 in BOTH regions:
        //
        //      front   p = ( signU * |u|, signV * |v|, 1 - |u| - |v| )
        //      back    p = ( signU * ( 1 - |v| ), signV * ( 1 - |u| ), 1 - |u| - |v| )
        //
        // The back form is the front form's reflection into the corners, which is what makes the two meet continuously along |u| + |v| = 1: there the front gives ( signU*|u|, signV*|v|, 0 ) and the back gives ( signU*( 1 - |v| ), signV*( 1 - |u| ), 0 ), and those are equal exactly when |u| + |v| = 1.
        //
        // Affine in the region's own coordinates and therefore valid OUTSIDE the region as well, which the tap construction and the solid-angle clipping both rely on: a tap offsets by a fraction of a texel and can leave the region, and a texel-sized box at the square's edge leaves the square.
        //-------------------------------------------------------------------------

        //  The region comes from the COORDINATE, not from the argument
        //-------------------------------------------------------------------------
        //  The face argument is redundant for this map: the front regions are the central diamond and the back ones are the corners, so where a coordinate IS says which formula names its direction.
        //
        //  That is not pedantry, it is the fold. A mip reduction's phantom texel is a real coordinate one step outside the texel it came from, and past the square's outer boundary the correct direction is the octant on the other side of the equator, mirrored.
        //  The back formula gives exactly that for a coordinate just past the edge: u = 1 + e becomes x = signU * ( 1 - |v| ), y = signV * e, z = -e, and resolving that direction back through GetFaceAndUVFromDirection lands inside the square on the neighbouring octant.
        //
        //  Deriving the formula from the CALLER'S face instead extrapolates the caller's own plane past the equator, which is a different point on the sphere, so a reduction averages a tap that is not a neighbour. That is the seam-local deviation this map's bspline check used to report, and it is why the deviation was the same size at every base resolution, training grid, start count and correction scale: it lives on one texel row per level, not in the fit.
        //
        //  Neither of the other maps has that problem - a cube face's coordinates continue onto the neighbouring face's plane, and a tetrahedral tile's edges are real adjacencies - so neither of them derives anything from the coordinate here.
        //-------------------------------------------------------------------------

        static void GetDirectionFromUV( double* pOutDirection, double u, double v, uint32_t face )
        {
            // Not consulted, and it stays in the signature because the contract has it and the other two maps genuinely need it. Every caller passes either a region vertex or a point inside the region, where the coordinate's own region and the argument agree.
            (void) face;

            GetDirectionFromUV( pOutDirection, u, v );
        }

        //---------------------------------------------------------------------

        static void GetDirectionFromUV( double* pOutDirection, double u, double v )
        {
            FF_ASSERT( pOutDirection != nullptr );

            uint32_t const face = GetFaceFromUV( u, v );

            double const signU = GetFaceSignU( face );
            double const signV = GetFaceSignV( face );

            double const magnitudeU = std::fabs( u );
            double const magnitudeV = std::fabs( v );

            if ( IsBackFace( face ) )
            {
                pOutDirection[0] = signU * ( 1.0 - magnitudeV );
                pOutDirection[1] = signV * ( 1.0 - magnitudeU );
            }
            else
            {
                pOutDirection[0] = signU * magnitudeU;
                pOutDirection[1] = signV * magnitudeV;
            }

            pOutDirection[2] = 1.0 - magnitudeU - magnitudeV;
        }

        //  Direction to coordinate
        //
        //  The inverse of the pair above, and the form the engine's shaders use: divide by the L1 norm, then either keep the x and y components or reflect them into the corners.
        //  A direction with a zero component keeps the coordinate it already has, which is what signNotZero is for; the only directions affected are exactly on a midline, where the coordinate is the same either way.
        //-------------------------------------------------------------------------

        static void GetFaceAndUVFromDirection( double const* pDirection, uint32_t& face, double& u, double& v )
        {
            FF_ASSERT( pDirection != nullptr );

            double const l1Norm = std::fabs( pDirection[0] ) + std::fabs( pDirection[1] ) + std::fabs( pDirection[2] );

            FF_ASSERT( l1Norm > 0.0 );

            double const inverse = 1.0 / l1Norm;

            double const normalizedX = pDirection[0] * inverse;
            double const normalizedY = pDirection[1] * inverse;
            double const normalizedZ = pDirection[2] * inverse;

            if ( normalizedZ >= 0.0 )
            {
                u = normalizedX;
                v = normalizedY;
                face = GetFaceFromSigns( ( u >= 0.0 ), ( v >= 0.0 ), false );
            }
            else
            {
                u = ( 1.0 - std::fabs( normalizedY ) ) * SignNotZero( normalizedX );
                v = ( 1.0 - std::fabs( normalizedX ) ) * SignNotZero( normalizedY );
                face = GetFaceFromSigns( ( u >= 0.0 ), ( v >= 0.0 ), true );
            }

            // The square's own bounds. The round trip is exact inside them; clamping only matters for a direction that is numerically at a corner.
            u = Clamp( u, -1.0, 1.0 );
            v = Clamp( v, -1.0, 1.0 );
        }

        //---------------------------------------------------------------------
        //  Frames
        //---------------------------------------------------------------------

        static void GetFrameUp( double* pOutUp, uint32_t frame )
        {
            FF_ASSERT( frame < NumFrames );
            FF_ASSERT( pOutUp != nullptr );

            pOutUp[0] = 0.0;
            pOutUp[1] = 0.0;
            pOutUp[2] = 0.0;

            pOutUp[frame] = 1.0;
        }

        //---------------------------------------------------------------------

        static double GetFrameWeight( double const* pDirection, uint32_t frame )
        {
            FF_ASSERT( frame < NumFrames );

            double major[3];
            MajorNormalized( major, pDirection );

            uint32_t const otherFrame0 = 1 - ( frame & 1u ) - ( frame >> 1 );
            uint32_t const otherFrame1 = 2 - ( frame >> 1 );

            double const absOther0 = std::fabs( major[otherFrame0] );
            double const absOther1 = std::fabs( major[otherFrame1] );

            double const maxOther = ( absOther0 > absOther1 ) ? absOther0 : absOther1;

            return ( maxOther - kBlendThreshold ) / ( 1.0 - kBlendThreshold );
        }

        //---------------------------------------------------------------------

        static void GetPolynomialArguments( double const* pDirection, uint32_t frame, double& theta, double& phi )
        {
            FF_ASSERT( frame < NumFrames );

            theta = 0.0;
            phi = 0.0;

            double major[3];
            MajorNormalized( major, pDirection );

            uint32_t const otherFrame0 = 1 - ( frame & 1u ) - ( frame >> 1 );
            uint32_t const otherFrame1 = 2 - ( frame >> 1 );

            double const nx = major[otherFrame0];
            double const ny = major[otherFrame1];
            double const nz = std::fabs( major[frame] );

            double const maxXY = ( std::fabs( ny ) > std::fabs( nx ) ) ? std::fabs( ny ) : std::fabs( nx );

            if ( maxXY <= 0.0 )
            {
                return;
            }

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

        //  The level correction
        //-------------------------------------------------------------------------
        //  A tap is placed in the frame's coordinates and read from a mip chain, and the two do not have the same density, so this converts a distance in frame coordinates into the mip level whose texel density matches there.
        //
        //  The generic law, from MapProjection.h, is
        //
        //      correction = -0.5 * log2( dOmega / dudv )
        //
        //  up to a constant that the table's own level coefficient absorbs.
        //
        //  DERIVATION
        //
        //  In one octant the map is affine: p ( u, v ) = ( signU * u', signV * v', 1 - u' - v' ) in the front, so
        //
        //      dp/du = ( signU, 0, -1 )        dp/dv = ( 0, signV, -1 )
        //      cross = ( signU * signV, signU * signV, signU * signV )      | cross | = sqrt( 3 )
        //      p . cross = signU * signV
        //
        //  and the solid-angle element is ( p . cross ) / | p |^3 dudv, so
        //
        //      dOmega / dudv = 1 / | p |^3
        //
        //  in the FRONT region. The back region gives the same expression: its p is ( signU * ( 1 - |v| ), signV * ( 1 - |u| ), 1 - |u| - |v| ), whose derivatives are ( 0, -signV, -1 ) and ( -signU, 0, -1 ), whose cross is ( signU * signV ) * ( 1, 1, -1 ) with the same length sqrt( 3 ), and whose p . cross is again signU * signV.
        //
        //  So the whole map obeys dOmega/dudv = 1 / | p |^3, and therefore
        //
        //      correction = 1.5 * log2( | p | )
        //
        //  with | p | the Euclidean length of the L1-normalized direction.
        //
        //  SIGN AND RANGE
        //
        //  | p | is at least 1/sqrt(3), at a point where the L1 mass is spread evenly, and at most 1, where it is all in one component: that is the diamond's four vertices and the square's four corners, both of which are where the map's texels are SMALLEST.
        //  So the correction is zero at the eight points where the texels are smallest and falls to 1.5 * log2( 1/sqrt(3) ) = -0.75 * log2( 3 ) in the middle of the four corner triangles, and it is never positive.
        //  That is the opposite sign to the cube's, whose correction is positive at a face corner - it has to be, because on a cube the corner is where the texels are smallest and the mip has to go coarser there. Here the smallest texels need no correction at all and every other point needs a finer mip.
        //-------------------------------------------------------------------------

        static double GetLevelCorrection( double const* pSampleDirection )
        {
            double normalized[3];
            L1Normalized( normalized, pSampleDirection );

            double const length = std::sqrt( ( normalized[0] * normalized[0] ) + ( normalized[1] * normalized[1] ) + ( normalized[2] * normalized[2] ) );

            if ( length <= 0.0 )
            {
                return 0.0;
            }

            return g_octahedralLevelCorrectionScale * 1.5 * std::log2( length );
        }

        //  1 / ( the solid angle one texel covers ), which is what a sample's own footprint is divided by to pick a source mip.
        //
        //  One texel covers ( dOmega/dudv ) * ( the uv area of one texel ), and this map's coordinate extent is 2 per axis, so that area is 4 / R^2 and one texel covers 4 / ( | p |^3 R^2 ).
        //  The reciprocal is therefore R^2 * | p |^3 / 4, whose whole variation is the +1.5 * log2( | p | ) that GetLevelCorrection adds - the two are readings of one quantity and cannot disagree in sign.
        //-------------------------------------------------------------------------

        static double GetInverseTexelSolidAngle( double const* pDirection, uint32_t baseResolution )
        {
            double normalized[3];
            L1Normalized( normalized, pDirection );

            double const lengthSquared = ( normalized[0] * normalized[0] ) + ( normalized[1] * normalized[1] ) + ( normalized[2] * normalized[2] );
            double const lengthCubed = lengthSquared * std::sqrt( lengthSquared );

            if ( lengthCubed <= 0.0 )
            {
                return 0.0;
            }

            double const resolution = static_cast<double>( baseResolution );

            return ( resolution * resolution ) * lengthCubed / 4.0;
        }

        //  The table level at which a zero-width ( mirror ) level samples the unfiltered source at every direction: the negative of the largest correction GetLevelCorrection can add.
        //
        //  That largest value is 0 here, at the diamond's vertices and the square's corners, so this is 0.
        //  It is not a default and not an oversight: a level is a constant added to every tap's correction, and the requirement is that level + correction <= 0 at every direction so the sampler stays at or below mip 0. Every correction here is already <= 0, so level 0 satisfies it, where a cubemap needs -0.75 * log2( 3 ) and a tetrahedral map needs -1.5 * log2( 3 ) because on both of those the correction goes POSITIVE somewhere.
        //-------------------------------------------------------------------------

        static double GetMirrorLevel()
        {
            return 0.0;
        }

        //---------------------------------------------------------------------
        //  Mapping: where the texels are and what they cover
        //---------------------------------------------------------------------

        static void GetDirection( double* pOutDirection, uint32_t slice, uint32_t texelX, uint32_t texelY, uint32_t resolution )
        {
            double centreU = 0.0;
            double centreV = 0.0;
            GetTexelCentreUV( &centreU, &centreV, texelX, texelY, resolution );

            GetTexelDirection( pOutDirection, slice, centreU, centreV );
        }

        //---------------------------------------------------------------------

        static void GetTexelDirection( double* pOutDirection, uint32_t slice, double u, double v )
        {
            // Single slice, so the slice is not part of the address and the coordinate names its own face
            (void) slice;

            GetDirectionFromUV( pOutDirection, u, v );
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

            // The map's extent is [-1, 1], and both axes run with the texel index, which is the single-slice convention: a cube's v runs down a face and this map's does not.
            *pOutU = ( ( 2.0 * static_cast<double>( texelX ) ) + 1.0 ) * inverseResolution - 1.0;
            *pOutV = ( ( 2.0 * static_cast<double>( texelY ) ) + 1.0 ) * inverseResolution - 1.0;
        }

        //---------------------------------------------------------------------

        static void GetTexelCoordinateFromUV( double* pOutX, double* pOutY, double u, double v, uint32_t resolution )
        {
            double const resolutionDouble = static_cast<double>( resolution );

            *pOutX = ( ( ( u + 1.0 ) * resolutionDouble ) - 1.0 ) * 0.5;
            *pOutY = ( ( ( v + 1.0 ) * resolutionDouble ) - 1.0 ) * 0.5;
        }

        //---------------------------------------------------------------------

        static void GetTexelStep( double* pOutStepU, double* pOutStepV, uint32_t resolution )
        {
            double const step = kExtent / static_cast<double>( resolution );

            *pOutStepU = step;
            *pOutStepV = step;
        }

        //---------------------------------------------------------------------

        static size_t GetTexelIndex( uint32_t slice, uint32_t texelX, uint32_t texelY, uint32_t resolution )
        {
            // Single slice, so the slice is not part of the address and the eight faces share the square's coordinates
            (void) slice;

            return ( static_cast<size_t>( texelY ) * resolution ) + texelX;
        }

        //---------------------------------------------------------------------

        static uint32_t GetSliceFaces( uint32_t slice, uint32_t* pOutFace )
        {
            FF_ASSERT( slice < NumSlices );

            for ( uint32_t face = 0; face < NumFaces; ++face )
            {
                pOutFace[face] = face;
            }

            return NumFaces;
        }

        //  The region of a box that one face owns
        //-------------------------------------------------------------------------
        //  A slice holds all eight faces, so a box straddles two of them wherever it crosses a midline or the diamond's edge.
        //  Each face's region is the intersection of three half-planes - two for the signs the octant fixes and one for the hemisphere - so this is the tetrahedral map's polygon clip with three planes instead of two.
        //-------------------------------------------------------------------------

        static uint32_t GetFaceRegion
        (
            uint32_t face, double minU, double minV, double maxU, double maxV,
            double* pOutU, double* pOutV
        )
        {
            FF_ASSERT( face < NumFaces );

            double halfPlanes[3][3];
            GetFaceHalfPlanes( face, halfPlanes );

            double currentU[8];
            double currentV[8];
            uint32_t currentCount = 0;

            double nextU[8];
            double nextV[8];
            uint32_t nextCount = 0;

            currentU[0] = minU;
            currentV[0] = minV;
            currentU[1] = maxU;
            currentV[1] = minV;
            currentU[2] = maxU;
            currentV[2] = maxV;
            currentU[3] = minU;
            currentV[3] = maxV;
            currentCount = 4;

            for ( uint32_t plane = 0; plane < 3; ++plane )
            {
                ClipPolygon
                (
                    currentU, currentV, currentCount,
                    halfPlanes[plane][0], halfPlanes[plane][1], halfPlanes[plane][2],
                    nextU, nextV, nextCount
                );

                for ( uint32_t index = 0; index < nextCount; ++index )
                {
                    currentU[index] = nextU[index];
                    currentV[index] = nextV[index];
                }

                currentCount = nextCount;

                if ( currentCount == 0 )
                {
                    break;
                }
            }

            for ( uint32_t index = 0; index < currentCount; ++index )
            {
                pOutU[index] = currentU[index];
                pOutV[index] = currentV[index];
            }

            return currentCount;
        }

        //  Exact solid angle of a coordinate box.
        //
        //  Independent of GetJacobian on purpose: the box is clipped into one convex piece per face, each piece is mapped through its own region and each is summed as spherical triangles.
        //  Integrating the Jacobian instead would make the law check agree by construction and it would be circular.
        //---------------------------------------------------------------------

        static double GetTexelSolidAngle( double minU, double minV, double maxU, double maxV )
        {
            double total = 0.0;

            for ( uint32_t face = 0; face < NumFaces; ++face )
            {
                double regionU[8];
                double regionV[8];
                uint32_t const regionCount = GetFaceRegion( face, minU, minV, maxU, maxV, regionU, regionV );

                if ( regionCount < 3 )
                {
                    continue;
                }

                double direction[8][3];

                for ( uint32_t index = 0; index < regionCount; ++index )
                {
                    GetDirectionFromUV( direction[index], regionU[index], regionV[index], face );
                }

                for ( uint32_t index = 1; ( index + 1 ) < regionCount; ++index )
                {
                    total += GetTriangleSolidAngle( direction[0], direction[index], direction[index + 1] );
                }
            }

            return total;
        }

        //  dOmega/dudv, in closed form rather than finite-differenced: the derivation is at GetLevelCorrection and it holds over both regions.
        //---------------------------------------------------------------------

        static double GetJacobian( double u, double v )
        {
            double direction[3];
            GetDirectionFromUV( direction, u, v );

            double const lengthSquared = ( direction[0] * direction[0] ) + ( direction[1] * direction[1] ) + ( direction[2] * direction[2] );
            double const lengthCubed = lengthSquared * std::sqrt( lengthSquared );

            if ( lengthCubed <= 0.0 )
            {
                return 0.0;
            }

            return 1.0 / lengthCubed;
        }

    private:

        //  A u + B v + C >= 0, three per face: the two sign half-planes and the hemisphere one.
        // One definition, because the region a face owns and the region whose exact solid angle is computed have to be the same region.
        //---------------------------------------------------------------------

        static void GetFaceHalfPlanes( uint32_t face, double( &outPlanes )[3][3] )
        {
            FF_ASSERT( face < NumFaces );

            double const signU = GetFaceSignU( face );
            double const signV = GetFaceSignV( face );

            outPlanes[0][0] = signU;
            outPlanes[0][1] = 0.0;
            outPlanes[0][2] = 0.0;

            outPlanes[1][0] = 0.0;
            outPlanes[1][1] = signV;
            outPlanes[1][2] = 0.0;

            if ( IsBackFace( face ) )
            {
                // Outside the diamond: signU * u + signV * v - 1 >= 0
                outPlanes[2][0] = signU;
                outPlanes[2][1] = signV;
                outPlanes[2][2] = -1.0;
            }
            else
            {
                // Inside the diamond: 1 - signU * u - signV * v >= 0
                outPlanes[2][0] = -signU;
                outPlanes[2][1] = -signV;
                outPlanes[2][2] = 1.0;
            }
        }

        //---------------------------------------------------------------------

        static double SignNotZero( double value )
        {
            // Zero is positive, which is the convention the engine's shaders use. The only coordinates this changes are on a midline, where both signs name the same direction.
            return ( value >= 0.0 ) ? 1.0 : -1.0;
        }

        //---------------------------------------------------------------------

        static double Clamp( double value, double minimum, double maximum )
        {
            return ( value < minimum ) ? minimum : ( ( value > maximum ) ? maximum : value );
        }

        //  Scales a direction by its L1 norm, which is the convention the MAP's law is written in: the parameterization maps an L1-normalized vector, and dOmega/dudv = 1 / | p |^3 is a statement about that p.
        //  Scale-invariant, so a unit direction or a region's affine output both land on the same p.
        //  This is a different normalization from MajorNormalized below, and using the wrong one is a silent 1.5 * log2( ratio ) error in every tap's mip, so the two are named apart deliberately.
        //---------------------------------------------------------------------

        static void L1Normalized( double* pOut, double const* pDirection )
        {
            FF_ASSERT( pOut != nullptr );

            double const l1Norm = std::fabs( pDirection[0] ) + std::fabs( pDirection[1] ) + std::fabs( pDirection[2] );

            if ( l1Norm <= 0.0 )
            {
                pOut[0] = 0.0;
                pOut[1] = 0.0;
                pOut[2] = 1.0;
                return;
            }

            double const inverse = 1.0 / l1Norm;

            pOut[0] = pDirection[0] * inverse;
            pOut[1] = pDirection[1] * inverse;
            pOut[2] = pDirection[2] * inverse;
        }

        //  Scales a direction so its largest component is 1, which is the convention the cube's frame rules are written against.
        //  The component is divided by its SIGNED value so the result is +1 there, matching a cube face coordinate, whose constant 1 is what makes the weight rule work.
        //  Scale-invariant, so it is correct for a unit direction, for an L1-normalized one and for a cube face vector alike.
        //---------------------------------------------------------------------

        static void MajorNormalized( double* pOut, double const* pDirection )
        {
            FF_ASSERT( pOut != nullptr );

            double const absX = std::fabs( pDirection[0] );
            double const absY = std::fabs( pDirection[1] );
            double const absZ = std::fabs( pDirection[2] );

            double const absMajor = ( absX > absY ) ? ( ( absX > absZ ) ? absX : absZ ) : ( ( absY > absZ ) ? absY : absZ );

            if ( absMajor <= 0.0 )
            {
                pOut[0] = 1.0;
                pOut[1] = 0.0;
                pOut[2] = 0.0;
                return;
            }

            double const major = ( absMajor == absX ) ? pDirection[0] : ( ( absMajor == absY ) ? pDirection[1] : pDirection[2] );
            double const inverse = 1.0 / major;

            pOut[0] = pDirection[0] * inverse;
            pOut[1] = pDirection[1] * inverse;
            pOut[2] = pDirection[2] * inverse;
        }

        //---------------------------------------------------------------------

        static double Dot3( double const* pLeft, double const* pRight )
        {
            return ( pLeft[0] * pRight[0] ) + ( pLeft[1] * pRight[1] ) + ( pLeft[2] * pRight[2] );
        }

        //---------------------------------------------------------------------

        static void Cross3( double* pOut, double const* pLeft, double const* pRight )
        {
            pOut[0] = ( pLeft[1] * pRight[2] ) - ( pLeft[2] * pRight[1] );
            pOut[1] = ( pLeft[2] * pRight[0] ) - ( pLeft[0] * pRight[2] );
            pOut[2] = ( pLeft[0] * pRight[1] ) - ( pLeft[1] * pRight[0] );
        }

        //  Sutherland-Hodgman against A u + B v + C >= 0.
        //  The tetrahedral projection has the same two helpers privately; they are repeated here rather than shared, because a projection is meant to be a self-contained answer to the contract in MapProjection.h and neither map should depend on the other.
        //---------------------------------------------------------------------

        static void ClipPolygon
        (
            double const* pInU, double const* pInV, uint32_t inCount,
            double a, double b, double c,
            double* pOutU, double* pOutV, uint32_t& outCount
        )
        {
            outCount = 0;

            if ( inCount == 0 )
            {
                return;
            }

            for ( uint32_t index = 0; index < inCount; ++index )
            {
                uint32_t const next = ( index + 1 ) % inCount;

                double const currentValue = ( a * pInU[index] ) + ( b * pInV[index] ) + c;
                double const nextValue = ( a * pInU[next] ) + ( b * pInV[next] ) + c;

                bool const currentInside = ( currentValue >= 0.0 );
                bool const nextInside = ( nextValue >= 0.0 );

                if ( currentInside )
                {
                    pOutU[outCount] = pInU[index];
                    pOutV[outCount] = pInV[index];
                    ++outCount;
                }

                if ( currentInside != nextInside )
                {
                    double const denominator = currentValue - nextValue;

                    if ( std::fabs( denominator ) > 1.0e-30 )
                    {
                        double const t = currentValue / denominator;

                        pOutU[outCount] = pInU[index] + ( t * ( pInU[next] - pInU[index] ) );
                        pOutV[outCount] = pInV[index] + ( t * ( pInV[next] - pInV[index] ) );
                        ++outCount;
                    }
                }
            }
        }

        //  Van Oosterom and Strackee. Exact for a planar triangle read from the origin, which every region triangle is.
        //---------------------------------------------------------------------

        static double GetTriangleSolidAngle( double const* pA, double const* pB, double const* pC )
        {
            double crossBC[3];
            Cross3( crossBC, pB, pC );

            double const numerator = Dot3( pA, crossBC );

            double const lengthA = std::sqrt( Dot3( pA, pA ) );
            double const lengthB = std::sqrt( Dot3( pB, pB ) );
            double const lengthC = std::sqrt( Dot3( pC, pC ) );

            double const denominator = ( lengthA * lengthB * lengthC )
                + ( Dot3( pA, pB ) * lengthC )
                + ( Dot3( pA, pC ) * lengthB )
                + ( Dot3( pB, pC ) * lengthA );

            return 2.0 * std::atan2( numerator, denominator );
        }
    };

    //-------------------------------------------------------------------------

    template<>
    struct ProbeMapOf< OctahedralProjection >
    {
        static constexpr ProbeMap Value = ProbeMap::Octahedral;
    };
}
