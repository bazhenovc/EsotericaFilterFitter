#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "Assert.h"
#include "ProbeMap.h"

// regular tetrahedron base map, single slice
//-------------------------------------------------------------------------
// Four triangular faces in ONE square texture.
// The square is cut by both diagonals into four triangles, each with two adjacent tile corners and the tile centre as its vertices, and each maps to one face of the tetrahedron:
//
//      T0   (0,0) (1,0) (1/2,1/2)      v <= u  and  u + v <= 1
//      T1   (1,0) (1,1) (1/2,1/2)      u >= v  and  u + v >= 1
//      T2   (1,1) (0,1) (1/2,1/2)      v >= u  and  u + v >= 1
//      T3   (0,1) (0,0) (1/2,1/2)      u <= v  and  u + v <= 1
//
// So the face is a consequence of the tile position and not an argument.
// That is what single-slice means: NumFaces is 4 and NumSlices is 1, and a consumersizes buffers by slices. See MapProjection.h.
//
// A point inside a tile triangle maps to its face by reusing the tile-space barycentric coordinates as weights on the face's three tetrahedron corners.
// The mapping is affine, which is what lets the Jacobian be exact rather than finite-differenced.
//
//  GEOMETRY
//
// Four vertices at distance 1 from the centre:
//
//      v0 = ( 1, 1, 1) / sqrt(3)      v2 = (-1, 1,-1) / sqrt(3)
//      v1 = ( 1,-1,-1) / sqrt(3)      v3 = (-1,-1, 1) / sqrt(3)
//
// Face k is opposite v_k: the plane p . v_k = -1/3, outward normal -v_k.
// The corner triples are wound so cross( b-a, c-a ) is antiparallel to v_k.
// Two of the four orderings a pattern suggests are inward, so each is stated rather than derived; an inward face makes every Jacobian on it negative.
//
// This orientation is a regular tetrahedron but it is NOT yet the one in Liao's tetrahedron mapping, which the TileBasedShadows reference demo implements and which is rotated so two face normals have no x component and two have no z.
// Switching is a change to GetVertex and to the tile-to-face assignment, and it waits on resolving that mapping's four rotation matrices, which are not pinned down. 
// Both the mapping and the implementation are shadow-map work rather than probe work; what this tool takes from them is the base map's geometry.
//
//  FRAMES
//
// Four frames, one per vertex axis, used directly rather than through abs: a tetrahedron has four distinct axis lines where a cube has three.
// The blend weight culls a frame the direction approaches:
//
//      weight_k = saturate( ( 0.75 - | d . v_k | ) / 0.75 )
//
// This keeps at least two frames active everywhere, and the threshold window is narrow.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    struct TetrahedralProjection
    {
        static constexpr uint32_t NumFaces = 4;
        static constexpr uint32_t NumFrames = 4;
        static constexpr uint32_t NumSlices = 1;

        static constexpr double kFaceOffset = 1.0 / 3.0;

        // Cull threshold on |d . v_k|. Above 2/3 keeps at least two frames active; at 1 or above all four can cull and the runtime divides by zero. 0.75 mirrors the cube.
        static constexpr double kBlendThreshold = 0.75;

        //---------------------------------------------------------------------

        static void GetVertex( uint32_t index, double* pOut )
        {
            static constexpr double kInverseRoot3 = 0.57735026918962576451;

            static constexpr int kSigns[4][3] =
            {
                {  1,  1,  1 },
                {  1, -1, -1 },
                { -1,  1, -1 },
                { -1, -1,  1 },
            };

            FF_ASSERT( index < 4 );

            pOut[0] = static_cast<double>( kSigns[index][0] ) * kInverseRoot3;
            pOut[1] = static_cast<double>( kSigns[index][1] ) * kInverseRoot3;
            pOut[2] = static_cast<double>( kSigns[index][2] ) * kInverseRoot3;
        }

        //---------------------------------------------------------------------

        static void GetFaceCorners( uint32_t face, double* pOutA, double* pOutB, double* pOutC )
        {
            static constexpr uint32_t kCorners[4][3] =
            {
                { 1, 3, 2 },
                { 0, 2, 3 },
                { 0, 3, 1 },
                { 0, 1, 2 },
            };

            FF_ASSERT( face < 4 );

            GetVertex( kCorners[face][0], pOutA );
            GetVertex( kCorners[face][1], pOutB );
            GetVertex( kCorners[face][2], pOutC );
        }

        //  The tile triangle owning a tile coordinate
        //---------------------------------------------------------------------

        static uint32_t GetFaceFromUV( double u, double v )
        {
            if ( u >= v )
            {
                return ( ( u + v ) >= 1.0 ) ? 1u : 0u;
            }

            return ( ( u + v ) >= 1.0 ) ? 2u : 3u;
        }

        //---------------------------------------------------------------------

        static void GetTileCorners( uint32_t face, double* pOutU, double* pOutV )
        {
            FF_ASSERT( face < 4 );

            static constexpr double kCorners[4][6] =
            {
                { 0.0, 0.0,  1.0, 0.0,  0.5, 0.5 },
                { 1.0, 0.0,  1.0, 1.0,  0.5, 0.5 },
                { 1.0, 1.0,  0.0, 1.0,  0.5, 0.5 },
                { 0.0, 1.0,  0.0, 0.0,  0.5, 0.5 },
            };

            pOutU[0] = kCorners[face][0];
            pOutV[0] = kCorners[face][1];
            pOutU[1] = kCorners[face][2];
            pOutV[1] = kCorners[face][3];
            pOutU[2] = kCorners[face][4];
            pOutV[2] = kCorners[face][5];
        }

        //  Affine, and valid outside the triangle as well: the tap construction and the solid-angle clipping both rely on the extrapolation.
        //---------------------------------------------------------------------

        static void GetDirectionFromUV( double* pOutDirection, double u, double v, uint32_t face )
        {
            double tileU[3];
            double tileV[3];
            GetTileCorners( face, tileU, tileV );

            double weight[3];
            SolveBarycentric2D( u, v, tileU, tileV, weight );

            double cornerA[3];
            double cornerB[3];
            double cornerC[3];
            GetFaceCorners( face, cornerA, cornerB, cornerC );

            double const* const corners[3] = { cornerA, cornerB, cornerC };

            for ( uint32_t component = 0; component < 3; ++component )
            {
                pOutDirection[component] = ( weight[0] * corners[0][component] ) + ( weight[1] * corners[1][component] ) + ( weight[2] * corners[2][component] );
            }
        }

        //  What a consumer calls: a tile coordinate names its own face
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
            (void) slice;

            GetDirectionFromUV( pOutDirection, u, v, GetFaceFromUV( u, v ) );
        }

        //  Direction to face and tile coordinate
        //---------------------------------------------------------------------

        static void GetFaceAndUVFromDirection( double const* pDirection, uint32_t& face, double& u, double& v )
        {
            double const length = std::sqrt( Dot3( pDirection, pDirection ) );
            FF_ASSERT( length > 0.0 );

            double const direction[3] = { pDirection[0] / length, pDirection[1] / length, pDirection[2] / length };

            // The face the ray meets is the one whose outward normal is most aligned with it, which is the smallest dot with the opposite vertex
            double bestAlignment = 1.0e30;
            face = 0;

            for ( uint32_t candidate = 0; candidate < 4; ++candidate )
            {
                double vertex[3];
                GetVertex( candidate, vertex );

                double const alignment = Dot3( direction, vertex );

                if ( alignment < bestAlignment )
                {
                    bestAlignment = alignment;
                    face = candidate;
                }
            }

            // Intersect the plane p . v_face = -1/3 with the ray t * d.
            // The offset is negative on that plane, so the ray length is positive exactly when the alignment is negative, which it is for a visible face.
            double const rayLength = -kFaceOffset / bestAlignment;
            double const point[3] = { direction[0] * rayLength, direction[1] * rayLength, direction[2] * rayLength };

            double cornerA[3];
            double cornerB[3];
            double cornerC[3];
            GetFaceCorners( face, cornerA, cornerB, cornerC );

            double weight[3];

            if ( !SolveBarycentric3D( point, cornerA, cornerB, cornerC, weight ) )
            {
                u = 0.0;
                v = 0.0;
                return;
            }

            double tileU[3];
            double tileV[3];
            GetTileCorners( face, tileU, tileV );

            u = ( weight[0] * tileU[0] ) + ( weight[1] * tileU[1] ) + ( weight[2] * tileU[2] );
            v = ( weight[0] * tileV[0] ) + ( weight[1] * tileV[1] ) + ( weight[2] * tileV[2] );

            u = Clamp01( u );
            v = Clamp01( v );
        }

        //---------------------------------------------------------------------

        static void GetTexelCentreUV( double* pOutU, double* pOutV, uint32_t texelX, uint32_t texelY, uint32_t resolution )
        {
            GetTexelCentreUVOutside( pOutU, pOutV, static_cast<int32_t>( texelX ), static_cast<int32_t>( texelY ), resolution );
        }

        //---------------------------------------------------------------------

        static void GetTexelCentreUVOutside( double* pOutU, double* pOutV, int32_t texelX, int32_t texelY, uint32_t resolution )
        {
            double const resolutionDouble = static_cast<double>( resolution );

            *pOutU = ( static_cast<double>( texelX ) + 0.5 ) / resolutionDouble;
            *pOutV = ( static_cast<double>( texelY ) + 0.5 ) / resolutionDouble;
        }

        //---------------------------------------------------------------------

        static void GetTexelCoordinateFromUV( double* pOutX, double* pOutY, double u, double v, uint32_t resolution )
        {
            double const resolutionDouble = static_cast<double>( resolution );

            *pOutX = ( u * resolutionDouble ) - 0.5;
            *pOutY = ( v * resolutionDouble ) - 0.5;
        }

        //  Both axes run with the texel index over this map's tile, which is the opposite v convention to a cube face.
        //---------------------------------------------------------------------

        static void GetTexelStep( double* pOutStepU, double* pOutStepV, uint32_t resolution )
        {
            double const inverseResolution = 1.0 / static_cast<double>( resolution );

            *pOutStepU = inverseResolution;
            *pOutStepV = inverseResolution;
        }

        //---------------------------------------------------------------------

        static size_t GetTexelIndex( uint32_t slice, uint32_t texelX, uint32_t texelY, uint32_t resolution )
        {
            // Single slice, so the slice is not part of the address and the four faces share the tile's coordinates
            (void) slice;

            return ( static_cast<size_t>( texelY ) * resolution ) + texelX;
        }

        //  The tile box a slice covers, and which part of it each face owns.
        //
        //  A slice holds all four faces, and a box straddles two of them wherever it crosses a diagonal.
        // Each face owns the part of the box inside its own wedge, which is the region the two half-planes select.
        //---------------------------------------------------------------------

        static uint32_t GetSliceFaces( uint32_t slice, uint32_t* pOutFace )
        {
            FF_ASSERT( slice < NumSlices );

            for ( uint32_t face = 0; face < 4; ++face )
            {
                pOutFace[face] = face;
            }

            return 4;
        }

        //---------------------------------------------------------------------

        static uint32_t GetFaceRegion
        (
            uint32_t face, double minU, double minV, double maxU, double maxV,
            double* pOutU, double* pOutV
        )
        {
            double halfPlanes[2][3];
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

            for ( uint32_t plane = 0; plane < 2; ++plane )
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
            }

            for ( uint32_t index = 0; index < currentCount; ++index )
            {
                pOutU[index] = currentU[index];
                pOutV[index] = currentV[index];
            }

            return currentCount;
        }

        // Exact solid angle of a tile box.
        //
        // Independent of GetJacobian on purpose.
        // The box is clipped into one convex piece per face, each piece is mapped through its own triangle, and each is summed as spherical triangles.
        // Integrating the Jacobian instead would make the law check agree by construction and it would be circular.
        //---------------------------------------------------------------------

        static double GetTexelSolidAngle( double minU, double minV, double maxU, double maxV )
        {
            double total = 0.0;

            for ( uint32_t face = 0; face < 4; ++face )
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

        //  dOmega/dudv. The mapping is affine per triangle, so the derivatives are constant and this is exact rather than finite-differenced.
        //---------------------------------------------------------------------

        static double GetJacobian( double u, double v )
        {
            uint32_t const face = GetFaceFromUV( u, v );

            double weightHere[3];
            double weightU[3];
            double weightV[3];

            GetBarycentricAt( face, u, v, weightHere );
            GetBarycentricAt( face, u + 1.0, v, weightU );
            GetBarycentricAt( face, u, v + 1.0, weightV );

            double cornerA[3];
            double cornerB[3];
            double cornerC[3];
            GetFaceCorners( face, cornerA, cornerB, cornerC );

            double const* const corners[3] = { cornerA, cornerB, cornerC };

            double point[3] = { 0.0, 0.0, 0.0 };
            double derivativeU[3] = { 0.0, 0.0, 0.0 };
            double derivativeV[3] = { 0.0, 0.0, 0.0 };

            for ( uint32_t corner = 0; corner < 3; ++corner )
            {
                for ( uint32_t component = 0; component < 3; ++component )
                {
                    point[component] += weightHere[corner] * corners[corner][component];
                    derivativeU[component] += ( weightU[corner] - weightHere[corner] ) * corners[corner][component];
                    derivativeV[component] += ( weightV[corner] - weightHere[corner] ) * corners[corner][component];
                }
            }

            double cross[3];
            Cross3( cross, derivativeU, derivativeV );

            double const determinant = Dot3( point, cross );
            double const lengthSquared = Dot3( point, point );
            double const lengthCubed = lengthSquared * std::sqrt( lengthSquared );

            if ( lengthCubed <= 0.0 )
            {
                return 0.0;
            }

            return determinant / lengthCubed;
        }

        //  Tap construction
        //---------------------------------------------------------------------

        static double GetFrameWeight( double const* pDirection, uint32_t frame )
        {
            double vertex[3];
            GetVertex( frame, vertex );

            double const length = std::sqrt( Dot3( pDirection, pDirection ) );

            if ( length <= 0.0 )
            {
                return 0.0;
            }

            double const alignment = std::fabs( Dot3( pDirection, vertex ) ) / length;

            return Clamp01( ( kBlendThreshold - alignment ) / kBlendThreshold );
        }

        //---------------------------------------------------------------------

        static void GetFrameUp( double* pOutUp, uint32_t frame )
        {
            GetVertex( frame, pOutUp );
        }

        // The two polynomial arguments.
        //
        // Provisional. The cube's are a bespoke pair built from the unnormalized face vector, and they decide fit quality; whether these do is answered by fitting rather than by reasoning.
        // What they must do here is vary smoothly across a face and stay bounded, which dividing by the largest tangent-frame component gives.
        //---------------------------------------------------------------------

        static void GetPolynomialArguments( double const* pDirection, uint32_t frame, double& theta, double& phi )
        {
            theta = 0.0;
            phi = 0.0;

            double up[3];
            GetFrameUp( up, frame );

            double const length = std::sqrt( Dot3( pDirection, pDirection ) );

            if ( length <= 0.0 )
            {
                return;
            }

            double const frameZ[3] = { pDirection[0] / length, pDirection[1] / length, pDirection[2] / length };

            double frameX[3];
            Cross3( frameX, up, frameZ );

            double const frameXLength = std::sqrt( Dot3( frameX, frameX ) );

            if ( frameXLength <= 0.0 )
            {
                return;
            }

            frameX[0] /= frameXLength;
            frameX[1] /= frameXLength;
            frameX[2] /= frameXLength;

            double frameY[3];
            Cross3( frameY, frameZ, frameX );

            double const componentX = Dot3( pDirection, frameX );
            double const componentY = Dot3( pDirection, frameY );
            double const componentZ = Dot3( pDirection, frameZ );

            double const largest = std::max( std::fabs( componentX ), std::max( std::fabs( componentY ), std::fabs( componentZ ) ) );

            if ( largest <= 0.0 )
            {
                return;
            }

            theta = componentX / largest;
            phi = componentY / largest;
        }

        // The level correction: -0.5 * log2( dOmega/dudv ), up to the constant that folds into the table's own level coefficient.
        // Both maps are  planar-faced projections, so both obey ( n . p ) / |p|^3, and on a tetrahedron face n . p is the constant 1/3 while |p| is ( 1/3 ) / ( n . L ).
        //
        // SIGN
        // 
        // The correction is read off the Jacobian, and a tap landing where the texels are SMALLER needs a COARSER mip, exactly as it does on a cube: on a cube the correction is 0.75 * log2( |d|^2 ) = -1.5 * log2( major ), which is positive at a corner, where the texels are smallest.
        // Here the texels are smallest at a face's corners, where n . L is 1/3, so the correction is -1.5 * log2( n . L ) - positive there and zero at a face centre.
        // The opposite sign is not a different convention, it is a mip chosen a full span away from the right one at every tap away from a face centre, and it is invisible to every invariant the construction checks: the taps still sum to one and a constant still blurs to a constant.
        //---------------------------------------------------------------------

        static double GetLevelCorrection( double const* pSampleDirection )
        {
            double const length = std::sqrt( Dot3( pSampleDirection, pSampleDirection ) );

            if ( length <= 0.0 )
            {
                return 0.0;
            }

            uint32_t face = 0;
            double u = 0.0;
            double v = 0.0;
            GetFaceAndUVFromDirection( pSampleDirection, face, u, v );

            double vertex[3];
            GetVertex( face, vertex );

            double const alignment = -Dot3( pSampleDirection, vertex ) / length;

            // Guarded: the alignment is a cosine against the face normal, so it is 1/3 at the face's corners and 1 at its centre
            double const clamped = ( alignment > 1.0e-6 ) ? alignment : 1.0e-6;

            return -1.5 * std::log2( clamped );
        }

        // 1 / (the solid angle one texel covers), which is what a sample's own footprint is divided by to pick a source mip.
        //
        // A slice is the whole tile, so one texel covers J / R^2 with R the base resolution and J the map's Jacobian - the tile is a unit square, so its  texel's uv area is 1 / R^2 rather than the cube's 4 / R^2.
        // J is 41.57 * ( n . L )^3 here ( 12*sqrt(3) is the same constant for a unit-area triangle, half of this one because the tile triangle is the quarter-area one ), so the reciprocal varies as +1.5 * log2( n . L ) ... which is the negative of GetLevelCorrection, as it has to be: the two are -0.5 * log2( J ) and R^2 / J.
        //---------------------------------------------------------------------

        static double GetInverseTexelSolidAngle( double const* pDirection, uint32_t baseResolution )
        {
            uint32_t face = 0;
            double u = 0.0;
            double v = 0.0;
            GetFaceAndUVFromDirection( pDirection, face, u, v );

            double const jacobian = GetJacobian( u, v );

            if ( jacobian <= 0.0 )
            {
                return 0.0;
            }

            double const resolution = static_cast<double>( baseResolution );

            return ( resolution * resolution ) / jacobian;
        }

        //  The table level at which a zero-width (mirror) level samples the unfiltered source at every direction: the negative of the largest correction GetLevelCorrection can add.
        //
        //  Here the correction is -1.5 * log2( n . L ) with n . L at least 1/3, at the face's corners, so the largest it adds is 1.5 * log2( 3 ) - exactly double the cube's 0.75 * log2( 3 ), and for the same reason: the tetrahedron's Jacobian varies over a 27:1 range where the cube's varies over 5.2:1.
        //---------------------------------------------------------------------

        static double GetMirrorLevel()
        {
            return -1.5 * std::log2( 3.0 );
        }

    private:

        //  A u + B v + C >= 0, two per face, being the two diagonals with the signs that select that face.
        // One definition, because the region a face owns and the region whose exact solid angle is computed have to be the same region.
        //---------------------------------------------------------------------

        static void GetFaceHalfPlanes( uint32_t face, double( &outPlanes )[2][3] )
        {
            FF_ASSERT( face < 4 );

            static constexpr double kHalfPlanes[4][2][3] =
            {
                { {  1.0, -1.0,  0.0 }, { -1.0, -1.0,  1.0 } },
                { {  1.0, -1.0,  0.0 }, {  1.0,  1.0, -1.0 } },
                { { -1.0,  1.0,  0.0 }, {  1.0,  1.0, -1.0 } },
                { { -1.0,  1.0,  0.0 }, { -1.0, -1.0,  1.0 } },
            };

            for ( uint32_t plane = 0; plane < 2; ++plane )
            {
                for ( uint32_t coefficient = 0; coefficient < 3; ++coefficient )
                {
                    outPlanes[plane][coefficient] = kHalfPlanes[face][plane][coefficient];
                }
            }
        }

        //---------------------------------------------------------------------

        static double Dot3( double const* pLeft, double const* pRight )
        {
            return ( pLeft[0] * pRight[0] ) + ( pLeft[1] * pRight[1] ) + ( pLeft[2] * pRight[2] );
        }

        static void Cross3( double* pOut, double const* pLeft, double const* pRight )
        {
            pOut[0] = ( pLeft[1] * pRight[2] ) - ( pLeft[2] * pRight[1] );
            pOut[1] = ( pLeft[2] * pRight[0] ) - ( pLeft[0] * pRight[2] );
            pOut[2] = ( pLeft[0] * pRight[1] ) - ( pLeft[1] * pRight[0] );
        }

        static double Clamp01( double value )
        {
            return ( value < 0.0 ) ? 0.0 : ( ( value > 1.0 ) ? 1.0 : value );
        }

        //---------------------------------------------------------------------

        static void GetBarycentricAt( uint32_t face, double u, double v, double* pOutWeight )
        {
            double tileU[3];
            double tileV[3];
            GetTileCorners( face, tileU, tileV );

            SolveBarycentric2D( u, v, tileU, tileV, pOutWeight );
        }

        //---------------------------------------------------------------------

        static void SolveBarycentric2D( double u, double v, double const* pTileU, double const* pTileV, double* pOutWeight )
        {
            double const edge1U = pTileU[1] - pTileU[0];
            double const edge1V = pTileV[1] - pTileV[0];
            double const edge2U = pTileU[2] - pTileU[0];
            double const edge2V = pTileV[2] - pTileV[0];

            double const relativeU = u - pTileU[0];
            double const relativeV = v - pTileV[0];

            double const denominator = ( edge1U * edge2V ) - ( edge1V * edge2U );

            if ( std::fabs( denominator ) < 1.0e-30 )
            {
                pOutWeight[0] = 1.0;
                pOutWeight[1] = 0.0;
                pOutWeight[2] = 0.0;
                return;
            }

            double const inverse = 1.0 / denominator;

            pOutWeight[1] = ( ( relativeU * edge2V ) - ( relativeV * edge2U ) ) * inverse;
            pOutWeight[2] = ( ( edge1U * relativeV ) - ( edge1V * relativeU ) ) * inverse;
            pOutWeight[0] = 1.0 - pOutWeight[1] - pOutWeight[2];
        }

        //---------------------------------------------------------------------

        static bool SolveBarycentric3D
        (
            double const* pPoint, double const* pA, double const* pB, double const* pC,
            double* pOutWeight
        )
        {
            double const v0[3] = { pB[0] - pA[0], pB[1] - pA[1], pB[2] - pA[2] };
            double const v1[3] = { pC[0] - pA[0], pC[1] - pA[1], pC[2] - pA[2] };
            double const v2[3] = { pPoint[0] - pA[0], pPoint[1] - pA[1], pPoint[2] - pA[2] };

            double const d00 = Dot3( v0, v0 );
            double const d01 = Dot3( v0, v1 );
            double const d11 = Dot3( v1, v1 );
            double const d20 = Dot3( v2, v0 );
            double const d21 = Dot3( v2, v1 );

            double const denominator = ( d00 * d11 ) - ( d01 * d01 );

            if ( std::fabs( denominator ) < 1.0e-30 )
            {
                pOutWeight[0] = 1.0;
                pOutWeight[1] = 0.0;
                pOutWeight[2] = 0.0;
                return false;
            }

            double const inverse = 1.0 / denominator;

            // point - A = weight1 * ( B - A ) + weight2 * ( C - A )
            pOutWeight[1] = ( ( d11 * d20 ) - ( d01 * d21 ) ) * inverse;
            pOutWeight[2] = ( ( d00 * d21 ) - ( d01 * d20 ) ) * inverse;
            pOutWeight[0] = 1.0 - pOutWeight[1] - pOutWeight[2];

            return true;
        }

        //  Sutherland-Hodgman against A u + B v + C >= 0
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

        //  Van Oosterom and Strackee. Exact for a planar triangle read from the origin, which every face triangle is.
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
    struct ProbeMapOf< TetrahedralProjection >
    {
        static constexpr ProbeMap Value = ProbeMap::Tetrahedron;
    };
}