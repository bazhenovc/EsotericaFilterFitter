#include "ChartReconstruction.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "Assert.h"
#include "MapProjection.h"
#include "OctahedralProjection.h"
#include "ProbeMap.h"
#include "TetrahedralProjection.h"

//-------------------------------------------------------------------------
//  See ChartReconstruction.h for what this is for. This file is the experiment.
//
//  TWO RULES KEEP THIS HONEST ACROSS THREE PROJECTIONS THAT DO NOT AGREE ON CONVENTIONS.
//
//  Nothing here assumes a chart's coordinate space or its texel-centre convention. A cube
//  spans [0,1] per face; an octahedral chart spans [-1,1] over the square, i.e. an extent of
//  two, because that is what makes its Jacobian constants clean. So every coordinate is
//  produced by the projection's own functions - GetTexelCentreUV for a texel's coordinate,
//  GetTexelCoordinateFromUV for the continuous coordinate a direction lands at - and never
//  by arithmetic on an index. Getting this wrong would not fail loudly; it would measure a
//  chart that does not exist.
//
//  And no chart-specific member is named in code that is instantiated for a chart that does
//  not have it. CubeProjection has no GetFaceFromUV, because a cube face is selected by
//  which face you are on rather than by where you are, while both square charts do. That
//  difference is handled with if constexpr so each instantiation only sees what exists.
//
//  THE METRIC IS A GLOBAL RELATIVE L1, NOT A PER-DIRECTION RATIO.
//-------------------------------------------------------------------------
//  Sum of absolute differences over the sum of absolute exact values, the same convention the
//  corpus metric uses. A per-direction |reconstructed - exact| / exact is unbounded on a field
//  with near-zero regions, and this field has four orders of magnitude between its lobes and
//  its antipodes: one early version of this file reported a mean "relative error" of 12.19 for
//  a cubemap that is nearly exact everywhere, and a max of 2589, because the divisor was
//  almost nothing at the dark end. No ratio of two per-direction errors is taken anywhere now.
//
//  THE EXPONENT IS log2( error at half the resolution / error at this one ).
//-------------------------------------------------------------------------
//  Bilinear reconstruction of a point-sampled smooth field errs like h squared, so a chart
//  that is limited by its texels scores 2 - four times better per doubling. A chart whose
//  reconstruction error comes from something that does not shrink with h, a crease or an
//  anisotropy at a join, scores 0: resolution is money burned there. That number is the point
//  of the experiment.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    namespace
    {
        constexpr uint32_t kResolutions[] = { 128u, 256u, 512u };
        constexpr uint32_t kNumResolutions = 3u;

        // Test directions. Fibonacci lattice: deterministic, cheap, and spread by area
        // rather than clustered at the poles, which matters because the charts disagree
        // most near their own joins and a clustered sample would weight those regions
        // differently for each chart.
        constexpr uint32_t kNumTestDirections = 100000u;

        //  The field
        //-------------------------------------------------------------------------
        //  Gaussians about three fixed, mutually oblique axes - one line of exact value, and
        //  a spectrum that is Gaussian, so the finest feature is set by one number rather
        //  than emerging from a sum. At 0.30 radians the smallest feature spans about ten
        //  texels at 128 and forty at 512, so every chart here can carry it. A field finer
        //  than a chart can represent would measure aliasing, which is not in question.
        //-------------------------------------------------------------------------

        // The field's bound, for the check in SampleChart: three lobes, each at most
        // exp( 2 / w^2 ) with w = 0.30, which is 4.477e9, times three for the sum, rounded up.
        constexpr double kFieldMaximumBound = 1.35e10;

        double EvaluateField( double const* pDirection )
        {
            static constexpr double kWidth = 0.30;
            static constexpr double kInverseWidthSquared = 1.0 / ( kWidth * kWidth );

            //  The field is defined on the SPHERE, so it normalizes what it is given.
            //
            //  The projections return UNNORMALIZED chart vectors on purpose: an octahedral
            //  texel's direction is ( u, v, 1 - |u| - |v| ), whose length runs from 1 to
            //  sqrt(3), and that length is exactly what their Jacobian 1/|p|^3 describes.
            //  Handing one of those to a field written in terms of unit alignments multiplies
            //  the exponent by the length, and exp(22) becomes exp(40).
            //
            //  That is not a small error. It is how this harness first reported a worst
            //  absolute error of 9.4e12 on a field whose true maximum is 4.5e9, with a global
            //  relative L1 above 1 - which no bounded field can produce, and which was the tell.
            double const length = std::sqrt( ( pDirection[0] * pDirection[0] ) + ( pDirection[1] * pDirection[1] ) + ( pDirection[2] * pDirection[2] ) );
            FF_ASSERT( length > 0.0 );

            double const unitDirection[3] = { pDirection[0] / length, pDirection[1] / length, pDirection[2] / length };

            static constexpr double kAxes[3][3] =
            {
                {  0.000000000000000,  0.000000000000000,  1.000000000000000 },
                {  0.801783725737273,  0.534522483824849,  0.267261241912424 },
                { -0.424264068711929,  0.565685424949238,  0.707106781186548 },
            };

            double value = 0.0;

            for ( uint32_t axis = 0; axis < 3; ++axis )
            {
                double const alignment = ( unitDirection[0] * kAxes[axis][0] )
                                       + ( unitDirection[1] * kAxes[axis][1] )
                                       + ( unitDirection[2] * kAxes[axis][2] );

                // 1 + alignment runs from 0 at the antipode to 2 at the axis, so the
                // exponent is bounded and no lobe underflows to exactly zero.
                value += std::exp( ( 1.0 + alignment ) * kInverseWidthSquared );
            }

            return value;
        }

        //  A chart, sampled
        //-------------------------------------------------------------------------

        struct ChartSamples
        {
            std::vector<double> m_texels;
            uint32_t            m_resolution = 0;
        };

        // Which slice holds a face. Nothing assumes an order: the face lists are asked for.
        template< typename TMap >
        uint32_t FindSliceForFace( uint32_t face )
        {
            for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
            {
                uint32_t faces[TMap::NumFaces] = {};

                uint32_t const count = TMap::GetSliceFaces( slice, faces );

                for ( uint32_t index = 0; index < count; ++index )
                {
                    if ( faces[index] == face )
                    {
                        return slice;
                    }
                }
            }

            FF_ASSERT( false );

            return 0;
        }

        // The face a chart coordinate is in. A cube face is chosen by being on it, so a
        // single-face slice - which every cube slice is - carries its own answer, and only
        // a slice holding several faces has to ask the projection where a coordinate is.
        template< typename TMap >
        uint32_t GetFaceAtUV( uint32_t slice, double u, double v )
        {
            uint32_t faces[TMap::NumFaces] = {};

            uint32_t const count = TMap::GetSliceFaces( slice, faces );

            if ( count == 1u )
            {
                return faces[0];
            }

            if constexpr ( ProbeMapOf< TMap >::Value == ProbeMap::Cube )
            {
                FF_ASSERT( false );

                return faces[0];
            }
            else
            {
                return TMap::GetFaceFromUV( u, v );
            }
        }

        // Returns false if any sampled value exceeded the field's bound. That can only happen
        // if a texel's direction is not a direction: the bound is the field's maximum over the
        // sphere for unit input, so nothing a chart can legitimately produce exceeds it.
        //
        // This is the harness validating itself, and it is the check whose absence let a 9.4e12
        // value reach the table. It also prints the offending direction's length, which names
        // the cause directly rather than leaving it to be inferred from a magnitude.
        template< typename TMap >
        bool SampleChart( ChartSamples* pPerSlice, uint32_t resolution )
        {
            bool bounded = true;

            for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
            {
                pPerSlice[slice].m_resolution = resolution;
                pPerSlice[slice].m_texels.assign( static_cast<size_t>( resolution ) * resolution, 0.0 );

                for ( uint32_t texelY = 0; texelY < resolution; ++texelY )
                {
                    for ( uint32_t texelX = 0; texelX < resolution; ++texelX )
                    {
                        double u = 0.0;
                        double v = 0.0;

                        TMap::GetTexelCentreUV( &u, &v, texelX, texelY, resolution );

                        double direction[3] = {};

                        TMap::GetTexelDirection( direction, slice, u, v );

                        double const length = std::sqrt( ( direction[0] * direction[0] ) + ( direction[1] * direction[1] ) + ( direction[2] * direction[2] ) );
                        double const value = EvaluateField( direction );

                        if ( value > kFieldMaximumBound )
                        {
                            if ( bounded )
                            {
                                std::printf
                                (
                                    "    FAIL  sampled field %.6e exceeds the field's bound %.6e at slice %u texel (%u, %u), whose direction has length %.6f\n",
                                    value, kFieldMaximumBound, slice, texelX, texelY, length
                                );
                            }

                            bounded = false;
                        }

                        pPerSlice[slice].m_texels[( static_cast<size_t>( texelY ) * resolution ) + texelX] = value;
                    }
                }
            }

            return bounded;
        }

        //  The reads
        //-------------------------------------------------------------------------
        //  Clamped is what hardware does: the bilinear cell is taken in chart coordinates
        //  and its indices are clamped to the slice, so a footprint running off the edge
        //  repeats the edge texel. On a cube that edge is a face boundary and the texture
        //  simply ends there, which is how a cubemap avoids blending two faces at all.
        //
        //  Resolved is the direction-space read the runtime gather was specified to use: the
        //  four footprint corners are converted to directions and each gathers the texel its
        //  own direction lands on. It cannot introduce a fold, because a direction has one
        //  home in the chart by construction.
        //-------------------------------------------------------------------------

        double ReadClamped( std::vector<double> const& texels, uint32_t resolution, double x, double y )
        {
            double const floorX = std::floor( x );
            double const floorY = std::floor( y );

            double const fractionX = x - floorX;
            double const fractionY = y - floorY;

            auto const texelAt = [&] ( int64_t texelX, int64_t texelY )
            {
                int64_t const limit = static_cast<int64_t>( resolution ) - 1;
                int64_t const clampedX = ( texelX < 0 ) ? 0 : ( ( texelX > limit ) ? limit : texelX );
                int64_t const clampedY = ( texelY < 0 ) ? 0 : ( ( texelY > limit ) ? limit : texelY );

                return texels[( static_cast<size_t>( clampedY ) * resolution ) + static_cast<size_t>( clampedX )];
            };

            int64_t const baseX = static_cast<int64_t>( floorX );
            int64_t const baseY = static_cast<int64_t>( floorY );

            double const top = ( texelAt( baseX, baseY ) * ( 1.0 - fractionX ) ) + ( texelAt( baseX + 1, baseY ) * fractionX );
            double const bottom = ( texelAt( baseX, baseY + 1 ) * ( 1.0 - fractionX ) ) + ( texelAt( baseX + 1, baseY + 1 ) * fractionX );

            return ( top * ( 1.0 - fractionY ) ) + ( bottom * fractionY );
        }

        // The texel a direction lands on, from the slice that holds it.
        template< typename TMap >
        double TexelAtDirection( ChartSamples const* pPerSlice, double const* pDirection )
        {
            uint32_t face = 0;
            double u = 0.0;
            double v = 0.0;

            TMap::GetFaceAndUVFromDirection( pDirection, face, u, v );

            uint32_t const slice = FindSliceForFace< TMap >( face );
            uint32_t const resolution = pPerSlice[slice].m_resolution;

            double x = 0.0;
            double y = 0.0;

            TMap::GetTexelCoordinateFromUV( &x, &y, u, v, resolution );

            int64_t const limit = static_cast<int64_t>( resolution ) - 1;
            int64_t const floorX = static_cast<int64_t>( std::floor( x ) );
            int64_t const floorY = static_cast<int64_t>( std::floor( y ) );
            int64_t const clampedX = ( floorX < 0 ) ? 0 : ( ( floorX > limit ) ? limit : floorX );
            int64_t const clampedY = ( floorY < 0 ) ? 0 : ( ( floorY > limit ) ? limit : floorY );

            return pPerSlice[slice].m_texels[( static_cast<size_t>( clampedY ) * resolution ) + static_cast<size_t>( clampedX )];
        }

        // The four footprint corners of a read and their bilinear weights, in chart
        // coordinates taken from the projection rather than computed from the index.
        template< typename TMap >
        void GetFootprintCorners
        (
            ChartSamples const* pPerSlice,
            double const* pDirection,
            uint32_t& slice,
            uint32_t& face,
            double( &cornerU )[2],
            double( &cornerV )[2],
            double( &weight )[4]
        )
        {
            double u = 0.0;
            double v = 0.0;

            TMap::GetFaceAndUVFromDirection( pDirection, face, u, v );

            slice = FindSliceForFace< TMap >( face );

            // The read's own resolution, not a constant: a footprint computed for a 256 chart
            // would classify and resolve reads on a chart that is not the one being measured.
            uint32_t const resolution = pPerSlice[slice].m_resolution;

            double x = 0.0;
            double y = 0.0;

            TMap::GetTexelCoordinateFromUV( &x, &y, u, v, resolution );

            double const floorX = std::floor( x );
            double const floorY = std::floor( y );

            double const fractionX = x - floorX;
            double const fractionY = y - floorY;

            int64_t const baseX = static_cast<int64_t>( floorX );
            int64_t const baseY = static_cast<int64_t>( floorY );

            // Corners by index, coordinates by the projection: this is the only way to be
            // right on a chart whose extent is not one.
            //
            // GetTexelCentreUVOutside, not GetTexelCentreUV: a footprint corner may sit one
            // texel outside the slice, which is the whole reason a read at an edge has to be
            // clamped, and the in-range call asserts on exactly that coordinate.
            for ( uint32_t corner = 0; corner < 4; ++corner )
            {
                TMap::GetTexelCentreUVOutside
                (
                    &cornerU[corner & 1u],
                    &cornerV[corner >> 1u],
                    static_cast<int32_t>( baseX ) + static_cast<int32_t>( corner & 1u ),
                    static_cast<int32_t>( baseY ) + static_cast<int32_t>( corner >> 1u ),
                    resolution
                );
            }

            weight[0] = ( 1.0 - fractionX ) * ( 1.0 - fractionY );
            weight[1] = fractionX * ( 1.0 - fractionY );
            weight[2] = ( 1.0 - fractionX ) * fractionY;
            weight[3] = fractionX * fractionY;
        }

        template< typename TMap >
        double ReadResolved( ChartSamples const* pPerSlice, double const* pDirection )
        {
            uint32_t slice = 0;
            uint32_t face = 0;
            double cornerU[2] = {};
            double cornerV[2] = {};
            double weight[4] = {};

            GetFootprintCorners< TMap >( pPerSlice, pDirection, slice, face, cornerU, cornerV, weight );

            double value = 0.0;

            for ( uint32_t corner = 0; corner < 4; ++corner )
            {
                double cornerDirection[3] = {};

                TMap::GetDirectionFromUV( cornerDirection, cornerU[corner & 1u], cornerV[corner >> 1u], face );

                value += weight[corner] * TexelAtDirection< TMap >( pPerSlice, cornerDirection );
            }

            return value;
        }

        //  Is this read crossing a crease
        //-------------------------------------------------------------------------
        //  A join read is one whose bilinear footprint does not stay inside ONE FACE of the
        //  chart - the chart's own region boundary, asked of each projection the way every
        //  other check asks it.
        //
        //  The first version of this tested whether the footprint's corners stayed inside one
        //  SLICE, which is the wrong question in both directions. A cubemap's six faces are
        //  six slices, so it could report a join; a square chart's eight faces live inside its
        //  ONE slice, so it could never report one - and the octahedral excess this experiment
        //  exists to measure sits exactly in those strips. The column read 0.000 for both
        //  square charts and the one number that mattered was invisible.
        //
        //  Two things this now measures, which are not the same and are both worth having:
        //  for a cubemap, faces are separate textures, so a read at a face edge clamps and the
        //  join column measures the cost of that clamp. For a square chart, the face boundary
        //  is a crease inside one texture, so the join column measures blending across it.
        //-------------------------------------------------------------------------

        template< typename TMap >
        bool IsJoinRead( ChartSamples const* pPerSlice, double const* pDirection )
        {
            uint32_t slice = 0;
            uint32_t face = 0;
            double cornerU[2] = {};
            double cornerV[2] = {};
            double weight[4] = {};

            GetFootprintCorners< TMap >( pPerSlice, pDirection, slice, face, cornerU, cornerV, weight );

            for ( uint32_t corner = 0; corner < 4; ++corner )
            {
                double cornerDirection[3] = {};

                TMap::GetDirectionFromUV( cornerDirection, cornerU[corner & 1u], cornerV[corner >> 1u], face );

                uint32_t cornerFace = 0;
                double chartU = 0.0;
                double chartV = 0.0;

                TMap::GetFaceAndUVFromDirection( cornerDirection, cornerFace, chartU, chartV );

                if ( cornerFace != face )
                {
                    return true;
                }
            }

            return false;
        }

        //  Chart distortion
        //-------------------------------------------------------------------------
        //  Two numbers per texel, separating the two things a projection can distort:
        //
        //      AREA   the solid angle a texel covers, from the chart's own Jacobian
        //      SHAPE  how far a texel's image is from a square, from the singular values of
        //             the chart's 2x2 derivative
        //
        //  Shape is the one that matters for this read, because a bilinear reconstruction
        //  assumes a square-ish footprint. A chart can be perfectly equal-area and still
        //  distort shape - the classical tradeoff, and the reason an equal-area chart is not
        //  automatically better here.
        //
        //  The singular values need no SVD: for derivative columns a and b, sigma1^2 +
        //  sigma2^2 is a.a + b.b and sigma1 * sigma2 is |a x b|, so the larger is the root of
        //  a quadratic and the ratio follows. Reported as a ratio, where 1.0 is a square.
        //-------------------------------------------------------------------------

        template< typename TMap >
        void MeasureChartDistortion( double& meanArea, double& areaRatio, double& meanAnisotropy, double& maxAnisotropy )
        {
            uint32_t const resolution = 256u;
            double const h = 1.0 / static_cast<double>( resolution );

            double areaSum = 0.0;
            double areaMinimum = 1.0e30;
            double areaMaximum = 0.0;
            double anisotropySum = 0.0;
            uint64_t count = 0u;

            maxAnisotropy = 0.0;

            for ( uint32_t slice = 0; slice < TMap::NumSlices; ++slice )
            {
                for ( uint32_t texelY = 0; texelY < resolution; ++texelY )
                {
                    for ( uint32_t texelX = 0; texelX < resolution; ++texelX )
                    {
                        double u = 0.0;
                        double v = 0.0;

                        TMap::GetTexelCentreUV( &u, &v, texelX, texelY, resolution );

                        uint32_t const face = GetFaceAtUV< TMap >( slice, u, v );

                        double const area = TMap::GetJacobian( u, v );

                        areaSum += area;
                        areaMinimum = ( area < areaMinimum ) ? area : areaMinimum;
                        areaMaximum = ( area > areaMaximum ) ? area : areaMaximum;

                        double directionU0[3] = {};
                        double directionU1[3] = {};
                        double directionV0[3] = {};
                        double directionV1[3] = {};

                        TMap::GetDirectionFromUV( directionU0, u - h, v, face );
                        TMap::GetDirectionFromUV( directionU1, u + h, v, face );
                        TMap::GetDirectionFromUV( directionV0, u, v - h, face );
                        TMap::GetDirectionFromUV( directionV1, u, v + h, face );

                        double const a[3] = { ( directionU1[0] - directionU0[0] ) / ( 2.0 * h ), ( directionU1[1] - directionU0[1] ) / ( 2.0 * h ), ( directionU1[2] - directionU0[2] ) / ( 2.0 * h ) };
                        double const b[3] = { ( directionV1[0] - directionV0[0] ) / ( 2.0 * h ), ( directionV1[1] - directionV0[1] ) / ( 2.0 * h ), ( directionV1[2] - directionV0[2] ) / ( 2.0 * h ) };

                        double const trace = ( a[0] * a[0] ) + ( a[1] * a[1] ) + ( a[2] * a[2] ) + ( b[0] * b[0] ) + ( b[1] * b[1] ) + ( b[2] * b[2] );

                        double const cross[3] =
                        {
                            ( a[1] * b[2] ) - ( a[2] * b[1] ),
                            ( a[2] * b[0] ) - ( a[0] * b[2] ),
                            ( a[0] * b[1] ) - ( a[1] * b[0] ),
                        };

                        double const crossLength = std::sqrt( ( cross[0] * cross[0] ) + ( cross[1] * cross[1] ) + ( cross[2] * cross[2] ) );

                        double const discriminant = ( trace * trace ) - ( 4.0 * crossLength * crossLength );
                        double const root = std::sqrt( ( discriminant > 0.0 ) ? discriminant : 0.0 );
                        double const largerSquared = 0.5 * ( trace + root );
                        double const smallerSquared = 0.5 * ( trace - root );

                        double anisotropy = 1.0;

                        if ( smallerSquared > 1.0e-30 )
                        {
                            anisotropy = std::sqrt( largerSquared / smallerSquared );
                        }

                        anisotropySum += anisotropy;
                        maxAnisotropy = ( anisotropy > maxAnisotropy ) ? anisotropy : maxAnisotropy;

                        ++count;
                    }
                }
            }

            meanArea = ( count > 0u ) ? ( areaSum / static_cast<double>( count ) ) : 0.0;
            areaRatio = ( areaMinimum > 0.0 ) ? ( areaMaximum / areaMinimum ) : 0.0;
            meanAnisotropy = ( count > 0u ) ? ( anisotropySum / static_cast<double>( count ) ) : 0.0;
        }

        //  One chart, one resolution
        //-------------------------------------------------------------------------
        //  Absolute differences and exact values are accumulated separately, and the reported
        //  error is their ratio over the whole region. That is what makes the number a norm
        //  rather than a mean of ratios, and it is why the dark half of the field can no
        //  longer dominate a chart's score.
        //-------------------------------------------------------------------------

        //  What the read actually touched
        //-------------------------------------------------------------------------
        //  Printed for the single worst test direction of each chart and resolution, so that a
        //  broken harness names its own cause instead of leaving it to be inferred from a
        //  magnitude. The footprint's four corners are shown with their coordinates, resolved
        //  indices, texel values and the length of the CHART VECTOR the projection returns for
        //  them. That length is deliberately not 1: every projection here returns an unnormalized
        //  chart vector whose length is the quantity its Jacobian is written against, so printing
        //  it shows the normalization mistake directly rather than encoding it.
        //-------------------------------------------------------------------------

        template< typename TMap >
        void PrintReadFootprint( ChartSamples const* pPerSlice, double const* pDirection )
        {
            uint32_t slice = 0;
            uint32_t face = 0;
            double cornerU[2] = {};
            double cornerV[2] = {};
            double weight[4] = {};

            GetFootprintCorners< TMap >( pPerSlice, pDirection, slice, face, cornerU, cornerV, weight );

            uint32_t const resolution = pPerSlice[slice].m_resolution;
            int64_t const limit = static_cast<int64_t>( resolution ) - 1;

            std::printf
            (
                "      footprint face %u, slice %u, resolution %u, weights %.4f %.4f %.4f %.4f\n",
                face, slice, resolution, weight[0], weight[1], weight[2], weight[3]
            );

            for ( uint32_t corner = 0; corner < 4; ++corner )
            {
                double const u = cornerU[corner & 1u];
                double const v = cornerV[corner >> 1u];

                double cornerDirection[3] = {};

                TMap::GetDirectionFromUV( cornerDirection, u, v, face );

                double const length = std::sqrt
                (
                    ( cornerDirection[0] * cornerDirection[0] )
                  + ( cornerDirection[1] * cornerDirection[1] )
                  + ( cornerDirection[2] * cornerDirection[2] )
                );

                double x = 0.0;
                double y = 0.0;

                TMap::GetTexelCoordinateFromUV( &x, &y, u, v, resolution );

                int64_t cornerX = static_cast<int64_t>( std::llround( x ) );
                int64_t cornerY = static_cast<int64_t>( std::llround( y ) );

                cornerX = ( cornerX < 0 ) ? 0 : ( ( cornerX > limit ) ? limit : cornerX );
                cornerY = ( cornerY < 0 ) ? 0 : ( ( cornerY > limit ) ? limit : cornerY );

                double const texel = pPerSlice[slice].m_texels[( static_cast<size_t>( cornerY ) * resolution ) + static_cast<size_t>( cornerX )];

                std::printf
                (
                    "        corner %u  uv (%.6f, %.6f)  index (%lld, %lld)  chart |vector| %.6f  texel %.6e\n",
                    corner, u, v, static_cast<long long>( cornerX ), static_cast<long long>( cornerY ), length, texel
                );
            }
        }

        struct ChartResult
        {
            double      m_interiorAbsolute = 0.0;
            double      m_interiorExact = 0.0;
            double      m_joinAbsolute = 0.0;
            double      m_joinExact = 0.0;
            double      m_resolvedAbsolute = 0.0;
            double      m_exact = 0.0;
            double      m_worstAbsolute = 0.0;
            uint32_t    m_interiorCount = 0u;
            uint32_t    m_joinCount = 0u;

            // The worst read, kept so the footprint can be printed for it afterwards.
            double      m_worstDirection[3] = {};
            double      m_worstExact = 0.0;
            double      m_worstReconstructed = 0.0;

            // False if the sampled field left its own bound, which is a defect in the
            // chart-to-direction mapping rather than in the reconstruction.
            bool        m_fieldBounded = true;

            double InteriorError() const { return ( m_interiorExact > 0.0 ) ? ( m_interiorAbsolute / m_interiorExact ) : 0.0; }
            double JoinError() const { return ( m_joinExact > 0.0 ) ? ( m_joinAbsolute / m_joinExact ) : 0.0; }
            double ResolvedError() const { return ( m_exact > 0.0 ) ? ( m_resolvedAbsolute / m_exact ) : 0.0; }
            double OverallError() const { return ( m_exact > 0.0 ) ? ( ( m_interiorAbsolute + m_joinAbsolute ) / m_exact ) : 0.0; }
            double JoinFraction() const { return ( ( m_interiorCount + m_joinCount ) > 0u ) ? ( static_cast<double>( m_joinCount ) / static_cast<double>( m_interiorCount + m_joinCount ) ) : 0.0; }
        };

        template< typename TMap >
        bool RunChartAtResolution( uint32_t resolution, ChartResult& result )
        {
            std::vector<ChartSamples> slices( TMap::NumSlices );

            bool const bounded = SampleChart< TMap >( slices.data(), resolution );

            result.m_fieldBounded = bounded;

            for ( uint32_t index = 0; index < kNumTestDirections; ++index )
            {
                // Fibonacci lattice over the sphere.
                double const golden = 3.14159265358979323846 * ( 3.0 - std::sqrt( 5.0 ) );
                double const z = 1.0 - ( 2.0 * ( static_cast<double>( index ) + 0.5 ) / static_cast<double>( kNumTestDirections ) );
                double const radius = std::sqrt( ( 1.0 - z ) * ( 1.0 + z ) );
                double const theta = golden * static_cast<double>( index );

                double const direction[3] = { radius * std::cos( theta ), radius * std::sin( theta ), z };

                double const exact = EvaluateField( direction );

                uint32_t face = 0;
                double u = 0.0;
                double v = 0.0;

                TMap::GetFaceAndUVFromDirection( direction, face, u, v );

                uint32_t const slice = FindSliceForFace< TMap >( face );
                uint32_t const sliceResolution = slices[slice].m_resolution;

                double x = 0.0;
                double y = 0.0;

                TMap::GetTexelCoordinateFromUV( &x, &y, u, v, sliceResolution );

                double const reconstructed = ReadClamped( slices[slice].m_texels, sliceResolution, x, y );
                double const resolved = ReadResolved< TMap >( slices.data(), direction );

                // Absolute differences, never a ratio of two per-direction values.
                double const absolute = std::fabs( reconstructed - exact );
                double const resolvedAbsolute = std::fabs( resolved - exact );

                result.m_resolvedAbsolute += resolvedAbsolute;
                result.m_exact += exact;

                if ( absolute > result.m_worstAbsolute )
                {
                    result.m_worstAbsolute = absolute;
                    result.m_worstDirection[0] = direction[0];
                    result.m_worstDirection[1] = direction[1];
                    result.m_worstDirection[2] = direction[2];
                    result.m_worstExact = exact;
                    result.m_worstReconstructed = reconstructed;
                }

                if ( IsJoinRead< TMap >( slices.data(), direction ) )
                {
                    result.m_joinAbsolute += absolute;
                    result.m_joinExact += exact;
                    ++result.m_joinCount;
                }
                else
                {
                    result.m_interiorAbsolute += absolute;
                    result.m_interiorExact += exact;
                    ++result.m_interiorCount;
                }
            }

            std::printf
            (
                "      worst read at resolution %u: direction (%.6f, %.6f, %.6f)  exact %.6e  reconstructed %.6e  error %.6e\n",
                resolution,
                result.m_worstDirection[0], result.m_worstDirection[1], result.m_worstDirection[2],
                result.m_worstExact, result.m_worstReconstructed, result.m_worstAbsolute
            );

            PrintReadFootprint< TMap >( slices.data(), result.m_worstDirection );

            return bounded;
        }

        // log2( error at half this resolution / error here ): 2 when the error falls with the
        // square of the texel size, which is what bilinear reconstruction of a smooth field
        // does when nothing else limits it, and 0 when it does not move at all, which is a
        // chart whose error comes from something that does not shrink with the texels.
        double ScalingExponent( double coarse, double fine )
        {
            if ( ( coarse <= 0.0 ) || ( fine <= 0.0 ) )
            {
                return 0.0;
            }

            return std::log2( coarse / fine );
        }

        //  The known answer, asserted
        //-------------------------------------------------------------------------
        //  A cubemap's row is not a measurement, it is a reference: six separate textures, a
        //  smooth field, a within-face bilinear read. So its interior error must be small, its
        //  join fraction small - its faces are separate textures, so it never blends across
        //  one - and its interior exponent near two, because nothing about it is scale free.
        //
        //  Without this the diagnostic prints a table that looks like a measurement even when
        //  its metric or its read path is broken, which is exactly what its first version did:
        //  it reported a mean relative error of 12.19 for the cubemap, an order of magnitude
        //  worse than a square chart, and printed it with no complaint.
        //-------------------------------------------------------------------------

        constexpr double kCubeInteriorErrorLimit = 0.01;
        constexpr double kCubeJoinFractionLimit = 0.02;
        constexpr double kCubeExponentMinimum = 1.5;
        constexpr double kCubeExponentMaximum = 2.5;

        bool CheckCubeSanity( ChartResult const* pResults )
        {
            uint32_t failures = 0u;

            for ( uint32_t index = 0; index < kNumResolutions; ++index )
            {
                if ( pResults[index].InteriorError() > kCubeInteriorErrorLimit )
                {
                    std::printf( "    FAIL  resolution %u: interior error %.6f exceeds the cubemap limit %.6f\n", kResolutions[index], pResults[index].InteriorError(), kCubeInteriorErrorLimit );
                    ++failures;
                }

                if ( pResults[index].JoinFraction() > kCubeJoinFractionLimit )
                {
                    std::printf( "    FAIL  resolution %u: join fraction %.4f exceeds the cubemap limit %.4f\n", kResolutions[index], pResults[index].JoinFraction(), kCubeJoinFractionLimit );
                    ++failures;
                }
            }

            for ( uint32_t index = 1u; index < kNumResolutions; ++index )
            {
                double const exponent = ScalingExponent( pResults[index - 1u].InteriorError(), pResults[index].InteriorError() );

                if ( ( exponent < kCubeExponentMinimum ) || ( exponent > kCubeExponentMaximum ) )
                {
                    std::printf( "    FAIL  resolution %u: interior exponent %.3f is outside [%.1f, %.1f]\n", kResolutions[index], exponent, kCubeExponentMinimum, kCubeExponentMaximum );
                    ++failures;
                }
            }

            if ( failures == 0u )
            {
                std::printf( "    cubemap reference row: PASS (interior small, joins rare, exponent near 2)\n" );

                return true;
            }

            std::printf( "    cubemap reference row: FAIL (%u clause%s)\n", failures, ( failures == 1u ) ? "" : "s" );

            return false;
        }

        template< typename TMap >
        bool RunChart( char const* pName )
        {
            std::printf( "\n  %s\n", pName );

            ChartResult results[kNumResolutions] = {};

            bool bounded = true;

            for ( uint32_t index = 0; index < kNumResolutions; ++index )
            {
                bounded = RunChartAtResolution< TMap >( kResolutions[index], results[index] ) && bounded;
            }

            double meanArea = 0.0;
            double areaRatio = 0.0;
            double meanAnisotropy = 0.0;
            double maxAnisotropy = 0.0;

            MeasureChartDistortion< TMap >( meanArea, areaRatio, meanAnisotropy, maxAnisotropy );

            std::printf
            (
                "    %-10s %-12s %-12s %-12s %-12s %-8s %-8s %-8s\n",
                "resolution", "interior", "join", "interior x", "join x", "resolved", "join%", "worst abs"
            );

            for ( uint32_t index = 0; index < kNumResolutions; ++index )
            {
                // The first row has nothing to compare against, and printing 0.000 there would
                // read as "scale free" rather than "not measured", so it prints a dash.
                char interiorExponentText[16] = {};
                char joinExponentText[16] = {};

                if ( index > 0u )
                {
                    std::snprintf( interiorExponentText, sizeof( interiorExponentText ), "%.3f", ScalingExponent( results[index - 1u].InteriorError(), results[index].InteriorError() ) );
                    std::snprintf( joinExponentText, sizeof( joinExponentText ), "%.3f", ScalingExponent( results[index - 1u].JoinError(), results[index].JoinError() ) );
                }
                else
                {
                    std::snprintf( interiorExponentText, sizeof( interiorExponentText ), "-" );
                    std::snprintf( joinExponentText, sizeof( joinExponentText ), "-" );
                }

                std::printf
                (
                    "    %-10u %-12.6f %-12.6f %-12s %-12s %-8.6f %-8.2f %-8.4f\n",
                    kResolutions[index],
                    results[index].InteriorError(),
                    results[index].JoinError(),
                    interiorExponentText,
                    joinExponentText,
                    results[index].ResolvedError(),
                    results[index].JoinFraction() * 100.0,
                    results[index].m_worstAbsolute
                );
            }

            std::printf
            (
                "    distortion  area mean %.6f, area max/min %.3f, anisotropy mean %.3f, max %.3f (1.0 is a square)\n",
                meanArea, areaRatio, meanAnisotropy, maxAnisotropy
            );

            // The field bound is judged before any number above is believed. An unbounded sample
            // means the chart-to-direction mapping itself is broken, and every error printed for
            // this chart is then a symptom of that rather than a measurement of the chart.
            if ( bounded )
            {
                std::printf( "    field bounded: PASS (every sampled value is within %.3e)\n", kFieldMaximumBound );
            }
            else
            {
                std::printf( "    field bounded: FAIL - a sampled value left the field's bound, so the numbers above are symptoms of a broken mapping, not measurements of this chart\n" );
            }

            bool passed = bounded;

            if constexpr ( ProbeMapOf< TMap >::Value == ProbeMap::Cube )
            {
                passed = CheckCubeSanity( results ) && passed;
            }

            return passed;
        }
    }

    //-------------------------------------------------------------------------

    bool RunChartReconstruction()
    {
        std::printf( "\n" );
        std::printf( "chart reconstruction\n" );
        std::printf( "  a smooth field sampled into each chart and read back the way a runtime reads it\n" );
        std::printf( "  interior and join are GLOBAL relative L1 over the test directions whose bilinear\n" );
        std::printf( "  footprint does or does not leave one face of the chart: sum of absolute errors over\n" );
        std::printf( "  the sum of exact values, the convention the corpus metric uses. No per-direction\n" );
        std::printf( "  ratios are taken anywhere, because this field spans four orders of magnitude and a\n" );
        std::printf( "  per-direction ratio is unbounded where it is small\n" );
        std::printf( "  resolved reads the same footprint with its corners gathered through direction space\n" );
        std::printf( "  x is log2( error at half this resolution / error here ): 2 is texel limited - four\n" );
        std::printf( "  times better per doubling - and 0 is scale free, where resolution buys nothing\n" );
        std::printf( "  every sampled value is read back through the chart's own direction mapping and must\n" );
        std::printf( "  land inside the field's bound; a chart failing that line has a broken mapping and its\n" );
        std::printf( "  error numbers are symptoms, not measurements, until the mapping is fixed\n" );
        std::printf( "  what this cannot conclude: it isolates the chart, not the table. A fitted table adds\n" );
        std::printf( "  error of its own from the tap layout and the reference chain, so a chart that wins\n" );
        std::printf( "  here can still lose in the corpus metric, and a table can beat its chart's\n" );
        std::printf( "  reconstruction floor. Rankings transfer between charts; absolute values do not\n" );

        bool passed = true;

        passed = RunChart< CubeProjection >( "cube, six faces in six slices" ) && passed;
        passed = RunChart< TetrahedralProjection >( "tetrahedral, four faces in one slice" ) && passed;
        passed = RunChart< OctahedralProjection >( "octahedral, eight faces in one slice" ) && passed;

        std::printf( "\n" );

        return passed;
    }
}
