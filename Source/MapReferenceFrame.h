#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "Assert.h"
#include "ReferenceFrame.h"

// The discretisation of the sphere a map defines
//-------------------------------------------------------------------------
// Every texel of a map at one resolution, with the unit direction it represents, the exact solid angle it subtends and the sphere-to-map density at its centre. 
// This is the measure every later stage depends on, so solid angles are analytic rather than the jacobian * texel-area approximation the vendored shader uses.
//
// A model of the frame contract in ReferenceFrame.h, written against the map contract in MapProjection.h and therefore not against a cube.
// Nothing here names a face count, a coordinate range or an axis direction: all of it comes from the map.
//
//  A LEVEL IS MADE OF SLICES, NOT FACES
// 
// Walking a level walks its slices and asks the map where each texel is, because a slice need not hold one face:
//
//      cubemap        6 slices of R x R, one face each
//      tetrahedral    1 slice  of R x R, all four faces
//
// So a level's texel count is NumSlices * R * R and not NumFaces * R * R, and the face a texel belongs to is resolved through the map rather than taken from the loop index.
//
// The coordinate convention is the map's own, down to the sign of the step from one texel to the next: a cube's v decreases down a face and a tetrahedral tile's increases.
// A consumer that needs to offset by a fraction of a texel asks the map for the step rather than deriving it, which is what keeps a frame from having to know which map it is.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    // One texel, whichever map produced it.
    // The record does not depend on the map: what varies between maps is how many of these there are and where they are, not what is known about each one.
    //-------------------------------------------------------------------------

    struct MapTexel
    {
        double      m_dir[3] = { 0.0, 0.0, 0.0 };           // Unit direction
        double      m_solidAngle = 0.0;                     // Exact, steradians
        double      m_jacobian = 0.0;                       // Density at the centre
        uint32_t    m_face = 0;                             // The face owning it
        uint32_t    m_slice = 0;                            // The slice holding it
        uint32_t    m_texelX = 0;
        uint32_t    m_texelY = 0;
    };

    //-------------------------------------------------------------------------

    template< typename TMap >
    class MapReferenceFrame
    {
    public:

        using Texel = MapTexel;

        // The map this frame is a discretisation of, so that a consumer holding a frame can name the same map without being told it twice
        using Map = TMap;

        static constexpr uint32_t NumFaces = TMap::NumFaces;
        static constexpr uint32_t NumSlices = TMap::NumSlices;

    public:

        void Initialize( uint32_t resolution )
        {
            FF_ASSERT( resolution > 0 );
            FF_ASSERT( ( resolution & ( resolution - 1 ) ) == 0 );

            m_resolution = resolution;
            m_totalSolidAngle = 0.0;

            double stepU = 0.0;
            double stepV = 0.0;
            TMap::GetTexelStep( &stepU, &stepV, resolution );

            m_stepU = stepU;
            m_stepV = stepV;
            m_texelArea = std::fabs( stepU * stepV );

            double const halfExtentU = 0.5 * std::fabs( stepU );
            double const halfExtentV = 0.5 * std::fabs( stepV );

            m_texels.clear();
            m_texels.resize( static_cast<size_t>( NumSlices ) * resolution * resolution );

            size_t texelIndex = 0;

            for ( uint32_t slice = 0; slice < NumSlices; ++slice )
            {
                for ( uint32_t texelY = 0; texelY < resolution; ++texelY )
                {
                    for ( uint32_t texelX = 0; texelX < resolution; ++texelX, ++texelIndex )
                    {
                        double centreU = 0.0;
                        double centreV = 0.0;
                        TMap::GetTexelCentreUV( &centreU, &centreV, texelX, texelY, resolution );

                        double direction[3];
                        TMap::GetTexelDirection( direction, slice, centreU, centreV );

                        double const directionLength = std::sqrt( ( direction[0] * direction[0] ) + ( direction[1] * direction[1] ) + ( direction[2] * direction[2] ) );

                        // The face is asked of the map.
                        // Deriving it from the slice would be right for a cubemap and wrong for a map that holds more than one face per slice.
                        uint32_t face = 0;
                        double faceU = 0.0;
                        double faceV = 0.0;
                        TMap::GetFaceAndUVFromDirection( direction, face, faceU, faceV );

                        MapTexel& texel = m_texels[texelIndex];
                        texel.m_dir[0] = direction[0] / directionLength;
                        texel.m_dir[1] = direction[1] / directionLength;
                        texel.m_dir[2] = direction[2] / directionLength;
                        texel.m_jacobian = TMap::GetJacobian( centreU, centreV );
                        texel.m_solidAngle = TMap::GetTexelSolidAngle( centreU - halfExtentU, centreV - halfExtentV, centreU + halfExtentU, centreV + halfExtentV );
                        texel.m_face = face;
                        texel.m_slice = slice;
                        texel.m_texelX = texelX;
                        texel.m_texelY = texelY;

                        m_totalSolidAngle += texel.m_solidAngle;
                    }
                }
            }
        }

        //---------------------------------------------------------------------

        inline bool                         IsInitialized() const { return m_resolution != 0; }
        inline uint32_t                     GetResolution() const { return m_resolution; }
        inline uint32_t                     GetNumTexels() const { return static_cast<uint32_t>( m_texels.size() ); }
        inline double                       GetTotalSolidAngle() const { return m_totalSolidAngle; }
        inline MapTexel const&              GetTexel( uint32_t texelIndex ) const { return m_texels[texelIndex]; }
        inline std::vector<MapTexel> const& GetTexels() const { return m_texels; }

        // Storage index of a texel, by SLICE.
        // A cubemap's slices are its faces, so a caller that has a face of a cubemap passes it here; a caller holding a face of a single-slice map has no use for this, because every face of such a map lives at the same indices.
        //---------------------------------------------------------------------

        inline uint32_t GetTexelIndex( uint32_t slice, uint32_t texelX, uint32_t texelY ) const
        {
            return ( slice * m_resolution * m_resolution ) + ( texelY * m_resolution ) + texelX;
        }

        //  Frame contract
        //---------------------------------------------------------------------

        double GetTexelMeasure( MapTexel const& texel, ErrorMeasure measure ) const
        {
            FF_ASSERT( m_resolution > 0 );

            if ( measure == ErrorMeasure::SolidAngle )
            {
                return texel.m_solidAngle;
            }

            if ( measure == ErrorMeasure::JacobianTexelArea )
            {
                // The density at the centre times the coordinate area one texel covers.
                // That product is the paper's 4 * J / R^2, and 4/R^2 is the coordinate area of a CUBE FACE'S texel, not a constant of the sphere: a cube face spans 2 in its own coordinates and a tetrahedral tile spans 1, so the same expression applied to a single-slice tetrahedral map overstates the total solid angle by exactly four. 
                // Measured by the frame check, which prints it.
                return texel.m_jacobian * m_texelArea;
            }

            // Uniform in the map's own face coordinates
            return m_texelArea;
        }

        // The direction at a fractional offset within a texel's own footprint.
        // Offsets are in units of one texel, so +/-0.5 reaches its edges. 
        // The offset is applied in face space, which is what makes the averaged preimage the texel average of the lobe rather than of its projection.
        //---------------------------------------------------------------------

        void GetTexelSampleDirection( double* pOutDir, uint32_t texelIndex, double offsetX, double offsetY ) const
        {
            FF_ASSERT( pOutDir != nullptr );
            FF_ASSERT( texelIndex < GetNumTexels() );

            MapTexel const& texel = m_texels[texelIndex];

            double centreU = 0.0;
            double centreV = 0.0;
            TMap::GetTexelCentreUV( &centreU, &centreV, texel.m_texelX, texel.m_texelY, m_resolution );

            // The step is signed, so this is the same expression for a map whose v decreases down a face and for one whose v increases
            double const sampleU = centreU + ( offsetX * m_stepU );
            double const sampleV = centreV + ( offsetY * m_stepV );

            // The map is handed the slice the texel came from.
            // A map that needs to be told which face to use reads it; a map that derives the face from the coordinate ignores it.
            // Handing it the FACE would work for a cube, where the two are the same number, and would be a latent bug in a map where they are not.
            TMap::GetTexelDirection( pOutDir, texel.m_slice, sampleU, sampleV );
        }

        // One separable (0.25, 0.5, 0.25) pass over each slice, clamped at the slice's edges. 
        // Broadens a field by roughly one texel.
        //
        // The neighbour structure is the map's STORAGE, which is the slice and not the face: a cubemap keeps one face per slice, so this never crosses a face border, and a single-slice map has no face border to cross because its four faces share one texture.
        //---------------------------------------------------------------------

        void SmoothInPlace( std::vector<double>& values ) const
        {
            FF_ASSERT( values.size() == GetNumTexels() );

            size_t const sliceTexels = static_cast<size_t>( m_resolution ) * m_resolution;
            std::vector<double> horizontal( values.size() );

            for ( uint32_t slice = 0; slice < NumSlices; ++slice )
            {
                size_t const sliceBase = static_cast<size_t>( slice ) * sliceTexels;

                for ( uint32_t y = 0; y < m_resolution; ++y )
                {
                    size_t const rowBase = sliceBase + ( static_cast<size_t>( y ) * m_resolution );

                    for ( uint32_t x = 0; x < m_resolution; ++x )
                    {
                        double const left = values[rowBase + ( ( x > 0 ) ? ( x - 1 ) : 0 )];
                        double const mid = values[rowBase + x];
                        double const right = values[rowBase + ( ( ( x + 1 ) < m_resolution ) ? ( x + 1 ) : ( m_resolution - 1 ) )];

                        horizontal[rowBase + x] = ( 0.25 * left ) + ( 0.5 * mid ) + ( 0.25 * right );
                    }
                }
            }

            for ( uint32_t slice = 0; slice < NumSlices; ++slice )
            {
                size_t const sliceBase = static_cast<size_t>( slice ) * sliceTexels;

                for ( uint32_t y = 0; y < m_resolution; ++y )
                {
                    uint32_t const above = ( y > 0 ) ? ( y - 1 ) : 0;
                    uint32_t const below = ( ( y + 1 ) < m_resolution ) ? ( y + 1 ) : ( m_resolution - 1 );

                    for ( uint32_t x = 0; x < m_resolution; ++x )
                    {
                        double const fromAbove = horizontal[sliceBase + ( static_cast<size_t>( above ) * m_resolution ) + x];
                        double const fromMid = horizontal[sliceBase + ( static_cast<size_t>( y ) * m_resolution ) + x];
                        double const fromBelow = horizontal[sliceBase + ( static_cast<size_t>( below ) * m_resolution ) + x];

                        values[sliceBase + ( static_cast<size_t>( y ) * m_resolution ) + x] = ( 0.25 * fromAbove ) + ( 0.5 * fromMid ) + ( 0.25 * fromBelow );
                    }
                }
            }
        }

    private:

        uint32_t                m_resolution = 0;
        double                  m_totalSolidAngle = 0.0;
        double                  m_stepU = 0.0;
        double                  m_stepV = 0.0;
        double                  m_texelArea = 0.0;
        std::vector<MapTexel>   m_texels;
    };
}
