//-------------------------------------------------------------------------
//  Reading a FilterFitter tetrahedral table
//-------------------------------------------------------------------------
//  What a runtime does with a table fitted for a single-slice tetrahedral map:
//  one output direction in, one filtered radiance out, as a handful of weighted
//  texture reads.
//
//  This mirrors BuildTableTaps in Source/TableHarness.cpp, which is the
//  authority. If the two disagree, that file is right.
//
//  Two numbers have to match the table being read, and both are in its header:
//
//      PROBE_TAPS_PER_AXIS             8, 16 or 32
//      PROBE_NUM_ACTIVE_COEFFICIENTS   1 for a constant table, 3 for a quadratic
//
//  The map itself is one texture per level. Its four faces are the four triangles
//  the square's diagonals cut it into, so a direction has to be turned into a tile
//  coordinate before anything can be read, and half of this file is that
//  conversion. A cubemap's reader is a separate file with its own frames and its
//  own lookup; nothing is shared between them.
//
//  NO INDEXED TABLES OF CONSTANTS. The four vertices are named values, the frame
//  loop is written out, and the face's corners are selected rather than looked up.
//  An array indexed by anything but a literal becomes a dynamic index, which a GPU
//  serves from local memory or from a chain of selects, and here the index would
//  come from a direction, so it would be both dynamic and non-uniform. The table
//  itself is indexed, which is a buffer read, and the sub-tap loops are unrolled
//  so even those subscripts are literals.
//
//  The shape below is compile-time for the same reason. Handed the tap count as a
//  function argument instead, the loops have no known trip count, do not unroll,
//  and the generated code carries a local array and one sample inside a loop
//  rather than the unrolled straight-line reads this file is meant to show.
//-------------------------------------------------------------------------

#ifndef PROBE_TAPS_PER_AXIS
#define PROBE_TAPS_PER_AXIS 8
#endif

#ifndef PROBE_NUM_ACTIVE_COEFFICIENTS
#define PROBE_NUM_ACTIVE_COEFFICIENTS 1
#endif

// Parameter order within a tap, as the table stores it
#define PARAM_DIR0   0
#define PARAM_DIR1   1
#define PARAM_DIR2   2
#define PARAM_LEVEL  3
#define PARAM_WEIGHT 4

static const uint NUM_AXES       = 4;
static const uint TAPS_PER_AXIS  = PROBE_TAPS_PER_AXIS;
static const uint NUM_ACTIVE     = PROBE_NUM_ACTIVE_COEFFICIENTS;
static const uint NUM_PARAMETERS = 5;
static const uint NUM_SUPER_TAPS = TAPS_PER_AXIS / 4;
static const uint NUM_INDICES    = ( NUM_AXES * TAPS_PER_AXIS ) / 4;

// The table's coefficients, as float4[ level ][ parameter ][ coefficient ][ index ].
// One float4 holds the four sub-taps of one index, and an index is
// NUM_SUPER_TAPS * axis + superTap. The payload starts at the header's
// payloadOffset; the engine reaches it through a bindless handle.
StructuredBuffer<float4> g_probeTable;

// One level of the environment's mip chain, one texture holding all four faces
Texture2D    g_source;
SamplerState g_sourceSampler;

//-------------------------------------------------------------------------
//  The tetrahedron
//-------------------------------------------------------------------------

// The four vertices, each at distance 1 from the centre. The product of a
// vertex's three signs is +1, which picks this orientation out of the two a
// regular tetrahedron can have.
static const float3 kVertex0 = float3(  0.57735027,  0.57735027,  0.57735027 );
static const float3 kVertex1 = float3(  0.57735027, -0.57735027, -0.57735027 );
static const float3 kVertex2 = float3( -0.57735027,  0.57735027, -0.57735027 );
static const float3 kVertex3 = float3( -0.57735027, -0.57735027,  0.57735027 );

// A frame is built around one vertex, and the face opposite that vertex is the one
// the frame belongs to.

// The vertex opposite a face, by selection rather than by a lookup
float3 FaceVertex( uint face )
{
    if( face == 1 )
    {
        return kVertex1;
    }

    if( face == 2 )
    {
        return kVertex2;
    }

    if( face == 3 )
    {
        return kVertex3;
    }

    return kVertex0;
}

// The face a direction reaches: the one whose vertex is most opposed to it. A face
// lies in the plane p . v = -1/3 for the vertex v opposite it, so the face the
// direction reaches first is the one with the most negative dot.
uint FaceFromDirection( float3 direction )
{
    float best = dot( direction, kVertex0 );
    uint  face = 0;

    float const alignment1 = dot( direction, kVertex1 );
    float const alignment2 = dot( direction, kVertex2 );
    float const alignment3 = dot( direction, kVertex3 );

    if( alignment1 < best )
    {
        best = alignment1;
        face = 1;
    }

    if( alignment2 < best )
    {
        best = alignment2;
        face = 2;
    }

    if( alignment3 < best )
    {
        best = alignment3;
        face = 3;
    }

    return face;
}

// The two coordinates the tap polynomials are evaluated at: the direction in the
// frame around this vertex, divided by the largest of its three components.
void PolynomialArguments( float3 direction, float3 axisVertex, float3 frameZ, out float theta, out float phi )
{
    float3 frameX = cross( axisVertex, frameZ );
    float const frameXLength = length( frameX );

    if( frameXLength <= 0.0 )
    {
        theta = 0.0;
        phi   = 0.0;
        return;
    }

    frameX /= frameXLength;
    float3 const frameY = cross( frameZ, frameX );

    float const x = dot( direction, frameX );
    float const y = dot( direction, frameY );
    float const z = dot( direction, frameZ );

    float const largest = max( abs( x ), max( abs( y ), abs( z ) ) );

    theta = ( largest > 0.0 ) ? ( x / largest ) : 0.0;
    phi   = ( largest > 0.0 ) ? ( y / largest ) : 0.0;
}

//-------------------------------------------------------------------------
//  Direction to tile coordinate
//-------------------------------------------------------------------------
//  The square holds four triangles: each face owns the triangle between two of
//  the square's corners and its centre. A direction resolves to a face by which
//  face plane it meets first, then to a coordinate inside that face's triangle.
//-------------------------------------------------------------------------

// A point's barycentric coordinates in a triangle
float3 Barycentric( float3 p, float3 a, float3 b, float3 c )
{
    float3 const v0 = b - a;
    float3 const v1 = c - a;
    float3 const v2 = p - a;

    float const d00 = dot( v0, v0 );
    float const d01 = dot( v0, v1 );
    float const d11 = dot( v1, v1 );
    float const d20 = dot( v2, v0 );
    float const d21 = dot( v2, v1 );

    float const denominator = ( d00 * d11 ) - ( d01 * d01 );
    float const v = ( ( d11 * d20 ) - ( d01 * d21 ) ) / denominator;
    float const w = ( ( d00 * d21 ) - ( d01 * d20 ) ) / denominator;

    return float3( 1.0 - v - w, v, w );
}

// Where a direction lands in the tile. Tile (0,0) is the top-left texel and v runs
// down, as it does in memory.
float2 TileCoordinate( float3 direction )
{
    float3 const unit = normalize( direction );
    uint const face = FaceFromDirection( unit );

    // Where the ray meets that face's plane, p . v = -1/3. point is a reserved
    // word in HLSL, hence the name.
    float3 const chartPoint = unit * ( -0.33333333 / dot( unit, FaceVertex( face ) ) );

    // The face's three corners, and where each of them sits in the tile. Selected
    // per face, because the tile corner a vertex maps to depends on the face and
    // not on the vertex alone.
    float3 a = 0.0;
    float3 b = 0.0;
    float3 c = 0.0;

    float2 tileA = 0.0;
    float2 tileB = 0.0;
    float2 tileC = 0.0;

    if( face == 0 )
    {
        a = kVertex1; b = kVertex3; c = kVertex2;
        tileA = float2( 0.0, 0.0 ); tileB = float2( 1.0, 0.0 ); tileC = float2( 0.5, 0.5 );
    }
    else if( face == 1 )
    {
        a = kVertex0; b = kVertex2; c = kVertex3;
        tileA = float2( 1.0, 0.0 ); tileB = float2( 1.0, 1.0 ); tileC = float2( 0.5, 0.5 );
    }
    else if( face == 2 )
    {
        a = kVertex0; b = kVertex3; c = kVertex1;
        tileA = float2( 1.0, 1.0 ); tileB = float2( 0.0, 1.0 ); tileC = float2( 0.5, 0.5 );
    }
    else
    {
        a = kVertex0; b = kVertex1; c = kVertex2;
        tileA = float2( 0.0, 1.0 ); tileB = float2( 0.0, 0.0 ); tileC = float2( 0.5, 0.5 );
    }

    // The barycentric coordinate of the point in the face is the same weighted
    // combination of the tile's own corners
    float3 const weight = Barycentric( chartPoint, a, b, c );

    return saturate( ( weight.x * tileA ) + ( weight.y * tileB ) + ( weight.z * tileC ) );
}

//-------------------------------------------------------------------------
//  The table
//-------------------------------------------------------------------------

// One of a sub-tap's five parameters, as a quadratic in theta^2 and phi^2. The
// quadratic has no linear term, which is what keeps the taps continuous where two
// frames meet.
float TapParameter( uint level, uint parameter, uint index, uint subTap, float theta2, float phi2 )
{
    float value = 0.0;

    [unroll]
    for( uint coefficient = 0; coefficient < NUM_ACTIVE; ++coefficient )
    {
        uint const slot = ( ( ( level * NUM_PARAMETERS ) + parameter ) * NUM_ACTIVE + coefficient ) * NUM_INDICES + index;

        float const basis = ( coefficient == 0 ) ? 1.0 : ( ( coefficient == 1 ) ? theta2 : phi2 );

        value += basis * g_probeTable[slot][subTap];
    }

    return value;
}

//-------------------------------------------------------------------------

// The taps of one frame, added to the running total. The frame's vertex is an
// argument, so the caller decides which axis this is by naming it.
void AccumulateFrame( uint level,
                      uint axisIndex,
                      float3 axisVertex,
                      float3 direction,
                      float3 frameZ,
                      inout float3 accumulated,
                      inout float weightSum )
{
    // A frame carries nothing where the direction is close to its own vertex, which
    // is where the tangent frame below degenerates
    float const frameWeight = saturate( ( 0.75 - abs( dot( direction, axisVertex ) ) ) / 0.75 );

    if( frameWeight <= 0.0 )
    {
        return;
    }

    float3 frameX = cross( axisVertex, frameZ );
    float const frameXLength = length( frameX );

    if( frameXLength <= 0.0 )
    {
        return;
    }

    frameX /= frameXLength;
    float3 const frameY = cross( frameZ, frameX );

    float theta = 0.0;
    float phi   = 0.0;
    PolynomialArguments( direction, axisVertex, frameZ, theta, phi );

    float const theta2 = theta * theta;
    float const phi2   = phi * phi;

    [unroll]
    for( uint superTap = 0; superTap < NUM_SUPER_TAPS; ++superTap )
    {
        uint const index = ( NUM_SUPER_TAPS * axisIndex ) + superTap;

        [unroll]
        for( uint subTap = 0; subTap < 4; ++subTap )
        {
            float3 tapDirection;
            tapDirection.x = TapParameter( level, PARAM_DIR0, index, subTap, theta2, phi2 );
            tapDirection.y = TapParameter( level, PARAM_DIR1, index, subTap, theta2, phi2 );
            tapDirection.z = TapParameter( level, PARAM_DIR2, index, subTap, theta2, phi2 );

            float const major = max( abs( tapDirection.x ), max( abs( tapDirection.y ), abs( tapDirection.z ) ) );

            if( major <= 0.0 )
            {
                continue;
            }

            float3 const unit = normalize( tapDirection / major );

            // How squarely the tap faces the face it landed on: 1 at a face centre
            // and 1/3 at a corner. It drives the level correction below, because a
            // tap near a corner reads a coarser mip - the texels there cover less
            // solid angle, so a coarser one matches the tap's own footprint.
            uint const face = FaceFromDirection( unit );

            float const alignment = max( -dot( unit, FaceVertex( face ) ), 1.0e-6 );

            float const levelCorrection = -1.5 * log2( alignment );

            float const tapLevel  = TapParameter( level, PARAM_LEVEL, index, subTap, theta2, phi2 ) + levelCorrection;
            float const tapWeight = TapParameter( level, PARAM_WEIGHT, index, subTap, theta2, phi2 ) * frameWeight;

            accumulated += tapWeight * g_source.SampleLevel( g_sourceSampler, TileCoordinate( unit ), tapLevel ).rgb;
            weightSum   += tapWeight;
        }
    }
}

//-------------------------------------------------------------------------

// The filtered radiance for an output direction, at one level of the output
// chain. n is the surface normal, which is also the direction the result is
// for. level is the output level the table row is being read for, 0 to 6.
float3 FilterProbeTetrahedral( float3 n, uint level )
{
    float3 const direction = normalize( n );
    float3 const frameZ    = direction;

    float3 accumulated = 0.0;
    float  weightSum   = 0.0;

    // One call per frame rather than a loop over a table of vertices
    AccumulateFrame( level, 0, kVertex0, direction, frameZ, accumulated, weightSum );
    AccumulateFrame( level, 1, kVertex1, direction, frameZ, accumulated, weightSum );
    AccumulateFrame( level, 2, kVertex2, direction, frameZ, accumulated, weightSum );
    AccumulateFrame( level, 3, kVertex3, direction, frameZ, accumulated, weightSum );

    // The runtime divides by the weight sum, so a constant environment stays
    // constant whatever the taps do
    return ( weightSum > 0.0 ) ? ( accumulated / weightSum ) : 0.0;
}
