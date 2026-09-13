//-------------------------------------------------------------------------
//  Reading a FilterFitter cubemap table
//-------------------------------------------------------------------------
//  What a runtime does with a table fitted for a cubemap: one output direction
//  in, one filtered radiance out, as a handful of weighted TextureCube reads.
//
//  This mirrors BuildTableTaps in Source/TableHarness.cpp, which is the
//  authority. If the two disagree, that file is right.
//
//  Two numbers have to match the table being read, and both are in its header:
//
//      PROBE_TAPS_PER_AXIS             8, 16 or 32
//      PROBE_NUM_ACTIVE_COEFFICIENTS   1 for a constant table, 3 for a quadratic
//
//  A cubemap has three axial frames, one per axis, and that is the only thing
//  this file knows about the map it reads. The tetrahedral map's reader is a
//  separate file with its own frames and its own coordinate system; nothing is
//  shared between them.
//
//  NO INDEXED TABLES OF CONSTANTS. The frames are named values and the loop over
//  them is written out, and the components a frame reads arrive as arguments
//  rather than as subscripts. An array indexed by anything but a literal becomes a
//  dynamic index, which a GPU serves from local memory or from a chain of selects,
//  and it costs more than the taps do. The table itself is indexed, which is a
//  buffer read, and the sub-tap loops are unrolled so even those subscripts are
//  literals.
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

static const uint NUM_AXES       = 3;
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

// One level of the environment's mip chain
TextureCube  g_source;
SamplerState g_sourceSampler;

// The three frames' axes, each a named value rather than a row of an array
static const float3 kAxisX = float3( 1.0, 0.0, 0.0 );
static const float3 kAxisY = float3( 0.0, 1.0, 0.0 );
static const float3 kAxisZ = float3( 0.0, 0.0, 1.0 );

//-------------------------------------------------------------------------

// The largest absolute component. The frame weights and the level correction are
// both written against it, so it is computed in one place.
float MajorComponent( float3 v )
{
    return max( abs( v.x ), max( abs( v.y ), abs( v.z ) ) );
}

// The two coordinates the tap polynomials are evaluated at, bounded by 1. The
// frame's own component and the two it reads from arrive as values, so there is
// nothing to index.
void PolynomialArguments( float otherX, float otherY, float own, out float theta, out float phi )
{
    float const maxXY    = max( abs( otherX ), abs( otherY ) );
    float const scaledX  = otherX / maxXY;
    float const scaledY  = otherY / maxXY;

    theta = ( scaledY < scaledX ) ? scaledY : -scaledY;

    // The frame's own component is 1 when this is the direction's major axis, and
    // then the second coordinate is the larger of the other two
    float const ownAbsolute = abs( own );

    phi = ( ownAbsolute >= 0.999 ) ? maxXY : ownAbsolute;
}

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

// The taps of one frame, added to the running total. The frame's axis and the
// components of the face vector it reads from are arguments, so the caller decides
// which axis this is by naming it.
void AccumulateFrame( uint level,
                      uint axisIndex,
                      float3 axis,
                      float otherX,
                      float otherY,
                      float own,
                      float3 frameZ,
                      inout float3 accumulated,
                      inout float weightSum )
{
    // A frame carries nothing where the direction is close to its own axis, which
    // is where the tangent frame below degenerates. Two frames always survive,
    // because a direction on a cubemap always has one component of magnitude 1.
    float const frameWeight = ( max( abs( otherX ), abs( otherY ) ) - 0.75 ) * 4.0;

    if( frameWeight <= 0.0 )
    {
        return;
    }

    float3 frameX = cross( axis, frameZ );
    float const frameXLength = length( frameX );

    if( frameXLength <= 0.0 )
    {
        return;
    }

    frameX /= frameXLength;
    float3 const frameY = cross( frameZ, frameX );

    float theta = 0.0;
    float phi   = 0.0;
    PolynomialArguments( otherX, otherY, own, theta, phi );

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

            float const major = MajorComponent( tapDirection );

            if( major <= 0.0 )
            {
                continue;
            }

            // Normalized by its largest component rather than by length, so the
            // level correction below reads the scale it is written against
            tapDirection /= major;

            // A cube texel covers a different solid angle across a face, so the mip
            // a tap reads is corrected for where it landed: 0 at a face centre,
            // 0.75 * log2( 3 ) at a corner.
            float const levelCorrection = 0.75 * log2( dot( tapDirection, tapDirection ) );

            float const tapLevel  = TapParameter( level, PARAM_LEVEL, index, subTap, theta2, phi2 ) + levelCorrection;
            float const tapWeight = TapParameter( level, PARAM_WEIGHT, index, subTap, theta2, phi2 ) * frameWeight;

            accumulated += tapWeight * g_source.SampleLevel( g_sourceSampler, tapDirection, tapLevel ).rgb;
            weightSum   += tapWeight;
        }
    }
}

//-------------------------------------------------------------------------

// The filtered radiance for an output direction, at one level of the output
// chain. n is the surface normal, which is also the direction the result is
// for. level is the output level the table row is being read for, 0 to 6.
float3 FilterProbeCube( float3 n, uint level )
{
    float3 const frameZ = normalize( n );

    // The face coordinate vector: the direction with its largest component divided
    // out, so one component is exactly 1. The frame weights and the polynomial
    // arguments are written against that scale rather than against length.
    float3 const faceVector = n / MajorComponent( n );

    float3 accumulated = 0.0;
    float  weightSum   = 0.0;

    // A frame's weight and its arguments are read from the two components it does
    // not own, so each call names the two it wants
    AccumulateFrame( level, 0, kAxisX, faceVector.y, faceVector.z, faceVector.x, frameZ, accumulated, weightSum );
    AccumulateFrame( level, 1, kAxisY, faceVector.x, faceVector.z, faceVector.y, frameZ, accumulated, weightSum );
    AccumulateFrame( level, 2, kAxisZ, faceVector.x, faceVector.y, faceVector.z, frameZ, accumulated, weightSum );

    // The runtime divides by the weight sum, so a constant environment stays
    // constant whatever the taps do
    return ( weightSum > 0.0 ) ? ( accumulated / weightSum ) : 0.0;
}
