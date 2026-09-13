#include "Assert.h"
#include "CoefficientTable.h"

#include <cstddef>

// The vendored tables are HLSL excerpts that declare static const float4 arrays and use float4( a, b, c, d ) constructor syntax, so they need a shim type with a matching constructor to be included as C++ at all.
// They also all declare coeffs, so each is included into its own namespace.
//
// The shim's constructors are constexpr so the arrays get static rather than dynamic initialisation - quad_32 alone is 2520 float4, 40320 bytes.

struct float4
{
    float x;
    float y;
    float z;
    float w;

    constexpr float4() : x( 0.0f ), y( 0.0f ), z( 0.0f ), w( 0.0f ) {}
    constexpr float4( float inX, float inY, float inZ, float inW ) : x( inX ), y( inY ), z( inZ ), w( inW ) {}

    float GetComponent( uint32_t component ) const
    {
        switch ( component )
        {
            case 0:  return x;
            case 1:  return y;
            case 2:  return z;
            default: return w;
        }
    }
};

namespace FilterFitter
{
    // The vendored files are Activision's data included as C++, and they are not ours to edit.
    // 
    // Their literals are full-precision decimals passed to a float constructor, so every value truncates and C4305 fires on all of them.
    // Suppressed around these four includes rather than with /wd4305 project-wide, so a genuine truncation in our own code still fails the build.
    #pragma warning( push, 0 )

    namespace RefConst8
    {
        #include "Reference/coeffs_const_8.txt"
    }

    namespace RefConst16
    {
        #include "Reference/coeffs_const_16.txt"
    }

    namespace RefConst32
    {
        #include "Reference/coeffs_const_32.txt"
    }

    namespace RefQuad32
    {
        #include "Reference/coeffs_quad_32.txt"
    }

    #pragma warning( pop )

    // Widening the two published shapes into one uniform layout
    //-------------------------------------------------------------------------
    // Both shapes reserve NumCoefficients slots per ( parameter, index ), even though a constant table only uses coefficient 0.
    // The unused slots stay zero, which is exactly what makes a constant table the degenerate quadratic with a1 = a2 = 0 and lets one evaluator read both.

    template <uint32_t NumIndices>
    static void CopyConstantTable
    (
        float4 const ( &source )[CoefficientTable::NumLevels][CoefficientTable::NumParameters][NumIndices],
        uint32_t numTaps,
        std::vector<double>& destination
    )
    {
        uint32_t const numIndices = ( CoefficientTable::NumCubeAxes * numTaps ) / 4;
        FF_ASSERT( numIndices == NumIndices );

        destination.assign( static_cast<size_t>( CoefficientTable::NumLevels ) * CoefficientTable::NumParameters * CoefficientTable::NumCoefficients * numIndices * 4, 0.0 );

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            for ( uint32_t parameter = 0; parameter < CoefficientTable::NumParameters; ++parameter )
            {
                for ( uint32_t index = 0; index < numIndices; ++index )
                {
                    for ( uint32_t subTap = 0; subTap < 4; ++subTap )
                    {
                        size_t const flat = ( ( ( static_cast<size_t>( level ) * CoefficientTable::NumParameters ) + parameter ) * CoefficientTable::NumCoefficients * numIndices * 4 ) + ( static_cast<size_t>( index ) * 4 ) + subTap;

                        destination[flat] = static_cast<double>( source[level][parameter][index].GetComponent( subTap ) );
                    }
                }
            }
        }
    }

    template <uint32_t NumIndices>
    static void CopyQuadraticTable
    (
        float4 const ( &source )[CoefficientTable::NumLevels][CoefficientTable::NumParameters][CoefficientTable::NumCoefficients][NumIndices],
        uint32_t numTaps,
        std::vector<double>& destination
    )
    {
        uint32_t const numIndices = ( CoefficientTable::NumCubeAxes * numTaps ) / 4;
        FF_ASSERT( numIndices == NumIndices );

        destination.assign( static_cast<size_t>( CoefficientTable::NumLevels ) * CoefficientTable::NumParameters * CoefficientTable::NumCoefficients * numIndices * 4, 0.0 );

        for ( uint32_t level = 0; level < CoefficientTable::NumLevels; ++level )
        {
            for ( uint32_t parameter = 0; parameter < CoefficientTable::NumParameters; ++parameter )
            {
                for ( uint32_t coefficient = 0; coefficient < CoefficientTable::NumCoefficients; ++coefficient )
                {
                    for ( uint32_t index = 0; index < numIndices; ++index )
                    {
                        for ( uint32_t subTap = 0; subTap < 4; ++subTap )
                        {
                            size_t const flat = ( ( ( ( static_cast<size_t>( level ) * CoefficientTable::NumParameters ) + parameter ) * CoefficientTable::NumCoefficients ) + coefficient ) * numIndices * 4 + ( static_cast<size_t>( index ) * 4 ) + subTap;

                            destination[flat] = static_cast<double>( source[level][parameter][coefficient][index].GetComponent( subTap ) );
                        }
                    }
                }
            }
        }
    }

    // TableShape
    //-------------------------------------------------------------------------

    static TableShape MakePublishedShape( uint32_t numTaps, uint32_t numActiveCoefficients, char const* pName )
    {
        TableShape shape;

        std::snprintf( shape.m_name, sizeof( shape.m_name ), "%s", pName );

        shape.m_numTapsPerAxis = numTaps;
        shape.m_numAxes = CoefficientTable::NumCubeAxes;
        shape.m_numActiveCoefficients = numActiveCoefficients;
        shape.m_projection = ProbeMap::Cube;

        return shape;
    }

    TableShape::TableShape( ReferenceTable table )
    {
        switch ( table )
        {
            case ReferenceTable::Const8:  *this = MakePublishedShape( 8, 1, "const_8" );  break;
            case ReferenceTable::Const16: *this = MakePublishedShape( 16, 1, "const_16" ); break;
            case ReferenceTable::Const32: *this = MakePublishedShape( 32, 1, "const_32" ); break;
            default:                      *this = MakePublishedShape( 32, 3, "quad_32" );  break;
        }
    }

    //-------------------------------------------------------------------------

    TableShape TableShape::ForMap( ProbeMap map, uint32_t numTapsPerAxis, uint32_t numActiveCoefficients, char const* pName )
    {
        FF_ASSERT( pName != nullptr );
        FF_ASSERT( numTapsPerAxis > 0 );
        FF_ASSERT( ( numTapsPerAxis % 4 ) == 0 );
        FF_ASSERT( ( numActiveCoefficients == 1 ) || ( numActiveCoefficients == CoefficientTable::NumCoefficients ) );

        TableShape shape;

        std::snprintf( shape.m_name, sizeof( shape.m_name ), "%s", pName );

        shape.m_numTapsPerAxis = numTapsPerAxis;
        shape.m_numAxes = GetProbeMapAxisCount( map );
        shape.m_numActiveCoefficients = numActiveCoefficients;
        shape.m_projection = map;

        FF_ASSERT( shape.m_numAxes > 0 );

        return shape;
    }

    //-------------------------------------------------------------------------

    uint32_t TableShape::GetPublishedShape() const
    {
        ReferenceTable const published[] = { ReferenceTable::Const8, ReferenceTable::Const16,
                                             ReferenceTable::Const32, ReferenceTable::Quad32 };

        for ( ReferenceTable candidate : published )
        {
            TableShape const shape( candidate );

            if ( shape == *this )
            {
                return static_cast<uint32_t>( candidate );
            }
        }

        return PublishedShapeNone;
    }

    //-------------------------------------------------------------------------

    bool TableShape::operator==( TableShape const& other ) const
    {
        return ( m_numTapsPerAxis == other.m_numTapsPerAxis )
            && ( m_numAxes == other.m_numAxes )
            && ( m_numActiveCoefficients == other.m_numActiveCoefficients )
            && ( m_projection == other.m_projection )
            && ( std::strcmp( m_name, other.m_name ) == 0 );
    }

    //-------------------------------------------------------------------------

    void CoefficientTable::Load( ReferenceTable table )
    {
        TableShape const shape( table );

        m_numAxes = shape.m_numAxes;
        m_numTaps = shape.m_numTapsPerAxis;
        m_numActiveCoefficients = shape.m_numActiveCoefficients;
        m_shape = shape;

        switch ( table )
        {
            case ReferenceTable::Const8:
            {
                m_pName = "const_8";
                CopyConstantTable( RefConst8::coeffs, m_numTaps, m_values );
            }
            break;

            case ReferenceTable::Const16:
            {
                m_pName = "const_16";
                CopyConstantTable( RefConst16::coeffs, m_numTaps, m_values );
            }
            break;

            case ReferenceTable::Const32:
            {
                m_pName = "const_32";
                CopyConstantTable( RefConst32::coeffs, m_numTaps, m_values );
            }
            break;

            default:
            {
                m_pName = "quad_32";
                CopyQuadraticTable( RefQuad32::coeffs, m_numTaps, m_values );
            }
            break;
        }
    }

    //-------------------------------------------------------------------------

    void CoefficientTable::Create( ReferenceTable table )
    {
        Create( TableShape( table ) );
    }

    //-------------------------------------------------------------------------

    void CoefficientTable::Create( TableShape const& shape )
    {
        FF_ASSERT( shape.m_numTapsPerAxis > 0 );
        FF_ASSERT( ( shape.m_numTapsPerAxis % 4 ) == 0 );
        FF_ASSERT( shape.m_numAxes > 0 );
        FF_ASSERT( ( shape.m_numActiveCoefficients == 1 ) || ( shape.m_numActiveCoefficients == NumCoefficients ) );

        // The map and the axis count are two statements about the same thing, so they are checked against each other rather than trusted.
        FF_ASSERT( GetProbeMapAxisCount( shape.m_projection ) == shape.m_numAxes );

        m_numTaps = shape.m_numTapsPerAxis;
        m_numAxes = shape.m_numAxes;
        m_numActiveCoefficients = shape.m_numActiveCoefficients;
        m_shape = shape;
        m_pName = m_shape.m_name;

        m_values.assign( GetStorageSize(), 0.0 );
    }

    //-------------------------------------------------------------------------

    size_t CoefficientTable::GetStorageSize() const
    {
        return static_cast<size_t>( NumLevels ) * NumParameters * NumCoefficients * GetNumIndices() * 4;
    }

    //-------------------------------------------------------------------------

    size_t CoefficientTable::GetStorageOffset( uint32_t level, uint32_t parameter, uint32_t coefficient, uint32_t index, uint32_t subTap ) const
    {
        return ( ( ( ( static_cast<size_t>( level ) * NumParameters ) + parameter ) * NumCoefficients ) + coefficient ) * GetNumIndices() * 4 + ( static_cast<size_t>( index ) * 4 ) + subTap;
    }

    //-------------------------------------------------------------------------

    double CoefficientTable::GetCoefficient( uint32_t level, uint32_t parameter, uint32_t coefficient, uint32_t index, uint32_t subTap ) const
    {
        FF_ASSERT( level < NumLevels );
        FF_ASSERT( parameter < NumParameters );
        FF_ASSERT( coefficient < NumCoefficients );
        FF_ASSERT( index < GetNumIndices() );
        FF_ASSERT( subTap < 4 );

        return m_values[GetStorageOffset( level, parameter, coefficient, index, subTap )];
    }

    void CoefficientTable::SetCoefficient( uint32_t level, uint32_t parameter, uint32_t coefficient, uint32_t index, uint32_t subTap, double value )
    {
        FF_ASSERT( level < NumLevels );
        FF_ASSERT( parameter < NumParameters );
        FF_ASSERT( coefficient < NumCoefficients );
        FF_ASSERT( index < GetNumIndices() );
        FF_ASSERT( subTap < 4 );

        m_values[GetStorageOffset( level, parameter, coefficient, index, subTap )] = value;
    }

    //  Tap addressing. Index enumerates ( axis, superTap ) and subTap the four within, so tap / 4 is the index and tap % 4 the sub-tap.
    //-------------------------------------------------------------------------

    double CoefficientTable::GetTapCoefficient( uint32_t level, uint32_t tap, uint32_t parameter, uint32_t coefficient ) const
    {
        FF_ASSERT( tap < GetTapCount() );

        return GetCoefficient( level, parameter, coefficient, tap / 4, tap % 4 );
    }

    void CoefficientTable::SetTapCoefficient( uint32_t level, uint32_t tap, uint32_t parameter, uint32_t coefficient, double value )
    {
        FF_ASSERT( tap < GetTapCount() );

        SetCoefficient( level, parameter, coefficient, tap / 4, tap % 4, value );
    }

    //-------------------------------------------------------------------------

    void CoefficientTable::GetLevelParameters( uint32_t level, std::vector<double>& values ) const
    {
        FF_ASSERT( level < NumLevels );

        values.resize( GetLevelParameterCount() );

        size_t writeIndex = 0;
        for ( uint32_t tap = 0; tap < GetTapCount(); ++tap )
        {
            for ( uint32_t parameter = 0; parameter < NumParameters; ++parameter )
            {
                for ( uint32_t coefficient = 0; coefficient < m_numActiveCoefficients; ++coefficient )
                {
                    values[writeIndex] = GetTapCoefficient( level, tap, parameter, coefficient );
                    ++writeIndex;
                }
            }
        }
    }

    void CoefficientTable::SetLevelParameters( uint32_t level, std::vector<double> const& values )
    {
        FF_ASSERT( level < NumLevels );
        FF_ASSERT( values.size() == GetLevelParameterCount() );

        size_t readIndex = 0;
        for ( uint32_t tap = 0; tap < GetTapCount(); ++tap )
        {
            for ( uint32_t parameter = 0; parameter < NumParameters; ++parameter )
            {
                for ( uint32_t coefficient = 0; coefficient < m_numActiveCoefficients; ++coefficient )
                {
                    SetTapCoefficient( level, tap, parameter, coefficient, values[readIndex] );
                    ++readIndex;
                }
            }
        }
    }

    //-------------------------------------------------------------------------

    bool CoefficientTable::ValidateShape() const
    {
        if ( m_numTaps == 0 )
        {
            return false;
        }

        if ( ( m_numTaps % 4 ) != 0 )
        {
            return false;
        }

        // The axis count is the frame's, so it is checked for being usable rather than for being three: a tetrahedral frame's four is as valid as a cubemap's three, and a zero would make the tap count zero. 
        // It also has to agree with the map the table names, because the two are read by different consumers: the axis count sizes the array, the map says what the array means.
        if ( m_numAxes == 0 )
        {
            return false;
        }

        if ( GetProbeMapAxisCount( m_shape.m_projection ) != m_numAxes )
        {
            return false;
        }

        if ( ( m_numActiveCoefficients != 1 ) && ( m_numActiveCoefficients != NumCoefficients ) )
        {
            return false;
        }

        return m_values.size() == GetStorageSize();
    }

    //-------------------------------------------------------------------------

    char const* GetReferenceTableName( ReferenceTable table )
    {
        switch ( table )
        {
            case ReferenceTable::Const8:    return "const_8";
            case ReferenceTable::Const16:   return "const_16";
            case ReferenceTable::Const32:   return "const_32";
            default:                        return "quad_32";
        }
    }
}
