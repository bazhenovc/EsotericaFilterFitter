#include "DFGTable.h"

#include "Assert.h"
#include "ParallelFor.h"

#include <cmath>

namespace FilterFitter
{
    //-------------------------------------------------------------------------

    DFGTexel EvaluateDFGTexel( double ndotV, double roughness, uint32_t sampleCount, DFGSampleSequence sequence )
    {
        DFGTexel result;

        if ( sampleCount == 0 )
        {
            return result;
        }

        // Held off an exact zero, which no texel centre produces and which the expressions below would divide by.
        double const cosV = ( ndotV < DFGIntegrand::MinCosine ) ? DFGIntegrand::MinCosine : ( ( ndotV > 1.0 ) ? 1.0 : ndotV );
        double const sinV = std::sqrt( ( 1.0 - cosV ) * ( 1.0 + cosV ) );

        // The engine's own mapping, and the only place this file chooses anything: the width is the roughness squared and the shadowing term is linear in the roughness.
        // A mirror falls out of it rather than being special cased - at roughness zero the sample is the normal itself, and the weight below is one.
        double const alpha = DFGIntegrand::EngineAlpha( roughness );
        double const alphaSquared = alpha * alpha;
        double const k = DFGIntegrand::EngineShadowingK( roughness );

        double const g1V = DFGIntegrand::GeometrySchlick( cosV, k );

        double scale = 0.0;
        double bias = 0.0;

        for ( uint32_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex )
        {
            double xiX = 0.0;
            double xiY = 0.0;

            if ( sequence == DFGSampleSequence::Hammersley )
            {
                DFGIntegrand::Hammersley( sampleIndex, sampleCount, xiX, xiY );
            }
            else
            {
                DFGIntegrand::Stratified( sampleIndex, sampleCount, xiX, xiY );
            }

            double half[3] = { 0.0, 0.0, 1.0 };
            DFGIntegrand::SampleHalfVector( xiX, xiY, alphaSquared, half );

            // H is built with phi measured from the x axis in the plane V lies in, and the surface frame is that same frame, so no rotation is needed.
            double const vdotH = ( sinV * half[0] ) + ( cosV * half[2] );
            double const dotNH = half[2];
            double const dotNL = ( 2.0 * vdotH * dotNH ) - cosV;

            double const visibility = DFGIntegrand::VisTerm( vdotH, cosV, dotNH, dotNL, g1V, k );

            if ( visibility <= 0.0 )
            {
                continue;
            }

            double const fresnel = DFGIntegrand::FresnelBase( vdotH );
            double const weight = visibility / static_cast<double>( sampleCount );

            scale += ( 1.0 - fresnel ) * weight;
            bias += fresnel * weight;
        }

        result.m_scale = scale;
        result.m_bias = bias;

        return result;
    }

    //-------------------------------------------------------------------------

    void BuildDFGTable( DFGOptions const& options, DFGTable& table )
    {
        FF_ASSERT( options.m_resolution > 0 );

        uint32_t const resolution = options.m_resolution;

        table.m_resolution = resolution;
        table.m_sampleCount = options.m_sampleCount;
        table.m_texels.assign( static_cast<size_t>( resolution ) * resolution, DFGTexel() );

        ParallelForRanges( resolution, [&] ( uint32_t begin, uint32_t end, uint32_t workerIndex )
        {
            (void) workerIndex;

            for ( uint32_t row = begin; row < end; ++row )
            {
                double const roughness = table.GetRoughness( row );

                for ( uint32_t column = 0; column < resolution; ++column )
                {
                    table.SetTexel
                    (
                        column, row,
                        EvaluateDFGTexel( table.GetNdotV( column ), roughness, options.m_sampleCount, DFGSampleSequence::Hammersley )
                    );
                }
            }
        } );
    }

    //-------------------------------------------------------------------------

    void GetHalfTexel( DFGTable const& table, uint32_t column, uint32_t row, DFGHalfTexel& texel )
    {
        DFGTexel const& value = table.GetTexel( column, row );

        texel.m_scale = FloatToHalf( static_cast<float>( value.m_scale ) );
        texel.m_bias = FloatToHalf( static_cast<float>( value.m_bias ) );
    }

    //  Half floats
    //-------------------------------------------------------------------------

    #pragma warning( push )
    #pragma warning( disable : 4201 )

    // Half precision floats based on // https://gist.github.com/rygorous/2156668

    union FP32
    {
        uint32_t u;
        float    f;
        struct
        {
            uint32_t m_mantissa : 23;
            uint32_t m_exponent : 8;
            uint32_t m_sign : 1;
        };
    };

    union FP16
    {
        uint16_t u;
        struct
        {
            uint32_t m_mantissa : 10;
            uint32_t m_exponent : 5;
            uint32_t m_sign : 1;
        };
    };

    #pragma warning( pop )

    uint16_t FloatToHalf( float const value )
    {
        FP32 f = { 0 };
        FP16 o = { 0 };

        f.f = value;

        // Based on ISPC reference code (with minor modifications)
        if ( f.m_exponent == 0 ) // Signed zero/denormal (which will underflow)
        {
            o.m_exponent = 0;
        }
        else if ( f.m_exponent == 255 ) // Inf or NaN (all exponent bits set)
        {
            o.m_exponent = 31;
            o.m_mantissa = f.m_mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
        }
        else // Normalized number
        {
            // Exponent unbias the single, then bias the halfp
            int const newExponent = f.m_exponent - 127 + 15;
            if ( newExponent >= 31 ) // Overflow, return signed infinity
            {
                o.m_exponent = 31;
            }
            else if ( newExponent <= 0 ) // Underflow
            {
                if ( ( 14 - newExponent ) <= 24 ) // Mantissa might be non-zero
                {
                    uint32_t const mantissa = f.m_mantissa | 0x800000; // Hidden 1 bit
                    o.m_mantissa = mantissa >> ( 14 - newExponent );
                    if ( ( mantissa >> ( 13 - newExponent ) ) & 1 ) // Check for rounding
                    {
                        o.u++; // Round, might overflow into exp bit, but this is OK
                    }
                }
            }
            else
            {
                o.m_exponent = newExponent;
                o.m_mantissa = f.m_mantissa >> 13;
                if ( f.m_mantissa & 0x1000 ) // Check for rounding
                {
                    o.u++; // Round, might overflow to inf, this is OK
                }
            }
        }

        o.m_sign = f.m_sign;
        return o.u;
    }

    //-------------------------------------------------------------------------

    float HalfToFloat( uint16_t const value )
    {
        FP16 h = { 0 };
        h.u = value;

        static constexpr FP32     MAGIC = { 113 << 23 };
        static constexpr uint32_t SHIFTED_EXP = 0x7c00 << 13; // exponent mask after shift

        FP32 o;
        o.u = ( h.u & 0x7fff ) << 13;           // exponent/mantissa bits
        uint32_t const exp = SHIFTED_EXP & o.u; // just the exponent
        o.u += ( 127 - 15 ) << 23;              // exponent adjust

        // handle exponent special cases
        if ( exp == SHIFTED_EXP ) // Inf/NaN?
        {
            o.u += ( 128 - 16 ) << 23; // extra exp adjust
        }
        else if ( exp == 0 ) // Zero/Denormal?
        {
            o.u += 1 << 23; // extra exp adjust
            o.f -= MAGIC.f; // renormalize
        }

        o.u |= ( h.u & 0x8000 ) << 16; // sign bit
        return o.f;
    }
}
