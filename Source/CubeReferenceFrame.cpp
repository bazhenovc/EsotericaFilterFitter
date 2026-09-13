#include "Assert.h"
#include "MapProjection.h"

#include <cmath>

namespace FilterFitter
{
    // Analytic solid angle
    //-------------------------------------------------------------------------
    // The cube-face area element is
    //
    //      1 / ( u^2 + v^2 + 1 )^(3/2)
    //
    // whose mixed antiderivative with respect to both u and v is
    //
    //      atan2( u * v, sqrt( u^2 + v^2 + 1 ) )
    //
    // so the solid angle of a face-space rectangle is that expression evaluated at the four corners with alternating signs.

    static double CubeAreaElement( double u, double v )
    {
        return std::atan2( u * v, std::sqrt( ( u * u ) + ( v * v ) + 1.0 ) );
    }

    //-------------------------------------------------------------------------

    double GetCubeTexelSolidAngle( double minU, double minV, double maxU, double maxV )
    {
        return ( CubeAreaElement( maxU, maxV ) - CubeAreaElement( minU, maxV ) ) - ( CubeAreaElement( maxU, minV ) - CubeAreaElement( minU, minV ) );
    }

    double GetCubeJacobian( double u, double v )
    {
        double const radiusSquared = ( u * u ) + ( v * v ) + 1.0;
        return 1.0 / ( radiusSquared * std::sqrt( radiusSquared ) );
    }

    void GetCubeFaceDirection( double* pOutDir, double u, double v, uint32_t face )
    {
        switch ( face )
        {
            case 0:
            {
                pOutDir[0] = 1.0;
                pOutDir[1] = v;
                pOutDir[2] = -u;
            }
            break;

            case 1:
            {
                pOutDir[0] = -1.0;
                pOutDir[1] = v;
                pOutDir[2] = u;
            }
            break;

            case 2:
            {
                pOutDir[0] = u;
                pOutDir[1] = 1.0;
                pOutDir[2] = -v;
            }
            break;

            case 3:
            {
                pOutDir[0] = u;
                pOutDir[1] = -1.0;
                pOutDir[2] = v;
            }
            break;

            case 4:
            {
                pOutDir[0] = u;
                pOutDir[1] = v;
                pOutDir[2] = 1.0;
            }
            break;

            default:
            {
                pOutDir[0] = -u;
                pOutDir[1] = v;
                pOutDir[2] = -1.0;
            }
            break;
        }
    }

    //-------------------------------------------------------------------------

    void GetCubeFaceAndUVFromDirection( double const* pDir, uint32_t& face, double& u, double& v )
    {
        double const absX = std::fabs( pDir[0] );
        double const absY = std::fabs( pDir[1] );
        double const absZ = std::fabs( pDir[2] );

        if ( ( absX >= absY ) && ( absX >= absZ ) )
        {
            if ( pDir[0] > 0.0 )
            {
                face = 0;
                u = -pDir[2] / pDir[0];
                v = pDir[1] / pDir[0];
            }
            else
            {
                face = 1;
                u = -pDir[2] / pDir[0];
                v = -pDir[1] / pDir[0];
            }
        }
        else if ( absY >= absZ )
        {
            if ( pDir[1] > 0.0 )
            {
                face = 2;
                u = pDir[0] / pDir[1];
                v = -pDir[2] / pDir[1];
            }
            else
            {
                face = 3;
                u = -pDir[0] / pDir[1];
                v = -pDir[2] / pDir[1];
            }
        }
        else
        {
            if ( pDir[2] > 0.0 )
            {
                face = 4;
                u = pDir[0] / pDir[2];
                v = pDir[1] / pDir[2];
            }
            else
            {
                face = 5;
                u = pDir[0] / pDir[2];
                v = -pDir[1] / pDir[2];
            }
        }
    }
}
