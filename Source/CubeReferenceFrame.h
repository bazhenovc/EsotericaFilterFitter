#pragma once

#include "MapProjection.h"

//  Cubemap's reference frame
//-------------------------------------------------------------------------
// The frame contract is implemented once, generically, by  MapReferenceFrame< TMap > in MapReferenceFrame.h. 
// This header is the cube's name for that instantiation, which is what the fit, the accumulator and the published-table checks are written against.
//
// The cube's face and UV convention - kept identical to Reference/downsample_cubemap.txt and Reference/filter_using_table_128.txt so that our tables and theirs can be compared directly - lives in CubeProjection in MapProjection.h:
//
//      face 0: (  1,  v, -u )      face 3: (  u, -1,  v )
//      face 1: ( -1,  v,  u )      face 4: (  u,  v,  1 )
//      face 2: (  u,  1, -v )      face 5: ( -u,  v, -1 )
//
//      u =   ( 2x + 1 ) / R - 1
//      v = - ( 2y + 1 ) / R + 1
//
//  u and v keep the vendored names deliberately, so this file can be read beside theirs.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    using CubeReferenceFrame = MapReferenceFrame< CubeProjection >;
}
