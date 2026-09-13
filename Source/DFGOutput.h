#pragma once

#include <cstdint>
#include <vector>

#include "DFGTable.h"

// The preintegrated split-sum DFG table as files
//-------------------------------------------------------------------------
// A binary the engine embeds, and a text header that carries the same numbers.
//
//  BINARY LAYOUT
//-------------------------------------------------------------------------
// A 64-byte header, then the payload. Little endian, matching the only platform this builds on. 
// Every field is a uint32 or a char array, so no compiler has to leave a hole anywhere for alignment and sizeof is the on-disk size.
//
//  offset  size    field
//  ------  ------  --------------------------------------------------------------
//       0       4  magic, written as 0x46444746 and read little endian as F G D F
//       4       4  version
//       8       4  resolution, the texel count per axis, the grid being square
//      12       4  sample count per texel in the table's own evaluation
//      16       4  values per texel, two
//      20       4  value format, DFGValueHalf or DFGValueFloat
//      24       4  payload offset in bytes, 64
//      28       4  payload size in bytes
//      32      24  name, NUL padded
//      56       8  reserved
//
//  THE PAYLOAD
//-------------------------------------------------------------------------
// Row major, then texel, then channel.
// The second axis is the row, so roughness increases with y, and the first is the column, so N dot V increases with x.
//
// One texel is TWO CHANNELS at the file's value format: channel zero is scale, channel one is bias, in that order, which is the layout of the RG16 render target the engine used to render this table into.
// The payload uploads as it stands.
//
// A texel's lookup coordinates are its centre:
//
//      N dot V   = ( column + 0.5 ) / resolution
//      roughness = ( row + 0.5 ) / resolution
//
// so no texel sits on either edge of the grid. The first row is the smoothest.

namespace FilterFitter
{
    static constexpr uint32_t DFGBinaryVersion = 2;

    static constexpr uint32_t DFGBinaryMagic = 0x46444746;  // reads as FGDF in a dump

    // Channels per texel, and the order that number implies.
    static constexpr uint32_t DFGChannelCount = 2;

    static constexpr uint32_t DFGValueHalf = 0;     // two IEEE 754 binary16
    static constexpr uint32_t DFGValueFloat = 1;    // two IEEE 754 binary32

    // What the table is, in the file, because a blob found later has to say what it is.
    // The runtime's conventions are not negotiable, so a name is all this carries.
    inline constexpr char const* DFGTableName = "esoterica";

    struct DFGBinaryHeader
    {
        uint32_t    m_magic;                            //   0
        uint32_t    m_version;                          //   4
        uint32_t    m_resolution;                       //   8
        uint32_t    m_sampleCount;                      //  12
        uint32_t    m_valuesPerTexel;                   //  16
        uint32_t    m_valueFormat;                      //  20
        uint32_t    m_payloadOffset;                    //  24
        uint32_t    m_payloadBytes;                     //  28
        char        m_name[24];                         //  32
        uint32_t    m_reserved[2];                      //  56
    };

    static_assert( sizeof( DFGBinaryHeader ) == 64, "the binary header is a file layout and must not gain padding" );
    static_assert( ( sizeof( DFGBinaryHeader ) % 16 ) == 0, "the payload must stay 16-byte aligned" );

    // Header, then the payload. The bytes are the file.
    void FormatDFGTableBinary( DFGTable const& table, std::vector<uint8_t>& bytes );

    bool WriteDFGTableBinary( DFGTable const& table, char const* pPath );

    // The same numbers as a C header: what the table is and what the binary holds, then the payload texel by texel in the order the binary stores it.
    void FormatDFGTableHeader( DFGTable const& table, std::vector<char>& text );

    bool WriteDFGTableHeader( DFGTable const& table, char const* pPath );

    // Formats a small grid both ways, reads the binary back, and requires every stored value to survive, the header's fields to agree, and the two encodings to hold the same numbers. 
    // Run with the rest of the startup self-checks.
    bool RunDFGWriterSelfCheck();
}
