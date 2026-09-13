cmake_minimum_required( VERSION 3.20 )

project( EsotericaFilterFitter LANGUAGES CXX )

#-------------------------------------------------------------------------
#  Esoterica FilterFitter
#-------------------------------------------------------------------------
#  Offline prefiltered-radiance table generator. Deliberately standalone: it
#  includes nothing from the engine, links nothing from it, and uses only the
#  standard library. That is what lets a fork refit the tables for its own BRDF
#  without a working engine build, and it is why this lives outside the solution
#  alongside the other standalone tools.
#
#  The settings below match the engine's property sheets wherever they change an
#  answer, so a table fitted here means the same thing as one fitted there: C++20,
#  warnings as errors, precise floating point, exception handling off.
#
#  Building is two commands, from any prompt - the Visual Studio generator finds the
#  toolchain itself, so no developer prompt and no vcvars are needed:
#
#      cmake -S External/FilterFitter -B External/FilterFitter/Build -G "Visual Studio 18 2026" -A x64
#      cmake --build External/FilterFitter/Build --config Release
#-------------------------------------------------------------------------

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# The vendored reference tables under Source/Reference are #included by
# CoefficientTable.cpp rather than compiled, so they are not listed here.
file( GLOB FF_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cpp" )
file( GLOB FF_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Source/*.h" )

# TinyEXR and stb are vendored under Source/TinyEXR and Source/stb, which the glob
# above deliberately does not reach: they are third-party, they need their own
# warning settings, and a glob that swept them in would also sweep in the next
# thing dropped there. They share one implementation unit because TinyEXR
# decompresses EXR ZIP blocks through stb's zlib rather than a bundled miniz,
# which is also what removes the external zlib dependency entirely.
set( FF_VENDORED_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/Source/ThirdParty/VendoredImplementations.cpp" )

add_executable( EsotericaFilterFitter ${FF_SOURCES} ${FF_HEADERS} ${FF_VENDORED_SOURCE} )

# Source is the include root, so the quoted includes stay as they are: "Assert.h"
# and "Reference/coeffs_const_8.txt" both resolve relative to it.
target_include_directories( EsotericaFilterFitter PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Source" )

target_compile_definitions( EsotericaFilterFitter PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    # Matches the engine. The STL substitutes abort for throw, which is what the
    # engine's build does, so a failure here behaves the way it would there.
    _HAS_EXCEPTIONS=0
)

if( MSVC )
    # CMake adds /EHsc by default and the engine builds with exception handling off,
    # so strip it rather than leave the two disagreeing about the model.
    string( REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )

    target_compile_options( EsotericaFilterFitter PRIVATE
        /W4
        /WX
        /fp:precise
    )

    # TinyEXR and stb are not ours to fix, and /W4 /WX is a statement about our
    # code. Both /w and /WX- are set because the order of target and source
    # options is the generator's business; either one alone is enough, and
    # neither relaxes anything for the files above.
    set_source_files_properties( ${FF_VENDORED_SOURCE} PROPERTIES COMPILE_OPTIONS "/w;/WX-" )
endif()

# The executable lands in Build/<Config> rather than in a generator-specific
# subdirectory, so the path to run is the same whichever generator is used. The
# generator expression is explicit because a multi-config generator would otherwise
# append the configuration itself and append it somewhere else.
set_target_properties( EsotericaFilterFitter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>"
)

message( STATUS "FilterFitter: output in ${CMAKE_BINARY_DIR}/<Config>" )
