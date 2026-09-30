#pragma once

//  Chart reconstruction diagnostic
//-------------------------------------------------------------------------
//  A self-contained experiment, with no fit and no corpus behind it: how faithfully can a
//  sphere-to-chart mapping be reconstructed by the reads a runtime actually performs?
//
//  WHY IT EXISTS
//
//  The octahedral reflection-probe map measures about twice the cubemap's error - 0.0910
//  against 0.0452 mean relative L1 over levels 1-6 on the 425-asset corpus - and the excess
//  sits in the strips along the map's joins, in all eight faces including the fold-free
//  diamond ones. The working explanation is chart anisotropy at those creases: a
//  tangent-plane tap offset lands on a bilinear footprint whose shape is least like the disk
//  the tangent plane assumes.
//
//  That explanation predicts the reconstruction error is SCALE-FREE, which is consistent
//  with 128 to 256 buying only 1.4% of corpus error, but it has never been tested directly
//  and no alternative chart has been compared. This mode tests exactly that, in isolation
//  from the fit, the tables and the environments.
//
//  WHAT IT MEASURES
//
//  A smooth analytic field on the sphere, known in closed form, is evaluated at every texel
//  centre of a chart at a given resolution. Test directions are then reconstructed by a
//  bilinear read of that chart and compared against the field's exact value. Three things
//  are reported per chart and resolution:
//
//      mean and max relative error, split into JOIN reads and INTERIOR reads
//      the scaling exponent between successive resolutions, so a scale-free chart is flat
//      and a texel-limited one is near 2 per doubling
//      the chart's own area and anisotropy statistics, so a chart's error can be attributed
//      to how much its texel SIZE varies against how much its texel SHAPE does
//
//  The join split is chart-agnostic and is the read's own property: a read is a join read
//  when the faces of its four footprint corners are not all the same face. That is the
//  crease crossing the read, which is what the octahedral excess was measured to be.
//
//  THE READS MODELLED ARE THE RUNTIME'S, NOT AN IDEALISED ONE
//
//  A chart is read the way the hardware would read it, and the two families differ:
//
//      cube        six faces in six slices, so a read filters within one face and clamps at
//                  its edge - hardware cube filtering never blends across a face
//      square      one slice holding every face, so a read filters across the whole square
//                  and clamps at its outer boundary
//
//  That difference is not incidental. It is the mechanism by which a six-face chart avoids
//  creases that a single-square chart cannot, so the harness reproduces it by walking slices
//  exactly as the projections define them rather than by flattening every chart to one image.
//
//  A second read mode is offered for the square charts: RESOLVED, which takes each of the
//  four footprint corners' directions and gathers the texels those directions land on,
//  instead of filtering in chart coordinates. That is the direction-space read the runtime
//  gather was specified to use, and it separates the chart's contribution from the read's.
//-------------------------------------------------------------------------

namespace FilterFitter
{
    // Runs every chart at every resolution and prints the table. Returns false if any check
    // inside it fails, so it can be wired into the tool's pass/fail summary.
    bool RunChartReconstruction();
}
