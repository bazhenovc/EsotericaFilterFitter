# FilterFitter

FilterFitter generates the small lookup tables that blur reflection probes.

It implements the method from Manson and Sloan, "Fast Filtering of Reflection Probes", EGSR 2016, for the Esoterica Engine. The paper published four tables, and none of them apply here: this engine's roughness curve differs from the paper's, and a table only works for the curve it was fitted for. So the tool fits its own, for any lobe shape, any roughness curve and either base map.

If you change the BRDF, refit your tables with this tool. It runs offline, on the CPU, once per profile and roughness curve.

Sections 1 and 2 set out the problem and the approximation, 3 and 4 the tool, and 5 the measurements. Sections 6 to 10 are the limits, the deviations, the usage, the options and the disclosure.

## 1. The problem

A shiny surface reflects its surroundings. The BRDF says how much light the surface sends back, and in which directions. A rough surface reflects a blurred version: the rougher the surface, the wider the blur.

A reflection probe is a picture of the surroundings, taken at one point in a scene, that is used to light shiny surfaces. Blurring that picture by the right amount is what makes a rough surface look rough rather than polished.

The blur has a shape, called the specular lobe. A mirror sends light out in one direction; a rough surface spreads it over a range of directions, because every point on the surface is tilted a little differently. The lobe says how much light goes in each direction within that range: narrow for a smooth surface, wide for a rough one. "Specular" is the mirror-like part of reflection, as opposed to the diffuse part, which spreads light in all directions.

For each direction `n` and each roughness, the renderer needs the average of the environment weighted by that lobe:

```
              integral over the hemisphere of  L_env(l) * D(h) * (n.l) dl
    L(n)  =   -----------------------------------------------------------
                       integral over the hemisphere of  D(h) * (n.l) dl

    h = normalize( n + l )
```

- `L_env(l)` is the environment map: the surroundings, stored as an image giving the light arriving from each direction `l`.
- `D` is the distribution of microfacet normals: how the tiny bumps on the surface are tilted. That is what decides the shape of the lobe. The engine uses GGX.
- `h` is the half vector, halfway between the view direction and the light direction.
- The denominator normalises, so a constant environment stays constant.

`n` is the surface normal and also the direction the result is for. In this method the two are the same.

The environment is a reflection probe: a base map holding the surroundings, plus a mip chain of successively blurred copies of it.

Two base maps are supported: a **cubemap** (six square faces in six textures) and a **single-slice tetrahedral map** (four triangles packed into one square texture, after Liao et al).

For a real environment map this integral has no closed form, and computing it by brute force takes thousands of samples per pixel. That is affordable offline and not affordable per frame.

## 2. What is approximated

The paper replaces the integral with a small, fixed number of texture reads. Each read is called a tap, and each tap needs three things: a direction, a mip level in the environment's chain, and a weight.

```
              sum over taps of  w_i * L_env( level_i, dir_i )
    L(n)  =   ------------------------------------------------
                    sum over taps of  w_i
```

The taps are not fixed. They depend on the direction `n`, and the shader computes them at run time from a polynomial:

```
    dir0    = c0 + c1 * theta^2 + c2 * phi^2
    dir1    = c0 + c1 * theta^2 + c2 * phi^2
    dir2    = c0 + c1 * theta^2 + c2 * phi^2
    level   = c0 + c1 * theta^2 + c2 * phi^2
    weight  = c0 + c1 * theta^2 + c2 * phi^2
```

`theta` and `phi` are two coordinates derived from `n`, and `dir0`, `dir1`, `dir2` are the components of the tap's direction in the local frame of the output texel. All five quantities are quadratics in `theta` and `phi` with no linear term, which is what keeps the taps continuous where two axial frames meet.

**The polynomial coefficients are what the table stores.** Nothing else about the filter is stored.

One table has seven rows, one per level of the output mip chain, from 128 texels down to 2. A level is six 128x128 faces for a cubemap, and a single 128x128 square holding four triangles for a tetrahedral map. Each row says how to blur the environment at that level.

The output chain stops at 2x2 rather than continuing to 1x1, because a 1x1 face holds a single direction and cannot describe a hemisphere. The chain the taps read still runs to 1x1: a table row reads its own level and coarser ones, and the coarsest row's taps reach well past their own level. That chain is therefore eight levels, 128 down to 1, one longer than the table. The eighth level is what keeps the coarsest row's taps from clamping early; the error at level 6 is 65% lower with it than without it.

Four table shapes come from the paper:

| shape      | taps per axis | total taps | polynomial terms         | coefficients per level |
|------------|---------------|------------|--------------------------|------------------------|
| `const_8`  | 8             | 24         | constant only            | 120                    |
| `const_16` | 16            | 48         | constant only            | 240                    |
| `const_32` | 32            | 96         | constant only            | 480                    |
| `quad_32`  | 32            | 96         | constant, theta^2, phi^2 | 1440                   |

A coefficient count is taps x 5 parameters x polynomial terms, and these are the numbers the fitter solves for. More taps and more terms mean more accuracy and more run-time cost.

### The two base maps, and why the table has an axis count

Each tap is placed in an **axial frame**: a tangent basis built around one axis of the base map. A direction selects the frames it needs, and the taps are divided between them.

How many frames exist is a property of the base map:

| map                      | frames                        | frames active at one direction | total taps, `const_8` |
|--------------------------|-------------------------------|--------------------------------|-----------------------|
| cubemap                  | 3, one per axis               | 2, and a 3rd near a corner     | 24                    |
| single-slice tetrahedral | 4, one per tetrahedron vertex | 2 to 4                         | 32                    |

A frame is dropped when its blend weight reaches zero. That weight is zero while the direction is close to the frame's axis and grows as the direction moves away from it. Both maps use 0.75 as the threshold, and each measures distance in its own coordinates: a cubemap uses the largest of the two off-axis components of the direction, each divided by the on-axis one, and a tetrahedral map uses `|d . v|`, the alignment with the frame's vertex.

On a cubemap this leaves two frames over most of a face and three near a corner.
On a tetrahedral map it leaves two to four.

The published tables are cubemap tables, so they have three axes. A tetrahedral table with the same tap count and the same polynomial order has the same structure with four. The axis count is therefore part of the table: it is stored in the binary and in the text header, and a checkpoint written with a different axis count is refused rather than reused.

The two maps also correct the sampled mip level differently, because a texel covers a different amount of the sphere in each. That amount is the texel's solid angle, the area of the sphere that the texel covers:

| map         | level added at a sampled direction                                      | range                   |
|-------------|-------------------------------------------------------------------------|-------------------------|
| cubemap     | `0.75 * log2( dot( d, d ) )`, with `d` divided by its largest component | 0 to `0.75 * log2( 3 )` |
| tetrahedral | `-1.5 * log2( n . L )`, with `L` the triangle the sample lands in       | 0 to `1.5 * log2( 3 )`  |

The tetrahedral span is exactly twice the cubemap's, which is the factor of two in its Jacobian: the ratio between area on the map and area on the sphere.
Nothing else about the construction changes between the maps.

**The exact integral is still computed, offline.** It is called the reference. It uses thousands of samples per texel on the CPU. The reference is the target: the tool fits the taps to it.

## 3. Why the tool exists, and what a profile is

### The goal

The engine cannot use the paper's tables, and the reason is the roughness curve. A table has seven rows, one per mip level, and each row was fitted for the roughness that level represents. The paper's curve is geometric; this engine's is linear:

| level              | 0      | 1      | 2      | 3      | 4      | 5      | 6     |
|--------------------|--------|--------|--------|--------|--------|--------|-------|
| paper's roughness  | 0.0526 | 0.0884 | 0.1487 | 0.2498 | 0.4190 | 0.6866 | 1.000 |
| engine's roughness | 0.000  | 0.167  | 0.333  | 0.500  | 0.667  | 0.833  | 1.000 |

Level 3 of the paper's table describes how to blur at roughness 0.2498. Level 3 of an engine table has to describe how to blur at 0.5000. Dropping the paper's table into the engine would blur every level by the wrong amount, and the rendered image would be wrong.

You have two options: change the engine's curve to match the paper's, or fit new tables for the curve the engine already has. This project does the second, which means implementing the paper's fitting method.

### What a profile is

A profile is the pair of things that define the blur at a given level:

1. **The lobe shape.** This is the NDF from section 1, the distribution of microfacet normals. This tool implements GGX and Beckmann.
2. **The width of that lobe at each level.** This is the roughness curve.

They are separate objects because you change them independently.

Everything else is shared and does not depend on the profile: how the environment is stored, how directions and solid angles are computed, how the error is measured, and the optimizer. Two profiles therefore use the same measure, and their errors are comparable.

## 4. What was achieved

The paper published four tables, fitted for its own GGX lobe and its own roughness curve. This tool fits any of the four shapes, for GGX or Beckmann, and for any roughness curve. What can be changed without touching the engine:

| piece           | options                                                          | what it is                                   |
|-----------------|------------------------------------------------------------------|----------------------------------------------|
| profile         | GGX or Beckmann, with any curve                                  | lobe shape plus width per level              |
| roughness curve | paper's gloss curve, linear-in-roughness, explicit seven numbers | the lobe width at each level                 |
| table shape     | `const_8`, `const_16`, `const_32`, `quad_32`                     | how many taps, and how many polynomial terms |
| base map        | cubemap, or a single-slice tetrahedral map                       | what the probe's mip chain is stored as      |

The base map is as much an input as the profile: a table does not know how the environment is stored, only where its taps point and which mip each one reads. A cubemap's table has three axial frames and a tetrahedral map's has four, so the axis count is part of the table, and the artifacts and checkpoints are named for the map they belong to.

Fork this project and implement your own profile if you need one. The engine does not need to know which profile produced a table, because a table holds only coefficients.

The fit is scored on the lobe shape, which is an intermediate quantity: a lobe that is 5% wrong can still produce an image that is 20% wrong. The tool therefore also convolves real environment maps both ways and compares the images, and the numbers in section 5 are those comparisons.

Measured against a brute-force reference: the table this tool fits for the paper's curve is 2% better than the published table under that curve, and the table it fits for this project's curve is 27% better across all seven levels.

## 5. Results

Every number here comes from a run of this tool. Section 5.1 checks the evaluator against the paper's own published scores; the rest are radiance comparisons against the brute-force reference of section 2, on 10 environment maps unless a subsection says otherwise.

### 5.1 Does the evaluator reproduce the paper's tables exactly

The paper's own four tables, run through this tool's evaluator:

| table      | this tool | paper published | ratio |
|------------|-----------|-----------------|-------|
| `const_8`  | 0.1372    | 0.1111          | 1.24  |
| `const_16` | 0.1192    | 0.0815          | 1.46  |
| `const_32` | 0.0993    | 0.0613          | 1.62  |
| `quad_32`  | 0.0642    | 0.0506          | 1.27  |

Lower is better. These are the paper's coefficients scored by this tool's error measure. The tool measures 24% to 62% more error than the paper reports, on the paper's own tables.
The ordering is right: fewer taps is worse, and `quad_32` is best.

Two readings are possible and this project has not separated them: this tool's error measure differs from the paper's, or the pipeline has a residual error.
Either way this tool does not claim to have reproduced the paper's tables exactly.
It claims to produce tables of comparable quality under a measure it applies consistently to everything.

### 5.2 Error in radiance

Measured on 10 reference HDRI maps.

The reference and the approximation are compared per level. Relative L1 means total absolute difference divided by the reference's own total, so 0.01 is a 1% average error.

```
table                     lvl 0    lvl 1    lvl 2    lvl 3    lvl 4    lvl 5    lvl 6    mean 1-6
const_8                   0.1422   0.0442   0.0525   0.0438   0.0483   0.0833   0.0575   0.0549
const_16                  0.1416   0.0328   0.0408   0.0332   0.0367   0.0607   0.0676   0.0453
const_32                  0.1411   0.0336   0.0361   0.0242   0.0296   0.0558   0.0540   0.0389
quad_32                   0.1382   0.0239   0.0210   0.0163   0.0232   0.0275   0.0457   0.0263
fitted_esoterica  cube    0.0000   0.0626   0.0523   0.0516   0.0545   0.0637   0.0598   0.0574
fitted_esoterica  tetra   0.0002   0.1041   0.1194   0.1520   0.1736   0.1582   0.1485   0.1426
fitted_paper(18)  cube    0.1496   0.0542   0.0504   0.0546   0.0448   0.0593   0.0598   0.0538
fitted_paper(18)  tetra   0.1218   0.1046   0.1223   0.1101   0.1635   0.1581   0.1485   0.1345
```

Each column is one level of the output chain. The first four rows are the paper's tables, scored under the paper's roughness curve. The last four are tables this tool fitted, one per base map, each scored under the curve it was fitted for.

- More taps give consistently lower error: `const_8` 0.055, `const_16` 0.045, `const_32` 0.039, `quad_32` 0.026.
- Level 0 is the special case. The paper's curve makes it a narrow lobe, which is the hardest level to approximate, so it dominates the published tables' error at 0.14. This project's curve makes level 0 a mirror: exact for the cubemap at 0.0000, and 0.0002 for the tetrahedral map, where a tap still covers a footprint even at zero width.
- Every fitted table has the `const_8` shape, so it is less accurate than `const_32` or `quad_32` by construction. Its gain is the curve, not the fitting.
- **The tetrahedral table is 2.5x worse than the cubemap one, and that is the map, not the table.** A tetrahedral level at resolution R holds R x R texels over the whole sphere where a cubemap level holds six of them, so at the same nominal resolution its texels are six times the solid angle and its narrow-lobe levels are much harder. The energy is right - the mean of the table's convolution over the reference is 1.00 to 1.14 at every level - so the 2.5x is a placement error, not an energy error. A runtime that wants tetrahedral probes as accurate as cubemap ones at the same lobe width has to use a higher base resolution.
- **A fitted table is reproducible only on the same machine and the same worker count.** The objective is summed in item order and does not depend on the worker count, but the gradient is accumulated per worker and reduced in worker order, and the optimizer is sensitive to that in the last bits. If your fork needs bit-identical tables, fix the worker count as well as the settings.

### 5.3 Their table against ours

Same shape (`const_8`), same roughness curve, same reference. The only difference is who fitted the coefficients.

| configuration                      | mean levels 1-6 | mean all 7 levels |
|------------------------------------|-----------------|-------------------|
| paper's table, paper's curve       | 0.0549          | 0.0674            |
| fitted table, paper's curve        | 0.0538          | 0.0675            |
| fitted table, this project's curve | 0.0574          | **0.0492**        |

- Row 1 against row 2: under identical conditions this tool's fit is 2% better on levels 1-6 and equal on all seven. That is smaller than the spread between environments in a 10-map sample. Fitting does not beat the paper.
- Row 2 against row 3: the roughness curve is worth 27% on all seven levels, and it comes from level 0 being exact. The curve matters more than the fitter.
- Levels 1-6 alone make this project's curve look worse (0.0574 against 0.0538). All seven levels make it look better. The paper's convention of averaging levels 1-6 leaves out the largest error term in the table.

### 5.4 Fireflies

A firefly is one texel whose error is much larger than its neighbours'. It does not move an average over 98304 texels, so average error cannot detect it. Two other numbers can:

- **max abs**: the largest single-texel error, in linear radiance, meaning the raw light values with no tone mapping applied.
- **peak ratio**: the brightest texel of the approximation divided by the brightest texel of the reference. Above 1 means the approximation's brightest texel is brighter than the reference's. Far below 1 means the reference has a spike the approximation smoothed away.

Both are printed per environment and per table, and the worst cases are listed at the end of a run.

**Where fireflies come from.** Measured over 425 environments, a firefly is a level 0 problem and it belongs to the paper's roughness curve, not to the paper's fit.

| level  | `const_8`, median peak ratio | assets above 1.2 |
|--------|------------------------------|------------------|
| 0      | 1.456, worst 1.752           | 298 of 425       |
| 1      | 1.078                        | 2                |
| 2 to 6 | 0.98 to 1.08                 | 0 to 24          |

Level 0 of the paper's curve is a near-delta lobe, with a GGX alpha of 0.00276, the square of its roughness 0.0526. The taps cannot represent a lobe that narrow, so on a bright small source they overshoot the brightest texel by about 45%.

Under the engine's linear curve, level 0 is a mirror and is exact, so the same failure cannot happen: the worst peak ratio over all 425 environments is 1.163 and no environment exceeds 1.2.

**Fitting does not fix this.** Under the paper's curve, the table this tool fits is worse at level 0 than the published one: median peak ratio 1.555 against 1.456, worst 2.321 against 1.752, 330 environments above 1.2 against 298, and a level-0 mean error of 0.1761 against 0.1225.
If the engine ever adopted the paper's curve, it would inherit a worse firefly than the paper's own table has.

Practical consequence: fireflies are avoided by the choice of roughness curve, not by fitting better.

### 5.5 Energy

A correctly normalised filter preserves the average brightness exactly. The tool prints the ratio of the approximation's mean to the reference's mean for each level, and it should read 1.0000 everywhere.

Measured: 0.983 to 1.039. The largest deviation was `const_32` at level 5, gaining 3.9% brightness, on the same level where its error is highest, so part of that level's error is an energy error rather than a shape error.

### 5.6 Consistency across environments

Across the reference environment maps the error is **not** consistent in absolute terms.
It varies by a factor of 15 from the easiest environment to the hardest, and the standard deviation is about 0.9 times the mean for every table.

What is consistent:

- The ordering. The correlation between per-environment errors of any two tables is 0.995 to 1.000, where 1.0 means identical. An environment that is hard for one table is hard for all of them.
- The advantage. `quad_32` is 2.34x better than `const_8`, with a spread of +/-13%.

Darker environments are harder. The correlation between error and average brightness is -0.55.
The hardest environments are the dark ones with a small, bright source.

## 6. What this tool does not claim

- **It does not reproduce the paper's published tables exactly.** See 5.1.
- **It does not beat the paper's fitter.** Measured, and it is level. See 5.3.
- **Comparisons against the published tables are biased in this tool's favour.** The fitted table was fitted against this tool's error measure, and the published table is scored by that same measure. If the 1.24x to 1.62x gap in 5.1 is systematic, the published tables are measured worse than they are, and the fitted table is measured by the measure it was fitted against.
- **The reference convolution is GGX only.** A Beckmann reference would need Beckmann importance sampling, which is not implemented.
- **The validated numbers assume the paper's 4-tap pass-1 downsample.** The engine currently uses a single bilinear tap for pass 1. The numbers here therefore do not exactly predict what the engine will produce.

## 7. Deviations from the paper

Every difference between this implementation and the paper, with the reason for it.

### Required by the engine

| deviation                                                           | reason                                                                                                                                                                                                                                |
|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Linear roughness curve instead of the paper's geometric gloss curve | Esoterica curve is linear in roughness, and a table is only valid for the curve it was fitted for. This is the reason the tool exists.                                                                                                |
| Level 0 is a mirror, not a narrow lobe                              | A linear curve starts at roughness 0. A zero-width lobe is the identity filter, which a continuous NDF cannot express, so the tool handles it as a special case in both the fit and the reference. The paper has no zero-width level. |
| Seven output levels, 128 down to 2                                  | Esoterica `PROBE_REFLECTION_MIPS`, which is also how many mips the engine's radiance target is created with. A table row per output level is what the method needs, and a 1x1 output level would hold a single direction.             |

### Where the paper's text and its reference code disagree

| deviation                                                  | reason                                                                                                                                                                                                                                                                     |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Pass-1 Jacobian weighting uses `J`, not `1/J`              | The paper's prose says `J`; the paper's reference code computes `1/J`. They are reciprocals. `J` scores 1.7x better on the published tables, so the prose is what produced the tables.                                                                                     |
| Cross-face taps are kept                                   | A coarse texel's taps stay inside its face, but their bilinear footprints do not. The reference code drops the ones that leave, which means a constant environment no longer blurs to a constant along every face edge. They are resolved through direction space instead. |
| The paper's table index layout is followed, not the code's | The two disagree on tap numbering.                                                                                                                                                                                                                                         |

### Chosen because they measured better

| deviation                                             | reason                                                                                                                                                                                                     |
|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Lobe convention is `D(h) * (n.l)` over the hemisphere | The paper's equation read literally is `D` over the whole sphere. The cosine form matches the engine's shader and scores 3.2x better.                                                                      |
| Texel solid angles are exact                          | The reference shader approximates with `jacobian * texel area`. The exact value and `4J/R^2` agree to four decimal places, and the exact one is used everywhere so the measure cannot vary between stages. |
| The reference preimage is point-sampled               | Confirmed by the level-0 width optimum.                                                                                                                                                                    |

### Method, where the paper is not specific

| deviation                                                                       | reason                                                                                                                                        |
|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| The table is fitted from an analytic seed, not from the published table         | The published table is only available for the paper's own curve and lobe. A generator needs a starting point that works for any profile.      |
| L-BFGS with a convex weight-polish stage, and several starting points per level | The paper does not specify its optimizer. The polish stage uses the fact that the weights enter linearly, which makes that subproblem convex. |
| The output format is this tool's own                                            | The paper publishes numbers, not a file format.                                                                                               |

## 8. Using the tool

Build the tool:

```
External\FilterFitter\BuildFilterFitter.bat
```

Fit a table, then write it out:

```
EsotericaFilterFitter.exe --fit --curve esoterica
EsotericaFilterFitter.exe --write-header --write-binary --curve esoterica
```

The fit is resumable: it checkpoints after every level. `--reset` starts it over from level 0. Outputs:

```
ReflectionProbeTable_ggx_esoterica_cube.h
ReflectionProbeTable_ggx_esoterica_cube.bin
FilterFitter_ggx_esoterica.fit          the checkpoint, cubemap
```

The binary is what the engine reads; the header is the same data as text.

For a tetrahedral probe, add `--projection tetrahedron` to every command. A tetrahedral fit takes minutes rather than tens of minutes, because a tetrahedral level is one slice where a cubemap level is six:

```
EsotericaFilterFitter.exe --fit --write-header --write-binary --projection tetrahedron --curve esoterica
```

```
ReflectionProbeTable_ggx_esoterica_tetrahedron.h
ReflectionProbeTable_ggx_esoterica_tetrahedron.bin
FilterFitter_ggx_esoterica_tetrahedron.fit
```

To validate a table against real environments:

```
EsotericaFilterFitter.exe --hdri-ingest   "E:\mtld\.mtld-texture-cache"
EsotericaFilterFitter.exe --hdri-validate "E:\mtld\.mtld-texture-cache" --curve esoterica
```

Ingest decodes each panorama into the selected base map and caches it. The two maps cache separately, under the same environment directory. Only equirectangular panoramas are supported.

Validate convolves every environment twice, once by brute force and once from the table, and reports the difference. Both stages cache their results, so re-running them is fast. A cubemap run validates the four published tables beside the fitted one; a tetrahedral run validates the fitted table and nothing else, because the published tables are cubemap data.

Results are written as EXR, one file per slice per level, under `.hdri/<environment>/<stage>/`.

### Reading a table at run time

The sampling side is in the provided HLSL files: one for a cubemap, one for a tetrahedral map.
Each takes one output direction, builds the taps the table prescribes, and sums the weighted reads. That is the same construction the fitter solves for, so a reader that follows it matches the tables.

What they need from the tool is one table. `--write-binary` writes a 144-byte header, then the coefficients as `float4`s in this order:

```
    float4[ level ][ parameter ][ coefficient ][ index ]
```

`index` is `numSuperTaps * axis + superTap`, and `numSuperTaps` is `numTapsPerAxis / 4`.

The four components of one `float4` are the four sub-taps of one index.

The header describes the layout, so a reader can check it instead of assuming it:

| offset | field                 | meaning                                                          |
|--------|-----------------------|------------------------------------------------------------------|
| 0      | magic                 | `'FTBL'`, so a wrong file is refused rather than interpreted     |
| 4      | version               | 4                                                                |
| 8      | numLevels             | 7                                                                |
| 12     | numParameters         | 5                                                                |
| 16     | numActiveCoefficients | 1 for a constant table, 3 for a quadratic one                    |
| 20     | numTapsPerAxis        | 8, 16 or 32                                                      |
| 24     | projection            | 0 cubemap, 1 tetrahedral                                         |
| 28     | numAxes               | 3 cubemap, 4 tetrahedral                                         |
| 32     | numIndices            | `numAxes * numTapsPerAxis / 4`                                   |
| 36     | numFloat4             | `numLevels * numParameters * numActiveCoefficients * numIndices` |
| 40     | payloadOffset         | 144                                                              |
| 44     | payloadBytes          | `numFloat4 * 16`                                                 |
| 48     | shapeName[16]         | text                                                             |
| 64     | profileName[48]       | text                                                             |
| 112    | widths[7]             | one float per level, for the curve the table was fitted for      |

Per output texel the shader normalises the output direction, keeps every axial frame whose blend weight is positive, and emits four sub-taps per super-tap.
Each sub-tap reads five quadratics in `( theta^2, phi^2 )`, forms a direction in the frame, adds the level correction above, and samples the environment.

The result is the sum of the weighted reads divided by the sum of the weights.

Two things the runtime has to decide that the provided files do not:

- **The tile's edges.** A single-slice tetrahedral map puts four different faces against one square's border, so a hardware bilinear read near the edge mixes two faces that do not meet there. The tool's own convolution resolves every tap through direction space instead, which is exact but costs four reads and a coordinate conversion per tap. Start with the hardware read and decide from your own measurements whether the seams need the resolved one;
- **Roughness zero.** A zero-width level is the identity, and the engine reaches it by sampling mip 0 directly rather than by reading the table. A table fitted with a zero-width level stores a mirror level (`-0.75 * log2( 3 )` for a cubemap, `-1.5 * log2( 3 )` for a tetrahedral map), which holds the sampler at or below mip 0 at every direction, so the table is correct there too, but skipping it costs less.

## 9. Advanced: command line options

### Fitting

| option                             | effect                                                                                                                                                          |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--fit`                            | Fit a table, one level at a time. Resumes from the checkpoint.                                                                                                  |
| `--reset`                          | Discard the checkpoint first, so the fit starts at level 0.                                                                                                     |
| `--fit-seeded`                     | Same fit, but seeded from the published `const_8` table instead of from the analytic seed. Cubemap only; a tetrahedral fit has no published table to seed from. |
| `--profile <ggx\|beckmann>`        | Which NDF to fit.                                                                                                                                               |
| `--curve <paper\|esoterica>`       | Which roughness curve. `paper` is the paper's gloss curve, `esoterica` is linear in roughness.                                                                  |
| `--projection <cube\|tetrahedron>` | Which base map. Default `cube`. It selects the frame, the fit's shape, every artifact and checkpoint name, and the map the HDRI stages are cached under.        |
| `--spec-power <n>`                 | Spec power for the paper curve. 18 reproduces the published tables.                                                                                             |
| `--widths <list>`                  | Explicit widths, one per level, comma separated. Overrides `--curve`.                                                                                           |

### Writing tables

| option                  | effect                                                                                       |
|-------------------------|----------------------------------------------------------------------------------------------|
| `--write-header [path]` | Write the C header. Path optional; omitted, the name is derived from profile, curve and map. |
| `--write-binary [path]` | Write the binary the engine reads.                                                           |

### Validation checks

| option             | effect                                                                            |
|--------------------|-----------------------------------------------------------------------------------|
| `--diagnostics`    | Perturb one modelling input at a time to see which ones matter. About 10 minutes. |
| `--benchmark`      | Measure the cost of one objective evaluation and project fit time.                |
| `--optimizer`      | Perturb a published level and require the optimizer to recover it.                |
| `--seed`           | Score the analytic starting point at every level, without fitting.                |
| `--weight-probe`   | Measure how much the weight coefficients alone can improve.                       |
| `--gradient-check` | Compare the analytic gradient against a finite difference.                        |
| `--irls`           | Compare the split optimizer against the joint one.                                |
| `--converge`       | Run each level to a larger budget and report the gradient.                        |
| `--sample-size`    | Re-score on a denser, held-out grid to measure overfitting.                       |

### HDRI radiance validation

| option                   | effect                                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------------------|
| `--hdri-scan <root>`     | List the environment maps in a dataset and what was rejected.                                                       |
| `--hdri-ingest <root>`   | Decode, downsample, project and cache every environment map.                                                        |
| `--hdri-validate <root>` | Reference and table convolutions, compared per level.                                                               |
| `--hdri-dir <path>`      | Where the projected maps are cached. Default `External/FilterFitter/hdri`.                                          |
| `--hdri-limit <n>`       | Stop after n environments. Stable order, so the same n is the same set.                                             |
| `--hdri-samples <n>`     | Reference samples per texel. Default 8192, which is enough to converge. 1024 reproduces the engine's own estimator. |
| `--hdri-equirect <w>`    | Equirect width before projection. Default 2048.                                                                     |
| `--hdri-force`           | Recompute cached stages instead of reusing them.                                                                    |
| `--hdri-showcase [dir]`  | Write the showcase EXRs as well: one image per interesting environment.                                             |
| `--hdri-csv <path>`      | Dump every per-environment, per-table, per-level number.                                                            |

### Showcase images

The validation writes EXR, one file per slice per level. To see whether a table blurs the way it should, you want the source, the filtered result and the table's result side by side at every level, in one image.

`--hdri-showcase [dir]` writes those images. Every level below 0 is point upsampled to the source resolution. Values are written linear and unclipped.

| file                        | what it is                                                                          |
|-----------------------------|-------------------------------------------------------------------------------------|
| `esoterica_best.exr`        | the environment the fitted esoterica table filters best, by mean L1 over levels 1-6 |
| `fireflies_paper.exr`       | the environment and the paper's table with the largest level-0 peak ratio           |
| `fireflies_esoterica.exr`   | that same environment under the fitted esoterica table, for comparison              |
| `tetrahedral_esoterica.exr` | the first environment with a result, for a tetrahedral run                          |

A cubemap gets one row per level, three results across it and its six faces across each block:

```
    row 0                 the source cubemap, its six faces in the first block; the other two blocks are black
    rows 1..7             level 0 at row 1 down to level 6 at row 7, each row holding the brute-force result, then the table's, then the naive mip chain
```

Comparing results at one level means reading across a row; comparing levels means reading down. The source's row leaves its other blocks black because there is no filtered result to compare it against.

A 128 cubemap gives a 2304x1024 image.

The third block is the **naive** result: the source chain's own level, with no lobe applied to it.

A tetrahedral map is one texture per level, so its levels go across and the source takes one tile:

```
    row 0, column 0       the source
    row 1                 the brute-force chain, one level per column, 0 to 6
    row 2                 the table's chain
    row 3                 the naive mip chain
```

The images go beside `--hdri-csv`, or into `--hdri-dir` when there is no CSV; passing `dir` overrides both.

### Running with no arguments

With no arguments the tool prints every self-check and the published-table conformance numbers. It takes a few seconds, and `OVERALL : PASS` means every check passed.

### Cost / performance, measured on a 32-core machine

| operation                                                                | time                                |
|--------------------------------------------------------------------------|-------------------------------------|
| self-checks                                                              | 3 s                                 |
| fit one table, 7 levels, `const_8`, cubemap                              | 13-16 min                           |
| fit one table, 7 levels, `const_8`, tetrahedral                          | 2 min                               |
| ingest one environment map                                               | 0.5 s                               |
| validate one environment map, 5 tables, 2 reference chains, 8192 samples | 7 s                                 |
| validate one environment map, 1 table, `--hdri-force`                    | 2 s                                 |
| full 425-environment corpus                                              | ingest 4 min, validate about 1 hour |
| cache size per environment map                                           | 12 MB per map                       |

A tetrahedral fit is minutes rather than tens of minutes for the same reason its radiance error is 2.5x: a tetrahedral level is one slice of R x R texels where a cubemap level is six faces of R x R, so the fit does a sixth of the work.

## 10. Disclosure

LLM assistance was used when developing this project.
