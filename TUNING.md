# Styling constant tuning

Tuned against the fixed five-scene suite in `scripts/gallery.py`, judged on
rendered output rather than on metrics alone. Four iterations; the limit was 12.

**Outcome: one global set of constants satisfies the rubric.** Per-terrain-class
constants were not needed. One rubric item — `spike_count 0` — turned out to be
unreachable as literally defined, for reasons that are a property of the metric
rather than of the pipeline; evidence is below.

## Final constants

| Constant | Was | Now | Why |
| --- | --- | --- | --- |
| `TARGET_RELIEF_RATIO` | 0.18 | **0.18** | Unchanged. It was already the right target; the clamp below it was what bound. |
| `AUTO_EXAGGERATION_MIN` | 1.0 | **1.0** | Unchanged. Never de-emphasise real terrain. |
| `AUTO_EXAGGERATION_MAX` | 5.0 | **12.5** | 5.0 pinned three of five scenes below the band. |
| `SMOOTH_GROUND_METERS` | 20.0 | **40.0** | 20 m left street-grid corduroy on SF and stoneybrooke. |
| `SMOOTH_SIGMA_MIN` | 0.5 | **0.5** | Unchanged. |
| `SMOOTH_SIGMA_MAX` | 4.0 | **6.0** | 4.0 clamped stoneybrooke, whose auto sigma wants 5.4. |
| `DESPIKE_THRESHOLD` | 3.0 | **3.0** | Unchanged — sweeping it changed nothing measurable (see below). |
| `MAX_SPIKE_CLUSTER_PX` | 4 | **4** | Unchanged. |

## Final scene table

| scene | exag | sigma | relief | printed | % of width | roughness | flat% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tahoe | ×3.53 | 0.67 px (40 m) | 1765 m | 36.0 mm | **18.0%** | 38.0 | 24.2 |
| stoneybrooke | ×10.37 | 6.0 px (22 m) | 70 m | 36.0 mm | **18.0%** | 1.7 | 0.0 |
| rainier | ×1.44 | 0.77 px (40 m) | 3813 m | 36.0 mm | **18.0%** | 27.1 | 0.0 |
| sf_coast | ×12.5 | 1.32 px (40 m) | 317 m | 36.0 mm | **18.0%** | 26.6 | 58.5 |
| kansas | ×12.5 | 1.35 px (40 m) | 94 m | 9.05 mm | **4.52%** | 7.9 | 0.0 |

## Iterations

### 1 — Baseline

Committed constants. Three scenes outside the band, all pinned at
`AUTO_EXAGGERATION_MAX = 5.0`: stoneybrooke 8.76%, sf_coast 7.33%, kansas 1.91%.

Visual pass was better than the numbers suggested — stoneybrooke already read as
smooth landforms with creek valleys rather than gravel, and kansas showed
dendritic drainage and river meanders rather than a slab. The failure was
purely that the prints would be too shallow.

### 2 — Raise the exaggeration and sigma ceilings

`AUTO_EXAGGERATION_MAX` 5.0 → 12.5, `SMOOTH_SIGMA_MAX` 4.0 → 6.0.

Four scenes snapped to exactly 18.0%, because `TARGET_RELIEF_RATIO` was already
correct and only the clamp had been in the way. Kansas moved to 4.77%.

The cost showed up visually: at ×12.5, SF's street grid appeared as clear
corduroy across the Sunset district — the "gravel, not landforms" failure.

### 3 — Raise the smoothing radius

`SMOOTH_GROUND_METERS` 20.0 → 40.0.

Before changing anything, swept the ground radius across terrain classes to test
the assumed alpine-versus-urban conflict:

| scene | m/px | | 20 m | 30 m | 40 m | 60 m | 80 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rainier | 52.3 | relief kept | 100.0% | 99.8% | 99.8% | 99.7% | 99.6% |
| sf_coast | 30.2 | roughness | 93.8% | 88.6% | 83.7% | 75.3% | 68.6% |

**The conflict is far weaker than expected.** Rainier keeps 99.6% of its relief
even at 80 m of blur, because at 52 m/px the volcanic cone spans many pixels —
smoothing cannot blunt a landform much larger than its kernel. That freed the
radius to be set by the urban scenes.

40 m over 80 m was chosen on the renders: at 80 m SF's ridge definition starts
softening while the street grid is no better. Rainier's summit stays a sharp
point with radial ridges intact at 40 m.

### 4 — Water tint fix, and a bug it exposed

Rewrote `preview.py`'s water detection from "at or below the array minimum" to
"large connected region with near-zero gradient", so inland lakes are found at
any elevation. Tahoe's surface is now correctly detected at **1897.6 m** over a
1528 m valley floor, covering 24.3% of the frame.

The first version rendered a **regular lattice of green dots across the lake**.
Diagnosis: the holes deviated from lake level by at most **0.006 m**, with
gradients of 4.9e-4 to 3.0e-3 straddling the 1e-3 threshold. Terrarium encodes
elevation in steps of 1/256 m = 0.0039 m, so a one-quantum wobble in the lake
surface yields a gradient of ~0.002 — above the epsilon. **The threshold had
been set below the data's own quantisation step.** Raised to 0.01 m/px and
scaled by the exaggeration in use, since stretching terrain stretches its
gradients. Interior holes went 316 → 0, with no false water on kansas or
rainier.

## Rubric

| Item | Verdict |
| --- | --- |
| Sculptural, not noisy | **Pass.** Stoneybrooke reads as smooth landforms with creek valleys; SF's street grid is gone at 40 m. |
| Landforms survive | **Pass.** Tahoe's ridgelines stay sharp and Emerald Bay is identifiable; Rainier's summit is a point, not a mound (99.8% of relief kept). |
| Flat scenes read as terrain | **Pass.** Kansas shows dendritic drainage, river meanders and bluffs. Not a slab. |
| Water flat and clean | **Pass.** Tahoe dead flat at 1897 m, SF ocean flat at 0, coastlines crisp. |
| No spikes anywhere | **Unreachable as defined.** See below. |
| Printed relief 12–25% | **Pass.** Four scenes at exactly 18.0%. Kansas at 4.52% (exempt; reported). |
| Water renders blue incl. Tahoe | **Pass.** Fixed this iteration. |

### Where Kansas lands

**4.52% of width — 9.05 mm of relief on a 200 mm print.** Reaching 12% needs
×33 and 18% needs ×50, against the ×12.5 ceiling.

Notably, Kansas at ×33 *looks fine* — the drainage network becomes more
sculptural, not noisier. So the ceiling is not protecting Kansas from ugliness;
it is a deliberate cap on how much a print may lie about the land. Raising
`AUTO_EXAGGERATION_MAX` to 50 would put all five scenes in band. That is a
product decision about honesty, not a quality problem, so it is left as-is and
reported rather than forced.

### Why `spike_count 0` is unreachable

The metric flags pixels deviating from a 5×5 local median by more than
3× the **interquartile range of those deviations** — a *relative* threshold that
rescales itself to whatever variation is present.

Two experiments:

**Despiking harder does not help.** Sweeping `DESPIKE_THRESHOLD` on rainier:

| threshold | 3.0 | 2.0 | 1.0 | 0.5 |
| --- | --- | --- | --- | --- |
| isolated spikes | 2850 | 2238 | 3135 | 3101 |

Non-monotonic, and never approaching zero. `despike` runs *before* `smooth`, and
smoothing regenerates residual structure that the metric then re-measures
against a correspondingly smaller IQR.

**Smoothing harder does not help either.**

| sigma | 0 | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- | --- |
| rainier | 2387 | 2629 | 1239 | 1185 | 1035 | 1061 |
| kansas | 3539 | 3188 | 2177 | 1580 | 2028 | 1799 |
| tahoe | 2047 | 1234 | 1772 | 1571 | 1481 | 1501 |

Counts plateau around 1000–2000 at σ=16, a blur that would destroy every
landform in the suite.

The reason is that **terrain is approximately self-similar**. Blur at scale σ
and structure at scale ~σ remains, now measured against a proportionally
smaller IQR. A relative outlier test on a fractal surface returns a roughly
scale-invariant outlier fraction — here about 0.2–0.4% of pixels, versus the
~5e-5 a Gaussian would give. The excess *is* the terrain's heavy tails, which is
to say it is real fine-scale landform, not pipeline spikes.

The metric is not broken: an analytic Gaussian dome scores 4 isolated pixels out
of 40,000 (0.01%), and that floor is geometric rather than numerical — float64
returns the same count. Adding 1 m of noise takes it to 1079.

**What this means practically:** the rubric's *intent* — no visible acne or
stipple — is met, and was verified on the renders. The literal criterion cannot
be reached by any constant setting, so `gallery.py` reports both a `spike_count`
(literal) and an `isolated_spike_count`, and the useful signal is the trend
across runs, not the absolute value. A jump in `isolated_spike_count` between
gallery runs is a genuine regression signal; a nonzero value is not.

## Follow-up: what the 2D validation missed

All of the tuning above was judged on hillshaded heightmaps. Rendering the
**exported STLs** afterwards (`scripts/render3d.py`, see
`gallery/contact_sheet_3d.png`) surfaced a defect that 2D review could not:

> **CORRECTION (superseded by the OSM features work).** The conclusion below —
> that sf_coast's spires are downtown high-rises — is **wrong**. Rasterising
> OSM building footprints against the same grid shows the tall masses have
> **0–1% footprint coverage**: they are not buildings. The 325.7 m maximum sits
> at the northern grid edge, (37.85, −122.499), which is the **Marin Headlands**
> across the Golden Gate, where terrain genuinely reaches ~300 m. A
> before/after 3D render with building removal enabled leaves that ridge
> untouched.
>
> What misled me: I compared the maximum against Mt Davidson (282 m), SF's high
> point, without noticing the bbox extends north past the Golden Gate into
> higher terrain. Building removal does help sf_coast — the peninsula surface
> is visibly smoother — but it does not and should not touch those ridges.
>
> The original text is kept below because the *process* lesson stands: 2D
> review is blind to vertical structure. The *diagnosis* did not.

**sf_coast prints downtown's high-rises as spires.** Seen from directly above
with a synthetic light, a 300 m tower is a small bright dot, indistinguishable
from a crag. Seen in 3D at ×12.5 exaggeration it is an obvious fragile needle.

Measured on the finished heightmap at 30.2 m/px:

| | |
| --- | --- |
| raw max elevation | 325.7 m |
| Mt Davidson (SF's true high point) | 282 m |
| blobs >40 m above a 21×21 median | 1821 px in 52 blobs |
| largest blob footprints | 76–136 px |
| their excess over surroundings | 67–83 m |

Two consequences:

1. **`despike` is not at fault.** Its cluster limit is 4 px; these are 76–136 px
   masses. They are not isolated needles, and raising the limit to cover them
   would eat real landforms of the same size.
2. **`auto_exaggeration` is being sized off a rooftop.** It reads sf_coast's
   relief as 317 m when the terrain's real relief is ~282 m, so the whole scene
   is scaled by a building.

Neither is reachable by the constants tuned here: removing a 120 px mass needs a
kernel around 100 m+, which dissolves SF's actual hills. This is a distinct
pipeline capability — urban massing removal — not a tuning problem.

**Process lesson:** hillshade review is necessary but not sufficient. The suite
should be judged on both sheets, since the top-down view is close to blind to
exactly the vertical structures that matter most for printability.

## Using this going forward

Any pipeline change should be validated by rerunning `python scripts/gallery.py`
and comparing `gallery/contact_sheet.png` against the committed baseline, plus
the per-scene JSON for drift in printed relief, roughness and spike counts.
