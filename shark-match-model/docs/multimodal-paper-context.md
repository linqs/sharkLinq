# Paper context: multimodal late fusion for individual shark re-identification

> **Status:** Research notes; not part of the supported Shark Matcher workflow.

## Purpose

A self-contained briefing for the paper-drafting agent working on the
multimodal chapter of the shark re-identification thesis.

## Thesis Contributions

The thesis has two contributions; this chapter covers the second:

1. **Image-only re-identification.** A backbone fine-tuned with LoRA and
   an ArcFace head on a proprietary white-shark photographic dataset,
   producing L2-normalized embeddings used for retrieval via cosine
   similarity. Covered in the prior chapter.
2. **Multimodal extension** (this chapter). A calibrated late-fusion
   pipeline that combines those frozen image embeddings with a small,
   carefully encoded set of per-photo and per-individual metadata
   fields, producing a fused embedding that re-uses the same retrieval
   stack at inference time and improves on every retrieval metric we
   measure — without retraining the backbone or training any new
   parameters.

The rest of this document covers the design, the experimental journey, the
final method, and the ablations.

---

## 1. Setting

White-shark re-identification at the individual level.

- Test queries are photos of individuals.
- The gallery is the corpus minus the query.
- A retrieval is "correct" if any photo of the same individual appears ahead of any photo of a different individual in the deduplicated-by-individual ranking.

### Metrics

* **Hits@k** — fraction of queries whose first correct individual lands
  in the top *k* of the dedup'd neighbor list (k ∈ {1, 5, 10, 25, 50, 100}).
* **Avg K** — mean rank of the first correct individual in the dedup'd
  list, capped at 100 (lower is better).
* **MRR** — mean reciprocal rank, `mean(1 / k)` over the same dedup'd
  ranking. The harmonic dual of Avg K; bounded in [0, 1]; standard ranking
  metric in retrieval/ReID literature, complementary to Avg K (which is
  tail-sensitive while MRR saturates).

### Test Splits

Two splits exist; they share the same underlying corpus
and differ only in which individuals are held out for test:
* `thesis_unmatch_random_out` — random hold-out (696 test queries after
  `min_instances ≥ 2` filtering). Primary split for this chapter.
* `best_unmatch_new_out_no_psl` — alternative hold-out (847 test queries).
  Secondary; used to confirm results transfer.

All results in this document are on the primary split unless noted.

---

## 2. The opportunity

The image-only baseline is strong (Hits@1 = 0.886, Hits@10 = 0.950), but the
long-rank tail is wide (Avg K ≈ 10).

- Low-ranked correct-match queries show a recurring failure: visually similar individuals from different parts of the dataset cluster together in embedding space.
- The image backbone, however well trained, cannot know that two visually similar fins belong to sharks of different sex, photographed at different sites, on different days, of different lengths.

The dataset has rich per-photo and per-individual metadata collected by expert
annotators (with occasional error). For each sighting:

| Field | Type | Granularity | Notes |
|---|---|---|---|
| `sex` | categorical {F, M, Unknown} | per individual | |
| `location` | categorical (six site codes) | per photo | |
| `type` | categorical {still, video, …} | per photo | |
| `quality` | categorical {good, medium, poor, unmatchable} | per photo | |
| `length` | continuous (feet) | per individual | expert estimate |
| `time` | UNIX timestamp | per photo | day-of-year + year |

This is the multimodal opportunity: each metadata field carries some
identity-discriminating information, and combining several fields pushes
retrieval beyond what the image alone can do.

---

## 3. Why this is non-trivial

Five frictions that any naive approach hits:

1. **Heterogeneous types.** Categorical, continuous, cyclical, sparse —
   all in one feature vector.
2. **Sparse / missing fields.** Most photos have unknown sex; many have
   unknown length; some have no recorded time. The encoding must handle
   missing-ness gracefully and not let it inject noise.
3. **Dimensionality mismatch.** The image embedding is 2152-D; the raw
   metadata vector after one-hot encoding is ~20-D. Naive concatenation
   would let the image dominate by sheer norm even at α = 0.5.
4. **Per-feature contribution varies.** Some fields (location, sex)
   carry useful identity signal; others (still vs video) don't. A
   one-α-fits-all weighting is leaving signal on the table.
5. **Limited compute.** The thesis's compute budget is small. We cannot
   afford backbone retraining or large-scale hyperparameter sweeps. The
   final method must work on a laptop in minutes, not on a cluster in
   days.

These constraints shape the entire design.

---

## 4. The journey: from trained fusion to calibrated late fusion

### Trained-Fusion Attempts

The first attempts were trained-fusion heads on top of frozen image
embeddings. Two are reproducible and appear as ablation rows in the paper:

* **Naive concatenation fusion.** Visual Feature Expert (small MLP
  projecting image embedding) and Textual Metadata Expert (small MLP
  projecting metadata vector) feeding a final projection head, end-to-
  end triplet-trained.
* **Residual fusion.** Final embedding = `L2norm(image + δ(image,
  metadata))` with δ's last linear zero-initialized. At step 0 the model
  is exactly the image-only baseline; training can only deviate when
  triplet loss decreases.

Other variants (gate, cross-attention, temperature scaling,
attention-pooling) were tested earlier in the project; all underperformed
image-only or these two and were dropped from the paper.

### What We Found

The headline table below lists all three methods with every metric.

* Naive concat regresses ~12 H@1 points below image-only. The head over-
  fits, and there is no path back to the image baseline.
* Residual fusion floors at image-only (by construction) and improves
  Avg K (10.12 -> 6.62), but the H@1 / MRR gains are within noise. The
  triplet-loss signal isn't strong enough to extract more.

### The Design Pivot

If a trained head can't reliably beat the image baseline on H@1, the right
move may be to not train a head at all. Instead, spend effort on:

- the *encoding* of the metadata;
- a small set of *fusion weights*, tuned by search over a precomputed similarity matrix.

That move was right: the final method outperforms both trained baselines on
every metric, with no learned parameters beyond a handful of α weights.

---

## 5. The final method: calibrated late fusion

### Summary

Frozen image embeddings and engineered metadata feature vectors are
independently L2-normalized per group. Each group is scaled by a tuned scalar
weight α_group. The weighted vectors are concatenated and re-L2-normalized to
produce a fused embedding.

Retrieval at inference is unchanged from the image-only pipeline (cosine on
the fused embedding).

### Pipeline Diagram

```
┌─────────────────────────────┐    ┌──────────────────────────────────┐
│ Image (frozen embedding)    │    │ Metadata record                  │
│ x_img ∈ ℝ^2152              │    │ (sex, location, length, time, …) │
└─────────────────────────────┘    └──────────────────────────────────┘
              │                                       │
              │                          ┌────────────┴───────────────┐
              │                          │ engineer_features (repC)   │
              │                          │   per-feature encoding,    │
              │                          │   produces e_g ∈ ℝ^{d_g}   │
              │                          │   for each feature group g │
              │                          └────────────┬───────────────┘
              │                                       │
              v                                       v
   x_img / ||x_img||₂                  for each group g:  e_g / ||e_g||₂
              │                                       │
              v                                       v
       α_image · x̂_img         α_g · ê_g     (one α per group)
              │                                       │
              └──────── concat ───────────────────────┘
                              │
                              v
                  fused = L2norm(concat[…])
                              │
                              v
                   retrieval (cosine top-K)
```

### Mathematical Form

Let G be the set of metadata feature groups
(sex, location, type, quality, length, time_day, time_year), let `ê_g`
be the L2-normalized encoded vector for group g, and let `x̂_img` be the
L2-normalized image embedding. The fused vector is

```
f = L2norm( [α_image · x̂_img,  α_g₁ · ê_{g₁},  α_g₂ · ê_{g₂},  …] )
```

Cosine similarity between two fused vectors decomposes as

```
cos(f_q, f_p) = (1/Z) · ( α_image² · cos(image_q, image_p)
                        + Σ_g α_g² · cos(g_q, g_p) )
```

Z is the product of fused-vector norms (a query-pair-dependent constant that
does not affect retrieval ranking). So **only the relative magnitudes of
α_image² and the α_g² matter**: the per-group α weights are precisely the
per-group cosine-similarity weights, up to a global scale that drops out at
retrieval time.

### Why No Trained Head

- The decomposition above is the design justification: a learnable fusion head over per-group similarities would just be a parameterization of these α².
- The α landscape on validation hits@k is shallow and amenable to random search + coordinate refinement; no gradient descent or millions of training samples are needed.
- Side benefit: every α has a direct interpretation (the contribution of that feature group to the final similarity), which makes ablation and explanation natural.

### What "Frozen" Means

- The image-only model is trained once, in the prior chapter, with no awareness of metadata.
- Its embeddings are dumped to a `.pt` file and never touched again.
- The multimodal pipeline reads that file and supplies the metadata stream alongside.

---

## 6. Metadata encoding: representation C

This is the heart of the technical contribution. Three encoding
representations were tried over the project. A and B are mentioned briefly to
motivate C, which is the one shipped and the one the paper centers.

### The Problem Each Encoding Solves

Per-group L2 normalization is mandatory for the fused-cosine-decomposition
argument above to hold. Two encoded-feature pathologies hurt downstream
retrieval:

- L2 normalizing a 1-dimensional length vector collapses every nonzero sample to 1.0, destroying the magnitude information completely.
- One-hot categoricals with an "Unknown" bin have odd semantics: two photos with unknown sex cosine-match at 1.0 (both have the same one-hot at the Unknown index).

### Representations A and B

* **Representation A** (the naive baseline). Hit by both pathologies above.
  * One-hot categoricals including an Unknown bin.
  * 1-dim normalized length.
  * 3-dim time block (sin day, cos day, year_norm).
* **Representation B** (transitional; mention briefly).
  * Length becomes `[value, present]` (2-dim), so the present-flag preserves magnitude through L2 normalization.
  * Time splits into `time_day = (sin, cos)` and `time_year = [year_norm, present]`, so each gets its own α.
  * Categoricals are unchanged from A.

### Representation C (Final)

Three parallel design choices, all motivated by "what does cosine-similarity
of two L2-normed vectors *mean* for this feature":

* **Categoricals** (sex, location, type, quality): drop the Unknown bin
  entirely. For known classes, encode the one-hot, then *center* by the
  population class proportions, then L2-normalize.
  * Two photos sharing a class get cosine ≈ +1.
  * Two photos in conflicting classes get cosine < 0.
  * An unknown sample is encoded as a zero vector and gives cosine = 0 against anything (no signal).
  * This is the semantics you actually want: same-class boost, conflict penalty, unknown abstains.
* **Continuous** (length, year): K-bin **RBF** encoding with σ = inter-center
  spacing, L2-normalized.
  * Two values close in the original space get cosine ≈ +1; far values cosine ≈ 0.
  * The encoding is *distance-graded* rather than binary-match.
  * Missing values are encoded as the L2-normalized population-mean RBF: two unknown samples cosine-match at 1, and unknown vs known gets a moderate match (population-average match strength).
  * This is a trade-off. We considered encoding unknown continuous as zero (no signal), but the population-mean choice gave better empirical retrieval and still reflects the prior "we don't know length, but values near the population mean are most likely."
* **Cyclical** (day-of-year): unchanged across representations.
  * `(sin θ, cos θ)` with θ = 2π·day/366.
  * Already on the unit circle, so L2-norm is a no-op.
  * It correctly puts December 30 next to January 2.

These three together are repC.

- The paper should present repC as the encoding contribution, and repA and repB as failed steps along the way.
- repC works because each encoding is built so that the *cosine similarity of two L2-normed encoded vectors* matches the *semantic similarity* a domain expert would assign to the underlying values.
- Match -> +1; conflict -> −1 (categoricals) or 0 (continuous); unknown -> 0 (categoricals) or moderate (continuous).

---

## 7. Tuning the α weights

The α tuning recipe is "Full preset" in the codebase:

* **Search space.** For each feature group g (image plus six metadata
  groups), α_g is searched in log-uniform [10⁻³, 1.0].
  * Time has two sub-groups, time_day and time_year, for a total of seven
    metadata coordinates.
  * The image weight α_image is searched in the same range.
  * Legacy behaviour fixed it at 1.0; tuning it helps.
* **Stage 1 — random search**, 1500 trials.
  * Each trial samples one α vector i.i.d. log-uniform over the search space,
    computes the fused similarity matrix on a held-out validation slice, and
    records the retrieval score.
  * Two anchor points (image-only and uniform-low) are added for sanity.
* **Stage 2 — coordinate refinement.** Starting from the random-search
  best, iterate over each coordinate, try a fixed multiplier set
  {0.25, 0.5, 0.75, 1.5, 2.0} on its current value (and zero), keep
  whichever wins, repeat for up to 4 passes or until no improvement.
* **Validation strategy.** Five independent stratified individual-level
  carves from the train split (each carve takes 20% of train *individuals*,
  not photos, so whole sharks move to validation; matches retrieval
  semantics). The objective is the average score across carves —
  reduces overfit to any single split. **The test split is never
  touched by the optimizer.**
* **Objective.** Avg K (continuous, gives richer gradient signal than
  hits@k thresholds). Other metrics are reported but not optimized.

The whole tuning takes ~30 seconds on an Apple-silicon laptop, because it
operates on precomputed per-group similarity matrices, not raw image forward
passes. This is the "limited compute" pitch in concrete terms.

---

## 8. Headline results

All numbers below are on the primary split (`thesis_unmatch_random_out`,
696 test queries) unless otherwise noted. Norm stats are fit on the
train split only.

| Method | Hits@1 | Hits@5 | Hits@10 | Hits@25 | Hits@50 | Hits@100 | Avg K | MRR |
|---|---|---|---|---|---|---|---|---|
| Image-only baseline | 0.886 | 0.930 | 0.950 | 0.963 | 0.971 | 0.983 | 10.12 | 0.908 |
| Trained naive concat | 0.763 | 0.864 | 0.894 | 0.921 | 0.937 | 0.953 | 14.86 | 0.809 |
| Trained residual | 0.882 | 0.941 | 0.961 | 0.968 | 0.980 | 0.987 | 6.62 | 0.911 |
| **Late fusion repC, full features** | **0.904** | 0.950 | 0.968 | 0.976 | 0.986 | 0.990 | **4.84** | **0.926** |
| **Late fusion repC, pruned (sex+loc only)** | **0.904** | 0.945 | 0.960 | 0.977 | 0.986 | 0.987 | 5.24 | 0.924 |

### Key Takeaways

* Calibrated late fusion (repC) beats both trained-fusion alternatives
  on every metric, with no parameters trained beyond a handful of α.
* Hits@1: 0.886 -> 0.904, a ~1.8 point absolute gain.
* Avg K: 10.12 -> 4.84 in the full version, 10.12 -> 5.24 in the pruned
  version. The multimodal extension more than halves the rank at which
  the correct individual is found, on average.
* The pruned version (only `sex` and `include_location` enabled in the
  metadata spec) achieves *identical Hits@1* to the full version with
  6 -> 2 features and 8 -> 3 tuned α parameters. The other four features
  contribute a small Avg K refinement (4.84 vs 5.24) but no precision
  gain.

---

## 9. Ablations

Three ablations support the design choices.

### 9.1 Single-feature contribution

Each row is "image embedding + ONLY one metadata feature", repC encoding, α
tuned end-to-end. This isolates the marginal value of each feature.

| Variant | Hits@1 | Avg K | MRR | ΔH@1 vs image-only |
|---|---|---|---|---|
| Image-only | 0.886 | 10.12 | 0.908 | — |
| + sex | 0.892 | 7.29 | 0.916 | +0.006 |
| + location | **0.898** | 6.59 | 0.918 | **+0.012** |
| + type | 0.876 | 13.52 | 0.901 | **−0.010** |
| + quality | 0.878 | 8.90 | 0.903 | −0.008 |
| + length | 0.884 | 8.90 | 0.907 | −0.002 |
| + time | 0.882 | 5.70 | 0.908 | −0.004 |
| Full repC (all six) | **0.904** | 4.84 | 0.926 | +0.018 |

#### Findings


* **Only sex and location individually help on Hits@1.** Type and quality
  individually hurt.
  * Type is the most striking: it drops H@1 by 0.010 *and* blows Avg K up to 13.52.
  * Type (still vs video) is largely uncorrelated with individual identity in this dataset, so adding it alone acts as noise.
  * The α-tuner finds a validation optimum, but that optimum doesn't transfer to test.
* **Cross-feature interactions are real.** Full repC beats every
  single-feature variant on every metric. The optimizer's α-balancing
  finds positive contributions from individually-harmful features when
  others are also present — likely because they act as tiebreakers
  once the higher-information features have narrowed the candidate set.

#### Pruning Recommendation

- This ablation drove the historical recommendation to enable only the features that produced a positive H@1 delta on their own.
- For the white-shark dataset, the pruned experiment used sex and location and matched the full repC's Hits@1.
- This is a paper result, not a statement of the current shared default; the maintained specification is owned by `multimodal_config.py`.

### 9.2 Trained fusion vs calibrated late fusion

The trained naive-concat and residual results are in the headline table
above. The framing for the paper:

* Naive concat catastrophically over-fits — the head finds a low-loss
  representation on train that doesn't transfer.
* Residual fusion is a strict improvement on naive concat (it can
  always fall back to image-only) but in practice extracts only modest
  Avg K gains and no H@1 gain.
* Calibrated late fusion outperforms both. The reason is in the design:
  the per-group α formulation is exactly the right inductive bias for
  this data scale — it gives the optimizer a small, well-shaped search
  space (one weight per feature group), it doesn't require gradients,
  and the fused similarity decomposes additively into per-group
  contributions that the optimizer can balance directly.

The paper should lean on the headline framing: "calibrated representation
outperforms trained fusion at this dataset scale".

### 9.3 Failure-mode taxonomy (subsection / appendix candidate)

For the 12 queries on the primary split where multimodal pulled the
correct individual *out* of the image-only top-1 (regressions), three
recurring failure modes account for all twelve:

1. **Look-alike + metadata coincidence (~7/12).** A visually similar
   shark from a different individual happens to share exact metadata
   with the query, while the correct same-individual photo has metadata
   mismatches (often because the correct individual was photographed
   years apart, or at a different site).
   * Cleanest example: query `PR09102101.jpeg`. The wrong-individual gallery
     photo agrees with the query on every known metadata field; the correct
     same-individual photo differs on `type` and `year`.
   * Image similarity ranks them roughly tied; metadata's perfect-match bonus
     goes to the wrong shark.
   * This is a fundamental limit of the modality: if two metadata-identical
     individuals exist in the gallery, no metadata-fusion approach
     disambiguates them without additional signal.
2. **Sparse query metadata (~2/12).** When the query has many unknown
   fields, the correct match's mismatches stand out more than they
   would if more known-positive agreements existed. Example
   `MB25090304.jpeg`: query sex unknown; correct match has unknown
   length and poor quality; wrong gallery photo (different individual)
   has exact length and matching quality. Two crisp positive signals
   beat one positive and one neutral.
3. **Large temporal gap on the correct match (~3/12).** Same individual
   photographed years apart; year-RBF distance penalizes the correct
   match more than a wrong match closer in time. Example
   `FAR06100307.jpeg`: correct match is 9 years later, wrong match
   only 2 years apart. The 9-year RBF penalty outweighs the
   length-1.0 mismatch on the wrong match.

#### Interpretation

- These are not bugs.
- (1) is a fundamental modality limit.
- (2) and (3) are the flip side of repC's design choices working correctly: sparse queries by construction get weaker signal, and year-RBF penalizing large gaps is the price of correctly grouping recent sightings.

#### Aggregate Trade

- The trade is favorable on both splits: ~2:1 promotions to regressions.
- Promotion magnitudes (e.g., k = 34 -> 1) are much larger than regression magnitudes (mostly k = 1 -> 2 or 1 -> 3).
- Late fusion accepts a bounded regression rate as the price of much larger gains.

This subsection can be presented inline in the paper or moved to an appendix;
it is informative but not load-bearing.

---

## 10. What's novel, what isn't

### Novel Claims This Chapter Can Defend

1. **Per-group L2 normalization makes heterogeneous metadata composable
   for late fusion.** A single α per feature group decomposes fused
   cosine similarity into a weighted sum of per-group similarities,
   each of which is independently interpretable.
2. **Centered + L2-normalized categorical encoding** correctly captures
   match / conflict / unknown semantics in a way that one-hot encoding
   cannot.
3. **RBF + L2-normalization for continuous fields** preserves
   distance-graded similarity through L2 normalization (which the
   naive 1-dim encoding destroys).
4. **Calibrated late fusion outperforms more-complex trained fusion**
   at this dataset scale — without backbone retraining, without head
   training, with α tuning that runs in seconds on a laptop.
5. **Per-feature ablation reveals individually-harmful features that
   the optimizer can still leverage in combination.** Gives a
   principled recipe for adapting the framework to new datasets:
   ablate single features, prune those that regress, tune the leaner
   spec.

### Honest Framing: Not Novel, Not the Contribution

* Late fusion as a concept is decades old. The contribution is the
  encoding design and the per-group α decomposition, not the existence
  of late fusion.
* The α search procedure (random search + coord refinement) is
  standard. The contribution is identifying it as the right tool for
  this problem scale.
* RBF encoding for continuous features and centered embeddings for
  categoricals are both common in machine learning. The contribution
  is composing them through the per-group L2 framework so they
  interact correctly in fused cosine similarity.

---

## 11. Limitations and open questions

* **Statistical significance not yet computed.** Bootstrap CIs and
  paired McNemar tests on the discordant queries are tractable on the
  existing eval pipeline (~30 lines of code) but were deferred. The
  Hits@1 deltas (~0.018) on N = 696 are point estimates; reviewers and
  committee members may want CIs.
* **One species, two splits of one corpus.** The framework-generality
  claim rests on the design (it's representation-agnostic and
  feature-set-agnostic) more than on cross-dataset evidence. A
  metadata-completeness ablation (random-mask each field at 10/25/50/
  75% and plot H@1/Avg K curves under repC vs repA/B) is the strongest
  single experiment that would shore up the generality claim.
* **Type and quality were uncorrelated with identity in this dataset.**
  Whether they would help on a dataset with stronger photographic-
  context structure (e.g., a fixed multi-camera setup where `type`
  encodes camera angle) is an open question.
* **The correlation between α-tuning validation score and test score is
  imperfect.** The single-feature ablation showed several cases where
  the tuner found a validation optimum that didn't transfer. With more
  validation seeds (we use 5) and stronger anti-overfit regularization
  on α, this could improve.
* **Geographic location is currently a one-hot of six site codes.**
  Replacing it with a (lat, lon) distance kernel is the largest
  remaining accuracy headroom on the dataset; site coordinates exist
  but the encoding hasn't been built.

---

## 12. Suggested narrative arc for the paper

A three-act structure that reflects the experimental journey while landing
the technical contribution cleanly.

### Act 1: Setup and Problem

* Brief recap of the image-only chapter (a paragraph).
* The opportunity: metadata is available, sparse, heterogeneous, and
  varies in identity-discriminating power. Show one motivating example
  from the dataset where two visually similar individuals are in
  different sites with different sexes.
* Frame the design space: trained fusion vs late fusion. State the
  compute constraint (laptop, no backbone retraining) and the design
  goal (interpretable, modular, framework-extensible).

### Act 2: The Journey

* Trained fusion attempts. State briefly: naive concat regresses, residual
  fusion floors at image-only, neither extracts substantial gain.
  Reproduce the table.
* Hypothesis: the bottleneck isn't model capacity, it's metadata
  encoding. Present repA and repB as transitional encodings that
  illustrate the failure modes (1-dim L2 collapse, unknown-bin
  semantics) and motivate repC. Brief — these aren't the
  contribution.
* Land on calibrated late fusion + repC as the design that resolves
  every issue and matches the compute constraint. State the
  decomposition argument: fused cosine = weighted sum of per-group
  cosines, so per-group α is the natural and interpretable parameterization.

### Act 3: The Method, Ablations, and Results

* Section 5 of this document: pipeline figure, math, and the encoding
  table for repC.
* Section 7: α tuning recipe.
* Section 9.1: single-feature ablation. Lead with the surprising
  finding that type and quality individually hurt; close with the
  pruning recommendation and the pruned-vs-full comparison.
* Section 9.2: trained-fusion comparison.
* Headline results table.
* Section 9.3: failure-mode taxonomy (inline or appendix).
* Section 11: limitations.

### Headline Framing

The chapter's headline framing, which the abstract should re-state and the
conclusion should close on:

> *"A small, carefully-encoded set of metadata fields, fused into the
> image embedding via per-group L2 normalization and a handful of
> tuned α weights, halves the average rank at which the correct
> individual is retrieved — and outperforms more complex trained
> fusion at this dataset scale, without retraining the image
> backbone."*

---

## 13. Pointers to code and reproduction

The current reproduction entry points and recorded results are in `README.md`,
which also identifies historical analyses whose helper is no longer retained.

### Relevant Code

* `metadata_features.py` — `MetadataFeatureSpec`, `fit_norm_stats`,
  `engineer_features`. Per-feature encoders for repA/B/C; per-feature
  toggles `repc_categoricals`, `repc_length`, `repc_time_year` (used
  for orthogonal ablations).
* `fusion_dataset.py` — `FusionDataset`. Pairs frozen image embeddings
  with engineered metadata, fits norm stats on train.
* `src/retrieval/embeddings.py` — builds the fused embedding artifacts
  from the feature specification in `multimodal_config.py`.
* `optimize_late_fusion.py` — α tuner. Random search + coordinate
  refinement, 5 stratified validation seeds, full-preset defaults.
* `multimodal_config.py` — current shared metadata-feature and named α
  configuration registries.
* The research-only discordant-pair helper used for the failure-mode taxonomy
  is not retained in this source tree; the recorded results remain historical
  evidence rather than a current reproduction path.
* `trained_fusion/` — isolated subdirectory housing the
  trained-fusion baselines (naive concat, residual). Has its own
  README; does not interfere with the late-fusion path.

### Canonical Configs

The two configs cited in the chapter:

* `thesis_fine_repC_noleak` — full repC, all six metadata features.
* `thesis_fine_repC_pruned` — repC with only sex and location enabled.
  This is the pruned historical experiment, not the current shared default.
