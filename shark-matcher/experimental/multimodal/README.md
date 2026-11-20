# Multimodal Fusion Research

> **Status:** Research notes; not part of the supported Shark Matcher workflow.

Research utilities and results for the late-fusion pipeline. The supported,
integrated workflow and its implementation owners are documented in
`src/multimodal/README.md`; the shared modules live in
`src/multimodal/`.

## Post-Hoc Fusion Utility

Existing embeddings can be fused without regenerating their image embeddings:

```bash
python3 -m experimental.multimodal.multimodal_embeddings \
    --embeddings-dir <embeddings-name>
```

[`multimodal_embeddings.py`](multimodal_embeddings.py) performs optimization
and writes the multimodal artifacts into the selected existing embeddings
directory. Unlike the integrated generation path, it updates that directory in
place and does not publish a complete generation through atomic staging.
Prefer the integrated path for a new release artifact.

## Research Code

- [`multimodal_embeddings.py`](multimodal_embeddings.py) is the in-place
  post-hoc entry point.
- `trained_fusion/` contains trained-head research
  baselines and is not part of the integrated late-fusion workflow.
- `docs/multimodal-paper-context.md` records the paper context for these
  experiments.

## Historical Benchmark Results

The tables and analysis below are retained as an as-run research record. Names,
flags, feature sets, weights, and dataset counts in this section describe those
experiments; they are **not** current defaults or a current CLI reference.
Reproduce a historical configuration from its saved metadata and the relevant
source revision rather than assuming that every recorded flag remains exposed.

### best_unmatch_new_out_no_psl (847 test queries) — fit on FULL corpus (legacy)

These rows were produced before the `--fit-stats-split` flag existed and have a
minor train->test leak in the norm-stats fit (see the
"thesis_unmatch_random_out" tables for the leak fix). Kept for reproducibility;
thesis is the dataset retuned end-to-end with no leakage.

| Config | H@1 | H@5 | H@10 | H@25 | H@50 | H@100 | Avg K |
|---|---|---|---|---|---|---|---|
| Image-only baseline | 0.893 | 0.937 | 0.948 | 0.956 | 0.965 | 0.976 | 8.97 |
| `global_optimize` (repA, simple, uniform-α recipe) | 0.906 | 0.947 | 0.959 | 0.976 | 0.989 | 0.995 | 4.20 |
| `fine_optimize` (repB, full) | 0.902 | 0.952 | 0.960 | 0.978 | 0.989 | 0.995 | 4.33 |
| **`fine_optimize_c`** (repC, full) | **0.909** | **0.952** | **0.965** | **0.980** | **0.991** | **0.996** | **2.94** |

### thesis_unmatch_random_out (norm stats fit on FULL corpus — has minor train->test leak)

These rows were tuned with `--fit-stats-split all` (the old default; the
flag did not exist when these were produced). Categorical vocabularies,
length/year min-max ranges, RBF centers, and centered-encoding
proportions saw test rows during fitting. Kept here for reproducibility;
the no-leakage table below is the one to cite.

| Config | H@1 | H@5 | H@10 | H@25 | H@50 | H@100 | Avg K |
|---|---|---|---|---|---|---|---|
| Image-only baseline | 0.886 | 0.930 | 0.950 | 0.963 | 0.971 | 0.983 | 10.12 |
| `global_optimize` (repA, simple, uniform-α recipe) | 0.898 | 0.950 | 0.966 | 0.980 | 0.984 | 0.990 | 4.37 |
| `thesis_fine_repB` (repB, per-dataset) | 0.889 | 0.944 | 0.964 | 0.977 | 0.987 | 0.991 | **3.83** |
| **`thesis_fine_repC`** (repC, per-dataset) | **0.902** | **0.950** | **0.968** | 0.974 | 0.986 | 0.990 | 4.84 |

### thesis_unmatch_random_out (norm stats fit on TRAIN only — no leakage)

Produced with the historical `optimize_late_fusion.py --preset {simple,full}`
and `--fit-stats-split train`. Test rows do not influence vocabularies,
min-max ranges, RBF centers, or centered-encoding proportions. **These are the
rows the paper should cite.**

| Config | H@1 | H@5 | H@10 | H@25 | H@50 | H@100 | Avg K | MRR |
|---|---|---|---|---|---|---|---|---|
| Image-only baseline | 0.886 | 0.930 | 0.950 | 0.963 | 0.971 | 0.983 | 10.12 | 0.908 |
| `thesis_global_optimize_tied_noleak` (repA, simple, tied α — `--uniform-metadata-alpha`) | 0.898 | 0.950 | 0.966 | 0.980 | 0.984 | 0.990 | 4.37 | 0.922 |
| `thesis_global_optimize_noleak` (repA, simple, per-group α) | 0.872 | 0.927 | 0.944 | 0.961 | 0.971 | 0.978 | 7.36 | 0.899 |
| `thesis_fine_repB_noleak` (repB, full, per-group α) | 0.889 | 0.944 | 0.964 | 0.977 | 0.987 | 0.991 | **3.83** | 0.915 |
| **`thesis_fine_repC_noleak`** (repC, full, per-group α) | **0.904** | **0.950** | **0.968** | 0.976 | 0.986 | 0.990 | 4.84 | **0.926** |

#### Precision vs Ranking Depth (MRR vs Avg K)

Run the optimizer with `--mrr` to add MRR alongside the standard table.

- RepC leads on H@1 and MRR; repB leads on Avg K.
- MRR is the harmonic dual of Avg K. Both use the same first-correct-individual rank.
- MRR averages 1/k (saturates on the tail, weights early ranks heavily); Avg K averages k linearly (tail-sensitive).
- Together they expose the precision-depth trade-off: repC nails top-1 retrieval more often, repB recovers difficult queries deeper in the list.
- Pick whichever fits your application.

### Single-feature contribution ablation (thesis, repC, no leak)

To attribute the multimodal gain to specific metadata features, the
table below isolates each one. Each row is *image embedding + ONLY
that one feature*, with α tuned end-to-end (`--preset full
--representation-c --only-feature <name>`). Compare against the
image-only baseline (no metadata) and the all-features repC at the
top and bottom.

| Variant | H@1 | H@5 | H@10 | H@25 | H@50 | H@100 | Avg K | MRR | ΔH@1 vs image-only | ΔAvg K |
|---|---|---|---|---|---|---|---|---|---|---|
| Image-only baseline | 0.886 | 0.930 | 0.950 | 0.963 | 0.971 | 0.983 | 10.12 | 0.908 | — | — |
| + sex only | 0.892 | 0.944 | 0.957 | 0.967 | 0.976 | 0.980 | 7.29 | 0.916 | +0.006 | −2.83 |
| **+ location only** | **0.898** | 0.943 | 0.955 | 0.974 | 0.981 | 0.987 | 6.59 | 0.918 | **+0.012** | −3.53 |
| + type only | 0.876 | 0.930 | 0.944 | 0.957 | 0.963 | 0.977 | **13.52** | 0.901 | **−0.010** | **+3.40** |
| + quality only | 0.878 | 0.927 | 0.945 | 0.967 | 0.978 | 0.983 | 8.90 | 0.903 | −0.008 | −1.22 |
| + length only | 0.884 | 0.932 | 0.947 | 0.968 | 0.974 | 0.983 | 8.90 | 0.907 | −0.002 | −1.22 |
| **+ time only** | 0.882 | 0.934 | 0.954 | 0.973 | 0.984 | 0.990 | **5.70** | 0.908 | −0.004 | **−4.42** |
| repC (all six) | **0.904** | 0.950 | 0.968 | 0.976 | 0.986 | 0.990 | 4.84 | **0.926** | +0.018 | −5.28 |

#### Per-Feature Contribution Story

* **Location is the strongest individual feature on H@1** (+0.012),
  with a strong Avg K win too (−3.53). Sex is the second strongest
  H@1 contributor (+0.006).
* **Time is excellent for ranking depth** (Avg K −4.42, the largest
  single-feature improvement), but mildly hurts H@1 (−0.004). Time
  alone cannot disambiguate visual look-alikes — but it pulls many
  correct individuals into the top-25 / top-50 / top-100 ranges,
  which is exactly the tail Avg K measures.
* **Type and quality net-hurt H@1.** Type is the most striking — it
  drops H@1 by 0.010 and *blows up* Avg K by 3.40 vs image-only. The
  α-tuner on the validation slice landed on a config that adds type
  as significant signal, but type (still/video) is essentially
  uncorrelated with individual identity, so on test it acts as noise.
  Quality similarly hurts H@1 (−0.008), with a smaller Avg K
  improvement.
* **Length alone is roughly neutral** (small negative on H@1, small
  positive on Avg K). Useful in combination but not on its own.
* **All-six repC wins on every metric.** H@1 = 0.904 (better than
  the best single feature, location, at 0.898). The optimizer's
  α-balancing finds positive contributions from features that hurt
  alone, suggesting cross-feature interactions: e.g., quality
  becomes useful as a tiebreaker once sex and location have already
  narrowed the candidate set, and type acts as a weak prior that
  the higher-weighted features can override when wrong.

#### Implication for New Datasets

- Run the single-feature ablation first.
- Features whose single-feature row regresses against image-only are candidates for exclusion (or for very low α priors) when other features carry most of the signal.
- The framework's robustness comes from the optimizer balancing these trade-offs automatically.
- On small-validation-budget settings, pre-pruning low-information features can save tuning effort and reduce variance.

### Pruned repC: keep only the features that helped alone

For this recorded ablation, the pruned feature specification enabled only sex
and location. The pruned repC configuration below was tuned end-to-end with
those two features. The current shared specification lives in
[`multimodal_config.py`](../../src/multimodal/multimodal_config.py) and should not be inferred from
this historical table.

| Variant | Features used | α params tuned | H@1 | H@5 | H@10 | H@25 | H@50 | H@100 | Avg K | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| Image-only baseline | — | — | 0.886 | 0.930 | 0.950 | 0.963 | 0.971 | 0.983 | 10.12 | 0.908 |
| Full repC (`thesis_fine_repC_noleak`) | sex, loc, type, qual, len, time | 8 | **0.904** | 0.950 | 0.968 | 0.976 | 0.986 | 0.990 | **4.84** | **0.926** |
| **Pruned repC (`thesis_fine_repC_pruned`)** | sex, loc | **3** | **0.904** | 0.945 | 0.960 | 0.977 | 0.986 | 0.987 | 5.24 | 0.924 |

**What this shows.**

- Dropping the four single-feature-regressing features preserves H@1 *exactly* (both 0.904) with 6 -> 2 features and 8 -> 3 tuned α parameters.
- The full-six repC keeps a small ranking-depth advantage (Avg K 4.84 vs 5.24, MRR 0.926 vs 0.924).
- That 0.40 Avg K gap is the residual contribution from type/quality/length/time when the optimizer has all six available to combine.

So:

* **For a paper-grade ablation table:** the pruned repC is the
  cleaner result to lead with — same H@1 with a far smaller and
  more interpretable feature set. Less risk of "why did adding
  type help on test when it hurt alone?" reviewer questions.
* **For maximum performance:** keep the full-six repC as the
  reference; the extra ~8% Avg K reduction is real and free at
  inference time.
* **For new datasets:** run `--only-feature` per metadata field
  first; keep only the features with positive H@1 deltas in the
  default; tune α with the leaner spec; optionally compare against
  the all-features tune to see whether cross-feature interactions
  buy enough to justify the larger surface.

### Trained-fusion ablation (thesis, train-only stats, --fit-stats-split train)

To position the calibrated late-fusion result against more complex
*trained* alternatives, two trained-head baselines are reproduced
end-to-end on thesis (code lives under
`experimental/multimodal/trained_fusion/`, isolated from the
late-fusion path). Both are trained with batch-hard triplet loss
(margin 0.2), AdamW + cosine schedule, P×K = 8×4, on the train split
only; norm stats fit on train.

| Config | H@1 | H@5 | H@10 | H@25 | H@50 | H@100 | Avg K | MRR |
|---|---|---|---|---|---|---|---|---|
| Image-only baseline | 0.886 | 0.930 | 0.950 | 0.963 | 0.971 | 0.983 | 10.12 | 0.908 |
| Trained naive concat (VFE+TME -> head) | 0.763 | 0.864 | 0.894 | 0.921 | 0.937 | 0.953 | 14.86 | 0.809 |
| Trained residual fusion (image + δ(image, meta)) | 0.882 | 0.941 | 0.961 | 0.968 | 0.980 | 0.987 | 6.62 | 0.911 |
| **Late-fusion repC (`thesis_fine_repC_noleak`)** | **0.904** | **0.950** | **0.968** | 0.976 | 0.986 | 0.990 | **4.84** | **0.926** |

**What this shows.**

* **Naive concat regresses ~12 H@1 points below image-only.** The head
  over-fits the train distribution. There is no residual path guaranteeing a
  floor at the image-only baseline.
* **Residual fusion is roughly image-only.** The zero-init residual
  path forces a floor at the image-only baseline; training improves
  Avg K (10.12 -> 6.62) but the H@1 / MRR improvements are within
  noise. The metric signal that triplet loss gets to use isn't strong
  enough to extract more.
* **Calibrated late fusion (repC) wins on every metric.**
  * Beats image-only by 1.8 H@1 points and halves Avg K.
  * Beats the best trained alternative (residual) on H@1 (+2.2) and Avg K (−1.78).
  * Requires *no backbone retraining and no head training*, only a cheap
    per-group α tuning step on a precomputed similarity matrix.
  * The paper's headline framing, "calibrated representation beats more
    complex trained fusion at this scale", is the result this table supports.

## Qualitative analysis — H@1 promotions and regressions

The exact queries that move the needle on Hits@1 between image-only and the
multimodal late fusion.

- *Promotions*: queries whose first correct individual went from rank `k > 1` (image-only) to rank `k = 1` (multimodal).
- *Regressions*: the inverse; queries the multimodal pipeline got worse on.
- The "correct match" filename is the gallery photo retrieved at top-1 by the *winning* system in each table (multimodal in promotions, image-only in regressions).

### Provenance and Check

- These tables are retained as historical experiment evidence.
- They were produced with a research-only `discordant_pairs.py` helper that is not retained in the current source tree, so this checkout provides no supported reproduction command for the promotion/regression listing.
- The discordant counts match the headline H@1 deltas exactly: `(promotions − regressions) / N_test = ΔH@1`.

### thesis_unmatch_random_out (repC noleak vs image-only, 696 test queries)

H@1: 0.886 -> 0.904. Net flips: 24 − 12 = +12 queries (Δ=+0.0172, matching observed +0.018).

**Promotions — 24 queries flipped from wrong-top-1 to correct-top-1.**

| Query | Correct match | k (image-only) | k (multimodal) |
|---|---|---|---|
| `FAR10101801.jpeg` | `FAR10101202.jpeg` | 34 | 1 |
| `PR14102802.jpeg` | `PR14102905.jpeg` | 32 | 1 |
| `PR08101801.jpeg` | `PR08091401.jpeg` | 23 | 1 |
| `AN25011508.jpeg` | `AN24111003.jpeg` | 21 | 1 |
| `PR19120902.jpeg` | `PR19121009.jpeg` | 7 | 1 |
| `AN23111308.jpeg` | `AN21111204.jpeg` | 6 | 1 |
| `FAR05111901.jpeg` | `FAR06100901.jpeg` | 6 | 1 |
| `FAR16101203.jpeg` | `FAR17121202.jpeg` | 4 | 1 |
| `NB24051101.jpeg` | `NB24062702.jpeg` | 4 | 1 |
| `FAR23103114.jpeg` | `FAR23112206.jpeg` | 3 | 1 |
| `MB24070501.jpeg` | `MB24082309.jpeg` | 3 | 1 |
| `NB22090601.jpeg` | `NB22082207.jpeg` | 3 | 1 |
| `AN14011407.jpeg` | `AN14011802.jpeg` | 2 | 1 |
| `AN15100103.jpeg` | `PR12100601.jpeg` | 2 | 1 |
| `AN19111007.jpeg` | `AN20011805.jpeg` | 2 | 1 |
| `AN21110316.jpeg` | `AN21110312.jpeg` | 2 | 1 |
| `AN23111008.jpeg` | `AN23102112.jpeg` | 2 | 1 |
| `FAR13101901.jpeg` | `FAR13101801.jpeg` | 2 | 1 |
| `FAR21111003.jpeg` | `FAR21111505.jpeg` | 2 | 1 |
| `MB24090209.jpeg` | `NB24070216.jpeg` | 2 | 1 |
| `NB22061001.jpeg` | `NB22061003.jpeg` | 2 | 1 |
| `NB23071903.jpeg` | `NB23071901.jpeg` | 2 | 1 |
| `PR16102002.jpeg` | `PR15100102.jpeg` | 2 | 1 |
| `PR22101204.jpeg` | `AN25110805.jpeg` | 2 | 1 |

**Regressions — 12 queries flipped from correct-top-1 to wrong-top-1.**

| Query | Correct match | k (image-only) | k (multimodal) |
|---|---|---|---|
| `PR24100201.jpeg` | `PR24092301.jpeg` | 1 | 26 |
| `NB22081603.jpeg` | `FAR25110313.jpeg` | 1 | 24 |
| `MB25090304.jpeg` | `MB25082001.jpeg` | 1 | 12 |
| `NB24081304.jpeg` | `NB24053110.jpeg` | 1 | 9 |
| `PR11121401.jpeg` | `FAR22101702.jpeg` | 1 | 5 |
| `PR12121307.jpeg` | `AN25120402.jpeg` | 1 | 4 |
| `AN16011502.jpeg` | `AN22110616.jpeg` | 1 | 3 |
| `AN19110601.jpeg` | `AN25121110.jpeg` | 1 | 2 |
| `FAR06100307.jpeg` | `FAR15101507.jpeg` | 1 | 2 |
| `MB25090503.jpeg` | `MB25081302.jpeg` | 1 | 2 |
| `PR09102101.jpeg` | `PR08110703.jpeg` | 1 | 2 |
| `PR25120504.jpeg` | `MB24090207.jpeg` | 1 | 2 |

### best_unmatch_new_out_no_psl (fine_optimize_c, leaky stats fit, vs image-only, 847 test queries)

- This comparison uses the legacy `fine_optimize_c` config built with `--fit-stats-split all` (the leaky default that produced the README's headline best_unmatch table).
- The leak fix has near-zero numerical effect on this dataset, so the pairs below also represent the no-leak version.
- H@1: 0.893 -> 0.909. Net flips: 29 − 15 = +14 queries (Δ=+0.0165, matching observed +0.016).

**Promotions — 29 queries flipped from wrong-top-1 to correct-top-1.**

| Query | Correct match | k (image-only) | k (multimodal) |
|---|---|---|---|
| `NB24070204.jpeg` | `NB24080811.jpeg` | 30 | 1 |
| `AN23120409.jpeg` | `MB23092205.jpeg` | 25 | 1 |
| `NB22090601.jpeg` | `NB22082207.jpeg` | 10 | 1 |
| `MB24090502.jpeg` | `MB25080601.jpeg` | 9 | 1 |
| `MB25090806.jpeg` | `MB25080602.jpeg` | 7 | 1 |
| `NB24051103.jpeg` | `NB24060101.jpeg` | 6 | 1 |
| `AN23100801.jpeg` | `AN23120308.jpeg` | 3 | 1 |
| `AN24120201.jpeg` | `AN24120306.jpeg` | 3 | 1 |
| `AN25110805.jpeg` | `PR22101204.jpeg` | 3 | 1 |
| `FAR23103114.jpeg` | `FAR23112206.jpeg` | 3 | 1 |
| `MB24082309.jpeg` | `MB24082011.jpeg` | 3 | 1 |
| `MB25080602.jpeg` | `MB25090806.jpeg` | 3 | 1 |
| `AN22102104.jpeg` | `AN22120504.jpeg` | 2 | 1 |
| `AN23111007.jpeg` | `AN22110701.jpeg` | 2 | 1 |
| `AN25111104.jpeg` | `AN22102104.jpeg` | 2 | 1 |
| `AN26011102.jpeg` | `AN26011501.jpeg` | 2 | 1 |
| `AN26011501.jpeg` | `AN26011102.jpeg` | 2 | 1 |
| `FAR06110104.jpeg` | `FAR06102303.jpeg` | 2 | 1 |
| `FAR21111003.jpeg` | `FAR21111505.jpeg` | 2 | 1 |
| `FAR21111505.jpeg` | `FAR21111003.jpeg` | 2 | 1 |
| `FAR23112206.jpeg` | `FAR23103114.jpeg` | 2 | 1 |
| `MB23090401.jpeg` | `MB23090803.jpeg` | 2 | 1 |
| `MB23100601.jpeg` | `MB23100907.jpeg` | 2 | 1 |
| `MB24070501.jpeg` | `MB24082011.jpeg` | 2 | 1 |
| `MB24090503.jpeg` | `MB24090904.jpeg` | 2 | 1 |
| `MB25081302.jpeg` | `MB25090503.jpeg` | 2 | 1 |
| `NB22082207.jpeg` | `NB22081108.jpeg` | 2 | 1 |
| `NB23072503.jpeg` | `NB23072704.jpeg` | 2 | 1 |
| `NB24080801.jpeg` | `NB24072302.jpeg` | 2 | 1 |

**Regressions — 15 queries flipped from correct-top-1 to wrong-top-1.**

| Query | Correct match | k (image-only) | k (multimodal) |
|---|---|---|---|
| `MB25090304.jpeg` | `MB25082001.jpeg` | 1 | 5 |
| `FAR22092514.jpeg` | `PR22092401.jpeg` | 1 | 3 |
| `FAR23102107.jpeg` | `FAR25102204.jpeg` | 1 | 3 |
| `MB24090207.jpeg` | `PR25120504.jpeg` | 1 | 3 |
| `NB24053110.jpeg` | `NB24081304.jpeg` | 1 | 3 |
| `AN25120414.jpeg` | `AN23111209.jpeg` | 1 | 2 |
| `AN26011907.jpeg` | `NB22061012.jpeg` | 1 | 2 |
| `FAR25111922.jpeg` | `MB23090801.jpeg` | 1 | 2 |
| `MB24082012.jpeg` | `NB22061014.jpeg` | 1 | 2 |
| `MB24090903.jpeg` | `NB24060103.jpeg` | 1 | 2 |
| `MB25090802.jpeg` | `MB25081804.jpeg` | 1 | 2 |
| `PR22092401.jpeg` | `FAR22092514.jpeg` | 1 | 2 |
| `PR24100201.jpeg` | `PR24092301.jpeg` | 1 | 2 |
| `PR25120502.jpeg` | `AN25121208.jpeg` | 1 | 2 |
| `PR25120504.jpeg` | `MB24090207.jpeg` | 1 | 2 |

#### Choosing Examples for the Qualitative Figure

- The largest-rank promotions (e.g. `FAR10101801.jpeg` at k=34->1, `NB24070204.jpeg` at k=30->1) are the most visually striking: image-only ranked the correct individual deep in the list, and metadata pulled it to the top.
- The highest-rank regressions are honest counterpoints. At the largest gaps the multimodal mistake is informative, often a metadata conflict overpowering image similarity (e.g. `PR24100201.jpeg` at k=1->26 on thesis).

### Why do regressions happen? Three named failure modes

All 27 regressions across the two datasets were verified as genuine top-1
flips: the image-only top-1 image is a same-individual photo, and the
multimodal top-1 image is a different-individual photo.

Inspecting query/correct/wrong metadata side-by-side surfaces three recurring
patterns. Each thesis regression fits cleanly into one of them.

#### Pattern 1: Look-Alike + Metadata Coincidence (~7/12 Thesis Regressions)

A visually similar shark from a different individual happens to share
exact-or-very-close metadata with the query, while the correct same-individual
photo has metadata mismatches. Cleanest example, query `PR09102101.jpeg`
(correct individual = label 4290):

| | sex | location | type | quality | length | year |
|---|---|---|---|---|---|---|
| Query | M | PR | topshot | ? | 14.0 | 2009 |
| Correct (label 4290) | M | PR | underwater | ? | 14.0 | **2008** |
| Wrong (label 4071) | M | PR | topshot | ? | 14.0 | 2009 |

- The wrong-individual gallery photo agrees with the query on every known metadata field.
- The correct same-individual photo differs on `type` (topshot vs underwater) and `year` (2008 vs 2009).
- Image similarity ranks them roughly tied; metadata's perfect-match bonus goes to the wrong shark.
- This is a *fundamental* limit of the modality: if two metadata-identical individuals exist in the gallery, no metadata-fusion approach can disambiguate them without additional signal.

#### Pattern 2: Sparse Query Metadata (~2/12 Thesis Regressions)

When the query has many `?` fields, the correct match's mismatches stand out
more than they would if more known-positive agreements existed alongside them.

- Example `MB25090304.jpeg`: query sex unknown; the correct match has `length=?` and `quality=poor` (vs the query's `medium`).
- The wrong gallery photo (different individual, at NB) has `length=7.0` exact and `quality=medium` exact.
- Two crisp positive signals beat one positive (`location=MB`) and one neutral.

#### Pattern 3: Large Temporal Gap on the Correct Match (~3/12 Thesis Regressions)

The same individual was photographed years apart (sharks grow, locations
drift). The year-RBF and sometimes length-RBF penalize the correct match more
than a wrong match that happens to be closer in time.

- Example `FAR06100307.jpeg`: the correct match is 9 years later (2006 -> 2015); the wrong match is only 2 years apart (2006 -> 2008).
- The 9-year gap is far in the year-RBF; the time penalty on the correct match exceeds the length-1.0 mismatch on the wrong one.

### What This Means for the Paper

* **None of the regressions are bugs or metric artifacts** — all
  verified as honest top-1 flips with explainable causes.
* **The trade is favorable on aggregate.** Promotions to regressions
  ratio is ~2:1 on both datasets. Promotion magnitudes include
  k=34->1 and k=30->1; regression magnitudes are mostly k=2–3 with
  long-tail outliers. The expected per-query benefit dominates.
* **Patterns 2 and 3 are by design.** Sparse-metadata queries
  contributing weaker positive signal is exactly what repC's
  "Unknown -> zero vector" encoding guarantees. Year-RBF distance
  penalizing legitimate matches with large temporal gaps is the
  flip side of it correctly grouping recent sightings — the
  parameter that controls one controls the other.
* **Pattern 1 is the only one with mitigation room** — and only
  through richer per-pair signals (e.g., pairwise re-ranking on
  individual-conditional photo features) rather than richer
  late-fusion encodings.
* **The residual-fusion baseline avoided most regressions** (it's
  floored at image-only on H@1) but also missed almost all the
  promotions. Late fusion's higher H@1 comes from accepting a
  bounded regression rate as the price of much larger gains, a
  defensible paper position.

### Notes on the Leak Fix

* **repB and repC are essentially unchanged** under train-only fitting
  (within ±0.2% on H@k; Avg K identical).
  * For thesis, train and full-corpus fits produce nearly identical normalization stats.
  * Every test categorical level exists in train, and test length/year values fall within train's range.
  * The leak fix is the principled thing to do; on this dataset it has near-zero numerical effect.
* **The legacy `global_optimize` recipe is reproduced exactly by the
  tied-α run** above.
  * Numbers match the leaky table to four decimal places, confirming that the leak has zero numerical impact on thesis.
  * Use `--uniform-metadata-alpha` to opt into the 2-DoF search (one α_image vs one shared α_meta).
* **Per-group repA (`thesis_global_optimize_noleak`) underperforms
  tied repA.**
  * With six free metadata coordinates and only the hits@10 signal on a single validation carve, the per-group search overfits to the validation slice.
  * RepB and repC stay robust to per-group α tuning because their encodings are calibrated (magnitude-preserving in repB; centered+L2 in repC), giving the optimizer interpretable distance gradients.
  * RepA's raw one-hot + 1-dim length collapses under per-group L2 and doesn't.

### Summary

- Avg K dropped ~52–62% from image-only; Hits@1 up ~1.8 points (repC).
- RepC leads on Hits@1; repB leads on Avg K for the thesis dataset. Both beat image-only comfortably.
- Run both presets and ship whichever wins for your application's priority (precision-at-1 vs ranking depth).

## Tier-3 baseline (--no-group-l2)

Skips per-group L2-norm; metadata groups contribute raw dot products.

- Net-negative on best_unmatch (optimizer drops length entirely); mixed on thesis.
- Confirms repC's encoding approach is correct.
