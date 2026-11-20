# PSL Rule Experimentation Log

> **Status:** Research notes; not part of the supported Shark Matcher workflow.

Documents PSL rule architectures tested on the shark re-identification task:
the reasoning behind each design, empirical results, and lessons learned.

## Baseline

`thesis_unmatch_random_out`, neural only:

| Hits@1 | Hits@5 | Hits@10 | Avg K |
|--------|--------|---------|-------|
| 0.886  | 0.930  | 0.950   | 10.12 |

---

## Ruleset A — Sex Disagreement (Direct Evidence)

**Status:** Retired — hurts results  
**Embeddings tested on:** `thesis_unmatch_random_out`

### Rules
```
neural_prior_forward:  NeuralSim(I1, I2) -> SameShark(I1, I2)          w=0.23
neural_prior_reverse:  SameShark(I1, I2) -> NeuralSim(I1, I2)          w=0.23
sex_disagreement_mf:   IsMale(I1) & IsFemale(I2) -> !SameShark(I1, I2) w=learned
sex_disagreement_fm:   IsFemale(I1) & IsMale(I2) -> !SameShark(I1, I2) w=learned
```

- Targets: forward pairs only `(q -> n)`. No backward targets. No symmetry.
- Sex encoding: hard (0.0, 0.0 for unknown sex).
- Weights: MLE-learned for sex rules; neural prior fixed at 0.23.

### Results

| Metric  | Broken (MLE killed neural prior) | Fixed (neural prior frozen) |
|---------|-----------------------------------|-----------------------------|
| Hits@1  | 0.818 (−7.7%)                     | 0.881 (−0.6%)               |
| Hits@5  | 0.866                             | 0.925                       |
| Hits@10 | 0.879                             | 0.947                       |
| Avg K   | 11.44                             | 10.23                       |

### Why It Underperformed

#### Rule Direction Mismatch for Re-Ranking

Direct evidence rules place sex observations in the rule body and SameShark in
the head:

```
body = max(0, IsMale(I1) + IsFemale(I2) − 1)   <- fixed observations
penalty = body² × w  applied to SameShark        <- ignores current SameShark value
```

- The rule fires at *full strength for every sex-conflicting pair regardless of neural similarity*.
- A pair ranked 95th out of 100 by neural similarity receives the same sex penalty as one ranked 2nd.
- For a re-ranker this is wasteful noise; only the top of the ranked list needs adjusting.

#### MLE Collapses Neural Prior and Sex Weights

- The training distribution has ~95% wrong-match pairs in the top-100, many with high NeuralSim.
- MLE minimises the energy across all pairs and finds that *neural prior weight -> 0* avoids penalising the dominant wrong-match case.
- MLE also learns a weak sex weight (~0.5): same-individual pairs always have matching sex, so there is no same-individual sex conflict to train from.

#### Lesson

- Direct evidence rules suit standalone classification.
- For re-ranking on top of a strong neural model, rules should condition on the current SameShark estimate, so biological evidence *modulates* neural similarity rather than overriding it uniformly.
- MLE is the wrong optimiser for weights on this type of constraint; see Ruleset B.

---

## Ruleset B — Sex Consistency + Symmetry (Propagation)

**Status:** Retired — hurts results (unknown-sex penalty problem)  
**Embeddings tested on:** `thesis_unmatch_random_out`

### Rules
```
neural_prior_forward:  NeuralSim(I1, I2) -> SameShark(I1, I2)               w=0.23
neural_prior_reverse:  SameShark(I1, I2) -> NeuralSim(I1, I2)               w=0.23
sex_consistency_m:     SameShark(I1, I2) & IsMale(I1) -> IsMale(I2)         w=10.0
sex_consistency_f:     SameShark(I1, I2) & IsFemale(I1) -> IsFemale(I2)     w=10.0
sex_symmetry:          SameShark(I1, I2) = SameShark(I2, I1)                w=10.0
```

- Targets: forward **and** backward pairs `(q -> n)` and `(n -> q)`.
- Sex encoding: hard (0.0, 0.0 for unknown sex).
- Weights: all fixed at DEFAULT_WEIGHTS; no MLE for sex rules.

### Penalty Mechanics

Consistency rules place SameShark in the body alongside the sex observation:

```
body = max(0, SameShark + IsMale(I1) − 1 − IsMale(I2))
```

- The penalty is *proportional to SameShark*: it fires hard only when the model already believes two images are the same shark AND their sexes disagree.
- For pairs ranked low by neural similarity (SameShark ≈ 0), the rule barely fires.
- This is the right behaviour for a re-ranker: only adjust plausible matches.

#### Symmetry and Backward Targets

The symmetry rule plus backward targets doubles the effective sex pressure:

- `(q -> n)`: `SameShark(q,n) & IsMale(q) -> IsMale(n)` fires
- `(n -> q)`: `SameShark(n,q) & IsFemale(n) -> IsFemale(q)` fires (via backward target)
- Symmetry forces `SameShark(q,n) = SameShark(n,q)`, so both penalties reinforce

#### Side Effect of One-Directional Observations

Only `NeuralSim(q,n)` is set; `NeuralSim(n,q)` defaults to 0.

- The symmetry rule pulls `SameShark(q,n)` slightly downward, because `SameShark(n,q)` has no upward pressure from the neural prior.
- All PSL scores end up in a compressed range (observed ~[0, 0.57]) rather than [0, 1].
- Relative ordering is preserved, so evaluation is unaffected (it uses list position, not absolute scores).

### Why Weights Are Fixed, Not MLE-Learned

Sex consistency rules cannot be learned from this data via MLE:
- Same-individual pairs *always* have matching sex (sharks have one sex)
- No same-individual sex-conflict pairs exist to provide MLE gradient
- MLE learns weight -> 0 since consistency constraints are never violated
  in the labelled data

This is fundamentally different from length or location rules, where
same-individual pairs can legitimately have different values. Sex (and the
symmetry constraint) must be hand-tuned or grid-searched.

### Results

| Metric  | Baseline | This ruleset |
|---------|----------|--------------|
| Hits@1  | 0.886    | **0.818** (−7.7%) |
| Hits@5  | 0.930    | 0.866            |
| Hits@10 | 0.950    | 0.879            |
| Avg K   | 10.12    | 11.30            |

### Why It Failed: the Unknown-Sex Penalty Problem

Consistency rules check `SameShark & IsMale(I1) -> IsMale(I2)`.

- With hard encoding (0.0 for unknown sex), an unknown-sex neighbor has `IsMale(I2) = 0.0`.
- The penalty is `max(0, SameShark + 1.0 − 1 − 0.0)² × 10 = SameShark² × 10`, **identical to a confirmed female neighbor**.
- All 17.6% unknown-sex images in the dataset are incorrectly treated as wrong-sex matches.

#### Encoding and Rule Direction

This is the opposite of the soft encoding problem in Ruleset A:

- Soft encoding + disagreement rules -> half-penalty on unknown (wrong)
- Hard encoding + consistency rules -> full-penalty on unknown (worse)
- **Hard encoding + disagreement rules -> zero penalty on unknown (correct)**

The disagreement rule `IsMale(I1) & IsFemale(I2) -> !SameShark` has `IsFemale(I2)` in the body. With hard encoding, unknown `IsFemale = 0.0` gives body `= max(0, 1.0 + 0.0 − 1) = 0`, so the rule is completely silent for unknown neighbors. This is the semantically correct behaviour.

#### Lesson

Rule direction and sex encoding are not independent choices; they must be matched.

- Consistency rules require a known-sex guard predicate or a different encoding to handle unknowns.
- Disagreement rules work correctly with hard (0.0) encoding out of the box.

---

## Ruleset C — Sex Disagreement, Hard Encoding, Fixed Weight 10.0

**Status:** Current / active  
**Embeddings tested on:** `thesis_unmatch_random_out`

### Rules
```
neural_prior_forward:  NeuralSim(I1, I2) -> SameShark(I1, I2)          w=0.23  (fixed)
neural_prior_reverse:  SameShark(I1, I2) -> NeuralSim(I1, I2)          w=0.23  (fixed)
sex_disagreement_mf:   IsMale(I1) & IsFemale(I2) -> !SameShark(I1, I2) w=10.0  (fixed)
sex_disagreement_fm:   IsFemale(I1) & IsMale(I2) -> !SameShark(I1, I2) w=10.0  (fixed)
```

- Targets: forward pairs only `(q -> n)`. No backward targets. No symmetry.
- Sex encoding: hard (0.0, 0.0 for unknown sex).
- Weights: all fixed, no MLE.
- Sex weight 10.0 is the same magnitude as the original working `best_unmatch_new_out_psl`, without the doubled symmetry effect (effectively 10.0 vs the original's effective ~20.0).

### Key Design Decisions vs. Ruleset A and B

- **vs. Ruleset A (disagreement + MLE):** Same rule structure, but weight is
  fixed at 10.0 instead of MLE-learned (~0.5). MLE cannot learn sex weights
  because same-individual pairs never have sex conflicts — no gradient signal.
- **vs. Ruleset B (consistency + symmetry):** Different rule direction.
  Disagreement rules correctly stay silent for unknown-sex pairs under hard
  encoding. Consistency rules incorrectly fire at full strength for unknown-sex.

### Results

| Metric  | Baseline | Ruleset A (MLE, w≈0.5) | This ruleset (fixed, w=10.0) |
|---------|----------|------------------------|------------------------------|
| Hits@1  | 0.886    | 0.881                  | **0.881**                    |
| Hits@5  | 0.930    | 0.925                  | **0.925**                    |
| Hits@10 | 0.950    | 0.947                  | **0.947**                    |
| Avg K   | 10.12    | 10.23                  | **10.23**                    |

Increasing sex weight from 0.5 -> 10.0 produced no change.

---

## Upcoming Experiments

- **Neural prior weight sweep** — grid search over `neural_prior_forward/reverse`
  in [0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 2.0] holding sex rules fixed at Ruleset B
  to find the optimal anchor strength.
- **Length rules** — add `length_agreement` / `length_disagreement` on top of Ruleset B.
  Need to decide: consistency propagation (like sex) or direct evidence? Length is
  continuous, so direct evidence (`SimilarLength -> SameShark`) is more natural —
  the disagreement direction is appropriate here unlike for binary sex.
- **Location and time rules** — same question: propagation vs. direct evidence.
- **Stacked rules** — combine the best-performing individual rules.
