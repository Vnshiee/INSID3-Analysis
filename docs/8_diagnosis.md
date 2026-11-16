# 8. Stage 1 — root-cause diagnosis of INSID3 (no algorithm change)

**Goal.** Find out *where* in the official INSID3 pipeline ground-truth object regions are lost, before any
new method is proposed. Nothing in the segmentation algorithm is changed: the diagnosis runs the official
code and records what happens inside it.

Files: `insid3_ext/diagnosis.py` (logic), `scripts/diagnose_segmentation.py` (run),
`scripts/analyze_diagnosis.py` (report), `run_diagnosis.sh` (stages), `tests/test_diagnosis.py` (65 tests).

---

## 8.1 What INSID3 decides, and what is logged

INSID3 clusters the target image (agglomerative, τ = 0.6, original features) and then decides for every
target cluster *k* whether it joins the mask (`models/insid3.py` L311-364):

| quantity | official definition | logged as |
|---|---|---|
| candidate | the cluster contains ≥ 1 patch with ⟨x, p⟩ > 0 (forward) **and** whose nearest reference patch lies in the reference mask (backward) | `candidate`, `fwd_frac`, `bwd_frac`, `cand_frac` |
| c_k | mean of ⟨x_debiased, p⟩ over the cluster's patches (reference similarity) | `c` |
| seed | the candidate cluster whose debiased mean is most similar to p | `is_seed`, `seed_score` |
| i_k | ⟨original mean of k, original mean of the seed⟩ (seed similarity) | `i` |
| a_k | share of the cluster's patches that are candidates; 0 for non-candidates; 1 for the seed | `a` |
| S_k | c_k · i_k · a_k; the cluster is kept if S_k > α = 0.2 (also the seed) | `S`, `selected` |

Also logged (not used by INSID3): `i_debiased` (seed similarity with debiased features), distance to the
seed (`dist_to_seed`, in patches), centroid, and whether one factor alone would rescue a rejected cluster
(`fix_by_c/i/a`: S_k / factor > α).

**Proof that the logged numbers are INSID3's.** The merge rule is recomputed with the official lines
(same dtypes and in-place operations) and compared bit for bit with the official `_seed_and_aggregate`
on every episode; the run stops on any difference. On the first N episodes the final mask is also
compared with the untouched official `predict_mask` end to end (`verified_full_pipeline`). On the COCO
tuning split the episodes are the ones of the round-1 baseline, and the report compares the IoU of every
episode with `results/round1/fail_spectral_k500_g4.csv` (`iou_a`).

## 8.2 Ground truth per cluster, and one reason per missed pixel

The GT is resized to the evaluation size like the official evaluation (nearest). Every pixel belongs to
the 16 × 16 patch it lies in, hence to that patch's cluster. **Purity** = object share of a cluster;
a *GT cluster* has purity ≥ 0.5. Every missed object pixel gets exactly one reason (they add up to all
missed pixels; tested):

| reason | meaning |
|---|---|
| `no_candidate_anywhere` | the target has no candidate patch at all → official empty mask |
| `mixed_cluster` | the pixel lies in a cluster that is mostly background (purity < 0.5): a clustering problem |
| `not_candidate` | GT cluster without a single candidate patch: it can never be merged |
| `seed_rejected` | GT cluster is the seed but its own score c_seed ≤ 0.2 |
| `negative_cross` / `negative_intra` | GT candidate cluster with c_k ≤ 0 / i_k ≤ 0 |
| `low_cross` / `low_intra` / `low_area` | all factors > 0, S_k ≤ 0.2; the named factor is the smallest of the three |

Wrong pixels are split into `fp_background_cluster` (a kept cluster that is mostly background) and
`fp_inside_gt_cluster` (background pixels inside a kept GT cluster).

Size buckets (object share of the image): small < 5 %, medium 5-25 %, large > 25 %.

## 8.3 Oracle ladder (upper bounds that use the ground truth)

Every variant is a patch mask that is upsampled and scored exactly like the real prediction (official
`_finalize_mask` and `Evaluator`), on the same episodes.

| level | variant(s) | what it removes |
|---|---|---|
| patch grid | `ub_patch_p<t>` | everything except the 16-px patch grid: every patch with object share ≥ t |
| best clusters | `ub_all_p<t>` | INSID3's choice among its own clusters: every cluster with purity ≥ t |
| best candidate clusters | `ub_cand_p<t>` | the same, but only among candidate clusters |
| best alpha on S_k | `alpha_<t>`, `actual` | INSID3's own score with the best single threshold |
| INSID3 | `actual` | — |

Why thresholds and not per-episode optima: mIoU adds intersections and unions over all episodes of a
class, so the per-episode best mask does not maximise mIoU. Maximising Σ I / Σ U (Dinkelbach) keeps every
region whose object share exceeds IoU*/(1 + IoU*), i.e. one purity threshold for all episodes; the report
takes the best t ∈ {0.3, 0.4, 0.5, 0.6} of each family for each scope. The exact per-episode optima
(`ub_clusters_all`, `ub_clusters_candidates`, `ub_best_alpha`, proven optimal and tested by brute force)
are reported as mean episode IoU.

The **gaps** between consecutive levels say how much mIoU each decision costs: clustering granularity,
candidate test, ranking by S_k, and the fixed threshold α = 0.2.

## 8.4 Counterfactuals (measurements of the existing rule, not proposals)

| variant | change |
|---|---|
| `cf_no_area` | S = c · i for candidates (the paper's Eq. 13-14; the area weight exists only in the code, D5) |
| `cf_no_intra` / `cf_no_cross` | S = c · a / S = i · a |
| `cf_seed_always` | official mask ∪ seed cluster (paper Eq. 14 keeps the seed always; code D6) |
| `cf_intra_debiased` | i_k computed with debiased instead of original features (position removed) |
| `oracle_seed` | the seed is the candidate cluster with the highest purity (uses GT) |
| `alpha_<t>` | threshold t ∈ {0.05, 0.10, 0.15, 0.25, 0.30} instead of 0.2 |
| `all_candidates` | every candidate cluster (the recall ceiling of the candidate test) |

Each is reported with a paired bootstrap 95 % interval against INSID3 and with recall and precision, overall and
per size bucket.

## 8.5 How to run

```
STAGE=check ./run_diagnosis.sh       # tests + official files + 5-episode GPU smoke run
STAGE=coco_tune ./run_diagnosis.sh   # 1,000 COCO tuning episodes (= round-1 baseline episodes)
STAGE=pascal ./run_diagnosis.sh      # PASCAL-Part test, 4 folds, 1-shot
STAGE=coco_test ./run_diagnosis.sh   # COCO-20i test, 4 folds, 1-shot
STAGE=pascal5 ./run_diagnosis.sh     # optional: PASCAL-Part fold 1, 5-shot
STAGE=report ./run_diagnosis.sh      # -> results/diagnosis/report.md, summary.json, tables, figures
STAGE=pack ./run_diagnosis.sh        # -> diagnosis_results.tar.gz
```

All stages resume after an interruption. Outputs per fold: `episodes.jsonl` (one record per episode,
including the inter/union of every variant), `clusters.csv` (one row per informative cluster: every
cluster that has object pixels, is a candidate or is kept), `metrics.json` (mIoU of every variant),
`figures/` (representative under-segmented episodes: reference, target + GT, prediction, cluster status
map, factor table).

## 8.6 Reading the result — decision rules fixed BEFORE the runs

To avoid reading the numbers selectively, the interpretation is fixed now. "Dominant" means the largest
share of missed object pixels in that size bucket, **and** a matching gap in the ladder.

| if the evidence shows | it means | next experiment (Stage 2, tuning split first) |
|---|---|---|
| `mixed_cluster` dominates and the clustering gap is large | objects are merged with background during clustering (τ, original features) | τ sweep for the target clustering; CRF / boundary refinement; per-size analysis |
| `not_candidate` dominates and the candidate-test gap is large | the backward nearest-neighbour or forward test discards object parts | split by `fwd_frac` vs `bwd_frac`: which half fails; study scale mismatch reference ↔ target |
| `low_area` dominates and `cf_no_area` gains | the code-only area weight (D5) removes partly matched object clusters | paper rule (no area weight) and a soft area weight, with bootstrap on tuning, then test |
| `low_intra` dominates, i falls with distance to the seed, and `cf_intra_debiased` gains | positional content of the original features penalises distant object parts | position-reduced seed similarity; multi-seed / region-growing aggregation |
| `low_intra` dominates but `cf_intra_debiased` does not gain | the object really looks different from the seed part | multi-seed / self-support aggregation from the target (SSP-style) |
| `low_cross` dominates | object parts look unlike the reference prototype (appearance gap) | only then: richer reference description (calibrated multi-prototype or soft dense voting) |
| ranking gap large, threshold gap small | the product c · i · a orders clusters badly, α is not the issue | another way to combine the three factors (training-free, chosen on tuning) |
| threshold gap large | α = 0.2 is miscalibrated (possibly per object size) | α per size / adaptive α, chosen on tuning only |
| seed often not an object cluster and `oracle_seed` gains | wrong seed | seed selection (e.g. consensus of seed score and candidate density) |

If none of these patterns is clear, the conclusion is that the loss is spread over several stages, and
Stage 2 should start with the largest gap only.

## 8.7 Limits of the diagnosis

* Reasons are assigned at patch level (each pixel → its patch); the bilinear upsampling only rounds the
  corners, so the pixel totals differ slightly from the exact pixel FN. All IoUs are exact.
* Purity 0.5 defines a GT cluster; a mixed cluster can still contain many object pixels — the
  `mixed_cluster` share measures exactly that.
* One factor is changed at a time; the factors interact (e.g. a different seed changes every i_k).
* Oracles use the ground truth: they bound what is *possible* with INSID3's regions, not what a method can
  reach.
* COCO ground truth contains all instances of the class; under-segmentation here includes missed instances.
* The counterfactuals use a float32 re-computation of S; the number of cluster decisions that differ from
  the official bf16 decision is reported (`float32_rule_mismatch_clusters`).
