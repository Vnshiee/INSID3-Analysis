"""Turn diagnosis runs (scripts/diagnose_segmentation.py) into the Stage-1 evidence report.

Usage
    python scripts/analyze_diagnosis.py --out results/diagnosis \
        --run "COCO tune=outputs/diag_baseline/coco-tune/fold0_shot1_seed0" \
        --run "PASCAL-Part test=outputs/diag_baseline/pascal_part/fold*_shot1_seed0" \
        --reference "COCO tune=results/round1/fail_spectral_k500_g4.csv"

    --run LABEL=GLOB        one or more fold directories (glob) that form one data set; repeatable
    --reference LABEL=CSV   optional per-episode table of an earlier baseline run on the same episodes
                            (columns idx, iou_a), to check that the diagnosis reproduces it

Writes into --out:  report.md, summary.json, tables/*.csv, figures/*.png

Every number is computed from the logs; the report states only what the numbers show. Sections:
  0 validity (verification, reproduction of the earlier baseline)
  1 oracle ladder: how far each stage of INSID3 is from the best it could do with its own regions
  2 where the missed object pixels go (one reason per pixel), by object size
  3 the merge factors c, i, a of object clusters that are rejected vs accepted, and how well each
    factor separates object clusters from background clusters (AUC)
  4 counterfactuals (one factor removed, other threshold, oracle seed) with paired bootstrap intervals
  5 seed and position checks
  6 representative failure episodes
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np

from insid3_ext.diagnosis import ALPHA_GRID, FN_REASONS, FP_SOURCES, PURITY_GRID
from insid3_ext.experiment import read_jsonl

SIZES = ("small", "medium", "large")
# Oracle ladder in mIoU. Each level is the best member of a family, chosen once for the whole scope
# (one threshold for all episodes; see insid3_ext/diagnosis.py for why per-episode optima are not used here).
LEVELS = {
    "patch grid": [f"ub_patch_p{t:.1f}" for t in PURITY_GRID],
    "best clusters": [f"ub_all_p{t:.1f}" for t in PURITY_GRID],
    "best candidate clusters": [f"ub_cand_p{t:.1f}" for t in PURITY_GRID],
    "best alpha on S_k": ["actual"] + [f"alpha_{t:.2f}" for t in ALPHA_GRID],
    "INSID3": ["actual"],
}
LADDER = tuple(LEVELS)
GAPS = (("clustering granularity", "patch grid", "best clusters"),
        ("candidate test", "best clusters", "best candidate clusters"),
        ("ranking by S_k", "best candidate clusters", "best alpha on S_k"),
        ("threshold alpha = 0.2", "best alpha on S_k", "INSID3"))
# per-episode views (mean episode IoU)
EPISODE_LADDER = ("ub_clusters_all", "ub_clusters_candidates", "ub_best_alpha", "actual")
CF = ["all_candidates", "oracle_seed", "cf_no_area", "cf_no_intra", "cf_no_cross", "cf_seed_always",
      "cf_intra_debiased"] + [f"alpha_{t:.2f}" for t in ALPHA_GRID]


# ───────────────────────────── loading ─────────────────────────────

def load(spec):
    label, pattern = spec.split("=", 1)
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isfile(os.path.join(d, "episodes.jsonl")))
    if not dirs:
        raise SystemExit(f"no diagnosis run found for '{pattern}'")
    eps, clusters = [], []
    for fold_i, d in enumerate(dirs):
        for r in read_jsonl(os.path.join(d, "episodes.jsonl")):
            r["_fold"] = fold_i
            r["_dir"] = d
            eps.append(r)
        path = os.path.join(d, "clusters.csv")
        if os.path.isfile(path):
            with open(path) as fp:
                for row in csv.DictReader(fp):
                    row["_fold"] = fold_i
                    clusters.append(row)
    return label.strip(), dirs, eps, clusters


def f(row, key):
    return float(row[key])


def b(row, key):
    return str(row[key]) in ("True", "true", "1")


# ───────────────────────────── metrics ─────────────────────────────

def miou(eps, variant, idx=None):
    """Official-style mIoU: class-wise IoU per fold, mean over the classes present, then mean over folds."""
    sel = eps if idx is None else [eps[i] for i in idx]
    by_fold = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r in sel:
        v = r["variants"][variant]
        acc = by_fold[r["_fold"]][r["class_id"]]
        acc[0] += v["inter"][1]
        acc[1] += v["union"][1]
    folds = [np.mean([i / max(u, 1.0) for i, u in cls.values()]) for cls in by_fold.values() if cls]
    return 100.0 * float(np.mean(folds)) if folds else float("nan")


def recall_precision(eps, variant):
    tp = sum(r["variants"][variant]["inter"][1] for r in eps)
    g = sum(r["gt_pixels"] for r in eps)
    p = sum(r["variants"][variant]["area"] for r in eps)
    return tp / max(g, 1), tp / max(p, 1)


def bootstrap_delta(eps, variant, ref="actual", n=1000, seed=0):
    """Paired bootstrap of mIoU(variant) - mIoU(ref), episodes resampled within each fold."""
    rng = np.random.default_rng(seed)
    folds = defaultdict(list)
    for i, r in enumerate(eps):
        folds[r["_fold"]].append(i)
    deltas = []
    for _ in range(n):
        idx = []
        for members in folds.values():
            idx.extend(rng.choice(members, size=len(members), replace=True).tolist())
        deltas.append(miou(eps, variant, idx) - miou(eps, ref, idx))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def auc(pos, neg):
    """Probability that a random positive scores higher than a random negative (ties count 1/2)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                         # average ranks of ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# ───────────────────────────── analysis of one data set ─────────────────────────────

def analyse(label, dirs, eps, clusters, reference, n_boot):
    S = {"label": label, "runs": dirs, "n_episodes": len(eps)}
    by_size = {s: [r for r in eps if r["size"] == s] for s in SIZES}
    S["n_by_size"] = {s: len(v) for s, v in by_size.items()}

    # 0. validity
    S["verified_episodes"] = sum(1 for r in eps if "verified_full_pipeline" in r)
    S["verify_mismatches"] = sum(1 for r in eps if r.get("verified_full_pipeline") is False)
    S["float32_rule_mismatch_clusters"] = sum(r.get("float32_rule_mismatch_clusters", 0) for r in eps)
    if reference:
        ref = {}
        with open(reference) as fp:
            for row in csv.DictReader(fp):
                ref[int(row["idx"])] = float(row["iou_a"])
        diffs = [abs(r["variants"]["actual"]["inter"][1] / max(r["variants"]["actual"]["union"][1], 1) - ref[r["idx"]])
                 for r in eps if r["idx"] in ref and r["_fold"] == 0]
        S["reference"] = {"file": reference, "compared": len(diffs),
                          "identical_iou_share": float(np.mean([d < 1e-6 for d in diffs])) if diffs else None,
                          "mean_abs_iou_diff": float(np.mean(diffs)) if diffs else None}

    # 1. ladder
    S["ladder"] = {}
    for scope, sub in [("all", eps)] + list(by_size.items()):
        if not sub:
            continue
        row, chosen = {}, {}
        for level, members in LEVELS.items():
            vals = {m: miou(sub, m) for m in members}
            chosen[level] = max(vals, key=vals.get)
            row[level] = vals[chosen[level]]
        row["chosen"] = chosen
        row["gaps"] = {name: row[hi] - row[lo] for name, hi, lo in GAPS}
        row["recall_precision"] = {level: recall_precision(sub, chosen[level]) for level in LEVELS}
        row["mean_episode_iou"] = {v: float(np.mean([r["variants"][v]["inter"][1] / max(r["variants"][v]["union"][1], 1)
                                                     for r in sub])) for v in EPISODE_LADDER}
        S["ladder"][scope] = row

    # 2. missed pixels and wrong pixels
    S["fn"], S["fp"], S["fn_episode_dominant"] = {}, {}, {}
    for scope, sub in [("all", eps)] + list(by_size.items()):
        if not sub:
            continue
        tot = {k: sum(r["fn"].get(k, 0.0) for r in sub) for k in FN_REASONS}
        g = sum(r["gt_pixels"] for r in sub)
        missed = sum(tot.values())
        S["fn"][scope] = {"missed_share_of_gt": missed / max(g, 1),
                          "by_reason": {k: v / max(missed, 1) for k, v in tot.items()}}
        fpt = {k: sum(r["fp"].get(k, 0.0) for r in sub) for k in FP_SOURCES}
        S["fp"][scope] = {k: v / max(sum(fpt.values()), 1) for k, v in fpt.items()}
        dom = defaultdict(int)
        for r in sub:
            if sum(r["fn"].values()) > 0:
                dom[max(r["fn"], key=r["fn"].get)] += 1
        S["fn_episode_dominant"][scope] = dict(dom)

    # 3. factors of object clusters (candidate, not the seed)
    S["factors"] = {}
    for scope in ("all",) + SIZES:
        rows = [c for c in clusters if scope == "all" or c["size"] == scope]
        gt_c = [c for c in rows if b(c, "is_gt") and b(c, "candidate") and not b(c, "is_seed")]
        bg_c = [c for c in rows if f(c, "purity") < 0.1 and b(c, "candidate") and not b(c, "is_seed")]
        rej = [c for c in gt_c if not b(c, "selected")]
        acc = [c for c in gt_c if b(c, "selected")]
        if not gt_c:
            continue
        w = np.array([f(c, "gt_pix") for c in rej]) if rej else np.zeros(0)

        def med(lst, k):
            return float(np.median([f(c, k) for c in lst])) if lst else float("nan")

        def wshare(flag):
            return float((w * np.array([b(c, flag) for c in rej])).sum() / max(w.sum(), 1)) if rej else float("nan")

        S["factors"][scope] = {
            "n_gt_candidate_clusters": len(gt_c), "n_rejected": len(rej), "n_background_candidate_clusters": len(bg_c),
            "median": {grp: {k: med(lst, k) for k in ("c", "i", "a", "S", "i_debiased", "dist_to_seed")}
                       for grp, lst in (("gt_accepted", acc), ("gt_rejected", rej), ("bg_candidates", bg_c))},
            "rejected_pixel_share_fixable_by_one_factor": {"c": wshare("fix_by_c"), "i": wshare("fix_by_i"),
                                                          "a": wshare("fix_by_a")},
            "auc_gt_vs_background": {k: auc([f(c, k) for c in gt_c], [f(c, k) for c in bg_c])
                                     for k in ("c", "i", "a", "S", "i_debiased", "seed_score")},
            "spearman_with_distance_to_seed": {
                "i": spearman(np.array([f(c, "dist_to_seed") for c in gt_c]), np.array([f(c, "i") for c in gt_c])),
                "i_debiased": spearman(np.array([f(c, "dist_to_seed") for c in gt_c]),
                                       np.array([f(c, "i_debiased") for c in gt_c]))},
            "background_candidates_accepted_share": float(np.mean([b(c, "selected") for c in bg_c])) if bg_c else float("nan"),
        }

    # 4. counterfactuals
    S["counterfactuals"] = {}
    for v in CF:
        entry = {"miou": miou(eps, v), "delta": miou(eps, v) - miou(eps, "actual"),
                 "recall_precision": recall_precision(eps, v),
                 "by_size_delta": {s: miou(sub, v) - miou(sub, "actual") for s, sub in by_size.items() if sub}}
        if n_boot:
            entry["ci95"] = bootstrap_delta(eps, v, n=n_boot)
        S["counterfactuals"][v] = entry

    # 5. seed
    S["seed"] = {}
    for scope, sub in [("all", eps)] + list(by_size.items()):
        sub = [r for r in sub if r.get("K", 0) and r.get("seed_cluster", -1) >= 0]
        if sub:
            S["seed"][scope] = {"seed_is_object_cluster": float(np.mean([r["seed_is_gt"] for r in sub])),
                                "seed_not_kept": float(np.mean([not r["seed_selected"] for r in sub])),
                                "median_seed_purity": float(np.median([r["seed_purity"] for r in sub]))}

    # 6. representative failures: most missed object pixels, per size
    S["worst"] = {}
    for s, sub in by_size.items():
        top = sorted(sub, key=lambda r: -sum(r["fn"].values()))[:10]
        S["worst"][s] = [{"idx": r["idx"], "fold_dir": r["_dir"], "class_id": r["class_id"],
                          "iou": r["variants"]["actual"]["inter"][1] / max(r["variants"]["actual"]["union"][1], 1),
                          "recall": r["variants"]["actual"]["inter"][1] / max(r["gt_pixels"], 1),
                          "ub_clusters_all": r["variants"]["ub_clusters_all"]["inter"][1] / max(r["variants"]["ub_clusters_all"]["union"][1], 1),
                          "dominant_reason": max(r["fn"], key=r["fn"].get) if sum(r["fn"].values()) else "",
                          "figure": os.path.join(r["_dir"], r["figure"]) if r.get("figure") else ""} for r in top]
    return S


# ───────────────────────────── report ─────────────────────────────

def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def num(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def verdict_lines(S):
    """Plain statements generated from the numbers (no hand-written conclusions)."""
    out = []
    if S["verify_mismatches"]:
        out.append(f"**INVALID**: {S['verify_mismatches']} episodes differ from the official predict_mask; "
                   "do not use the numbers below until this is explained.")
    for scope in ("large", "medium", "small", "all"):
        fn = S["fn"].get(scope)
        if not fn:
            continue
        top = sorted(fn["by_reason"].items(), key=lambda kv: -kv[1])[:2]
        gaps = S["ladder"][scope]["gaps"]
        g = max(gaps, key=gaps.get)
        out.append(f"- **{scope}** objects: INSID3 misses {pct(fn['missed_share_of_gt'])} of the object pixels. "
                   f"Largest reasons: {top[0][0]} ({pct(top[0][1])}), {top[1][0]} ({pct(top[1][1])}). "
                   f"Largest gap in the oracle ladder: {g} ({num(gaps[g])} mIoU).")
    label_free = {k: v for k, v in S["counterfactuals"].items() if k.startswith(("cf_", "alpha_"))}
    for name, e in sorted(label_free.items(), key=lambda kv: -kv[1]["delta"])[:3]:
        ci = e.get("ci95")
        out.append(f"- Counterfactual {name}: {e['delta']:+.2f} mIoU"
                   + (f" (95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}])" if ci else "")
                   + f", recall {pct(e['recall_precision'][0])}, precision {pct(e['recall_precision'][1])}.")
    if "oracle_seed" in S["counterfactuals"]:
        e = S["counterfactuals"]["oracle_seed"]
        out.append(f"- Oracle seed (uses the ground truth to pick the seed): {e['delta']:+.2f} mIoU.")
    out.append("- Counterfactuals are measurements of the existing rule, not proposals.")
    return out


def write_report(path, summaries):
    L = ["# Stage 1: root-cause diagnosis of INSID3 (official algorithm, unchanged)", "",
         "Generated by `scripts/analyze_diagnosis.py` from `scripts/diagnose_segmentation.py` runs. "
         "Size buckets: object share of the image < 5 % (small), 5-25 % (medium), > 25 % (large). "
         "A *GT cluster* is a target cluster with at least 50 % object pixels. mIoU = official class-wise "
         "definition per fold, mean over folds (within a size bucket: over the classes present).", ""]
    for S in summaries:
        L += [f"## {S['label']}", "", f"Runs: {', '.join(S['runs'])}  ", f"Episodes: {S['n_episodes']} "
              f"(small {S['n_by_size']['small']}, medium {S['n_by_size']['medium']}, large {S['n_by_size']['large']})", ""]
        L += ["### Summary (computed)", ""] + verdict_lines(S) + [""]

        L += ["### 0. Validity", "",
              "- Merge decision re-computed and compared with the official `_seed_and_aggregate`: every episode "
              "(the run stops on any difference).",
              f"- End-to-end comparison with the official `predict_mask`: {S['verified_episodes']} episodes, "
              f"{S['verify_mismatches']} mismatches.",
              f"- float32 re-computation of the rule vs the official bf16 decision: "
              f"{S['float32_rule_mismatch_clusters']} cluster decisions differ in total (affects only counterfactuals)."]
        if "reference" in S:
            r = S["reference"]
            L.append(f"- Same episodes as `{r['file']}`: {r['compared']} compared, identical IoU in "
                     f"{pct(r['identical_iou_share'])}, mean |ΔIoU| {num(r['mean_abs_iou_diff'], 4)}.")
        L.append("")

        L += ["### 1. Oracle ladder (mIoU)", "",
              "Each level removes one decision of INSID3 and replaces it by the best possible choice (oracle, uses "
              "the ground truth). *patch grid*: every patch whose object share exceeds a threshold. *best clusters*: "
              "every target cluster whose object share exceeds a threshold. *best candidate clusters*: the same, "
              "only among candidate clusters. *best alpha on S_k*: INSID3's own score with the best single "
              "threshold. Each level takes the best threshold for the whole scope.", "",
              "| scope | patch grid | best clusters | best candidate clusters | best alpha on S_k | INSID3 | recall / precision of INSID3 |",
              "|---|---|---|---|---|---|---|"]
        for scope, row in S["ladder"].items():
            rp = row["recall_precision"]["INSID3"]
            L.append(f"| {scope} | " + " | ".join(num(row[v]) for v in LADDER) + f" | {pct(rp[0])} / {pct(rp[1])} |")
        L += ["", "Chosen member of each family (purity threshold / alpha): " +
              "; ".join(f"{scope}: " + ", ".join(row["chosen"][lv] for lv in LADDER[:-1]) for scope, row in S["ladder"].items()), "",
              "Mean per-episode IoU (per-episode optima): ", "",
              "| scope | " + " | ".join(EPISODE_LADDER) + " |", "|---|" + "---|" * len(EPISODE_LADDER)]
        for scope, row in S["ladder"].items():
            L.append(f"| {scope} | " + " | ".join(num(row["mean_episode_iou"][v], 3) for v in EPISODE_LADDER) + " |")
        L += ["", "| scope | " + " | ".join(g[0] for g in GAPS) + " |", "|---|" + "---|" * len(GAPS)]
        for scope, row in S["ladder"].items():
            L.append(f"| {scope} | " + " | ".join(num(row["gaps"][g[0]]) for g in GAPS) + " |")
        L.append("")

        L += ["### 2. Where the missed object pixels go (share of missed pixels)", "",
              "| reason | " + " | ".join(S["fn"].keys()) + " |", "|---|" + "---|" * len(S["fn"])]
        L.append("| *missed share of all object pixels* | " + " | ".join(pct(v["missed_share_of_gt"]) for v in S["fn"].values()) + " |")
        for k in FN_REASONS:
            L.append(f"| {k} | " + " | ".join(pct(v["by_reason"][k]) for v in S["fn"].values()) + " |")
        L += ["", "Wrong (false-positive) pixels by source:", "",
              "| source | " + " | ".join(S["fp"].keys()) + " |", "|---|" + "---|" * len(S["fp"])]
        for k in FP_SOURCES:
            L.append(f"| {k} | " + " | ".join(pct(v[k]) for v in S["fp"].values()) + " |")
        L.append("")

        L += ["### 3. Merge factors of object clusters (candidate, not the seed)", ""]
        for scope, fa in S["factors"].items():
            L += [f"**{scope}** — {fa['n_gt_candidate_clusters']} object clusters ({fa['n_rejected']} rejected), "
                  f"{fa['n_background_candidate_clusters']} background candidate clusters (purity < 0.1), "
                  f"{pct(fa['background_candidates_accepted_share'])} of them accepted.", "",
                  "| median | c | i | a | S | i (debiased) | distance to seed (patches) |", "|---|---|---|---|---|---|---|"]
            for grp, m in fa["median"].items():
                L.append(f"| {grp} | " + " | ".join(num(m[k], 3) for k in ("c", "i", "a", "S", "i_debiased", "dist_to_seed")) + " |")
            fx = fa["rejected_pixel_share_fixable_by_one_factor"]
            au = fa["auc_gt_vs_background"]
            sp = fa["spearman_with_distance_to_seed"]
            L += ["", f"Rejected object pixels that one factor alone would rescue (that factor set to 1): "
                  f"c {pct(fx['c'])}, i {pct(fx['i'])}, a {pct(fx['a'])}.  ",
                  "AUC object vs background clusters: " + ", ".join(f"{k} {num(v, 3)}" for k, v in au.items()) + ".  ",
                  f"Spearman correlation with distance to the seed: i {num(sp['i'], 3)}, i (debiased) {num(sp['i_debiased'], 3)}.", ""]

        L += ["### 4. Counterfactuals (measurements, not proposals)", "",
              "| variant | mIoU | Δ vs INSID3 | 95% CI | recall | precision | Δ small | Δ medium | Δ large |",
              "|---|---|---|---|---|---|---|---|---|"]
        for v, e in S["counterfactuals"].items():
            ci = e.get("ci95")
            L.append(f"| {v} | {num(e['miou'])} | {e['delta']:+.2f} | "
                     + (f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "") + f" | {pct(e['recall_precision'][0])} | "
                     f"{pct(e['recall_precision'][1])} | " + " | ".join(f"{e['by_size_delta'].get(s, float('nan')):+.2f}" for s in SIZES) + " |")
        L.append("")

        L += ["### 5. Seed", "", "| scope | seed is an object cluster | seed not kept | median seed purity |", "|---|---|---|---|"]
        for scope, sd in S["seed"].items():
            L.append(f"| {scope} | {pct(sd['seed_is_object_cluster'])} | {pct(sd['seed_not_kept'])} | {num(sd['median_seed_purity'])} |")
        L.append("")

        L += ["### 6. Representative failure episodes (most missed object pixels)", "",
              "| size | episode | IoU | recall | best clusters IoU | dominant reason | figure |", "|---|---|---|---|---|---|---|"]
        for s, lst in S["worst"].items():
            for w in lst[:5]:
                L.append(f"| {s} | {w['idx']} | {num(w['iou'])} | {num(w['recall'])} | {num(w['ub_clusters_all'])} | "
                         f"{w['dominant_reason']} | {w['figure']} |")
        L.append("")
    with open(path, "w") as fp:
        fp.write("\n".join(L) + "\n")


def write_figures(out, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for S in summaries:
        scopes = [s for s in SIZES if s in S["fn"]]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bottom = np.zeros(len(scopes))
        for k in FN_REASONS:
            vals = np.array([100 * S["fn"][s]["by_reason"][k] * S["fn"][s]["missed_share_of_gt"] for s in scopes])
            if vals.sum() == 0:
                continue
            ax.bar(scopes, vals, bottom=bottom, label=k)
            bottom += vals
        ax.set_ylabel("% of all object pixels that are missed")
        ax.set_title(f"{S['label']}: missed object pixels by reason")
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "figures", f"fn_by_reason_{S['label'].replace(' ', '_')}.png"), dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for s in ["all"] + scopes:
            ax.plot(range(len(LADDER)), [S["ladder"][s][v] for v in LADDER], marker="o", label=s)
        ax.set_xticks(range(len(LADDER)))
        ax.set_xticklabels(["patch grid", "best clusters", "best cand.\nclusters", "best alpha\non S_k", "INSID3"])
        ax.set_ylabel("mIoU")
        ax.set_title(f"{S['label']}: oracle ladder")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "figures", f"ladder_{S['label'].replace(' ', '_')}.png"), dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=GLOB")
    ap.add_argument("--reference", action="append", default=[], metavar="LABEL=CSV")
    ap.add_argument("--out", default="results/diagnosis")
    ap.add_argument("--bootstrap", type=int, default=1000, help="paired bootstrap resamples (0 = off)")
    args = ap.parse_args()
    refs = dict(r.split("=", 1) for r in args.reference)
    refs = {k.strip(): v for k, v in refs.items()}
    summaries = []
    for spec in args.run:
        label, dirs, eps, clusters = load(spec)
        summaries.append(analyse(label, dirs, eps, clusters, refs.get(label), args.bootstrap))
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(summaries, fp, indent=1, default=float)
    for S in summaries:
        tag = S["label"].replace(" ", "_")
        with open(os.path.join(args.out, "tables", f"fn_by_reason_{tag}.csv"), "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["reason"] + list(S["fn"]))
            for k in FN_REASONS:
                w.writerow([k] + [f"{v['by_reason'][k]:.4f}" for v in S["fn"].values()])
        with open(os.path.join(args.out, "tables", f"ladder_{tag}.csv"), "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["scope"] + list(LADDER))
            for scope, row in S["ladder"].items():
                w.writerow([scope] + [f"{row[v]:.2f}" for v in LADDER])
        with open(os.path.join(args.out, "tables", f"counterfactuals_{tag}.csv"), "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["variant", "miou", "delta", "ci_lo", "ci_hi", "recall", "precision"] + [f"delta_{s}" for s in SIZES])
            for v, e in S["counterfactuals"].items():
                ci = e.get("ci95") or (float("nan"), float("nan"))
                w.writerow([v, f"{e['miou']:.2f}", f"{e['delta']:.2f}", f"{ci[0]:.2f}", f"{ci[1]:.2f}",
                            f"{e['recall_precision'][0]:.4f}", f"{e['recall_precision'][1]:.4f}"]
                           + [f"{e['by_size_delta'].get(s, float('nan')):.2f}" for s in SIZES])
    write_report(os.path.join(args.out, "report.md"), summaries)
    write_figures(args.out, summaries)
    print(f"wrote {args.out}/report.md")


if __name__ == "__main__":
    main()
