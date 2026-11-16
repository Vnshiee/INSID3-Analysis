"""Stage-1 root-cause diagnosis of the ORIGINAL INSID3 (no change to the algorithm).

For every episode it logs, for every target cluster that contains object pixels or is a candidate:
candidate status, c_k (reference similarity), i_k (seed similarity), a_k (area factor), the merge score
S_k = c_k * i_k * a_k, whether it was kept, and the reason why a missed object cluster was rejected.
It also evaluates oracle upper bounds and counterfactuals (see insid3_ext/diagnosis.py, VARIANT_NAMES).

Guarantees checked while running
  * every episode: the re-computed merge decision equals the official _seed_and_aggregate bit for bit
  * the first --verify-episodes episodes: the final mask equals the official predict_mask end to end
  * same episodes as every other run (same seeding and order; fingerprint per episode)

Output (resumable; next to the usual run files):
  <output_root>/<exp_name>/<dataset>[-tune]/fold<f>_shot<k>_seed<s>/
      episodes.jsonl     one record per episode (fn / fp attribution, all variant inter/union, ...)
      clusters.csv       one row per informative target cluster
      metrics.json       mIoU of every variant
      figures/           representative failure episodes (JPEG)

Examples
    python scripts/diagnose_segmentation.py --set data.split=tune data.dataset=coco data.tune_episodes=1000
    python scripts/diagnose_segmentation.py --set data.dataset=pascal_part --folds all
"""
import argparse
import csv
import json
import os

import _bootstrap  # noqa: F401

import torch
import torch.nn.functional as F
from tqdm import tqdm

from insid3_ext.config import DATASET_FOLDS, load_config

CLUSTER_FIELDS = ["idx", "class_id", "size", "k", "n_patches", "n_pix", "gt_pix", "purity", "is_gt", "candidate",
                  "fwd_frac", "bwd_frac", "cand_frac", "c", "i", "a", "S", "i_debiased", "seed_score", "is_seed",
                  "selected", "reason", "fix_by_c", "fix_by_i", "fix_by_a", "dist_to_seed", "cy", "cx"]


def check_official(cfg) -> None:
    """The diagnosis is about the official algorithm: refuse any modified setting."""
    problems = []
    if cfg.debias.mode != "hard":
        problems.append(f"debias.mode={cfg.debias.mode} (official: hard)")
    if cfg.debias.center:
        problems.append("debias.center=true")
    if cfg.prototypes.enabled:
        problems.append("prototypes.enabled=true")
    if cfg.model.basis_eval_mode:
        problems.append("model.basis_eval_mode=true")
    if problems:
        raise SystemExit("diagnose_segmentation.py diagnoses the OFFICIAL INSID3 only; remove: " + ", ".join(problems))


def save_figure(path, ref_image, ref_mask, tgt_image, gt_px, maps, rec, rows) -> None:
    """Five panels: reference, target + GT, official prediction, cluster status, factor table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.data import denormalize

    def rgb(t):
        return denormalize(t.float().cpu()).clamp(0, 1).permute(1, 2, 0).numpy()

    def overlay(img, mask, color, alpha=0.5):
        out = img.copy()
        out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color)
        return out

    H, W = gt_px.shape
    ref = rgb(ref_image)
    tgt = rgb(tgt_image)
    gt = gt_px.cpu().numpy().astype(bool)
    labels = maps["labels"]
    pred = maps["official_patch_mask"]
    up = lambda m: F.interpolate(m[None, None].float(), size=(H, W), mode="nearest")[0, 0].cpu().numpy() > 0.5
    colours = {"selected_gt": (0.1, 0.8, 0.1), "selected_bg": (0.9, 0.1, 0.1), "not_candidate": (0.55, 0.2, 0.8),
               "seed_rejected": (0.5, 0.3, 0.1), "negative_cross": (1.0, 0.45, 0.0), "negative_intra": (1.0, 0.45, 0.0),
               "low_cross": (1.0, 0.55, 0.0), "low_intra": (1.0, 0.85, 0.0), "low_area": (1.0, 0.4, 0.7),
               "mixed_cluster": (0.6, 0.6, 0.6)}
    status = tgt * 0.35
    for row in rows:
        key = row["reason"]
        if key == "selected":
            key = "selected_gt" if row["is_gt"] else "selected_bg"
        if key in colours:
            status = overlay(status, up(labels == row["k"]), colours[key], 0.85)

    fig, ax = plt.subplots(1, 5, figsize=(26, 5.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.35]})
    ax[0].imshow(overlay(ref, ref_mask.cpu().numpy().astype(bool), (1.0, 0.5, 0.0)))
    ax[0].set_title("reference + mask")
    ax[1].imshow(overlay(tgt, gt, (0.0, 0.8, 1.0)))
    ax[1].set_title(f"target + GT ({100 * rec['gt_frac']:.0f}% of image, {rec['size']})")
    act = rec["variants"]["actual"]
    iou = act["inter"][1] / max(act["union"][1], 1)
    ax[2].imshow(overlay(tgt, up(pred), (0.1, 0.8, 0.1)))
    ax[2].contour(gt, levels=[0.5], colors="cyan", linewidths=1)
    ax[2].set_title(f"official prediction, IoU {iou:.2f}")
    ax[3].imshow(status)
    if labels is not None:
        ax[3].contour(up(labels == rec["seed_cluster"]), levels=[0.5], colors="white", linewidths=2)
    ax[3].contour(gt, levels=[0.5], colors="cyan", linewidths=1)
    ax[3].set_title("clusters: green kept/GT, red kept/bg, purple not candidate,\n"
                    "orange low c, yellow low i, pink low a, grey mixed; white = seed", fontsize=9)
    for a_ in ax[:4]:
        a_.axis("off")
    gt_rows = sorted([r for r in rows if r["gt_pix"] > 0], key=lambda r: -r["gt_pix"])[:14]
    lines = [f"{'k':>4} {'GT%':>5} {'pur':>5} {'cand':>4} {'c':>6} {'i':>6} {'a':>5} {'S':>6}  reason"]
    for r in gt_rows:
        lines.append(f"{r['k']:>4} {100 * r['gt_pix'] / max(rec['gt_pixels'], 1):>5.1f} {r['purity']:>5.2f} "
                     f"{'y' if r['candidate'] else 'n':>4} {r['c']:>6.3f} {r['i']:>6.3f} {r['a']:>5.2f} "
                     f"{r['S']:>6.3f}  {r['reason']}{' (seed)' if r['is_seed'] else ''}")
    fn = rec["fn"]
    tot = sum(fn.values()) or 1
    lines += ["", "missed object pixels by reason:"]
    lines += [f"  {k:22s} {100 * v / tot:5.1f}%" for k, v in sorted(fn.items(), key=lambda kv: -kv[1]) if v > 0]
    ax[4].text(0, 1, "\n".join(lines), family="monospace", fontsize=8.5, va="top")
    ax[4].axis("off")
    fig.suptitle(f"episode {rec['idx']}  class {rec['class_id']}  K={rec['K']} clusters, "
                 f"{rec.get('n_matched', 0)} candidate, seed purity {rec.get('seed_purity', 0):.2f}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=80, pil_kwargs={"quality": 80})   # JPEG keeps the archive small
    plt.close(fig)


def run_fold(cfg, model, args) -> dict:
    from insid3_ext.datasets_ext import build_eval_dataset
    from insid3_ext.diagnosis import VARIANT_NAMES, diagnose_episode
    from insid3_ext.experiment import (_log, _open_for_append, _prepare_run, compute_seg_metrics,
                                       episode_fingerprint, read_jsonl)
    from insid3_ext.runtime import CudaTimer

    out, jsonl, log = _prepare_run(cfg, "seg")
    ds = build_eval_dataset(cfg)
    done = {r["idx"]: r for r in read_jsonl(jsonl)}
    n_total = len(ds) if cfg.run.max_episodes is None else min(len(ds), cfg.run.max_episodes)
    _log(log, f"[diagnosis] run_dir={out}  episodes={n_total}  already_done={len(done)}")
    csv_path = os.path.join(out, "clusters.csv")
    new_csv = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    if not cfg.run.resume and os.path.isfile(csv_path):
        os.replace(csv_path, csv_path + ".bak")
        new_csv = True
    viz_ids = {int(v) for v in args.viz_ids.split(",") if v.strip()}
    viz_count = {"small": 0, "medium": 0, "large": 0}
    for r in done.values():                          # resume: count figures that already exist
        if r.get("figure_auto"):
            viz_count[r["size"]] += 1
    n_verified = n_mismatch = 0

    with _open_for_append(jsonl) as fp, open(csv_path, "a", newline="") as fc:
        writer = csv.DictWriter(fc, fieldnames=CLUSTER_FIELDS)
        if new_csv:
            writer.writeheader()
        for idx in tqdm(range(n_total), ncols=80):
            batch = ds[idx]                           # always drawn: keeps the episode sequence aligned
            fpr = episode_fingerprint(batch)
            if idx in done:
                if done[idx].get("fingerprint") != fpr:
                    raise RuntimeError(f"Episode {idx} fingerprint mismatch on resume; re-run with run.resume=false")
                continue
            for ref_img, ref_mask in zip(batch["ref_imgs"], batch["ref_masks"]):
                model.set_reference(ref_img, ref_mask)
            model.set_target(batch["tgt_img"])
            ref_images, ref_masks, tgt_image = model._ref_images, model._ref_masks, model._tgt_image
            model.reset_state()
            size = tuple(tgt_image.shape[-2:])
            gt_px = F.interpolate(batch["tgt_mask"].unsqueeze(0).unsqueeze(0).float(), size=size,
                                  mode="nearest").squeeze(0).squeeze(0) > 0.5
            ignore = batch.get("tgt_ignore_idx")
            with CudaTimer(cfg.model.device) as t:
                rec, rows, maps = diagnose_episode(model, ref_images, ref_masks, tgt_image,
                                                   gt_px.to(tgt_image.device), ignore)
            rec.update(idx=idx, fingerprint=fpr, class_id=int(batch["class_id"]), time_ms=t.ms)
            if idx < args.verify_episodes:
                # End-to-end check against the untouched official method.
                official = model.predict_mask(ref_images, ref_masks, tgt_image)
                mine = model._finalize_mask(maps["official_patch_mask"], tgt_image)
                same = bool(torch.equal(official.cpu(), mine.cpu()))
                rec["verified_full_pipeline"] = same
                n_verified += 1
                n_mismatch += int(not same)
                if not same:
                    _log(log, f"[diagnosis] WARNING episode {idx}: official predict_mask differs from the "
                              f"step-by-step pipeline ({int((official.cpu() != mine.cpu()).sum())} pixels)")
            act = rec["variants"]["actual"]
            g = rec["gt_pixels"]
            inter_fg = act["inter"][1]
            recall = inter_fg / g if g else 1.0
            iou = inter_fg / max(act["union"][1], 1)
            want = idx in viz_ids or (args.viz_auto > 0 and viz_count[rec["size"]] < args.viz_auto
                                      and iou < 0.6 and recall < 0.6 and g > 0)
            if want:
                if idx not in viz_ids:
                    viz_count[rec["size"]] += 1
                    rec["figure_auto"] = True
                fname = os.path.join(out, "figures", f"{rec['size']}_{idx:05d}.jpg")
                try:
                    save_figure(fname, ref_images[0], ref_masks[0], tgt_image[0], gt_px, maps,
                                rec, rows)
                    rec["figure"] = os.path.relpath(fname, out)
                except Exception as e:  # a figure must never stop the diagnosis
                    _log(log, f"[diagnosis] figure for episode {idx} failed: {e}")
            for row in rows:
                row.update(idx=idx, class_id=rec["class_id"], size=rec["size"])
                writer.writerow(row)
            fc.flush()
            fp.write(json.dumps(rec) + "\n")
            fp.flush()
            done[idx] = rec

    records = [done[i] for i in sorted(done) if i < n_total]
    metrics = {"n_episodes": len(records), "n_expected": len(ds), "complete": len(records) == len(ds),
               "verified_this_session": n_verified, "verify_mismatches_this_session": n_mismatch,
               "verified_total": sum(1 for r in records if "verified_full_pipeline" in r),
               "verify_mismatches_total": sum(1 for r in records if r.get("verified_full_pipeline") is False),
               "dataset": cfg.data.dataset, "split": cfg.data.split, "fold": cfg.data.fold, "shots": cfg.data.shots,
               "miou": {}}
    for name in VARIANT_NAMES:
        flat = [{"inter": r["variants"][name]["inter"], "union": r["variants"][name]["union"],
                 "class_id": r["class_id"]} for r in records]
        metrics["miou"][name] = compute_seg_metrics(flat, ds.class_ids, cfg.data.dataset)["miou"]
    with open(os.path.join(out, "metrics.json"), "w") as fm:
        json.dump(metrics, fm, indent=2)
    _log(log, "[diagnosis] mIoU  " + "  ".join(f"{k}={v:.2f}" for k, v in metrics["miou"].items()))
    _log(log, f"[diagnosis] end-to-end verification: {metrics['verified_total']} episodes, "
              f"{metrics['verify_mismatches_total']} mismatches")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--folds", default=None, help="'all' or comma list; default: data.fold of the config")
    ap.add_argument("--exp-name", default="diag_baseline")
    ap.add_argument("--verify-episodes", type=int, default=50,
                    help="compare with the official predict_mask end to end on the first N episodes")
    ap.add_argument("--viz-ids", default="", help="comma list of episode indices to draw")
    ap.add_argument("--viz-auto", type=int, default=12,
                    help="also draw the first N under-segmented episodes (IoU and recall < 0.6) per size bucket")
    args = ap.parse_args()

    overrides = args.set + [f"run.exp_name={args.exp_name}", "run.full_diagnostics=false"]
    cfg = load_config(args.config, overrides)
    check_official(cfg)
    if args.folds is None:
        folds = [cfg.data.fold]
    elif args.folds == "all":
        folds = list(range(DATASET_FOLDS.get(cfg.data.dataset, 1))) if cfg.data.split == "test" else [0]
    else:
        folds = [int(f) for f in args.folds.split(",")]

    from insid3_ext.builder import build_model
    from insid3_ext.runtime import resolve_device, set_seed

    cfg.model.device = resolve_device(cfg.model.device)
    set_seed(cfg.run.seed)                            # before the model, exactly like run_segmentation.py
    model = build_model(cfg)
    model.to(cfg.model.device)
    model.eval()
    summary = {}
    for f in folds:
        fold_cfg = load_config(args.config, overrides + [f"data.fold={f}"])
        fold_cfg.model.device = cfg.model.device
        summary[f] = run_fold(fold_cfg, model, args)
    print(json.dumps({f: {"actual": round(m["miou"]["actual"], 2), "ub_clusters_all": round(m["miou"]["ub_clusters_all"], 2),
                          "verify_mismatches": m["verify_mismatches_total"]} for f, m in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
