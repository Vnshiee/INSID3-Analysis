"""Stage-1 root-cause diagnosis of the ORIGINAL INSID3 segmentation (no algorithm change).

QUESTION
For every ground-truth object region that INSID3 misses: which step of the pipeline drops it?

HOW (per episode)
1. Run the official pipeline step by step, calling only the official methods of models/insid3.py
   (_extract_features, _debias_features, _locate_candidates, agglomerative_clustering,
   compute_cluster_prototypes, _seed_and_aggregate, _finalize_mask). The prediction we evaluate is the
   official one.
2. Re-compute the merge rule of the official _seed_and_aggregate (models/insid3.py L321-363) with the
   SAME lines, but keep the intermediate numbers of every target cluster k:
       c_k  cross-image score   = mean of <x_debiased, reference prototype> over the cluster's patches
       i_k  seed similarity     = <original cluster mean, original seed-cluster mean>
       a_k  area factor         = share of the cluster's patches that are candidates (seed forced to 1)
       S_k  final merge score   = c_k * i_k * a_k            cluster kept if S_k > alpha (0.2)
   The re-computed decision must equal the official decision bit for bit; this is checked on EVERY
   episode (verify_decisions), so the logged numbers are provably the ones INSID3 used.
3. Compare every cluster with the ground truth (GT): pixels, GT pixels, purity = GT share.
   A cluster with purity >= 0.5 is called a GT cluster.
4. Give every missed GT pixel exactly one reason (see FN_REASONS) and every wrong pixel a source.
5. Oracle upper bounds and counterfactuals: other patch masks built from the SAME clusters and scores,
   upsampled and scored with the official Evaluator exactly like the real prediction (VARIANTS).
   These are measurements only; INSID3's own prediction is never changed.

The GT is compared at the evaluation resolution (image_size x image_size, nearest-resized GT as in the
official evaluation). Each pixel belongs to the patch it lies in (16 x 16 pixels); the bilinear upsampling
of a patch mask moves only the corners of that assignment, so the pixel accounting is at "patch-membership"
level while every IoU in VARIANTS is the exact official pixel IoU.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import downsample_mask
from utils.metrics import Evaluator

GT_PURITY = 0.5                                    # a cluster is a "GT cluster" if >= 50 % of it is object
SIZE_BUCKETS = ((0.05, "small"), (0.25, "medium"), (float("inf"), "large"))   # GT share of the image
ALPHA_GRID = (0.05, 0.10, 0.15, 0.25, 0.30)       # counterfactual merge thresholds (official: 0.20)

PURITY_GRID = (0.3, 0.4, 0.5, 0.6)

# Every episode gets every variant (so that each variant has an mIoU over the same episodes).
#   actual                  the official INSID3 prediction
# Oracles (use the ground truth). Two kinds, because mIoU adds up intersections and unions over all
# episodes of a class: the per-episode optimum is NOT the mIoU optimum. The mIoU optimum keeps every
# region whose object share exceeds IoU*/(1 + IoU*) (Dinkelbach), i.e. a purity threshold that is the same
# for all episodes; the analysis takes the best threshold of each family at data-set level.
#   ub_patch_p<t>           every PATCH with object share >= t       (limit of any patch-level mask)
#   ub_all_p<t>             every target CLUSTER with purity >= t    (limit set by the clustering)
#   ub_cand_p<t>            every CANDIDATE cluster with purity >= t (limit set by clustering + candidate test)
#   ub_clusters_all         per-episode IoU optimum over all clusters (exact; for per-episode IoU)
#   ub_clusters_candidates  per-episode IoU optimum over candidate clusters
#   ub_best_alpha           per-episode best threshold on INSID3's own score S_k (limit of its ranking)
#   all_candidates          every candidate cluster (what the candidate test lets through)
#   oracle_seed             the official rule, but the seed is the candidate cluster with the most object
#   cf_no_area / cf_no_intra / cf_no_cross   the official rule with one factor removed (set to 1)
#   cf_seed_always          the official mask plus the seed cluster (paper Eq. 14 keeps the seed always)
#   cf_intra_debiased       seed similarity computed with debiased instead of original features
#   alpha_<t>               the official score with merge threshold t instead of 0.2
VARIANT_NAMES = (("actual", "ub_clusters_all", "ub_clusters_candidates", "ub_best_alpha", "all_candidates",
                  "oracle_seed", "cf_no_area", "cf_no_intra", "cf_no_cross", "cf_seed_always", "cf_intra_debiased")
                 + tuple(f"alpha_{t:.2f}" for t in ALPHA_GRID)
                 + tuple(f"ub_{fam}_p{t:.1f}" for fam in ("patch", "all", "cand") for t in PURITY_GRID))

# Reasons for missed GT pixels (exactly one per pixel; they sum to the patch-level FN count)
FN_REASONS = (
    "no_candidate_anywhere",    # not a single candidate patch in the target: official empty prediction
    "mixed_cluster",            # pixel lies in a cluster that is mostly background (purity < 0.5)
    "not_candidate",            # GT cluster without a single candidate patch (cannot be merged at all)
    "seed_rejected",            # GT cluster is the seed, but its own score c_seed <= alpha
    "negative_cross",           # GT candidate cluster with c_k <= 0
    "negative_intra",           # GT candidate cluster with i_k <= 0 (and c_k > 0)
    "low_cross",                # all factors > 0, S_k <= alpha, and c_k is the smallest factor
    "low_intra",                # ... i_k is the smallest factor
    "low_area",                 # ... a_k is the smallest factor
)
# Sources of wrong (false-positive) pixels
FP_SOURCES = ("fp_background_cluster",   # a selected cluster that is mostly background
              "fp_inside_gt_cluster")    # background pixels inside a selected GT cluster


def rejection_reason(gt_pix: float, purity: float, candidate: bool, is_seed: bool,
                     c: float, i: float, a: float) -> str:
    """Why a cluster that contains object pixels was NOT kept (rules in FN_REASONS, checked in this order)."""
    if gt_pix <= 0:
        return "background"
    if purity < GT_PURITY:
        return "mixed_cluster"
    if not candidate:
        return "not_candidate"
    if is_seed:
        return "seed_rejected"
    if c <= 0:
        return "negative_cross"
    if i <= 0:
        return "negative_intra"
    return ("low_cross", "low_intra", "low_area")[min(range(3), key=lambda j: (c, i, a)[j])]


def size_bucket(gt_fraction: float) -> str:
    for limit, name in SIZE_BUCKETS:
        if gt_fraction < limit:
            return name
    return SIZE_BUCKETS[-1][1]


# ════════════════════════════════════════════════════════════════════════════════════════
# 1. The official pipeline, step by step (only official methods)
# ════════════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def official_intermediates(model, ref_images: torch.Tensor, ref_masks: torch.Tensor,
                           tgt_image: torch.Tensor) -> Dict[str, Any]:
    """The official predict_mask (models/insid3.py L121-189) up to the merge step, keeping every tensor.

    Lines are copied from the official method; every computation is an official method call.
    """
    S = ref_images.shape[0]
    imgs = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)
    fmaps = model._extract_features(imgs)
    fmaps_norm = F.normalize(fmaps, p=2, dim=2)
    _, _, C, h, w = fmaps_norm.shape
    ref_masks = ref_masks.unsqueeze(1)
    feat_tgt = fmaps_norm[:, S]

    fmaps_debiased = model._debias_features(fmaps_norm)
    feat_refs_deb = fmaps_debiased[:, :S]
    feat_tgt_deb = fmaps_debiased[:, S]

    ref_prototypes = []
    for s in range(S):
        mask_s = downsample_mask(ref_masks[s:s + 1], h, w)
        fg = feat_refs_deb[0, s, :, mask_s]
        if fg.shape[1] > 0:
            ref_prototypes.append(fg.mean(dim=1))
    ref_prototype = F.normalize(torch.stack(ref_prototypes).mean(dim=0), p=2, dim=0).unsqueeze(1)

    sim_maps = []
    for m in range(S):
        sim_maps.append(torch.einsum('bchw,bcxy->bhwxy', feat_refs_deb[:, m], feat_tgt_deb))
    candidate_mask = model._locate_candidates(sim_maps, ref_masks, feat_tgt_deb, ref_prototype, h, w)

    forward_mask, backward_mask, sim_fwd = candidate_halves(sim_maps, ref_masks, feat_tgt_deb, ref_prototype, h, w)
    if not torch.equal(forward_mask & backward_mask, candidate_mask):
        raise RuntimeError("diagnosis: forward & backward does not reproduce _locate_candidates")

    out = dict(S=S, C=C, h=h, w=w, feat_tgt=feat_tgt, feat_tgt_deb=feat_tgt_deb, ref_prototype=ref_prototype,
               candidate_mask=candidate_mask, forward_mask=forward_mask, backward_mask=backward_mask,
               sim_fwd=sim_fwd, cluster_labels=None, K=0, cluster_protos=None)
    if candidate_mask.sum() == 0:                     # official early exit (L170-171): empty prediction
        return out

    feat_tgt_flat = feat_tgt[0].reshape(C, -1).permute(1, 0)
    cluster_labels = agglomerative_clustering(feat_tgt_flat, model.tau).reshape(h, w)
    K = int(cluster_labels.max().item()) + 1
    feat_tgt_deb_flat = feat_tgt_deb[0].reshape(C, -1).permute(1, 0)
    cluster_protos = compute_cluster_prototypes(feat_tgt_deb_flat, cluster_labels.view(-1), K)
    out.update(cluster_labels=cluster_labels, K=K, cluster_protos=cluster_protos)
    return out


def candidate_halves(sim_maps, ref_masks, feat_tgt_deb, ref_prototype, h: int, w: int):
    """The two halves of the official candidate test (models/insid3.py L285-305), kept apart for logging.

    forward:  <x, reference prototype> > 0 (else the top 10 %)
    backward: the nearest reference patch lies inside the reference mask (majority over the shots)
    """
    sim_fwd = torch.einsum('bchw,cd->bhw', feat_tgt_deb, ref_prototype).squeeze(0)
    forward_mask = sim_fwd > 0
    if forward_mask.sum() == 0:
        forward_mask = sim_fwd > float(torch.quantile(sim_fwd, 0.9))
    votes = torch.zeros((h, w), dtype=torch.int32, device=sim_maps[0].device)
    for m, sim_m in enumerate(sim_maps):
        sim0 = sim_m[0]
        Hs, Ws = sim0.shape[:2]
        best_idx = sim0.permute(2, 3, 0, 1).reshape(h, w, -1).argmax(dim=2)
        ref_mask_m = downsample_mask(ref_masks[m:m + 1], Hs, Ws).squeeze(0)
        votes += ref_mask_m[best_idx // Ws, best_idx % Ws].to(torch.int32)
    backward_mask = votes >= math.ceil(len(sim_maps) / 2)
    return forward_mask, backward_mask, sim_fwd


@torch.no_grad()
def official_merge_terms(model, it: Dict[str, Any]) -> Dict[str, Any]:
    """models/insid3.py L321-363 (_seed_and_aggregate), line by line, returning its intermediate numbers.

    Same dtypes and the same in-place operations as the official code, so `final_mask` is bit-identical.
    """
    candidate_mask, cluster_labels, K = it["candidate_mask"], it["cluster_labels"], it["K"]
    cluster_protos, ref_prototype = it["cluster_protos"], it["ref_prototype"]
    feat_tgt, feat_tgt_deb, h, w = it["feat_tgt"], it["feat_tgt_deb"], it["h"], it["w"]

    matched_mask = candidate_mask & (cluster_labels >= 0)
    matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)
    all_areas = cluster_labels[cluster_labels >= 0].unique(return_counts=True)[1]
    per_cluster = torch.zeros(K, device=cluster_labels.device)
    per_cluster[matched_ids] = n_pixels.float()
    area_weights = per_cluster / all_areas
    area_raw = area_weights.clone()                                    # before the seed is forced to 1

    protos_matched = cluster_protos[matched_ids]
    cross_sim_matched = (ref_prototype.T @ protos_matched.T).squeeze(0)
    seed_idx = int(torch.argmax(cross_sim_matched).item())
    seed_id = matched_ids[seed_idx].item()

    feat_tgt_flat = feat_tgt[0].reshape(feat_tgt.shape[1], -1).permute(1, 0)
    orig_protos = compute_cluster_prototypes(feat_tgt_flat, cluster_labels.view(-1), K)
    intra_sim = torch.einsum('c,kc->k', orig_protos[seed_id], orig_protos)

    fg_sim = torch.einsum('bchw,cd->bhw', feat_tgt_deb, ref_prototype).squeeze(0)
    cross_sim = torch.empty(K, device=fg_sim.device, dtype=fg_sim.dtype)
    for k in range(K):
        idx = (cluster_labels == k)
        cross_sim[k] = fg_sim[idx].mean() if idx.any() else 0.0

    combined = cross_sim * intra_sim
    area_weights[seed_id] = 1.0
    combined *= area_weights

    final_mask = torch.zeros(h, w, dtype=torch.bool, device=cluster_labels.device)
    valid = cluster_labels >= 0
    final_mask[valid] = combined[cluster_labels[valid]] > model.merge_threshold

    # Seed score of EVERY cluster (the official one is only computed for the matched clusters).
    seed_score_all = (ref_prototype.T @ cluster_protos.T).squeeze(0)
    # Seed similarity with DEBIASED cluster means (not used by INSID3; logged to test the position hypothesis).
    intra_deb = cluster_protos.float() @ cluster_protos[seed_id].float()
    return dict(matched_ids=matched_ids, area_raw=area_raw, area=area_weights, seed_id=int(seed_id),
                intra=intra_sim, cross=cross_sim, combined=combined, final_mask=final_mask,
                orig_protos=orig_protos, seed_score_all=seed_score_all, intra_deb=intra_deb)


# ════════════════════════════════════════════════════════════════════════════════════════
# 2. Ground truth per patch and per cluster
# ════════════════════════════════════════════════════════════════════════════════════════

def patch_counts(gt_px: torch.Tensor, h: int, w: int, ignore_px: Optional[torch.Tensor] = None):
    """GT pixels and valid pixels of every patch, (h, w) each. gt_px is (H, W) bool at evaluation size."""
    H, W = gt_px.shape
    if H % h or W % w:
        raise ValueError(f"evaluation size {H}x{W} is not a multiple of the patch grid {h}x{w}")
    fy, fx = H // h, W // w
    valid = torch.ones_like(gt_px, dtype=torch.bool) if ignore_px is None else ~(ignore_px > 0)
    gt = (gt_px > 0) & valid
    gt_c = gt.float().view(h, fy, w, fx).sum(dim=(1, 3))
    valid_c = valid.float().view(h, fy, w, fx).sum(dim=(1, 3))
    return gt_c, valid_c


def per_cluster_sums(labels: torch.Tensor, values: torch.Tensor, K: int) -> torch.Tensor:
    """Sum of `values` (h, w) over the patches of each cluster -> (K,) float64."""
    return torch.bincount(labels.reshape(-1), weights=values.reshape(-1).double(), minlength=K)


def best_prefix(order: Sequence[int], gt_pix: torch.Tensor, n_pix: torch.Tensor, G: float) -> List[int]:
    """Best union of clusters taken in the given order: the prefix with the highest IoU.

    IoU of a union of disjoint clusters = sum(gt) / (G + sum(non-gt)). Used for every oracle.
    """
    if len(order) == 0 or G <= 0:
        return []
    o = torch.as_tensor(list(order), dtype=torch.long)
    tp = torch.cumsum(gt_pix[o], 0)
    fp = torch.cumsum(n_pix[o] - gt_pix[o], 0)
    iou = tp / (G + fp)
    j = int(torch.argmax(iou))
    return o[: j + 1].tolist() if float(iou[j]) > 0 else []


def ratio_order(ids: Sequence[int], gt_pix: torch.Tensor, n_pix: torch.Tensor) -> List[int]:
    """Order that makes best_prefix the exact optimum over ALL subsets of `ids`.

    Maximising sum(a)/(G + sum(b)) over subsets: the optimum takes every cluster with a_k / b_k above the
    optimal value (Dinkelbach), i.e. a prefix of the clusters sorted by a_k / b_k (a = GT, b = non-GT).
    """
    a = gt_pix[list(ids)]
    b = (n_pix - gt_pix)[list(ids)]
    key = torch.where(b > 0, a / b.clamp_min(1e-12), torch.full_like(a, float("inf")))
    key = torch.where(a > 0, key, torch.full_like(a, -1.0))           # clusters without GT never help
    return [ids[i] for i in torch.argsort(key, descending=True).tolist()]


# ════════════════════════════════════════════════════════════════════════════════════════
# 3. One episode
# ════════════════════════════════════════════════════════════════════════════════════════

def _eval(model, patch_mask: torch.Tensor, tgt_image: torch.Tensor, gt_px: torch.Tensor, ignore_px):
    """Official upsampling (_finalize_mask) + official Evaluator; returns (inter[2], union[2], area)."""
    up = model._finalize_mask(patch_mask, tgt_image)
    inter, union = Evaluator.classify_prediction(up, gt_px, tgt_ignore_idx=ignore_px)
    return inter.flatten().cpu().tolist(), union.flatten().cpu().tolist(), int(up.sum())


def _mask_from_ids(labels: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    keep = torch.zeros(int(labels.max()) + 1, dtype=torch.bool, device=labels.device)
    if len(ids):
        keep[torch.as_tensor(list(ids), device=labels.device)] = True
    return keep[labels]


@torch.no_grad()
def diagnose_episode(model, ref_images: torch.Tensor, ref_masks: torch.Tensor, tgt_image: torch.Tensor,
                     gt_px: torch.Tensor, ignore_px: Optional[torch.Tensor] = None,
                     alpha_grid: Sequence[float] = ALPHA_GRID) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Diagnose one episode.

    ref_images (S, 3, H, W), ref_masks (S, H, W) bool, tgt_image (1, 3, H, W) as set by model.set_reference /
    set_target; gt_px (H, W) bool at the evaluation size. Returns (episode record, cluster rows, maps).
    """
    alpha = float(model.merge_threshold)
    it = official_intermediates(model, ref_images, ref_masks, tgt_image)
    h, w = it["h"], it["w"]
    gt_c, valid_c = patch_counts(gt_px, h, w, ignore_px)
    G = float(gt_c.sum())
    gt_frac = G / float(valid_c.sum())
    rec: Dict[str, Any] = {
        "gt_pixels": int(G), "gt_frac": gt_frac, "size": size_bucket(gt_frac),
        "forward_tokens": int(it["forward_mask"].sum()), "backward_tokens": int(it["backward_mask"].sum()),
        "candidate_tokens": int(it["candidate_mask"].sum()),
        "gt_tokens_majority": int((gt_c >= 0.5 * valid_c).sum()),
        "gt_candidate_tokens": int((it["candidate_mask"] & (gt_c >= 0.5 * valid_c)).sum()),
    }
    variants: Dict[str, Dict[str, Any]] = {}

    def add(name: str, patch_mask: torch.Tensor) -> None:
        inter, union, area = _eval(model, patch_mask, tgt_image, gt_px, ignore_px)
        variants[name] = {"inter": inter, "union": union, "area": area}

    # Patch-grid oracles (independent of the clustering).
    for t in PURITY_GRID:
        add(f"ub_patch_p{t:.1f}", (gt_c >= t * valid_c.clamp_min(1)) & (gt_c > 0))

    if it["cluster_labels"] is None:
        # No candidate patch at all: the official prediction is empty and nothing can be merged. The
        # clustering-only oracles are still computed (official clustering), every other variant is empty.
        empty = torch.zeros(h, w, dtype=torch.bool, device=it["candidate_mask"].device)
        feat_tgt_flat = it["feat_tgt"][0].reshape(it["C"], -1).permute(1, 0)
        labels = agglomerative_clustering(feat_tgt_flat, model.tau).reshape(h, w)
        K = int(labels.max().item()) + 1
        n_pix = per_cluster_sums(labels, valid_c, K).cpu()
        gt_pix = per_cluster_sums(labels, gt_c, K).cpu()
        purity = gt_pix / n_pix.clamp_min(1)
        add("ub_clusters_all", _mask_from_ids(labels, best_prefix(ratio_order(list(range(K)), gt_pix, n_pix), gt_pix, n_pix, G)))
        for t in PURITY_GRID:
            add(f"ub_all_p{t:.1f}", _mask_from_ids(labels, [k for k in range(K) if float(purity[k]) >= t and gt_pix[k] > 0]))
        for name in VARIANT_NAMES:
            if name not in variants:
                add(name, empty)
        fn = {r: 0.0 for r in FN_REASONS}
        fn["no_candidate_anywhere"] = G
        rec.update(K=K, n_matched=0, seed_cluster=-1, seed_purity=0.0, seed_is_gt=False, seed_selected=False,
                   seed_cross=0.0, n_gt_clusters=int((purity >= GT_PURITY).sum()), n_gt_clusters_selected=0,
                   n_selected=0, fn=fn, fp={s: 0.0 for s in FP_SOURCES}, float32_rule_mismatch_clusters=0,
                   oracle_seed_purity=0.0)
        rec["variants"] = variants
        return rec, [], {"labels": labels, "official_patch_mask": empty, "it": it}

    mt = official_merge_terms(model, it)
    official = model._seed_and_aggregate(it["candidate_mask"], it["cluster_labels"], it["cluster_protos"],
                                         it["K"], it["ref_prototype"], it["feat_tgt"], it["feat_tgt_deb"], h, w)
    if not torch.equal(official, mt["final_mask"]):
        raise RuntimeError("diagnosis: re-computed merge decision differs from the official _seed_and_aggregate")

    labels, K = it["cluster_labels"], it["K"]
    lab = labels.reshape(-1)
    n_patch = torch.bincount(lab, minlength=K).double()
    n_pix = per_cluster_sums(labels, valid_c, K)
    gt_pix = per_cluster_sums(labels, gt_c, K)
    purity = gt_pix / n_pix.clamp_min(1)
    fwd_n = per_cluster_sums(labels, it["forward_mask"].float(), K)
    bwd_n = per_cluster_sums(labels, it["backward_mask"].float(), K)
    cand_n = per_cluster_sums(labels, it["candidate_mask"].float(), K)
    matched = torch.zeros(K, dtype=torch.bool)
    matched[mt["matched_ids"].cpu()] = True
    c = mt["cross"].float().cpu()
    i = mt["intra"].float().cpu()
    a = mt["area"].float().cpu()
    Sk = mt["combined"].float().cpu()
    selected = torch.zeros(K, dtype=torch.bool)
    sel_ids = torch.unique(labels[official]).cpu() if official.any() else torch.zeros(0, dtype=torch.long)
    selected[sel_ids] = True
    seed = mt["seed_id"]
    n_pix, gt_pix, purity, fwd_n, bwd_n, cand_n, n_patch = (t.cpu() for t in (n_pix, gt_pix, purity, fwd_n, bwd_n, cand_n, n_patch))

    # Cluster geometry: centroid (in patches) and distance to the seed's centroid.
    ys, xs = torch.meshgrid(torch.arange(h, device=labels.device), torch.arange(w, device=labels.device), indexing="ij")
    cy = (per_cluster_sums(labels, ys.float(), K).cpu() / n_patch)
    cx = (per_cluster_sums(labels, xs.float(), K).cpu() / n_patch)
    dist_seed = torch.sqrt((cy - cy[seed]) ** 2 + (cx - cx[seed]) ** 2)

    # ── one reason per missed GT pixel ──
    fn = {r: 0.0 for r in FN_REASONS}
    fp = {s: 0.0 for s in FP_SOURCES}
    reasons: List[str] = [""] * K
    for k in range(K):
        g, nn_ = float(gt_pix[k]), float(n_pix[k])
        if selected[k]:
            is_gt = float(purity[k]) >= GT_PURITY
            fp["fp_inside_gt_cluster" if is_gt else "fp_background_cluster"] += nn_ - g
            reasons[k] = "selected"
            continue
        r = rejection_reason(g, float(purity[k]), bool(matched[k]), k == seed, float(c[k]), float(i[k]), float(a[k]))
        reasons[k] = r
        if r != "background":
            fn[r] += g

    # ── variants: oracles and counterfactuals (all evaluated exactly like the real prediction) ──
    add("actual", official)
    all_ids = list(range(K))
    cand_ids = mt["matched_ids"].cpu().tolist()
    add("ub_clusters_all", _mask_from_ids(labels, best_prefix(ratio_order(all_ids, gt_pix, n_pix), gt_pix, n_pix, G)))
    add("ub_clusters_candidates", _mask_from_ids(labels, best_prefix(ratio_order(cand_ids, gt_pix, n_pix), gt_pix, n_pix, G)))
    s_order = [k for k in torch.argsort(Sk, descending=True).tolist() if matched[k]]
    add("ub_best_alpha", _mask_from_ids(labels, best_prefix(s_order, gt_pix, n_pix, G)))
    for t in PURITY_GRID:
        add(f"ub_all_p{t:.1f}", _mask_from_ids(labels, [k for k in all_ids if float(purity[k]) >= t and gt_pix[k] > 0]))
        add(f"ub_cand_p{t:.1f}", _mask_from_ids(labels, [k for k in cand_ids if float(purity[k]) >= t and gt_pix[k] > 0]))
    add("all_candidates", _mask_from_ids(labels, cand_ids))

    a_area = a.clone()                       # counterfactual factor sets (float32; seed area already 1)
    m_f = matched.float()

    def keep(score: torch.Tensor, thr: float = alpha) -> torch.Tensor:
        return _mask_from_ids(labels, [k for k in all_ids if float(score[k]) > thr])

    add("cf_no_area", keep(c * i * m_f))                 # paper Eq. 13-14: area weight removed (candidates only)
    add("cf_no_intra", keep(c * a_area))
    add("cf_no_cross", keep(i * a_area))
    add("cf_seed_always", official | (labels == seed))   # paper Eq. 14: the seed is always kept
    for t in alpha_grid:
        add(f"alpha_{t:.2f}", keep(Sk, t))
    # Oracle seed: the candidate cluster with the most GT (by purity, then GT pixels); same merge rule.
    if cand_ids:
        best_seed = max(cand_ids, key=lambda k: (float(purity[k]), float(gt_pix[k])))
        i_o = mt["orig_protos"].float().cpu() @ mt["orig_protos"][best_seed].float().cpu()
        a_o = mt["area_raw"].float().cpu().clone()
        a_o[best_seed] = 1.0
        add("oracle_seed", keep(c * i_o * a_o))
        rec["oracle_seed_purity"] = float(purity[best_seed])
    else:                                          # cannot happen here (official code has >= 1 candidate)
        add("oracle_seed", official)
    # Seed similarity in the debiased space (position removed), everything else unchanged.
    add("cf_intra_debiased", keep(c * mt["intra_deb"].float().cpu() * a_area))
    # float32 re-computation of the official rule (should match; differences only from bf16 rounding)
    rec["float32_rule_mismatch_clusters"] = int(((Sk > alpha) != selected).sum())

    rec.update(K=K, n_matched=len(cand_ids), seed_cluster=seed, seed_purity=float(purity[seed]),
               seed_is_gt=bool(purity[seed] >= GT_PURITY), seed_selected=bool(selected[seed]),
               seed_cross=float(c[seed]), n_gt_clusters=int((purity >= GT_PURITY).sum()),
               n_gt_clusters_selected=int(((purity >= GT_PURITY) & selected).sum()),
               n_selected=int(selected.sum()), fn=fn, fp=fp)
    rec["variants"] = variants
    assert set(variants) == set(VARIANT_NAMES), sorted(set(VARIANT_NAMES) ^ set(variants))

    i_deb = mt["intra_deb"].float().cpu()                # one device->host copy instead of one per cluster
    seed_score_all = mt["seed_score_all"].float().cpu()
    rows = []
    for k in range(K):
        if gt_pix[k] == 0 and not matched[k] and not selected[k]:
            continue                                      # pure background, never a candidate: uninformative
        S_ck = float(Sk[k])
        rows.append({
            "k": k, "n_patches": int(n_patch[k]), "n_pix": float(n_pix[k]), "gt_pix": float(gt_pix[k]),
            "purity": float(purity[k]), "is_gt": bool(purity[k] >= GT_PURITY), "candidate": bool(matched[k]),
            "fwd_frac": float(fwd_n[k] / n_patch[k]), "bwd_frac": float(bwd_n[k] / n_patch[k]),
            "cand_frac": float(cand_n[k] / n_patch[k]),
            "c": float(c[k]), "i": float(i[k]), "a": float(a[k]), "S": S_ck,
            "i_debiased": float(i_deb[k]), "seed_score": float(seed_score_all[k]),
            "is_seed": k == seed, "selected": bool(selected[k]), "reason": reasons[k],
            "fix_by_c": bool(c[k] > 0 and i[k] > 0 and a[k] > 0 and S_ck / float(c[k]) > alpha),
            "fix_by_i": bool(c[k] > 0 and i[k] > 0 and a[k] > 0 and S_ck / float(i[k]) > alpha),
            "fix_by_a": bool(c[k] > 0 and i[k] > 0 and a[k] > 0 and S_ck / float(a[k]) > alpha),
            "dist_to_seed": float(dist_seed[k]), "cy": float(cy[k]) / h, "cx": float(cx[k]) / w,
        })
    return rec, rows, {"labels": labels, "official_patch_mask": official, "it": it, "mt": mt, "reasons": reasons, "purity": purity,
                       "selected": selected, "c": c, "i": i, "a": a, "S": Sk, "gt_c": gt_c, "valid_c": valid_c}
