"""Stage-1 diagnosis: it must observe the official INSID3 without changing it.

Checked on CPU with the fake encoder:
  * the prediction the diagnosis evaluates is the official predict_mask (several episodes, 1 and 2 shots)
  * every missed / wrong object pixel gets exactly one reason (the totals add up)
  * the oracles are real upper bounds and the cluster oracle is the exact optimum (brute force)
  * the logged c, i, a, S are the numbers of the official merge rule
  * the episode without any candidate patch is handled
  * the command-line loop: runs, resumes, writes clusters.csv / metrics.json, verification passes
"""

import copy
import csv
import itertools
import json
import os

import numpy as np
import pytest
import torch

from conftest import IMAGE_SIZE, ROOT, FakeEncoder, make_image, make_mask
from insid3_ext import datasets_ext, diagnosis
from insid3_ext.diagnosis import FN_REASONS, FP_SOURCES, VARIANT_NAMES, best_prefix, diagnose_episode, ratio_order
from insid3_ext.model import INSID3Ext
from models.insid3 import INSID3

KW = dict(image_size=IMAGE_SIZE, svd_components=8, tau=0.6, merge_threshold=0.2,
          mask_refiner="bilinear", resize_to_orig_size=False, device="cpu")


def _episode(seed, shots=1):
    refs = torch.cat([make_image(box=(16 + 4 * s, 16, 64 + 4 * s, 64), seed=10 + seed + 7 * s) for s in range(shots)])
    masks = torch.stack([make_mask((16 + 4 * s, 16, 64 + 4 * s, 64)) for s in range(shots)])
    tb = (40 + 4 * seed, 32, 104, 100 + 2 * seed)
    return refs, masks, make_image(box=tb, seed=20 + seed), make_mask(tb)


def _fiou(v):
    return v["inter"][1] / max(v["union"][1], 1)


@pytest.mark.parametrize("seed,shots", [(0, 1), (1, 1), (2, 1), (3, 2), (4, 2)])
def test_diagnosis_sees_the_official_prediction(encoder, seed, shots):
    official = INSID3(copy.deepcopy(encoder), **KW).eval()
    model = INSID3Ext(copy.deepcopy(encoder), **KW).eval()
    refs, masks, tgt, gt = _episode(seed, shots)
    rec, rows, maps = diagnose_episode(model, refs, masks, tgt, gt)
    assert set(rec["variants"]) == set(VARIANT_NAMES)
    expected = official.predict_mask(refs, masks, tgt)
    assert torch.equal(model._finalize_mask(maps["official_patch_mask"], tgt), expected)
    from utils.metrics import Evaluator
    inter, union = Evaluator.classify_prediction(expected, gt)
    assert rec["variants"]["actual"]["inter"] == inter.flatten().tolist()
    assert rec["variants"]["actual"]["union"] == union.flatten().tolist()


@pytest.mark.parametrize("seed", range(5))
def test_attribution_adds_up(encoder, seed):
    model = INSID3Ext(copy.deepcopy(encoder), **KW).eval()
    refs, masks, tgt, gt = _episode(seed)
    rec, rows, maps = diagnose_episode(model, refs, masks, tgt, gt)
    pred = maps["official_patch_mask"]
    gt_c, valid_c = maps["gt_c"], maps["valid_c"]
    assert sum(rec["fn"].values()) == pytest.approx(float(gt_c[~pred].sum()))
    assert sum(rec["fp"].values()) == pytest.approx(float((valid_c - gt_c)[pred].sum()))
    assert set(rec["fn"]) == set(FN_REASONS) and set(rec["fp"]) == set(FP_SOURCES)
    for r in rows:                                   # every GT cluster that was not kept has a reason
        if r["gt_pix"] > 0 and not r["selected"]:
            assert r["reason"] in FN_REASONS
        assert r["selected"] == (r["S"] > 0.2) or rec["float32_rule_mismatch_clusters"] > 0


@pytest.mark.parametrize("seed", range(5))
def test_logged_factors_are_the_official_merge_rule(encoder, seed):
    model = INSID3Ext(copy.deepcopy(encoder), **KW).eval()
    refs, masks, tgt, gt = _episode(seed)
    rec, rows, maps = diagnose_episode(model, refs, masks, tgt, gt)
    mt = maps["mt"]
    for r in rows:
        k = r["k"]
        assert r["S"] == pytest.approx(r["c"] * r["i"] * r["a"], abs=1e-5)
        assert r["S"] == pytest.approx(float(mt["combined"][k]), abs=1e-6)
        if not r["candidate"]:
            assert r["a"] == 0.0 and not r["selected"]
    seed_rows = [r for r in rows if r["is_seed"]]
    assert len(seed_rows) == 1 and seed_rows[0]["a"] == 1.0 and seed_rows[0]["i"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_oracles_are_upper_bounds(encoder, seed):
    model = INSID3Ext(copy.deepcopy(encoder), **KW).eval()
    refs, masks, tgt, gt = _episode(seed)
    rec, _, _ = diagnose_episode(model, refs, masks, tgt, gt)
    v = {k: _fiou(x) for k, x in rec["variants"].items()}
    tol = 0.02                                       # bilinear upsampling rounds patch corners
    assert v["ub_clusters_all"] + tol >= v["ub_clusters_candidates"]
    assert v["ub_clusters_candidates"] + tol >= v["ub_best_alpha"]
    assert v["ub_best_alpha"] + tol >= v["actual"]
    assert v["ub_clusters_all"] + tol >= max(v[n] for n in VARIANT_NAMES if not n.startswith("ub_patch"))


def test_ratio_prefix_is_the_exact_optimum():
    g = torch.Generator().manual_seed(0)
    for trial in range(200):
        K = int(torch.randint(1, 8, (1,), generator=g))
        n = torch.randint(1, 50, (K,), generator=g).double()
        gt = torch.floor(n * torch.rand(K, generator=g, dtype=torch.float64))
        G = float(gt.sum()) + float(torch.randint(0, 30, (1,), generator=g))
        if G == 0:
            continue
        best = 0.0
        for r in range(1, K + 1):
            for sub in itertools.combinations(range(K), r):
                s = list(sub)
                best = max(best, float(gt[s].sum() / (G + (n[s] - gt[s]).sum())))
        pick = best_prefix(ratio_order(list(range(K)), gt, n), gt, n, G)
        got = float(gt[pick].sum() / (G + (n[pick] - gt[pick]).sum())) if pick else 0.0
        assert got == pytest.approx(best, abs=1e-12)


def test_episode_without_candidates(encoder, monkeypatch):
    model = INSID3Ext(copy.deepcopy(encoder), **KW).eval()
    refs, masks, tgt, gt = _episode(0)
    zeros = lambda *a, **k: torch.zeros(IMAGE_SIZE // 16, IMAGE_SIZE // 16, dtype=torch.bool)  # noqa: E731
    monkeypatch.setattr(model, "_locate_candidates", zeros)
    monkeypatch.setattr(diagnosis, "candidate_halves",
                        lambda *a, **k: (zeros(), zeros(), torch.zeros(IMAGE_SIZE // 16, IMAGE_SIZE // 16)))
    rec, rows, maps = diagnose_episode(model, refs, masks, tgt, gt)
    assert rows == [] and set(rec["variants"]) == set(VARIANT_NAMES)
    assert rec["variants"]["actual"]["inter"][1] == 0
    assert rec["fn"]["no_candidate_anywhere"] == pytest.approx(float(gt.sum()))
    assert _fiou(rec["variants"]["ub_clusters_all"]) > 0


# ───────────── command-line loop ─────────────

class ToyDataset:
    class_ids = [0, 1]

    def __len__(self):
        return 6

    def __getitem__(self, idx):
        c = int(np.random.choice(self.class_ids))
        off = int(np.random.randint(0, 40))
        tb = (off, off, off + 56, off + 56)
        return {"ref_imgs": [make_image(box=(16, 16, 64, 64), seed=idx)],
                "ref_masks": [make_mask((16, 16, 64, 64)).float()],
                "tgt_img": make_image(box=tb, seed=100 + idx), "tgt_mask": make_mask(tb).float(),
                "class_id": torch.tensor(c)}


def test_cli_loop_runs_resumes_and_verifies(tmp_path, monkeypatch):
    import argparse
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import diagnose_segmentation as ds_cli
    from insid3_ext.builder import build_model
    from insid3_ext.config import load_config
    from insid3_ext.experiment import read_jsonl, run_dir

    monkeypatch.setattr(datasets_ext, "build_eval_dataset", lambda cfg: ToyDataset())
    over = [f"run.output_root={tmp_path}", "model.device=cpu", f"model.image_size={IMAGE_SIZE}",
            "model.svd_components=8", "run.exp_name=diag"]
    args = argparse.Namespace(verify_episodes=4, viz_ids="1", viz_auto=1)
    cfg = load_config(os.path.join(ROOT, "configs", "baseline.yaml"), over + ["run.max_episodes=3"])
    ds_cli.check_official(cfg)
    m = ds_cli.run_fold(cfg, build_model(cfg, encoder=FakeEncoder().eval()), args)
    assert m["n_episodes"] == 3 and m["verify_mismatches_total"] == 0
    cfg = load_config(os.path.join(ROOT, "configs", "baseline.yaml"), over)
    m = ds_cli.run_fold(cfg, build_model(cfg, encoder=FakeEncoder().eval()), args)
    out = run_dir(cfg)
    recs = read_jsonl(os.path.join(out, "episodes.jsonl"))
    assert len(recs) == 6 and m["complete"] and m["verified_total"] == 4 and m["verify_mismatches_total"] == 0
    with open(os.path.join(out, "clusters.csv")) as fp:
        rows = list(csv.DictReader(fp))
    assert rows and {int(r["idx"]) for r in rows} <= set(range(6))
    assert os.path.isfile(os.path.join(out, "figures", f"{recs[1]['size']}_00001.jpg"))
    with open(os.path.join(out, "metrics.json")) as fp:
        assert set(json.load(fp)["miou"]) == set(VARIANT_NAMES)

    # the diagnosed mIoU equals a normal baseline run on the same episodes
    from insid3_ext import experiment
    base = load_config(os.path.join(ROOT, "configs", "baseline.yaml"),
                       [f"run.output_root={tmp_path}", "model.device=cpu", f"model.image_size={IMAGE_SIZE}",
                        "model.svd_components=8", "run.exp_name=base"])
    mb = experiment.run_segmentation(base, model=build_model(base, encoder=FakeEncoder().eval()))
    assert m["miou"]["actual"] == pytest.approx(mb["miou"])


def test_refuses_modified_algorithm():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import diagnose_segmentation as ds_cli
    from insid3_ext.config import load_config
    cfg = load_config(os.path.join(ROOT, "configs", "baseline.yaml"), ["prototypes.enabled=true"])
    with pytest.raises(SystemExit):
        ds_cli.check_official(cfg)


# ───────────── the reason rule, branch by branch ─────────────

@pytest.mark.parametrize("args,expected", [
    ((0, 0.0, True, False, 0.5, 0.5, 0.5), "background"),
    ((10, 0.3, True, False, 0.5, 0.5, 0.5), "mixed_cluster"),
    ((10, 0.9, False, False, 0.5, 0.5, 0.0), "not_candidate"),
    ((10, 0.9, True, True, 0.1, 1.0, 1.0), "seed_rejected"),
    ((10, 0.9, True, False, -0.1, 0.5, 0.5), "negative_cross"),
    ((10, 0.9, True, False, 0.3, -0.2, 0.5), "negative_intra"),
    ((10, 0.9, True, False, 0.2, 0.6, 0.9), "low_cross"),
    ((10, 0.9, True, False, 0.6, 0.2, 0.9), "low_intra"),
    ((10, 0.9, True, False, 0.6, 0.9, 0.2), "low_area"),
])
def test_rejection_reason(args, expected):
    assert diagnosis.rejection_reason(*args) == expected


# ───────────── random cluttered scenes: invariants hold everywhere ─────────────

def _cluttered(seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda lo, hi: int(torch.randint(lo, hi, (1,), generator=g))   # noqa: E731
    col = lambda: tuple(float(x) for x in torch.rand(3, generator=g))   # noqa: E731
    obj = col()
    ref_box = (r(0, 40), r(0, 40), 0, 0)
    ref_box = (ref_box[0], ref_box[1], ref_box[0] + r(24, 80), ref_box[1] + r(24, 80))
    tgt_box = (r(0, 64), r(0, 64), 0, 0)
    tgt_box = (tgt_box[0], tgt_box[1], min(128, tgt_box[0] + r(16, 120)), min(128, tgt_box[1] + r(16, 120)))
    ref = make_image(color_box=obj, box=ref_box, bg=col(), seed=seed)
    tgt = make_image(color_box=obj, box=tgt_box, bg=col(), seed=seed + 1000)
    # a distractor square of another colour in the target
    d = (r(0, 100), r(0, 100))
    tgt[0, :, d[0]:d[0] + 24, d[1]:d[1] + 24] = torch.randn(3, 1, 1, generator=g)
    gt = make_mask(tgt_box)
    gt[d[0]:d[0] + 24, d[1]:d[1] + 24] = False
    return ref, make_mask(ref_box).unsqueeze(0), tgt, gt


@pytest.mark.parametrize("seed", range(30))
def test_invariants_on_random_scenes(encoder, seed):
    official = INSID3(copy.deepcopy(encoder), **KW).eval()
    model = INSID3Ext(copy.deepcopy(encoder), **KW).eval()
    ref, m, tgt, gt = _cluttered(seed)
    rec, rows, maps = diagnose_episode(model, ref, m, tgt, gt)
    assert torch.equal(model._finalize_mask(maps["official_patch_mask"], tgt), official.predict_mask(ref, m, tgt))
    pred = maps["official_patch_mask"]
    if "gt_c" in maps:
        assert sum(rec["fn"].values()) == pytest.approx(float(maps["gt_c"][~pred].sum()))
    v = {k: _fiou(x) for k, x in rec["variants"].items()}
    assert v["ub_clusters_all"] + 0.03 >= v["ub_clusters_candidates"] + 0.0
    assert v["ub_clusters_candidates"] + 0.03 >= v["ub_best_alpha"] >= v["actual"] - 0.03


def test_random_scenes_cover_many_reasons(encoder):
    """The random scenes reach most reasons, so the tests above are not only about easy cases."""
    model = INSID3Ext(copy.deepcopy(encoder), **KW).eval()
    seen = set()
    for seed in range(30):
        rec, rows, _ = diagnose_episode(model, *_cluttered(seed))
        seen |= {k for k, v in rec["fn"].items() if v > 0}
    assert len(seen) >= 3, seen


def test_analysis_report(tmp_path, monkeypatch):
    """scripts/analyze_diagnosis.py turns diagnosis runs into report.md / summary.json / tables."""
    import argparse
    import subprocess
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import diagnose_segmentation as ds_cli
    from insid3_ext.builder import build_model
    from insid3_ext.config import load_config

    monkeypatch.setattr(datasets_ext, "build_eval_dataset", lambda cfg: ToyDataset())
    for fold in (0, 1):
        cfg = load_config(os.path.join(ROOT, "configs", "baseline.yaml"),
                          [f"run.output_root={tmp_path}", "model.device=cpu", f"model.image_size={IMAGE_SIZE}",
                           "model.svd_components=8", "run.exp_name=diag", "data.dataset=pascal_part", f"data.fold={fold}"])
        ds_cli.run_fold(cfg, build_model(cfg, encoder=FakeEncoder().eval()),
                        argparse.Namespace(verify_episodes=2, viz_ids="", viz_auto=0))
    out = tmp_path / "report"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze_diagnosis.py"), "--out", str(out),
                    "--bootstrap", "20", "--run", f"Toy={tmp_path}/diag/pascal_part/fold*_shot1_seed0"],
                   check=True, cwd=ROOT)
    summary = json.load(open(out / "summary.json"))[0]
    assert summary["n_episodes"] == 12 and summary["verify_mismatches"] == 0
    ladder = summary["ladder"]["all"]
    # INSID3's own alpha is a member of the "best alpha" family, so that level can never be lower
    assert ladder["best alpha on S_k"] >= ladder["INSID3"] - 1e-9
    assert abs(sum(summary["fn"]["all"]["by_reason"].values()) - 1.0) < 1e-6 or summary["fn"]["all"]["missed_share_of_gt"] == 0
    text = open(out / "report.md").read()
    for section in ("0. Validity", "1. Oracle ladder", "2. Where the missed", "3. Merge factors", "4. Counterfactuals"):
        assert section in text
