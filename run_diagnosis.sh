#!/usr/bin/env bash
# Stage 1 — root-cause diagnosis of the ORIGINAL INSID3 (the algorithm is not changed).
# Usage:   STAGE=<name> ./run_diagnosis.sh          (every stage is resumable: just start it again)
#
#   check       tests of the diagnosis code + official files unchanged + 5-episode GPU smoke run
#   coco_tune   1,000 COCO tuning episodes: the SAME episodes as the round-1 baseline, which is used to check
#               that the diagnosis reproduces it (results/round1/fail_spectral_k500_g4.csv)   (~15 min)
#   pascal      PASCAL-Part, 4 folds, 1-shot, official test episodes                          (~1.5-2 h)
#   coco_test   COCO-20i, 4 folds, 1-shot, official test episodes                            (~50-60 min)
#   pascal5     PASCAL-Part fold 1, 5-shot (where under-segmentation was strongest; optional)  (~20 min)
#   report      tables, figures and report.md   -> results/diagnosis/      (CPU, minutes)
#   pack        diagnosis_results.tar.gz (report + figures + COCO-tune logs) to send back
#   all         check, coco_tune, pascal, coco_test, report, pack
#
# Outputs: $INSID3_OUTPUT_ROOT/diagnosis/diag_baseline/<dataset>/fold<f>_shot<k>_seed0/
#            episodes.jsonl  clusters.csv  metrics.json  figures/*.jpg  log.txt  config.yaml  env.json

set -euo pipefail
cd "$(dirname "$0")"
if [[ -f env.sh ]]; then source env.sh; else source env.example.sh; fi
PY="${INSID3_PYTHON:-python}"
STAGE="${STAGE:-}"
ROOT="$INSID3_OUTPUT_ROOT/diagnosis"
RUNS="$ROOT/diag_baseline"
mkdir -p "$ROOT/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$ROOT/logs/run_diagnosis.log"; }
require_cuda() {
  "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "ERROR: CUDA is not available to $PY (run scripts/check_env.py)."; exit 1; }
}
diag() {   # diag <name> [arguments of scripts/diagnose_segmentation.py]
  local name="$1"; shift
  log "diagnose $name"
  "$PY" scripts/diagnose_segmentation.py --config configs/baseline.yaml --exp-name diag_baseline \
      "$@" 2>&1 | tee -a "$ROOT/logs/$name.log"
}

stage_check() {
  "$PY" -m pytest tests/test_diagnosis.py tests/test_baseline_equivalence.py -q
  "$PY" scripts/verify_upstream.py
  require_cuda
  "$PY" scripts/diagnose_segmentation.py --config configs/baseline.yaml --exp-name diag_smoke \
      --verify-episodes 5 --viz-auto 2 \
      --set data.split=tune data.dataset=coco data.tune_episodes=1000 run.max_episodes=5 \
            run.output_root="$ROOT" run.resume=false
}
stage_coco_tune() {
  require_cuda
  diag coco_tune --verify-episodes 100 --viz-auto 12 \
      --set data.split=tune data.dataset=coco data.tune_episodes=1000 run.output_root="$ROOT"
}
stage_pascal() {
  require_cuda
  diag pascal --folds all --verify-episodes 25 --viz-auto 4 --set data.dataset=pascal_part run.output_root="$ROOT"
}
stage_coco_test() {
  require_cuda
  diag coco_test --folds all --verify-episodes 25 --viz-auto 4 --set data.dataset=coco run.output_root="$ROOT"
}
stage_pascal5() {
  require_cuda
  diag pascal5 --folds 1 --verify-episodes 25 --viz-auto 10 \
      --set data.dataset=pascal_part data.shots=5 run.output_root="$ROOT"
}
stage_report() {
  local args=()
  [[ -d "$RUNS/coco-tune/fold0_shot1_seed0" ]] && args+=(--run "COCO tune=$RUNS/coco-tune/fold0_shot1_seed0"
                                                       --reference "COCO tune=results/round1/fail_spectral_k500_g4.csv")
  compgen -G "$RUNS/pascal_part/fold*_shot1_seed0" > /dev/null && args+=(--run "PASCAL-Part test=$RUNS/pascal_part/fold*_shot1_seed0")
  compgen -G "$RUNS/coco/fold*_shot1_seed0" > /dev/null && args+=(--run "COCO-20i test=$RUNS/coco/fold*_shot1_seed0")
  [[ -d "$RUNS/pascal_part/fold1_shot5_seed0" ]] && args+=(--run "PASCAL-Part f1 5-shot=$RUNS/pascal_part/fold1_shot5_seed0")
  [[ ${#args[@]} -gt 0 ]] || { echo "no diagnosis runs found under $RUNS"; exit 1; }
  "$PY" scripts/analyze_diagnosis.py --out results/diagnosis "${args[@]}"
}
stage_pack() {
  local tmp; tmp="$(mktemp -d)"
  mkdir -p "$tmp/diagnosis_results"
  cp -r results/diagnosis "$tmp/diagnosis_results/report"
  for d in "$RUNS"/*/fold*_shot*_seed0; do
    [[ -d "$d" ]] || continue
    rel="${d#"$RUNS"/}"
    mkdir -p "$tmp/diagnosis_results/runs/$rel"
    cp "$d"/metrics.json "$d"/log.txt "$d"/config.yaml "$d"/env.json "$tmp/diagnosis_results/runs/$rel/" 2>/dev/null || true
    [[ -d "$d/figures" ]] && cp -r "$d/figures" "$tmp/diagnosis_results/runs/$rel/"
  done
  # full per-episode / per-cluster logs only for the COCO tuning run (the others are large)
  t="$RUNS/coco-tune/fold0_shot1_seed0"
  if [[ -d "$t" ]]; then
    gzip -c "$t/episodes.jsonl" > "$tmp/diagnosis_results/runs/coco-tune/fold0_shot1_seed0/episodes.jsonl.gz"
    gzip -c "$t/clusters.csv" > "$tmp/diagnosis_results/runs/coco-tune/fold0_shot1_seed0/clusters.csv.gz"
  fi
  cp "$ROOT/logs/"*.log "$tmp/diagnosis_results/" 2>/dev/null || true
  tar -czf diagnosis_results.tar.gz -C "$tmp" diagnosis_results
  rm -rf "$tmp"
  log "wrote $(pwd)/diagnosis_results.tar.gz ($(du -h diagnosis_results.tar.gz | cut -f1))"
  if [[ $(stat -c %s diagnosis_results.tar.gz) -gt 29000000 ]]; then
    log "WARNING: larger than 29 MB; send report/ and the COCO-tune logs first, figures separately"
  fi
}

case "$STAGE" in
  check) stage_check ;;
  coco_tune) stage_coco_tune ;;
  pascal) stage_pascal ;;
  coco_test) stage_coco_test ;;
  pascal5) stage_pascal5 ;;
  report) stage_report ;;
  pack) stage_pack ;;
  all) stage_check; stage_coco_tune; stage_pascal; stage_coco_test; stage_report; stage_pack ;;
  *) sed -n '2,19p' "$0"; exit 1 ;;
esac
