#!/usr/bin/env bash
# 프롬프트 버전 실험 실행 스크립트
# 사용법:  bash scripts/run_prompt_experiment.sh [dataset] [judge_model]
#   dataset 기본값: eval/data/golden/subset_nokg.jsonl (30문항, 크롤링 데이터 불필요)
#   전체 160문항: eval/data/golden/laneige_golden_v2.jsonl (data/amore_data.db, chroma/ 필요)
# 사전 조건: .env에 OPENAI_API_KEY, config/feature_flags.json의 use_centralized_prompts=true
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
DATASET="${1:-eval/data/golden/subset_nokg.jsonl}"
JUDGE_MODEL="${2:-gpt-4.1-mini}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.1}"   # 전 버전 동일 고정
STAMP="$(date +%Y%m%d_%H%M)"
OUT="eval_output/exp_${STAMP}"
mkdir -p "$OUT"

# 0) 스위치 확인
if ! grep -q '"use_centralized_prompts": true' config/feature_flags.json; then
  echo "[중단] config/feature_flags.json 의 use_centralized_prompts 를 true 로 바꾸십시오." >&2; exit 1
fi
if [ -z "${OPENAI_API_KEY:-}" ] && ! grep -q '^OPENAI_API_KEY=sk-' .env 2>/dev/null; then
  echo "[중단] OPENAI_API_KEY 가 없습니다 (.env 또는 환경변수)." >&2; exit 1
fi
cp prompts/agents/chatbot_system.txt "$OUT/chatbot_system.original.txt"   # 원본 백업

run_version () {
  local v="$1"
  echo "==== $v ===="
  cp "prompts/agents/variants/chatbot_system_${v}.txt" prompts/agents/chatbot_system.txt
  python eval/cli.py run --dataset "$DATASET" --judge llm --judge-model "$JUDGE_MODEL" \
    --top-k 8 --concurrency 1 --save-traces --out "$OUT/$v" 2>&1 | tee "$OUT/$v.log" | tail -25
}

run_version v0
python eval/cli.py set-baseline --name "v0_${STAMP}" --report "$OUT/v0/report.json" || echo "[경고] set-baseline 실패 — 수동 비교로 진행"
for v in v1 v2 v3 v4; do
  run_version "$v"
  python eval/cli.py compare --baseline-name "v0_${STAMP}" --report "$OUT/$v/report.json" > "$OUT/$v/compare_vs_v0.md" 2>&1 || echo "[경고] compare 실패($v)"
done

cp "$OUT/chatbot_system.original.txt" prompts/agents/chatbot_system.txt   # 원본 복구
python scripts/summarize_prompt_experiment.py "$OUT" | tee "$OUT/RESULTS.md"
echo "완료: $OUT/RESULTS.md 를 docs/experiments/ 로 복사해 커밋하십시오."
