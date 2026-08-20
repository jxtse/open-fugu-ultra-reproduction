#!/bin/bash
set -euo pipefail
B=/vePFS-Mindverse/share/richard/open-fugu-ultra
R=$B/runs/phase-b-grpo-infra-safe
cd $B/repos/prime-rl
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=$B/uv-cache
export WANDB_MODE=offline
if [ -f $B/.env ]; then set -a; . $B/.env; set +a; fi
exec uv run rl @ $R/rl.toml --output-dir $R/output --ckpt.resume-step -1 > $B/logs/phase-b-grpo-infra-safe.log 2>&1
