# Open Fugu-Ultra reproduction: policy-collapse evidence bundle

This is a **private handoff bundle** for independently inspecting and rerunning our
minimal Open Fugu-Ultra / Conductor-style reproduction. It is not a polished public
release.

## What we observed

A 100-step GRPO run on Qwen3.5-9B improved matched held-out answer accuracy from
40.0% to 81.7%, but the gain is almost entirely workflow-format reliability rather
than better orchestration:

- Correctness conditioned on a valid workflow: 96.0% (base) vs 96.1% (trained).
- Base valid workflows: 50/120, all 50 structurally unique and multi-step.
- Trained valid workflows: 102/120; 101 are one-step, 100 use the same generic
  subtask, 102/103 worker calls choose `worker_a`, and `worker_c` disappears.
- During training, routing entropy falls from 1.57 bits in steps 1–20 to 0.28 bits
  in steps 86–100.

Interpretation: with homogeneous math tasks, a strong universal worker, binary
answer reward, and no reward for decomposition/routing/cost, the shortest valid
workflow is a rational objective exploit. The run demonstrates format learning and
policy collapse, not learned task-conditional multi-agent orchestration.

## Recompute the evidence

Python 3.11+ is enough for the analysis:

```bash
uv run python analysis/analyze_policy_collapse.py
```

Expected output is checked in at `analysis/expected_output.txt`.

Key evidence:

- `runs/heldout-eval/base-v1/traces.jsonl`: frozen matched base evaluation, 120 traces.
- `runs/heldout-eval/step100-v1/traces.jsonl`: frozen trained evaluation, 120 traces.
- `evidence/train_rollouts_steps1-100.tar.gz`: all/effective training traces for all
  100 optimizer steps (78 MB compressed).
- `runs/heldout-eval/base-structured-v2/`: an incomplete follow-up structured-base
  probe (47/120 traces); do not substitute it for the matched comparison.

## Code and exact versions

- `environments/open_fugu_env/`: the custom Verifiers environment, workflow parser,
  reward, manifest loaders, scripts, and tests.
- `patches/*.commit`: exact upstream revisions of Prime-RL, Verifiers, and rlm-harness.
- `patches/*.patch` and `prime-rl-new-*.py`: local changes needed by the run.
- `runs/phase-b-grpo-infra-safe/`: training configuration and launch script.
- `data/`: frozen training, Phase A, and held-out manifests plus provenance summaries.

Clone the upstream projects, check out the revisions in `patches/`, apply the patches,
and install `environments/open_fugu_env/`. The checked-in run files preserve our
machine paths and model/service names as provenance; adapt them to your cluster.
Worker credentials are read from `OPENFUGU_WORKER_KEY` and are not included.

The original run used 8×A800 80 GB: four training GPUs and four rollout GPUs. The
coordinator was Qwen3.5-9B with thinking enabled. The three frozen workers were
frontier endpoints. All training data came from DAPO-Math-17k.

## Suggested independent checks

1. Confirm the checked-in metrics from raw traces.
2. Inspect the step-window transition and identify the first stable collapse point.
3. Check whether prompt leakage from the one-step schema example accelerates collapse.
4. Rerun with the same reward but without the generic one-step example.
5. Rerun on heterogeneous tasks/workers and score regret against each task's best
   directly measured worker. Do not reward diversity for its own sake.
6. Test whether a cost penalty makes the one-step policy even more dominant.

## Important limitations

- The trained checkpoint is not included: it is about 100+ GB and should be transferred
  separately only if needed. Raw training and evaluation traces are included.
- Frontier worker endpoints/models may not be independently available; substitute
  workers will test the mechanism but will not be an exact numerical replication.
- This is one 100-step seed on a homogeneous math distribution.
- Do not summarize the result as “trained orchestration beats base by 41.7 points”
  without the conditional-correctness and collapse diagnostics above.

## Upstream projects

- Prime-RL: https://github.com/PrimeIntellect-ai/prime-rl
- Verifiers: https://github.com/PrimeIntellect-ai/verifiers
- rlm-harness: https://github.com/PrimeIntellect-ai/rlm-harness
- DAPO-Math-17k: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k
