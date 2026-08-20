"""Replay/verify open-fugu-env traces: rewards, workflow validity, worker routing."""

import collections
import json
import sys


def main() -> None:
    path = sys.argv[1]
    rows = [json.loads(line) for line in open(path)]
    print("episodes:", len(rows))
    reward_hist = collections.Counter()
    for row in rows:
        info = row.get("info", {})
        reward = (
            row.get("rewards", {}).get("conductor_reward", {}).get("score")
        )
        reward_hist[reward] += 1
        workflow = info.get("workflow") or {}
        steps = workflow.get("steps", [])
        models = [call.get("model") for call in row.get("calls", [])]
        print(
            f"- reward={reward} valid={info.get('workflow_valid')}"
            f" steps={len(steps)} workers={[s.get('worker') for s in steps]}"
            f" models={models}"
            f" err={str(info.get('workflow_error'))[:60]}"
            f" ans={str(info.get('workflow_answer'))[:40]}"
        )
    print("reward histogram:", dict(reward_hist))


if __name__ == "__main__":
    main()
