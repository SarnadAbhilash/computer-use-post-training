#!/usr/bin/env python3
"""Build GRPO training rows with signed weights from rollout result folders.

Paths below are local defaults from the original training machine — point them
at your own rollout/data directories (or set them via argv) before running.
Requires a sibling reward helper `select_3b.reward_3b` on PYTHONPATH.

Reward r = success - 0.5*false_done - 0.05*invalid - 0.2*steps/max_steps + 0.1*verified_before_done
Advantage A = (r - mean_group) / (std_group + 0.1); uniform groups dropped.
weight = clip(A / mean|A|, -1, 2), negatives x NEG_SCALE (0.5). + REPLAY_FRAC of SFT rows.
Usage: grpo_select.py <tag> <run> [sft_jsonl_for_replay]"""
import json, glob, os, sys, random, statistics, collections
sys.path.insert(0, "/workspace/stage3/rl"); from select_3b import reward_3b
from PIL import Image
NEG = float(os.environ.get("NEG_SCALE", "0.5")); REP = float(os.environ.get("REPLAY_FRAC", "0.2"))
T = {t["id"]: t for p in ("/workspace/stage4/tasks_train", "/workspace/stage3/train_tasks") for t in json.load(open(f"{p}/tasks.json"))}
def load(tag):
    G = collections.defaultdict(list)
    for f in glob.glob(f"/workspace/stage4/rollouts/{tag}/*/g*/result.json"):
        r = json.load(open(f)); d = os.path.dirname(f); st = [json.loads(l) for l in open(f"{d}/steps.jsonl")]
        r["r"], _ = reward_3b(r, st, T[r["task"]]["max_steps"]); G[r["task"]].append((r, d, st))
    return G
def stats(tag):
    rs = [r for g in load(tag).values() for r, _, _ in g]; n = max(1, len(rs))
    return {"tag": tag, "n": len(rs), "success_rate": round(sum(r["success"] for r in rs) / n, 4), "mean_reward": round(sum(r["r"] for r in rs) / n, 4),
            "false_done_rate": round(sum(r["final"] == "done" and not r["success"] for r in rs) / n, 4)}
if __name__ == "__main__":
    tag, run = sys.argv[1], sys.argv[2]; rep_src = sys.argv[3] if len(sys.argv) > 3 else "/workspace/stage4/data/sft_v4.jsonl"
    O = "/workspace/stage4/data"; os.makedirs(f"{O}/{run}/img", exist_ok=True); rows, ng = [], 0
    for task, g in sorted(load(tag).items()):
        rs = [r["r"] for r, _, _ in g]
        if len(rs) < 2 or statistics.pstdev(rs) < 1e-6: continue
        mu, sd = statistics.mean(rs), statistics.pstdev(rs); ng += 1
        for r, d, st in g:
            A = (r["r"] - mu) / (sd + 0.1)
            if abs(A) < 0.05: continue
            for s in st:
                if s["action"] is None: continue
                name = f"{task}_{os.path.basename(d)}_{s['step']:02d}.png"
                Image.open(s["screenshot"]).convert("RGB").resize((1260, 784)).save(f"{O}/{run}/img/{name}")
                rows.append({"image": f"{run}/img/{name}", "system": s["system"], "user": s["user"], "target": s["raw"], "adv": A, "demo": d})
    m = sum(abs(r["adv"]) for r in rows) / max(1, len(rows))
    for r in rows: r["weight"] = round(max(-1.0, min(2.0, r["adv"] / m)) * (NEG if r["adv"] < 0 else 1.0), 3)
    random.seed(1); rep = [json.loads(l) for l in open(rep_src)]; rep = random.sample(rep, min(len(rep), max(1, int(len(rows) * REP))))
    with open(f"{O}/{run}/{run}.jsonl", "w") as f:
        for r in rows + rep: f.write(json.dumps(r) + "\n")
    s = {"tag": tag, "run": run, "practice": stats(tag), "groups_with_signal": ng, "pos_rows": sum(r["weight"] > 0 for r in rows), "neg_rows": sum(r["weight"] < 0 for r in rows), "replay_rows": len(rep)}
    json.dump(s, open(f"{O}/{run}/select.json", "w"), indent=1); print(json.dumps(s))
