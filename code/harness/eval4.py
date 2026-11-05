"""Stage 4 eval runner (model served anywhere with an OpenAI-compatible vLLM endpoint).
  --suite old24 : frozen 24-task held-out set (tests/original-24)
  --suite x35   : frozen 35-task debugging set (tests/debug-35)
  --repeat k    : repeat index (runs/<suite>/<model_tag>/r<k>/...); use EVAL_TEMPERATURE>0 (e.g. 0.2) for k>0 so repeats differ.
Env: VLLM_BASE_URL, VLLM_API_KEY (optional), EVAL_TEMPERATURE, EVAL_ROOT (optional output root)."""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import heldout
from checkers4 import CHECKERS
heldout.CHECKERS = CHECKERS
SUITES = {
    "old24": os.path.join(REPO, "tests", "original-24"),
    "x35": os.path.join(REPO, "tests", "debug-35"),
}
def verify(d):
    r = subprocess.run("sha256sum -c --quiet FROZEN.sha256", shell=True, cwd=d, capture_output=True, text=True)
    if r.returncode: sys.exit(f"FROZEN check failed in {d}: {r.stdout}{r.stderr}")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=SUITES, required=True); p.add_argument("--model_tag", required=True)
    p.add_argument("--base_url", default=os.environ.get("VLLM_BASE_URL")); p.add_argument("--display", default=":94")
    p.add_argument("--repeat", type=int, default=0); p.add_argument("--reverse", action="store_true"); p.add_argument("--video_tasks", default="")
    p.add_argument("--only", default=None)
    p.add_argument("--eval_root", default=os.environ.get("EVAL_ROOT", os.path.join(REPO, "runs")))
    p.add_argument("--sandbox_root", default=os.environ.get("SANDBOX_ROOT", os.path.join(REPO, "sandboxes")))
    a = p.parse_args(); d = SUITES[a.suite]; verify(d)
    tasks = json.load(open(f"{d}/tasks.json"))[:: -1 if a.reverse else 1]
    root = f"{a.eval_root}/{a.suite}/{a.model_tag}/r{a.repeat}"
    for t in tasks:
        if a.only and t["id"] not in a.only.split(","): continue
        out = f"{root}/{t['id']}"
        if os.path.exists(f"{out}/result.json"): continue
        try: os.makedirs(f"{out}/.claim")            # atomic claim -> any number of workers can share a suite
        except FileExistsError: continue
        t = dict(t, tier=t.get("tier", "T2X"))
        # heldout.setup copies start files from <task_dir>/files: point it at this suite
        heldout.TASK_DIR = d
        sandbox = f"{a.sandbox_root}/sb{a.display.lstrip(':')}"
        res = heldout.run_task(t, out, a.base_url, a.display, sandbox, video=t["id"] in a.video_tasks.split(","))
        print(json.dumps({k: res[k] for k in ("task", "final", "steps", "success")}), flush=True)
