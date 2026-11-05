"""Held-out eval runner: frozen tasks.json -> start-state setup -> agent loop (same format as training) -> checker."""
import base64, io, json, os, re, sys, time, subprocess, argparse, shutil
from openai import OpenAI
sys.path.insert(0, os.path.dirname(__file__))
import agent as A
from env import DesktopEnv
from checkers import CHECKERS

def model_size(w, h):
    if (w, h) == (1280, 800): return 1260, 784          # identical to training
    return round(w / 28) * 28, round(h / 28) * 28

def system_prompt(mw, mh):
    return A.SYSTEM.replace("1260x784", f"{mw}x{mh}")   # identical text at 1280x800

def parse(text, sx, sy):
    try: a = A.parse(text)  # maps with 1280/1260, 800/784; undo and redo with real scale
    except Exception: return None   # malformed (e.g. list coords) -> invalid_output
    if not isinstance(a, dict) or "action" not in a: return None
    if a:
        for k, f in (("x", sx), ("y", sy)):
            if k + "_model" in a: a[k] = round(float(a[k + "_model"]) * f)
    return a

BASHRC = 'PROMPT_COMMAND=\'history 1 | sed "s/^ *[0-9]* *//" >> ~/.cmdlog\'\n'

def setup(env, st, task_dir):
    """Build the frozen start state. Returns first screenshot."""
    env.w, env.h = st.get("w", 1280), st.get("h", 800)
    env.env["SCREEN_W"], env.env["SCREEN_H"] = str(env.w), str(env.h)
    term = st.get("term", [40, 40])
    # files are created before the desktop starts (reset wipes workdir, so write after rmtree)
    env.close(); time.sleep(2)
    for _ in range(5):
        shutil.rmtree(env.workdir, ignore_errors=True)
        if not os.path.exists(env.workdir): break
        time.sleep(2)
    os.makedirs(env.workdir, exist_ok=True)
    open(f"{env.workdir}/.bashrc", "w").write(BASHRC)
    for name in st.get("files", []):
        shutil.copy(f"{task_dir}/files/{name}", f"{env.workdir}/{name}")
    _reset_keep(env, term)
    if st.get("jupyter_open"):
        env._spawn(["jupyter", "lab"]); _wait_title(env, "JupyterLab", 40); time.sleep(4)
    if st.get("filebrowser"):
        env._spawn(["thunar", env.workdir]); time.sleep(3)
    if st.get("distractor"):
        env._spawn(["thunar", "/usr/share"]); time.sleep(3)
        env.xdo("search", "--sync", "--name", "share", "windowmove", "%@", 300, 150)
    if st.get("popup"):
        env._spawn(["xmessage", "-center", "-buttons", "Dismiss:0",
                    "System notice: a software update is available. Click Dismiss to continue."]); time.sleep(2)
    time.sleep(1)
    return env.screenshot()

def _wait_title(env, t, n):
    for _ in range(n):
        if env.xdo("search", "--name", t).stdout.strip(): return
        time.sleep(1)

def _reset_keep(env, term):
    """Like DesktopEnv.reset but does not wipe workdir; term=None -> no terminal."""
    n = env.display.lstrip(":")
    subprocess.run(["pkill", "-f", f"Xvfb {env.display}"]); time.sleep(0.5)
    for f in (f"/tmp/.X{n}-lock", f"/tmp/.X11-unix/X{n}"):
        try: os.remove(f)
        except FileNotFoundError: pass
    env._spawn(["Xvfb", env.display, "-screen", "0", f"{env.w}x{env.h}x24", "-nolisten", "tcp"]); time.sleep(1.5)
    env._spawn(["xfwm4", "--compositor=off"]); time.sleep(1)
    subprocess.run(["xsetroot", "-solid", "#2d4a6b"], env=env.env)
    if term is not None:
        env._spawn(["xfce4-terminal", "--disable-server", f"--geometry=110x30+{term[0]}+{term[1]}",
                    "--working-directory", env.workdir, "--title", "Terminal"]); time.sleep(2)

def run_task(t, out, base_url, display, workdir, video=False, model="qwen25vl"):
    os.makedirs(f"{out}/screens", exist_ok=True)
    tdir = globals().get("TASK_DIR") or os.path.dirname(os.path.abspath(__file__)) + "/../tasks"
    cli = OpenAI(base_url=base_url, api_key=os.environ.get("VLLM_API_KEY", "x"), timeout=600)
    env = DesktopEnv(display=display, workdir=workdir)
    img = setup(env, t["start"], tdir)
    mw, mh = model_size(env.w, env.h); sx, sy = env.w / mw, env.h / mh; system = system_prompt(mw, mh)
    if video: env.start_video(f"{out}/run.mp4")
    hist, log, final = [], open(f"{out}/steps.jsonl", "w"), "max_steps"
    for i in range(t["max_steps"]):
        sp = f"{out}/screens/step_{i:02d}.png"; img.save(sp)
        buf = io.BytesIO(); img.convert("RGB").resize((mw, mh)).save(buf, "PNG")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": A.user_text(t["instruction"], hist, i)}]}]
        t0 = time.time()
        r = cli.chat.completions.create(model=model, messages=msgs, max_tokens=300, temperature=float(os.environ.get("EVAL_TEMPERATURE", "0.0")))
        raw = r.choices[0].message.content; a = parse(raw, sx, sy)
        rec = {"step": i, "screenshot": sp, "raw": raw, "action": a, "latency_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if a is None: hist.append({"action": "invalid_output"}); img = env.screenshot(); continue
        hist.append(A.hist_entry(a))
        if a.get("action") in ("done", "fail"): final = a["action"]; break
        try: env.step(a)
        except Exception as e: rec["exec_error"] = str(e)
        img = env.screenshot()
    time.sleep(3); env.screenshot(f"{out}/screens/final.png"); env.stop_video()
    chk = CHECKERS[t["checker"]["type"]](workdir, **t["checker"]["args"])
    res = {"task": t["id"], "tier": t["tier"], "final": final, "steps": len(hist), "success": chk["success"], "check": chk}
    json.dump(res, open(f"{out}/result.json", "w"), indent=1, default=str)
    env.close(); return res

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_tag", required=True); p.add_argument("--base_url", required=True)
    p.add_argument("--display", default=":97"); p.add_argument("--only", default=None)
    p.add_argument("--video_tasks", default="")
    p.add_argument("--sandbox", default=None); p.add_argument("--reverse", action="store_true")
    a = p.parse_args()
    root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    tasks = json.load(open(f"{root}/tasks/tasks.json"))
    if a.reverse: tasks = tasks[::-1]
    for t in tasks:
        if a.only and t["id"] not in a.only.split(","): continue
        out = f"{root}/runs/{a.model_tag}/{t['id']}"
        if os.path.exists(f"{out}/result.json") or (a.reverse and os.path.exists(out)): continue   # resumable; reverse worker skips claimed
        res = run_task(t, out, a.base_url, a.display, a.sandbox or f"/workspace/eval_heldout/sb_{a.model_tag}",
                       video=t["id"] in a.video_tasks.split(","))
        print(json.dumps({k: res[k] for k in ("task", "tier", "final", "steps", "success")}), flush=True)
