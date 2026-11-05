"""Agent loop: screenshot -> Qwen2.5-VL (vLLM/OpenAI API) -> JSON action -> xdotool. Logs steps.jsonl."""
import base64, io, json, os, re, sys, time, argparse
from openai import OpenAI
sys.path.insert(0, os.path.dirname(__file__))
from env import DesktopEnv

# Screen 1280x800 is resized to 1260x784 (multiples of 28) so Qwen's absolute coords == pixels of the image it sees.
MW, MH = 1260, 784
SYSTEM = f"""You are a computer-use agent controlling a Linux desktop (screenshot is {MW}x{MH} pixels).
Each turn, output exactly ONE action as a JSON object on a single line, inside <action></action> tags. You may first write one short sentence of reasoning.
Actions (x,y are absolute pixel coordinates in the screenshot):
{{"action":"click","x":int,"y":int}}
{{"action":"double_click","x":int,"y":int}}
{{"action":"right_click","x":int,"y":int}}
{{"action":"type","text":"string (use \\n for Enter)"}}
{{"action":"key","keys":"xdotool keysym combo e.g. Return, ctrl+s, shift+Return, ctrl+a"}}
{{"action":"scroll","x":int,"y":int,"direction":"up|down","amount":int}}
{{"action":"wait","seconds":number}}
{{"action":"done"}}  (task completed)
{{"action":"fail"}}  (task impossible)"""

def b64(img):
    buf = io.BytesIO(); img.convert("RGB").resize((MW, MH)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def parse(text):
    m = re.search(r"<action>(.*?)</action>", text, re.S) or re.search(r"\{.*\}", text, re.S)
    if not m: return None
    s = m.group(1) if m.re.pattern.startswith("<") else m.group(0)
    try: a = json.loads(s.strip())
    except Exception:
        try: a = json.loads(re.search(r"\{.*\}", s, re.S).group(0))
        except Exception: return None
    for k, f in (("x", 1280 / MW), ("y", 800 / MH)):
        if k in a: a[k + "_model"] = a[k]; a[k] = round(float(a[k]) * f)
    return a

def user_text(task, hist, i):
    h = "\n".join(f"{j}: {json.dumps(x)}" for j, x in enumerate(hist[-15:])) or "(none)"
    return f"Task: {task}\nPrevious actions:\n{h}\nStep {i}. Output the next action."

def hist_entry(a):
    return {k: v for k, v in a.items() if not k.endswith("_model")}

def run(task, out, base_url, model="qwen25vl", max_steps=30, term_xy=(40, 40), video=None, display=":99", workdir="/workspace/eval_heldout/sandbox"):
    os.makedirs(f"{out}/screens", exist_ok=True)
    cli = OpenAI(base_url=base_url, api_key=os.environ.get("VLLM_API_KEY", "x"), timeout=600)
    env = DesktopEnv(display=display, workdir=workdir); img = env.reset(term_xy)
    if video: env.start_video(video)
    hist = []; log = open(f"{out}/steps.jsonl", "w")
    final = "max_steps"
    for i in range(max_steps):
        sp = f"{out}/screens/step_{i:02d}.png"; img.save(sp)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(img)}},
                    {"type": "text", "text": user_text(task, hist, i)}]}]
        t0 = time.time()
        r = cli.chat.completions.create(model=model, messages=msgs, max_tokens=300, temperature=0.0)
        raw = r.choices[0].message.content; a = parse(raw)
        rec = {"step": i, "screenshot": sp, "raw": raw, "action": a, "latency_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec)[:400], flush=True)
        if a is None: hist.append({"action": "invalid_output"}); img = env.screenshot(); continue
        hist.append(hist_entry(a))
        if a.get("action") in ("done", "fail"): final = a["action"]; break
        try: env.step(a)
        except Exception as e: rec["exec_error"] = str(e)
        img = env.screenshot()
    time.sleep(3); env.screenshot(f"{out}/screens/final.png"); env.stop_video()
    from judge import judge
    res = {"final": final, "steps": len(hist), "judge": judge(env.workdir)}
    json.dump(res, open(f"{out}/result.json", "w"), indent=1); env.close()
    print("RESULT", json.dumps(res)[:600]); return res

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", default="/workspace/stage1")
    p.add_argument("--base_url", required=True); p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--model", default="qwen25vl"); p.add_argument("--term_x", type=int, default=40)
    p.add_argument("--term_y", type=int, default=40); p.add_argument("--video", default=None)
    p.add_argument("--display", default=":99"); p.add_argument("--workdir", default="/workspace/eval_heldout/sandbox")
    p.add_argument("--task", default="Open a coding environment (Jupyter or VS Code), write a Python function that adds two numbers, run it, and fix any errors.")
    a = p.parse_args(); run(a.task, a.out, a.base_url, a.model, a.max_steps, (a.term_x, a.term_y), a.video, a.display, a.workdir)
