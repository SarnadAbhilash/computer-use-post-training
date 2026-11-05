"""Deterministic state checkers. They inspect files on disk and re-execute code on HIDDEN inputs; model claims are ignored."""
import json, glob, os, subprocess, sys

def _exec_tests(src, func, tests, timeout=20):
    prog = src + "\n\nimport json as _j\n_r=[]\nfor _a,_e in " + repr(tests) + ":\n" \
        "    try:\n        _v=" + func + "(*_a); _r.append(bool(_v==_e if not isinstance(_e,float) else abs(_v-_e)<1e-6))\n" \
        "    except Exception as _x: _r.append(False)\nprint('__T__'+_j.dumps(_r))"
    try:
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=timeout, cwd="/tmp")
    except subprocess.TimeoutExpired: return [False] * len(tests), "timeout"
    line = [l for l in r.stdout.splitlines() if l.startswith("__T__")]
    return (json.loads(line[-1][5:]) if line else [False] * len(tests)), r.stderr[-300:]

def _cells(nb):
    return [c for c in json.load(open(nb)).get("cells", []) if c["cell_type"] == "code"]

def _out_text(c):
    return "".join("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", "")) for o in c.get("outputs", []))

def notebook_function(workdir, func, tests, expected_output, notebook=None, strict_name=False):
    """A saved notebook defines `func`; it passes hidden tests; the defining cell was executed with no error
    and a printed output containing expected_output."""
    cands = [f"{workdir}/{notebook}"] if notebook and os.path.exists(f"{workdir}/{notebook}") else []
    cands += [f for f in glob.glob(f"{workdir}/*.ipynb") if f not in cands]
    best = {"success": False, "reason": "no notebook found", "files": [os.path.basename(f) for f in cands]}
    for nb in cands:
        cs = _cells(nb); src = "\n".join("".join(c["source"]) for c in cs)
        if f"def {func}" not in src: best.setdefault("seen", []).append(os.path.basename(nb)); continue
        ok, err = _exec_tests(src, func, tests)
        defc = [c for c in cs if f"def {func}" in "".join(c["source"])]
        executed = any(c.get("execution_count") for c in defc)
        errors = any(o.get("output_type") == "error" for c in cs if c.get("execution_count") for o in c.get("outputs", []))
        out_ok = any(expected_output in _out_text(c) for c in cs if c.get("execution_count"))
        r = {"file": os.path.basename(nb), "name_ok": notebook is None or os.path.basename(nb) == notebook,
             "hidden_tests": ok, "executed": executed, "error_in_outputs": errors, "expected_output_seen": out_ok,
             "stderr": err}
        r["success"] = all(ok) and executed and not errors and out_ok and (r["name_ok"] or not strict_name)
        if r["success"] or best.get("reason"): best = r
        if r["success"]: break
    return best

def _cmds(workdir):
    p = f"{workdir}/.cmdlog"
    return [l.strip() for l in open(p)] if os.path.exists(p) else []

def py_script(workdir, file, expected_stdout, func=None, tests=None, min_runs=1):
    """file exists; `python3 file` prints expected_stdout now; terminal log shows it was run (>= min_runs times)."""
    path = f"{workdir}/{file}"
    r = {"exists": os.path.exists(path)}
    if not r["exists"]: return {**r, "success": False}
    p = subprocess.run([sys.executable, file], capture_output=True, text=True, timeout=20, cwd=workdir)
    r["stdout_ok"] = expected_stdout in p.stdout; r["stdout"] = p.stdout[-200:]; r["stderr"] = p.stderr[-200:]
    runs = [c for c in _cmds(workdir) if ("python3 " in c or "python " in c) and file in c]
    r["runs_logged"] = len(runs)
    if func: r["hidden_tests"], _ = _exec_tests(open(path).read().replace("print(", "(lambda *a,**k:None)("), func, tests)
    r["success"] = r["stdout_ok"] and len(runs) >= min_runs and (not func or all(r["hidden_tests"]))
    return r

def text_file(workdir, file, must_contain):
    path = f"{workdir}/{file}"
    if not os.path.exists(path): return {"exists": False, "success": False}
    s = open(path).read(); return {"exists": True, "content": s[:200], "success": must_contain in s}

CHECKERS = {"notebook_function": notebook_function, "py_script": py_script, "text_file": text_file}
