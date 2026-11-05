"""Stage 4 checkers = Stage-3 checkers + 'test line intact': the original print/test line must still be present, unchanged, in the fixed file."""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from checkers import CHECKERS as BASE, notebook_function, py_script
def _nb_src(workdir, notebook):
    p = f"{workdir}/{notebook}"
    if not os.path.exists(p): return ""
    return "\n".join("".join(c["source"]) for c in json.load(open(p))["cells"] if c["cell_type"] == "code")
def notebook_function_intact(workdir, test_line, **kw):
    r = notebook_function(workdir, **kw); src = _nb_src(workdir, kw["notebook"])
    r["test_line_intact"] = test_line in [l.rstrip() for l in src.splitlines()]
    r["success"] = bool(r["success"]) and r["test_line_intact"]; return r
def py_script_intact(workdir, test_line, **kw):
    r = py_script(workdir, **kw); p = f"{workdir}/{kw['file']}"
    r["test_line_intact"] = os.path.exists(p) and test_line in [l.rstrip() for l in open(p).read().splitlines()]
    r["success"] = bool(r["success"]) and r["test_line_intact"]; return r
CHECKERS = dict(BASE, notebook_function_intact=notebook_function_intact, py_script_intact=py_script_intact)
