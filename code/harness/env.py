"""Sandboxed GUI env on Xvfb for computer-use agents (reusable for SFT data collection / RL rollouts)."""
import os, subprocess, time, shutil, signal
from PIL import ImageGrab

class DesktopEnv:
    def __init__(self, display=":99", w=1280, h=800, workdir=None):
        # Local path defaults can be overridden with WORKDIR / VENV_BIN / BROWSER env vars.
        workdir = workdir or os.environ.get("WORKDIR", os.path.abspath("./sandbox"))
        venv_bin = os.environ.get("VENV_BIN", "")
        browser = os.environ.get("BROWSER", "google-chrome")
        self.display, self.w, self.h, self.workdir = display, w, h, workdir
        self.procs = []
        path = (venv_bin + os.pathsep) if venv_bin else ""
        path += os.environ.get("PATH", "")
        self.env = dict(os.environ, DISPLAY=display, HOME=workdir, PATH=path, BROWSER=browser)
        # Never forward cloud tokens into the sandboxed desktop process.
        for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "RUNPOD_API_KEY", "RUNPOD_API_KEY_2", "VLLM_API_KEY"):
            self.env.pop(k, None)

    def _spawn(self, cmd):
        p = subprocess.Popen(cmd, env=self.env, cwd=self.workdir, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        self.procs.append(p); return p

    def reset(self, term_xy=(40, 40)):
        self.close()
        shutil.rmtree(self.workdir, ignore_errors=True); os.makedirs(self.workdir)
        n = self.display.lstrip(":")
        subprocess.run(["pkill", "-f", f"Xvfb {self.display}"]); time.sleep(0.5)
        for f in (f"/tmp/.X{n}-lock", f"/tmp/.X11-unix/X{n}"):
            try: os.remove(f)
            except FileNotFoundError: pass
        self._spawn(["Xvfb", self.display, "-screen", "0", f"{self.w}x{self.h}x24", "-nolisten", "tcp"]); time.sleep(1.5)
        self._spawn(["xfwm4", "--compositor=off"]); time.sleep(1)
        subprocess.run(["xsetroot", "-solid", "#2d4a6b"], env=self.env)
        self._spawn(["xfce4-terminal", "--disable-server", f"--geometry=110x30+{term_xy[0]}+{term_xy[1]}",
                     "--working-directory", self.workdir, "--title", "Terminal"]); time.sleep(2)
        return self.screenshot()

    def screenshot(self, path=None):
        img = ImageGrab.grab(xdisplay=self.display)
        if path: img.save(path)
        return img

    def xdo(self, *args):
        return subprocess.run(["xdotool", *map(str, args)], env=self.env, capture_output=True, text=True)

    def step(self, a):
        t = a.get("action")
        if t in ("click", "double_click", "right_click"):
            self.xdo("mousemove", int(a["x"]), int(a["y"]))
            self.xdo("click", *(["--repeat", 2] if t == "double_click" else []), 3 if t == "right_click" else 1)
        elif t == "type":
            self.xdo("type", "--delay", 20, "--", a["text"])
        elif t == "key":
            self.xdo("key", "--", *a["keys"].replace(" ", "").split(",") if "," in a["keys"] else [a["keys"].strip()])
        elif t == "scroll":
            if "x" in a: self.xdo("mousemove", int(a["x"]), int(a["y"]))
            self.xdo("click", "--repeat", abs(int(a.get("amount", 3))), 5 if a.get("direction", "down") == "down" else 4)
        elif t == "wait":
            time.sleep(float(a.get("seconds", 2)))
        time.sleep(1.5)

    def start_video(self, path, fps=8):
        self.vid = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", str(fps),
            "-video_size", f"{self.w}x{self.h}", "-i", self.display, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", path], stdin=subprocess.PIPE)

    def stop_video(self):
        if getattr(self, "vid", None):
            try: self.vid.communicate(b"q", timeout=20)
            except Exception: self.vid.kill()
            self.vid = None

    def close(self):
        for p in self.procs:
            try: os.killpg(p.pid, signal.SIGTERM)
            except Exception: pass
        self.procs = []
        # kill only processes belonging to THIS env (cwd or chrome profile inside workdir)
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or int(pid) == os.getpid(): continue
            try:
                cwd = os.readlink(f"/proc/{pid}/cwd"); cmd = open(f"/proc/{pid}/cmdline").read()
            except Exception: continue
            if cwd.startswith(self.workdir) or f"user-data-dir={self.workdir}" in cmd:
                try: os.kill(int(pid), signal.SIGTERM)
                except Exception: pass
